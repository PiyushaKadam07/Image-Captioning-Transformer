"""Greedy and beam-search caption generation with n-gram blocking."""
from __future__ import annotations

import torch


def _banned_tokens(seq: list[int], n: int) -> set[int]:
    """Tokens that would complete an n-gram already present in seq."""
    if n <= 0 or len(seq) < n:
        return set()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    return banned


@torch.no_grad()
def generate(model, feats: torch.Tensor, bos_id: int, eos_id: int, beam_size: int = 3,
             max_len: int = 30, length_penalty: float = 1.0, no_repeat_ngram: int = 3,
             return_attention: bool = False):
    """Generate one caption per image.

    feats: projected image features (B, R, H). Returns list of token-id lists
    (without BOS/EOS) and, if requested, per-token cross-attention maps over regions.
    """
    B, R, H = feats.shape
    k = beam_size
    device = feats.device
    feats_k = feats.repeat_interleave(k, dim=0)                     # (B*k, R, H)
    seqs = torch.full((B * k, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(B, k, device=device)
    scores[:, 1:] = -1e9                                            # start from a single live beam
    finished = torch.zeros(B * k, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model.decode_logits(seqs, feats_k).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)                    # (B*k, V)
        V = logp.size(-1)
        for row in range(B * k):
            if finished[row]:
                logp[row] = -1e9
                logp[row, eos_id] = 0.0                             # finished beams only extend with EOS
            else:
                banned = _banned_tokens(seqs[row, 1:].tolist(), no_repeat_ngram)
                if banned:
                    logp[row, list(banned)] = -1e9
        cand = (scores.view(-1, 1) + logp).view(B, k * V)
        top_scores, top_idx = cand.topk(k, dim=-1)
        beam_idx = top_idx // V
        token_idx = top_idx % V
        base = (torch.arange(B, device=device) * k).unsqueeze(1)
        rows = (base + beam_idx).view(-1)
        seqs = torch.cat([seqs[rows], token_idx.view(-1, 1)], dim=1)
        finished = finished[rows] | (token_idx.view(-1) == eos_id)
        scores = top_scores
        if finished.all():
            break

    # Rank finished hypotheses with a length penalty so short captions are not unfairly favoured.
    results, best_rows = [], []
    for b in range(B):
        best, best_val = None, -float("inf")
        for j in range(k):
            row = b * k + j
            toks = seqs[row, 1:].tolist()
            if eos_id in toks:
                toks = toks[: toks.index(eos_id)]
            length = max(len(toks), 1)
            val = scores[b, j].item() / (length ** length_penalty)
            if val > best_val:
                best, best_val, best_row = toks, val, row
        results.append(best)
        best_rows.append(best_row)

    if not return_attention:
        return results

    # Re-run the chosen captions once to collect cross-attention (averaged over layers and heads).
    attn_maps = []
    for b, toks in enumerate(results):
        ids = torch.tensor([[bos_id] + toks], device=device)
        out = model.decode_logits(ids, feats[b:b + 1], output_attentions=True)
        stacked = torch.stack(out.cross_attentions)                 # (layers, 1, heads, T, R)
        attn_maps.append(stacked.mean(dim=(0, 2))[0, : len(toks)].cpu())  # (T, R)
    return results, attn_maps
