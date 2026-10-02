"""Throwaway diagnostics for reverse retrieval on a small corpus.

python scripts/diagnose_reverse.py <model> <q_net_index_dir> <ir_dataset_name>
"""

import sys

import ir_datasets
import ir_measures
import torch
from ir_measures import R, RR, nDCG

from hypencoder_cb.inference.reverse import (
    HypencoderReverseRetriever,
    unflatten_q_net_params,
)
from hypencoder_cb.inference.shared import TextQuery
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

model_name, index_dir, dataset_name = sys.argv[1:4]
ds = ir_datasets.load(dataset_name)
queries = [TextQuery(id=q.query_id, text=q.text) for q in ds.queries_iter()]
qrels = list(ds.qrels_iter())

r = HypencoderReverseRetriever(model_name, index_dir, query_max_length=512)
dual = HypencoderDualEncoder.from_pretrained(model_name).to(r.device).eval()
tok = r.tokenizer
ids, texts = r.index.ids, r.index.texts


def tokenize(batch, max_length=512):
    return tok(
        batch,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    ).to(r.device)


with torch.no_grad():
    q_emb = r._encode_queries(queries)  # (Q, 768)

    # Reverse score matrix from the stored index: S[q, d] = qnet_d(emb_q).
    cols = []
    for start, params in r.index.iter_batches(64):
        params = params.to(r.device, dtype=r.dtype)
        m, v = unflatten_q_net_params(
            params, r.index.matrix_shapes, r.index.vector_shapes
        )
        net = r.converter(m, v, is_training=False)
        n = params.size(0)
        cols.append(net(q_emb.unsqueeze(0).expand(n, -1, -1)).squeeze(-1).T)
    S = torch.cat(cols, 1).float().cpu()

    # 1. Bug check: stock Hypencoder forward on the first docs, no index.
    t = tokenize(texts[:8])
    net = dual.query_encoder(t["input_ids"], t["attention_mask"]).representation
    direct = net(q_emb.unsqueeze(0).expand(8, -1, -1)).squeeze(-1).T.cpu()
    print(
        "[bug check] max |index - direct| ="
        f" {(S[:, :8] - direct).abs().max():.4g}"
        f" (score scale {S.abs().mean():.4g})"
    )

    # Forward (normal) score matrix for comparison: F[q, d] = qnet_q(emb_d).
    d_emb = []
    for i in range(0, len(texts), 128):
        t = tokenize(texts[i : i + 128])
        d_emb.append(
            dual.passage_encoder(
                t["input_ids"], t["attention_mask"]
            ).representation
        )
    d_emb = torch.cat(d_emb)
    rows = []
    for q in queries:
        t = tokenize([q.text])
        qn = dual.query_encoder(t["input_ids"], t["attention_mask"])
        rows.append(qn.representation(d_emb.unsqueeze(0)).view(-1))
    F = torch.stack(rows).float().cpu()

qid_pos = {q.id: i for i, q in enumerate(queries)}
did_pos = {d: i for i, d in enumerate(ids)}
rel = torch.zeros_like(S, dtype=torch.bool)
for qr in qrels:
    if qr.relevance > 0 and qr.query_id in qid_pos and qr.doc_id in did_pos:
        rel[qid_pos[qr.query_id], did_pos[qr.doc_id]] = True


def evaluate(name, M):
    top = M.topk(min(1000, M.size(1)), dim=1)
    run = {
        q.id: {ids[j]: s for s, j in zip(vals.tolist(), idx.tolist())}
        for q, vals, idx in zip(queries, top.values, top.indices)
    }
    res = ir_measures.calc_aggregate([nDCG @ 10, RR, R @ 1000], qrels, run)
    print(f"[{name}] " + ", ".join(f"{k}={v:.4f}" for k, v in res.items()))


def per_doc_auc(M):
    """How well each doc's column separates its relevant queries."""
    ranks = M.argsort(0).argsort(0).float()
    n_rel = rel.sum(0).float()
    n_non = rel.size(0) - n_rel
    ok = (n_rel > 0) & (n_non > 0)
    pairs = (ranks * rel).sum(0) - n_rel * (n_rel - 1) / 2
    return (pairs[ok] / (n_rel[ok] * n_non[ok])).mean().item()


def per_query_auc(M):
    """How well each query's row separates its relevant docs."""
    ranks = M.argsort(1).argsort(1).float()
    n_rel = rel.sum(1).float()
    n_non = rel.size(1) - n_rel
    ok = (n_rel > 0) & (n_non > 0)
    pairs = (ranks * rel).sum(1) - n_rel * (n_rel - 1) / 2
    return (pairs[ok] / (n_rel[ok] * n_non[ok])).mean().item()


def zscore_cols(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-6)


for name, M in [("forward", F), ("reverse", S)]:
    between = M.mean(0).std().item()  # spread of per-doc mean score
    within = M.std(0).mean().item()  # spread across queries for one doc
    top1 = M.argmax(1).unique().numel()
    print(
        f"\n== {name} ==\n"
        f"per-doc mean score std (doc prior) = {between:.4g}\n"
        f"mean per-doc std across queries   = {within:.4g}\n"
        f"distinct top-1 docs over {M.size(0)} queries = {top1}\n"
        f"AUC within each query (ranking docs)  = {per_query_auc(M):.4f}\n"
        f"AUC within each doc (ranking queries) = {per_doc_auc(M):.4f}"
    )
    evaluate(f"{name} raw", M)
    evaluate(f"{name} z-scored per doc", zscore_cols(M))
