"""Throwaway experiment: how small can reverse-retrieval q-nets get?

Compresses every item's stored q-net several ways, then runs calibrated
reverse retrieval with each and reports effectiveness against bytes per item.

python scripts/compress_reverse.py --model_name_or_path=$MODEL \
    --encoded_item_path=$WORK/qnet_index --output_path=$WORK/compression.json

Each large (768x768) q-net matrix M is split into the corpus mean matrix
(shared, stored once) plus a per-item part D = M - mean, which is compressed:
    svd: per-item truncated SVD of D, storing two (768, r) factors.
    shared: D projected onto a corpus-wide PCA basis for whichever side
        captures more energy, storing one (768, r) factor.
    tucker: D projected onto corpus-wide PCA bases on both sides, storing an
        (r, r) core.
The small matrices and bias vectors are always stored in full. With int8,
every stored tensor is quantized symmetrically with one fp16 scale per vector
along its longest axis.
"""

import json
import math
import random
from typing import Dict, Sequence

import fire
import ir_datasets
import ir_measures
import torch
from ir_measures import R, RR, nDCG

from hypencoder_cb.inference.reverse import (
    HypencoderReverseRetriever,
    load_query_texts,
    unflatten_q_net_params,
)
from hypencoder_cb.inference.shared import TextQuery
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

MSMARCO_PASSAGES = 8_841_823
ENERGY_LEVELS = (0.9, 0.99, 0.999)


def quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """Simulated int8 round trip, one scale per vector along the longest
    non-batch axis."""
    dim = 1 if x.size(1) >= x.size(2) else 2
    scale = x.abs().amax(dim, keepdim=True).clamp_min(1e-12) / 127
    return (x / scale).round().clamp(-127, 127) * scale


def stored_bytes(shape: Sequence[int], int8: bool) -> int:
    n = math.prod(shape)
    if not int8:
        return 2 * n
    return n + 2 * (n // max(shape))


def rank_for_energy(values: torch.Tensor, level: float) -> int:
    """Smallest rank whose top values hold `level` of the total energy."""
    energy = values.sort(descending=True).values.clamp_min(0)
    cumulative = energy.cumsum(0) / energy.sum()
    return int((cumulative < level).sum().item()) + 1


def maybe_quantize(x: torch.Tensor, int8: bool) -> torch.Tensor:
    return quantize_int8(x) if int8 else x


def main(
    model_name_or_path: str,
    encoded_item_path: str,
    output_path: str,
    ir_dataset_name: str = "beir/nfcorpus/test",
    calibration_queries: str = "beir/nfcorpus/train",
    num_calibration_queries: int = 1000,
    ranks: Sequence[int] = (1, 4, 16, 64),
    tucker_ranks: Sequence[int] = (16, 64, 256),
    methods: Sequence[str] = ("svd", "shared", "tucker"),
    query_max_length: int = 512,
    item_batch_size: int = 32,
    seed: int = 0,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True

    r = HypencoderReverseRetriever(
        model_name_or_path,
        encoded_item_path,
        item_batch_size=item_batch_size,
        query_max_length=query_max_length,
        use_calibration=False,
    )
    index, device = r.index, r.device
    projections = HypencoderDualEncoder.from_pretrained(
        model_name_or_path
    ).query_encoder.weight_hyper_projection

    # The (768, 768) matrices; the final (768, 1) one is stored in full.
    big = [i for i, s in enumerate(index.matrix_shapes) if min(s) > 1]

    # ---- Generator projection spectra (no data needed). ----
    print("\n== Hypernetwork projection spectra (rank for energy level) ==")
    for i, proj in enumerate(projections):
        sv2 = torch.linalg.svdvals(proj.weight.detach().double()) ** 2
        print(
            f"layer {i}: "
            + ", ".join(
                f"{lvl:.1%}: {rank_for_energy(sv2, lvl)}"
                for lvl in ENERGY_LEVELS
            )
        )

    # ---- Pass 1: corpus mean and second moments of each big matrix. ----
    n = len(index)
    sums = {i: 0 for i in big}
    left = {i: 0 for i in big}
    right = {i: 0 for i in big}
    for _, params in index.iter_batches(item_batch_size):
        params = params.to(device, dtype=torch.float64)
        matrices, _ = unflatten_q_net_params(
            params, index.matrix_shapes, index.vector_shapes
        )
        for i in big:
            M = matrices[i]
            sums[i] = sums[i] + M.sum(0)
            left[i] = left[i] + torch.einsum("bio,bjo->ij", M, M)
            right[i] = right[i] + torch.einsum("bio,bip->op", M, M)

    means, bases = {}, {}
    print("\n== Per-item part of each matrix (corpus PCA) ==")
    for i in big:
        mean = sums[i] / n
        cov_l = left[i] - n * mean @ mean.T
        cov_r = right[i] - n * mean.T @ mean
        eval_l, vec_l = torch.linalg.eigh(cov_l)
        eval_r, vec_r = torch.linalg.eigh(cov_r)
        eval_l, vec_l = eval_l.flip(0), vec_l.flip(1)
        eval_r, vec_r = eval_r.flip(0), vec_r.flip(1)
        means[i] = mean.float()
        bases[i] = (vec_l.float(), vec_r.float(), eval_l, eval_r)
        frac = (cov_l.trace() / left[i].trace()).item()
        print(
            f"layer {i}: per-item share of weight energy {frac:.2%};"
            " left basis "
            + ", ".join(
                f"{lvl:.1%}: {rank_for_energy(eval_l, lvl)}"
                for lvl in ENERGY_LEVELS
            )
            + "; right basis "
            + ", ".join(
                f"{lvl:.1%}: {rank_for_energy(eval_r, lvl)}"
                for lvl in ENERGY_LEVELS
            )
        )

    # ---- Configurations. ----
    configs = [{"method": "full", "rank": None}]
    for method in methods:
        for rank in tucker_ranks if method == "tucker" else ranks:
            configs.append({"method": method, "rank": rank})
    configs = [dict(c, int8=q) for c in configs for q in (False, True)]

    def compress(config, i, M, svd):
        """Returns the reconstruction and stored shapes for big matrix i."""
        method, k, q = config["method"], config["rank"], config["int8"]
        if method == "full":
            return maybe_quantize(M, q), [list(M.shape[1:])]

        D = M - means[i]
        vec_l, vec_r, eval_l, eval_r = bases[i]
        if method == "svd":
            U, S, Vh = svd[i]
            a = maybe_quantize(U[:, :, :k] * S[:, None, :k], q)
            b = maybe_quantize(Vh[:, :k, :], q)
            recon = a @ b
            shapes = [[M.size(1), k], [k, M.size(2)]]
        elif method == "shared":
            if eval_l[:k].sum() >= eval_r[:k].sum():
                L = vec_l[:, :k]
                f = maybe_quantize(L.T @ D, q)
                recon = L @ f
                shapes = [[k, M.size(2)]]
            else:
                Rb = vec_r[:, :k]
                f = maybe_quantize(D @ Rb, q)
                recon = f @ Rb.T
                shapes = [[M.size(1), k]]
        elif method == "tucker":
            L, Rb = vec_l[:, :k], vec_r[:, :k]
            core = maybe_quantize(L.T @ D @ Rb, q)
            recon = L @ core @ Rb.T
            shapes = [[k, k]]
        else:
            raise ValueError(method)
        return means[i] + recon, shapes

    # ---- Queries. ----
    ds = ir_datasets.load(ir_dataset_name)
    test_queries = [
        TextQuery(id=q.query_id, text=q.text) for q in ds.queries_iter()
    ]
    calib_texts = load_query_texts(calibration_queries)
    calib_texts = random.Random(seed).sample(
        calib_texts, min(num_calibration_queries, len(calib_texts))
    )
    embeddings = r._encode_queries(
        test_queries + [TextQuery(text=t) for t in calib_texts]
    )
    num_test = len(test_queries)

    # ---- Pass 2: compress, score, and measure reconstruction error. ----
    scores = [[] for _ in configs]
    err = [0.0 for _ in configs]
    per_item_energy = 0.0
    item_bytes = [None for _ in configs]
    with torch.no_grad():
        for start, params in index.iter_batches(item_batch_size):
            params = params.to(device, dtype=torch.float32)
            matrices, vectors = unflatten_q_net_params(
                params, index.matrix_shapes, index.vector_shapes
            )
            svd = {}
            if "svd" in methods:
                svd = {
                    i: torch.linalg.svd(
                        matrices[i] - means[i], full_matrices=False
                    )
                    for i in big
                }
            per_item_energy += sum(
                ((matrices[i] - means[i]) ** 2).sum().item() for i in big
            )

            for c, config in enumerate(configs):
                q = config["int8"]
                new_matrices, shapes = [], []
                for i, M in enumerate(matrices):
                    if i in big:
                        recon, s = compress(config, i, M, svd)
                        err[c] += ((recon - M) ** 2).sum().item()
                    else:
                        recon, s = maybe_quantize(M, q), [list(M.shape[1:])]
                    new_matrices.append(recon)
                    shapes += s
                new_vectors = [maybe_quantize(v, q) for v in vectors]
                shapes += [list(v.shape[1:]) for v in vectors]
                item_bytes[c] = sum(stored_bytes(s, q) for s in shapes)

                q_nets = r.converter(
                    new_matrices, new_vectors, is_training=False
                )
                num_items = params.size(0)
                scores[c].append(
                    torch.cat(
                        [
                            q_nets(e.unsqueeze(0).expand(num_items, -1, -1))
                            .squeeze(-1)
                            .T.float()
                            .cpu()
                            for e in torch.split(embeddings, 512)
                        ]
                    )
                )
            print(f"scored items {start + params.size(0)}/{n}", end="\r")

    # ---- Evaluate each configuration with calibration. ----
    qrels = list(ds.qrels_iter())

    def evaluate(S: torch.Tensor) -> Dict[str, float]:
        test, calib = S[:num_test], S[num_test:]
        z = (test - calib.mean(0)) / calib.std(0).clamp_min(1e-6)
        top = z.topk(min(1000, z.size(1)), dim=1)
        run = {
            query.id: {
                index.ids[j]: s for s, j in zip(v.tolist(), ix.tolist())
            }
            for query, v, ix in zip(test_queries, top.values, top.indices)
        }
        res = ir_measures.calc_aggregate([nDCG @ 10, RR, R @ 1000], qrels, run)
        return {str(k): v for k, v in res.items()}

    results = []
    print()
    header = (
        f"{'method':8} {'rank':>5} {'int8':>5} {'KB/item':>9}"
        f" {'MSMARCO':>9} {'err':>7} {'nDCG@10':>8} {'RR':>7} {'R@1000':>7}"
    )
    print(header)
    for c, config in enumerate(configs):
        metrics = evaluate(torch.cat(scores[c], dim=1))
        row = dict(
            config,
            bytes_per_item=item_bytes[c],
            msmarco_tb=item_bytes[c] * MSMARCO_PASSAGES / 1e12,
            relative_error=err[c] / per_item_energy,
            **metrics,
        )
        results.append(row)
        print(
            f"{config['method']:8} {str(config['rank']):>5}"
            f" {str(config['int8']):>5} {row['bytes_per_item'] / 1024:9.1f}"
            f" {row['msmarco_tb']:8.2f}T {row['relative_error']:7.3f}"
            f" {metrics['nDCG@10']:8.4f} {metrics['RR']:7.4f}"
            f" {metrics['R@1000']:7.4f}"
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    print(
        "err = squared reconstruction error of the big matrices divided by"
        " the energy of their per-item part (0 = exact, 1 = mean only)."
        " Shared means and bases are stored once and not counted."
    )


if __name__ == "__main__":
    fire.Fire(main)
