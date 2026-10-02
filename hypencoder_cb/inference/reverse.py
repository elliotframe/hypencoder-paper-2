"""Reverse Hypencoder retrieval: a q-net per item, a vector per query.

The hypernetwork side of a HypencoderDualEncoder (normally the query encoder)
is run on every item at encoding time and the generated q-net parameters are
written to disk. At retrieval time each query is encoded with the vector side
of the model (normally the passage encoder) and fed into every item's q-net.

A q-net index is a directory containing:
    q_nets.bin: Raw array with shape (num_items, num_params) holding each
        item's flattened q-net matrices followed by its flattened vectors.
    items.jsonl: One line per item, in the same order, with "id" and "text".
    meta.json: The number of items and parameters, the storage dtype, and the
        matrix and vector shapes needed to rebuild the q-nets.

Every q-net is large (about 3.5M parameters, 7 MB in fp16, for the 6 layer
model), so this is only practical for small corpora.
"""

import json
import math
from pathlib import Path
from typing import Iterable, List, Protocol, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import BaseRetriever, Item, TextQuery
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.iterator_utils import BackgroundGenerator, batchify
from hypencoder_cb.utils.jsonl_utils import JsonlReader, JsonlWriter
from hypencoder_cb.utils.torch_utils import dtype_lookup

Q_NETS_FILE = "q_nets.bin"
ITEMS_FILE = "items.jsonl"
META_FILE = "meta.json"

STORAGE_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


class QNetParameterEncoder(Protocol):
    def batch_encode(
        self, texts: List[str]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]: ...


def flatten_q_net_params(
    matrices: List[torch.Tensor], vectors: List[torch.Tensor]
) -> torch.Tensor:
    """Flattens per-item q-net matrices and vectors into one row per item.

    Args:
        matrices (List[torch.Tensor]): Tensors with shape (bs, ...).
        vectors (List[torch.Tensor]): Tensors with shape (bs, ...).

    Returns:
        torch.Tensor: Shape (bs, num_params).
    """
    return torch.cat([t.flatten(1) for t in matrices + vectors], dim=1)


def unflatten_q_net_params(
    params: torch.Tensor,
    matrix_shapes: List[List[int]],
    vector_shapes: List[List[int]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Inverse of `flatten_q_net_params`.

    Args:
        params (torch.Tensor): Shape (bs, num_params).
        matrix_shapes (List[List[int]]): Per-item shape of each matrix.
        vector_shapes (List[List[int]]): Per-item shape of each vector.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: The matrices and
            vectors, each with a leading batch dimension.
    """
    shapes = matrix_shapes + vector_shapes
    chunks = torch.split(params, [math.prod(s) for s in shapes], dim=1)
    tensors = [
        chunk.reshape(params.size(0), *shape)
        for chunk, shape in zip(chunks, shapes)
    ]
    return tensors[: len(matrix_shapes)], tensors[len(matrix_shapes) :]


def encode_q_nets_to_disk(
    encoder: QNetParameterEncoder,
    items: Iterable[Item],
    output_dir: str,
    batch_size: int = 32,
    storage_dtype: str = "fp16",
) -> None:
    """Generates a q-net for every item and writes them as a q-net index.

    Args:
        encoder (QNetParameterEncoder): Returns the q-net matrices and vectors
            for a batch of texts.
        items (Iterable[Item]): The items to encode.
        output_dir (str): Directory to write the q-net index to.
        batch_size (int, optional): The batch size used while encoding.
            Defaults to 32.
        storage_dtype (str, optional): The dtype the parameters are stored
            in, either "fp16" or "fp32". Defaults to "fp16".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = STORAGE_DTYPES[storage_dtype]

    meta = None
    num_items = 0
    with open(output_dir / Q_NETS_FILE, "wb") as param_file, JsonlWriter(
        output_dir / ITEMS_FILE
    ) as item_writer, tqdm() as pbar:
        for batch in BackgroundGenerator(batchify(items, batch_size), 10):
            matrices, vectors = encoder.batch_encode(
                [item.text for item in batch]
            )

            if meta is None:
                meta = {
                    "matrix_shapes": [list(m.shape[1:]) for m in matrices],
                    "vector_shapes": [list(v.shape[1:]) for v in vectors],
                    "storage_dtype": storage_dtype,
                }
                meta["num_params"] = sum(
                    math.prod(s)
                    for s in meta["matrix_shapes"] + meta["vector_shapes"]
                )
                item_mb = (
                    meta["num_params"]
                    * (torch.finfo(torch_dtype).bits // 8)
                    / 1024**2
                )
                print(
                    f"Each q-net has {meta['num_params']:,} parameters"
                    f" ({item_mb:.1f} MiB per item on disk)."
                )

            params = flatten_q_net_params(matrices, vectors)
            param_file.write(params.to(torch_dtype).cpu().numpy().tobytes())

            for item in batch:
                item_writer.write({"id": item.id, "text": item.text})

            num_items += len(batch)
            pbar.update(len(batch))

    meta["num_items"] = num_items
    with open(output_dir / META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


class QNetIndex:
    def __init__(self, index_dir: str) -> None:
        """Read-only view of a q-net index written by `encode_q_nets_to_disk`.

        The parameters are memory mapped, so they are read from disk as each
        batch is requested rather than loaded into RAM up front.

        Args:
            index_dir (str): Directory containing the q-net index.

        Raises:
            FileNotFoundError: If `index_dir` is not a q-net index.
        """
        index_dir = Path(index_dir)
        if not (index_dir / META_FILE).exists():
            raise FileNotFoundError(
                f"{index_dir} is not a q-net index (no {META_FILE}). Encode"
                " with `encode.py --representation_type=q_net` first."
            )

        with open(index_dir / META_FILE) as f:
            meta = json.load(f)

        self.matrix_shapes = meta["matrix_shapes"]
        self.vector_shapes = meta["vector_shapes"]
        self.num_items = meta["num_items"]

        self.params = np.memmap(
            index_dir / Q_NETS_FILE,
            dtype={"fp16": np.float16, "fp32": np.float32}[
                meta["storage_dtype"]
            ],
            mode="r",
            shape=(self.num_items, meta["num_params"]),
        )

        self.ids = []
        self.texts = []
        with JsonlReader(index_dir / ITEMS_FILE) as reader:
            for line in reader:
                self.ids.append(line["id"])
                self.texts.append(line["text"])

    def __len__(self) -> int:
        return self.num_items

    def iter_batches(
        self, batch_size: int, pin_memory: bool = False
    ) -> Iterable[Tuple[int, torch.Tensor]]:
        """Reads the parameters in order, prefetching in a background thread.

        Args:
            batch_size (int): Number of items per batch.
            pin_memory (bool, optional): Whether to pin each batch for faster
                copies to the GPU. Defaults to False.

        Yields:
            Tuple[int, torch.Tensor]: The index of the first item in the batch
                and its flattened parameters with shape (bs, num_params).
        """

        def read():
            for start in range(0, self.num_items, batch_size):
                batch = np.array(self.params[start : start + batch_size])
                batch = torch.from_numpy(batch)
                yield start, batch.pin_memory() if pin_memory else batch

        return BackgroundGenerator(read(), 2)


class HypencoderReverseRetriever(BaseRetriever):
    implements_retrieve_batch = True

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        item_batch_size: int = 64,
        query_batch_size: int = 1024,
        device: str = "cuda",
        dtype: Union[torch.dtype, str] = "fp32",
        query_max_length: int = 32,
        ignore_same_id: bool = False,
    ) -> None:
        """Scores each query vector with every item's q-net.

        Work is done item-major: each batch of q-nets is read from disk once
        and applied to all queries, so a whole query set costs one pass over
        the index.

        Args:
            model_name_or_path (str): Name or path to a HypencoderDualEncoder
                checkpoint, the same one used to build the q-net index.
            encoded_item_path (str): Path to the q-net index directory.
            item_batch_size (int, optional): Number of item q-nets applied at
                once. Defaults to 64.
            query_batch_size (int, optional): Number of query vectors fed to
                each batch of q-nets at once. Defaults to 1024.
            device (str, optional): The device to use. Defaults to "cuda".
            dtype (Union[torch.dtype, str], optional): The dtype for the model
                and q-nets. Options are "fp16", "fp32", and "bf16". Defaults
                to "fp32".
            query_max_length (int, optional): Maximum length of the query.
                Defaults to 32.
            ignore_same_id (bool, optional): Whether to ignore retrievals
                with the same ID as the query. Defaults to False.
        """
        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.item_batch_size = item_batch_size
        self.query_batch_size = query_batch_size
        self.query_max_length = query_max_length
        self.ignore_same_id = ignore_same_id

        model = HypencoderDualEncoder.from_pretrained(model_name_or_path)
        self.vector_encoder = (
            model.passage_encoder.to(device, dtype=self.dtype).eval()
        )
        self.converter = model.query_encoder.weight_to_model_converter
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.index = QNetIndex(encoded_item_path)

    def _encode_queries(self, queries: List[TextQuery]) -> torch.Tensor:
        embeddings = []
        for batch in batchify(queries, self.query_batch_size):
            tokenized = self.tokenizer(
                [query.text for query in batch],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.query_max_length,
            ).to(self.device)

            with torch.no_grad():
                embeddings.append(
                    self.vector_encoder(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                    ).representation
                )

        return torch.cat(embeddings, dim=0)

    def retrieve_batch(
        self, queries: List[TextQuery], top_k: int
    ) -> List[List[Item]]:
        query_embeddings = self._encode_queries(queries)
        num_queries = len(queries)

        # One extra so dropping the query's own item still leaves top_k.
        k = min(top_k + int(self.ignore_same_id), len(self.index))
        top_scores = torch.full(
            (num_queries, k), -float("inf"), device=self.device
        )
        top_indices = torch.full(
            (num_queries, k), -1, dtype=torch.long, device=self.device
        )

        with torch.no_grad(), tqdm(
            total=len(self.index), desc="Scoring items"
        ) as pbar:
            for start, params in self.index.iter_batches(
                self.item_batch_size,
                pin_memory=torch.device(self.device).type == "cuda",
            ):
                params = params.to(
                    self.device, dtype=self.dtype, non_blocking=True
                )
                num_items = params.size(0)
                matrices, vectors = unflatten_q_net_params(
                    params, self.index.matrix_shapes, self.index.vector_shapes
                )
                q_nets = self.converter(matrices, vectors, is_training=False)
                item_indices = torch.arange(
                    start, start + num_items, device=self.device
                )

                for q_start in range(0, num_queries, self.query_batch_size):
                    q_end = q_start + self.query_batch_size
                    batch_queries = query_embeddings[q_start:q_end]

                    # Each item's q-net sees every query in the batch:
                    # (num_items, num_queries, dim) -> (num_queries, items).
                    scores = (
                        q_nets(
                            batch_queries.unsqueeze(0).expand(
                                num_items, -1, -1
                            )
                        )
                        .squeeze(-1)
                        .transpose(0, 1)
                        .float()
                    )

                    merged_scores, merged_positions = torch.topk(
                        torch.cat([top_scores[q_start:q_end], scores], dim=1),
                        k,
                        dim=1,
                    )
                    merged_indices = torch.cat(
                        [
                            top_indices[q_start:q_end],
                            item_indices.expand(scores.size(0), -1),
                        ],
                        dim=1,
                    ).gather(1, merged_positions)

                    top_scores[q_start:q_end] = merged_scores
                    top_indices[q_start:q_end] = merged_indices

                pbar.update(num_items)

        results = []
        for query, scores, indices in zip(
            queries, top_scores.cpu().tolist(), top_indices.cpu().tolist()
        ):
            items = []
            for item_idx, score in zip(indices, scores):
                item_id = self.index.ids[item_idx]
                if self.ignore_same_id and item_id == query.id:
                    continue

                items.append(
                    Item(
                        text=self.index.texts[item_idx],
                        id=item_id,
                        score=score,
                        type="hypencoder_reverse_retriever",
                    )
                )
            results.append(items[:top_k])

        return results

    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        return self.retrieve_batch([query], top_k)[0]
