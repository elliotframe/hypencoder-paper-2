import copy
import os
import pickle
from collections import defaultdict
from queue import PriorityQueue
from typing import Dict, List, Optional, Union

import fire
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.entry_points import (
    BM25EntryPoints,
    EntryPointSelector,
    RandomEntryPoints,
)
from hypencoder_cb.inference.retrieve import do_retrieval_shared
from hypencoder_cb.inference.shared import (
    BaseRetriever,
    Item,
    TextQuery,
    load_encoded_items_from_disk,
)
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.jsonl_utils import JsonlReader
from hypencoder_cb.utils.torch_utils import dtype_lookup


class HypecoderGraphRetriever(BaseRetriever):

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        item_neighbors_path: str,
        batch_size: int = 100_000,
        device: str = "cuda",
        query_max_length: int = 32,
        cache_file: Optional[str] = None,
        num_entry_points: int = 10_000,
        ncandidates: int = 50,
        max_iter: int = 16,
        early_stop: bool = True,
        dtype: Union[torch.dtype, str] = "float32",
        ignore_same_id: bool = False,
        entry_points: str = "random",
        bm25_index_path: Optional[str] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """

        Args:
            model_name_or_path (str): The HypencoderDualEncoder model to use,
                this should match the model used for encoding the items.
            encoded_item_path (str): The path to the encoded items.
            item_neighbors_path (str): The path to the item neighbors JSONL.
                Should have the keys "item_id" and "neighbors".
            batch_size (int, optional): The batch size to use for inference.
                Defaults to 100_000.
            device (str, optional): The device to use to store the embeddings
                and to run the model. Defaults to "cuda".
            query_max_length (int, optional): The maximum length of the query.
                Defaults to 32.
            cache_file (Optional[str], optional): If provided, the cache file
                to use for loading the encoded items and item neighbors. If
                the file does not exist, it will be created and the data will
                stored in it. Defaults to None.
            num_entry_points (int, optional): The number of initial entry
                points. This is equal to len(initial_candidates). Defaults
                to 10_000.
            ncandidates (int, optional): The number of candidates to explore
                as each step. Defaults to 50.
            max_iter (int, optional): The maximum number of candidate expansion
                steps to do. Defaults to 16.
            early_stop (bool, optional): Whether to stop early if no new
                candidates are added to the queue at a given step. Defaults
                to True.
            dtype (Union[torch.dtype, str], optional): The dtype to use for
                the model and embeddings. Defaults to "float32".
            ignore_same_id (bool, optional): Whether to ignore retrievals
                with the same ID as the query. This is only relevant for
                certain datasets. Defaults to False.
            entry_points (str, optional): How entry points are chosen.
                "random" uses the same random items for every query. "bm25"
                uses the query's top BM25 results, falling back to the random
                items when BM25 matches nothing. Defaults to "random".
            bm25_index_path (Optional[str], optional): Directory of the PISA
                index used when `entry_points` is "bm25". It is built from the
                encoded items if it does not exist. Defaults to None.
            bm25_k1 (float, optional): BM25 k1. Defaults to 1.5.
            bm25_b (float, optional): BM25 b. Defaults to 0.75.
        """

        if entry_points not in ("random", "bm25"):
            raise ValueError(f"Unknown entry_points type: {entry_points}")

        if entry_points == "bm25" and bm25_index_path is None:
            raise ValueError(
                "bm25_index_path must be provided when entry_points is 'bm25'."
            )

        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.encoded_item_path = encoded_item_path
        self.num_entry_points = num_entry_points
        self.ncandidates = ncandidates
        self.max_iter = max_iter
        self.query_max_length = query_max_length
        self.early_stop = early_stop
        self.ignore_same_id = ignore_same_id

        print(model_name_or_path)
        self.model = (
            HypencoderDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading from cache {cache_file}")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)

                self.ids = cache["item_ids"]
                self.encoded_item_embeddings = cache["encoded_item_embeddings"]
                self.item_id_to_index = cache["item_id_to_index"]
                self.item_id_to_content = cache["item_id_to_content"]
                self.item_neighbor_ids = cache["item_neighbor_ids"]
                self.item_id_to_neighbor_indices = cache[
                    "item_id_to_neighbor_indices"
                ]

        else:
            self.encoded_items = load_encoded_items_from_disk(
                encoded_item_path
            )

            self.encoded_item_embeddings = torch.stack(
                [
                    torch.tensor(x.representation)
                    for x in tqdm(
                        self.encoded_items, desc="Item Embeddings to Tensor"
                    )
                ]
            ).to(self.device, dtype=self.dtype)

            self.item_id_to_index = {
                item.id: idx
                for idx, item in tqdm(
                    enumerate(self.encoded_items), desc="Item ID to Index"
                )
            }

            self.ids = [item.id for item in self.encoded_items]

            self.item_id_to_content = {
                item.id: item.text
                for item in tqdm(self.encoded_items, desc="Item ID to Content")
            }

            with JsonlReader(item_neighbors_path) as reader:
                self.item_neighbor_ids = {
                    line["item_id"]: line["neighbors"]
                    for line in tqdm(reader, desc="Loading Item Graph")
                }

            self.item_id_to_neighbor_indices = defaultdict(list)

            for item_id, neighbors in tqdm(
                self.item_neighbor_ids.items(),
                desc="Building Neighbor Indices",
            ):
                self.item_id_to_neighbor_indices[item_id] = [
                    self.item_id_to_index[neighbor] for neighbor in neighbors
                ]

            if cache_file is not None:
                print(f"Caching to {cache_file}")
                cache_values = {
                    "item_ids": self.ids,
                    "encoded_item_embeddings": self.encoded_item_embeddings,
                    "item_id_to_index": self.item_id_to_index,
                    "item_id_to_content": self.item_id_to_content,
                    "item_neighbor_ids": self.item_neighbor_ids,
                    "item_id_to_neighbor_indices": self.item_id_to_neighbor_indices,
                }

                with open(cache_file, "wb") as f:
                    pickle.dump(cache_values, f)

        self.random_entry_points = RandomEntryPoints(
            self.ids, self.num_entry_points
        )

        self.entry_point_selector: EntryPointSelector = (
            self.random_entry_points
        )
        if entry_points == "bm25":
            self.entry_point_selector = BM25EntryPoints(
                index_path=bm25_index_path,
                items=self.item_id_to_content.items(),
                num_entry_points=self.num_entry_points,
                k1=bm25_k1,
                b=bm25_b,
            )

    def _get_entry_point_ids(self, query: TextQuery) -> List[str]:
        entry_point_ids = self.entry_point_selector.select(query)

        unknown_ids = [
            item_id
            for item_id in entry_point_ids
            if item_id not in self.item_id_to_index
        ]
        if unknown_ids:
            raise ValueError(
                f"Entry points {unknown_ids[:5]} are not in the encoded items."
                " Was the BM25 index built from a different corpus?"
            )

        if not entry_point_ids:
            return self.random_entry_points.select(query)

        return entry_point_ids

    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            query_output = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            )
            query_model = query_output.representation

        final_queue = PriorityQueue(maxsize=top_k)

        candidates = self._get_entry_point_ids(query)
        explored = set(candidates)

        curr_iter = 0
        while curr_iter < self.max_iter:
            candidate_embeddings = self.encoded_item_embeddings[
                [self.item_id_to_index[x] for x in candidates]
            ]
            candidate_embeddings = candidate_embeddings.unsqueeze(0)
            similarity_matrix = query_model(candidate_embeddings).view(-1)

            ncandidates = min(
                max(self.ncandidates, top_k), similarity_matrix.shape[0]
            )
            values, indices = torch.topk(similarity_matrix, ncandidates, dim=0)

            indices = indices.view(-1).cpu()
            values = values.view(-1).cpu()

            prev_candidates = copy.deepcopy(candidates)
            candidates = []

            added_candidates = 0
            added_to_queue = 0
            for i, idx in enumerate(indices):
                idx = idx.item()

                item_id = prev_candidates[idx]
                score = similarity_matrix[idx].item()

                if final_queue.full():
                    # If queue is full, only add item to queue if score is
                    # greater than the minimum score in the queue
                    current_min = final_queue.get()
                    if score > current_min[0]:
                        added_to_queue += 1
                        final_queue.put((score, item_id))
                    else:
                        final_queue.put(current_min)
                else:
                    # If queue is not full add item to queue regardless of
                    # score
                    added_to_queue += 1
                    final_queue.put((score, item_id))

                if i < self.ncandidates:
                    for neighbor in self.item_neighbor_ids[item_id]:
                        if neighbor in explored:
                            continue

                        candidates.append(neighbor)
                        explored.add(neighbor)
                        added_candidates += 1

            if added_to_queue == 0 and self.early_stop:
                break

            curr_iter += 1

        items = []
        while not final_queue.empty():
            score, item_id = final_queue.get()
            if self.ignore_same_id and query.id == item_id:
                continue

            items.append(
                Item(
                    text=self.item_id_to_content[item_id],
                    id=item_id,
                    score=score,
                    type="hypecoder_graph_retriever",
                )
            )

        return sorted(items, key=lambda x: x.score, reverse=True)


def do_retrieval(
    model_name_or_path: str,
    encoded_item_path: str,
    item_neighbors_path: str,
    output_dir: str,
    num_entry_points: int = 10_000,
    ncandidates: int = 50,
    max_iter: int = 16,
    early_stop: bool = True,
    device: str = "cuda",
    cache_file: Optional[str] = None,
    ir_dataset_name: Optional[str] = None,
    query_jsonl: Optional[str] = None,
    qrel_json: Optional[str] = None,
    query_id_key: str = "id",
    query_text_key: str = "text",
    dtype: str = "fp32",
    top_k: int = 1000,
    batch_size: int = 100_000,
    retriever_kwargs: Optional[Dict] = None,
    query_max_length: int = 64,
    include_content: bool = True,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
    ignore_same_id: bool = False,
    entry_points: str = "random",
    bm25_index_path: Optional[str] = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> None:
    """Does retrieval and optionally evaluation.

    Args:
        model_name_or_path (str): Name or path to a HypencoderDualEncoder
            checkpoint.
        encoded_item_path (str): Path to the encoded items.
        item_neighbors_path (str): Path to the item neighbors JSONL.
            Should have the keys "item_id" and "neighbors".
        output_dir (str): Path to the output directory which will contain the
            retrieval results and optionally the evaluation results.
        num_entry_points (int, optional): The number of initial entry
            points. This is equal to len(initial_candidates). Defaults
            to 10_000.
        ncandidates (int, optional): The number of candidates to explore
            as each step. Defaults to 50.
        max_iter (int, optional): The maximum number of candidate expansion
            steps to do. Defaults to 16.
        early_stop (bool, optional): Whether to stop early if no new
            candidates are added to the queue at a given step. Defaults
            to True.
        device (str, optional): The device to use to store the embeddings
            and to run the model. Defaults to "cuda".
        cache_file (Optional[str], optional): If provided, the cache file
            to use for loading the encoded items and item neighbors. If
            the file does not exist, it will be created and the data will
            stored in it. Defaults to None.
        ir_dataset_name (Optional[str], optional): If provided is used to
            get the queries used for retrieval and qrels used for evaluation.
            If None, then `query_jsonl` must be provided and `qrel_json` must
            be provided if `do_eval` is True. Defaults to None.
        query_jsonl (Optional[str], optional): If provided is used as the
            queries for retrieval. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        qrel_json (Optional[str], optional): If provided is used as the qrels
            for evaluation. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        query_id_key (str, optional): The key in `query_jsonl` for the
            query ID. Not used if `ir_dataset_name` is provided. Defaults to
            "id".
        query_text_key (str, optional): The key in `query_jsonl` for the
            query text. Not used if `ir_dataset_name` is provided. Defaults to
            "text".
        dtype (str, optional): The dtype to use for the model and embedded
            items. Options are "fp16", "fp32", and "bf16". Defaults to "fp32".
        top_k (int, optional): The number of top items to retrieve. Defaults to
            1000.
        batch_size (int, optional): The batch size to use for retrieval.
            Defaults to 100,000.
        retriever_kwargs (Optional[Dict], optional): Additional keyword
            arguments to pass to the retriever. Defaults to None.
        query_max_length (int, optional): Maximum length of the query.
            Defaults to 64.
        include_content (bool, optional): Whether to include the content of the
            retrieved items in the output. Defaults to True.
        do_eval (bool, optional): Whether to do evaluation. Defaults to True.
        metric_names (Optional[List[str]], optional): A list of metrics to
            compute. These are passed to IR-Measures so should be compatible.
            If None, a default set of metrics is found. Defaults to None.
        ignore_same_id (bool, optional): Whether to ignore retrievals with the
            same ID as the query. This is only relevant for certain datasets.
            Defaults to False.
        entry_points (str, optional): "random" or "bm25". See
            `HypecoderGraphRetriever` for details. Defaults to "random".
        bm25_index_path (Optional[str], optional): Directory of the PISA
            index, required when `entry_points` is "bm25". Built from the
            encoded items if it does not exist. Defaults to None.
        bm25_k1 (float, optional): BM25 k1. Defaults to 1.5.
        bm25_b (float, optional): BM25 b. Defaults to 0.75.

    Raises:
        ValueError: If both `query_jsonl` and `ir_dataset_name` are provided.
        ValueError: If `do_eval` is True and `ir_dataset_name` is None and
            `qrel_json` is None.
    """

    retriever_kwargs = retriever_kwargs if retriever_kwargs is not None else {}

    do_retrieval_shared(
        retriever_cls=HypecoderGraphRetriever,
        retriever_kwargs=dict(
            model_name_or_path=model_name_or_path,
            encoded_item_path=encoded_item_path,
            dtype=dtype,
            batch_size=batch_size,
            query_max_length=query_max_length,
            ignore_same_id=ignore_same_id,
            item_neighbors_path=item_neighbors_path,
            num_entry_points=num_entry_points,
            ncandidates=ncandidates,
            max_iter=max_iter,
            early_stop=early_stop,
            device=device,
            cache_file=cache_file,
            entry_points=entry_points,
            bm25_index_path=bm25_index_path,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            **retriever_kwargs,
        ),
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        query_jsonl=query_jsonl,
        qrel_json=qrel_json,
        query_id_key=query_id_key,
        query_text_key=query_text_key,
        top_k=top_k,
        include_content=include_content,
        do_eval=do_eval,
        metric_names=metric_names,
    )


if __name__ == "__main__":
    fire.Fire(do_retrieval)