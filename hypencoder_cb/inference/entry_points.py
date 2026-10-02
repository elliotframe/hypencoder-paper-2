import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from tqdm import tqdm

from hypencoder_cb.inference.shared import TextQuery


class EntryPointSelector:
    """Chooses the item IDs a graph search starts from for a given query."""

    def select(self, query: TextQuery) -> List[str]:
        raise NotImplementedError


class RandomEntryPoints(EntryPointSelector):
    def __init__(
        self,
        item_ids: Sequence[str],
        num_entry_points: int,
        random_seed: int = 43,
    ) -> None:
        """The same random sample of items is used for every query.

        Args:
            item_ids (Sequence[str]): The IDs of all items that can be
                retrieved.
            num_entry_points (int): The number of items to sample. Capped at
                the number of items.
            random_seed (int, optional): Seed for the sample. Defaults to 43.
        """
        num_entry_points = min(num_entry_points, len(item_ids))
        indices = random.Random(random_seed).sample(
            range(len(item_ids)), num_entry_points
        )
        self.entry_point_ids = [item_ids[idx] for idx in indices]

    def select(self, query: TextQuery) -> List[str]:
        return list(self.entry_point_ids)


def load_bm25_index(
    index_path: str,
    items: Iterable[Tuple[str, str]],
    threads: int = 1,
):
    """Loads the PISA index at `index_path`, building it from `items` first if
    it does not exist.

    Requires `python-terrier` and `pyterrier-pisa`.

    Args:
        index_path (str): Directory of the PISA index.
        items (Iterable[Tuple[str, str]]): (item ID, item text) pairs. Only
            consumed when the index is built. Should be the encoded items so
            BM25 sees the same IDs and text as the graph search.
        threads (int, optional): Threads used to build the index. Defaults
            to 1.

    Returns:
        PisaIndex: The loaded index.
    """
    try:
        from pyterrier_pisa import PisaIndex
    except ImportError as e:
        raise ImportError(
            "BM25 entry points require `python-terrier` and"
            " `pyterrier-pisa`: pip install python-terrier pyterrier-pisa"
        ) from e

    index = PisaIndex(index_path, threads=threads)

    if index.built():
        print(f"Loading existing PISA index from {index_path}")
    else:
        print(f"Building PISA index at {index_path}")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        index.index(
            {"docno": item_id, "text": text}
            for item_id, text in tqdm(items, desc="Indexing for BM25")
        )

    return index


class BM25EntryPoints(EntryPointSelector):
    def __init__(
        self,
        index,
        num_entry_points: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Uses the top BM25 results for the query as its entry points.

        Args:
            index (PisaIndex): Index from `load_bm25_index`. Creating a
                selector is cheap, so one index can back several selectors
                with different parameters.
            num_entry_points (int): The number of BM25 results to use.
            k1 (float, optional): BM25 term frequency saturation. Defaults
                to 1.5.
            b (float, optional): BM25 length normalization. Defaults to 0.75.
        """
        self.bm25 = index.bm25(k1=k1, b=b, num_results=num_entry_points)

    def select(self, query: TextQuery) -> List[str]:
        return self.bm25.search(query.text)["docno"].tolist()
