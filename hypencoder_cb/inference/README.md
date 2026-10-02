## Overview
This directory contains the files to easily encode and retrieve items using Hypencoder models. As well as providing evaluation metrics.

### Encoding and Retrieving
If the queries and documents you want to retrieve exist as a dataset in the IR Dataset library no additional work is needed to encode and retrieve from the dataset. If the data is not a part of this library you will need two JSONL files for the documents and queries. These must have the format:
```
{"<id_key>": "afei1243", "<text_key>": "This is some text"}
...
```
where `<id_key>` and `<text_key>` can be any string and do not have to be the same for the document and query file.

#### Encoding
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
python hypencoder_cb/inference/encode.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--output_path=$ENCODING_PATH \
--jsonl_path=path/to/documents.jsonl \
--item_id_key=<id_key> \
--item_text_key=<text_key>
```
For all the arguments and information on using IR Datasets type:
`python hypencoder_cb/inference/encode.py --help`.

#### Retrieve
The values of `ENCODING_PATH` and `MODEL_NAME_OR_PATH` should be the same as
those used in the encoding step.
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export RETRIEVAL_DIR="..."
python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--query_jsonl=path/to/queries.jsonl \
--do_eval=False \
--query_id_key=<id_key> \
--query_text_key=<text_key> \
--query_max_length=64 \
--top_k=1000
```
For all the arguments and information on using IR Datasets type:
`python hypencoder_cb/inference/retrieve.py --help`.

#### Evaluation
Evaluation is done automatically when `hypencoder_cb/inference/retrieve.py` is called so long as `--do_eval=True`. If you are not using an IR Dataset you will need to provide the qrels with the argument `--qrel_json`. The qrels JSON should be in the format:
```
{
    "qid1": {
        "pid8": relevance_value (float),
        "pid65": relevance_value (float),
        ...
    }.
    "qid2": {
        ...
    },
    ...
}
```

#### Reverse Retrieval
Reverse retrieval swaps the roles of queries and items. The hypernetwork generates a q-net for every item at encoding time, and each query is encoded as a vector that is fed into every item's q-net. The pretrained models were trained the normal way round, so expect much lower effectiveness from them in reverse.

Each q-net is stored in full, which takes about 7 MB per item in fp16 for the 6 layer model, so this only suits small corpora (BEIR NFCorpus, with about 3.6k documents, needs about 25 GB). Generating many q-nets at once also needs a lot of GPU memory, so use a small `--batch_size`.
```
export QNET_INDEX_DIR="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export RETRIEVAL_DIR="..."
python hypencoder_cb/inference/encode.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--output_path=$QNET_INDEX_DIR \
--ir_dataset_name=beir/nfcorpus/test \
--representation_type=q_net \
--batch_size=16

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$QNET_INDEX_DIR \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=beir/nfcorpus/test \
--query_max_length=512 \
--reverse=True
```
`--output_path` is a directory holding `q_nets.bin` (memory mapped at retrieval time), `items.jsonl`, and `meta.json`. Retrieval reads each batch of `--batch_size` q-nets (default 64) once and scores all queries with it, so a whole query set costs one pass over the index. `timing.json` therefore records the total time for the query set rather than per-query latency.

#### Approximate Retrieval
##### Getting a Item Neighbor Graph
Approximate retrieval requires an item-to-item graph. To get this graph use the following command:
```
export ENCODING_PATH="..."
export ITEM_NEIGHBOR_GRAPH="..."
python hypencoder_cb/inference/neighbor_graph.py \
--encoded_items_path=$ENCODING_PATH \
--output_path=$ITEM_NEIGHBOR_GRAPH \
--batch_size=100 \
--top_k=100 \
--device=cuda
```

##### Doing approximate retrieval
The values of `ENCODING_PATH` and `MODEL_NAME_OR_PATH` should be the same as
those used in the encoding step. Similarly `ENCODING_PATH` should be the same as the one used to construct the neighbor graph.
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ITEM_NEIGHBOR_GRAPH="..."
export RETRIEVAL_DIR="..."
python hypencoder_cb/inference/approx_retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--query_jsonl=path/to/queries.jsonl \
--do_eval=False \
--query_id_key=<id_key> \
--query_text_key=<text_key> \
--query_max_length=64 \
--top_k=1000
```

##### Seeding the search with BM25
By default the graph search starts from the same `--num_entry_points` random items for every query. With `--entry_points=bm25` it starts from the query's top `--num_entry_points` BM25 results. If BM25 matches nothing for a query, the search falls back to the random entry points. This requires `python-terrier` and `pyterrier-pisa`. The first run builds a PISA index from the encoded items at `--bm25_index_path`, so BM25 sees the same IDs and text as the neural search. Later runs reuse that index. Use a separate index path for each corpus.
```
python hypencoder_cb/inference/approx_retrieve.py \
... # Same arguments as above
--entry_points=bm25 \
--bm25_index_path=path/to/pisa_index \
--num_entry_points=1000 \
--bm25_k1=1.5 \
--bm25_b=0.75
```

##### Reusing a loaded retriever for parameter sweeps
Loading the model, encoded items, neighbor graph, and BM25 index is slow, so a sweep should load them once. `with_search_params` returns a copy of the retriever with different search parameters that shares all of that loaded state. The parameters it can change are listed in `HypecoderGraphRetriever.SEARCH_PARAMS`. Pass the copy to `do_retrieval_shared` with `retriever=` instead of `retriever_cls`. Each run writes its results, metrics, and `timing.json` to its own `output_dir`.
```python
from itertools import product

from hypencoder_cb.inference.approx_retrieve import HypecoderGraphRetriever
from hypencoder_cb.inference.retrieve import do_retrieval_shared

base = HypecoderGraphRetriever(
    model_name_or_path="jfkback/hypencoder.6_layer",
    encoded_item_path="...",
    item_neighbors_path="...",
    bm25_index_path="...",  # Only needed for entry_points="bm25"
)

for entry_points, num_entry_points, ncandidates in product(
    ["random", "bm25"], [100, 1000], [16, 64]
):
    retriever = base.with_search_params(
        entry_points=entry_points,
        num_entry_points=num_entry_points,
        ncandidates=ncandidates,
    )
    do_retrieval_shared(
        retriever=retriever,
        output_dir=f"runs/{entry_points}-{num_entry_points}-{ncandidates}",
        ir_dataset_name="msmarco-passage/trec-dl-2019/judged",
        top_k=1000,
    )
```
