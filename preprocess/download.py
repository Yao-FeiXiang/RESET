from beir import util
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset
import os


def download_beir():
    BEIR_DATASETS = [
        # "cqadupstack", "fever", "hotpotqa",
        "msmarco",
        "nq",
    ]

    base_dir = "../ir_datasets"
    os.makedirs(base_dir, exist_ok=True)

    for d in BEIR_DATASETS:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{d}.zip"
        out_dir = os.path.join(base_dir, d)
        util.download_and_unzip(url, out_dir)


# def load_lotte(domain="science", task="search", split="test"):
#     prefix = f"{domain}_{task}"
#     corpus = load_dataset("mteb/LoTTE", f"{prefix}-corpus", split=split)
#     queries = load_dataset("mteb/LoTTE", f"{prefix}-queries", split=split)
#     qrels = load_dataset("mteb/LoTTE", f"{prefix}-qrels", split=split)
#     return corpus, queries, qrels


# dataset = load_lotte()
if __name__ == "__main__":
    download_beir()
