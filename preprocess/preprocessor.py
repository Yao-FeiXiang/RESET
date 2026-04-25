import os
import json
import time
from abc import ABC
from tqdm import tqdm
from collections import defaultdict
from util import *
from math import gcd


class Processor(ABC):
    def __init__(self, save_dir=None):
        pass

    def build_all(self, data_path):
        raise NotImplementedError("Processor.build_all() should be implemented in subclasses.")


class GraphProcessor(Processor):
    def __init__(self, save_dir=None):
        self.edges = []
        self.nodes = set()
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def add_edge(self, u, v):
        if u == v:
            return
        self.edges.append((u, v))
        self.nodes.update([u, v])

    def remap_nodes(self):
        sorted_nodes = sorted(self.nodes)
        node_map = {node: idx for idx, node in enumerate(sorted_nodes)}
        num_nodes = len(sorted_nodes)
        step = 10007
        while gcd(step, num_nodes) != 1:
            step += 1
        perm = [(i * step) % num_nodes for i in range(num_nodes)]
        perm_map = {i: perm[i] for i in range(num_nodes)}
        self.edges = [(perm_map[node_map[u]], perm_map[node_map[v]]) for u, v in self.edges]
        self.nodes = set(perm_map.values())

    def read_edges_from_file(self, file_path, remap=False):
        try:
            if os.path.getsize(file_path) == 0:
                return
        except OSError:
            return

        total_lines = None
        try:
            use_tqdm = os.path.getsize(file_path) >= 1_000_000
            if use_tqdm:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines = sum(1 for _ in f)
        except Exception:
            use_tqdm = False

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            it = (
                tqdm(f, total=total_lines, desc=f"Reading {os.path.basename(file_path)}")
                if use_tqdm
                else f
            )
            for line in it:
                parts = line.strip().split()
                if len(parts) < 2 or parts[0].startswith("#"):
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    self.add_edge(u, v)
                except ValueError:
                    continue
        if remap:
            self.remap_nodes()

    def read_edges_from_directory(self, directory_path, remap=True):
        for filename in os.listdir(directory_path):
            if filename.endswith(".edges") or filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                self.read_edges_from_file(file_path, remap=False)
        if remap:
            self.remap_nodes()

    def get_edge_list(self):
        return self.edges

    def get_nodes(self):
        return self.nodes

    def _build_adj(self, make_undirected: bool = True):
        num_nodes = len(self.nodes)
        adj = [[] for _ in range(num_nodes)]
        for u, v in self.edges:
            if u == v:
                continue
            adj[u].append(v)
            if make_undirected:
                adj[v].append(u)

        for i in range(num_nodes):
            nbrs = adj[i]
            nbrs = sorted(set(nbrs))
            adj[i] = nbrs
        return adj, num_nodes

    def export_to_bin(self):
        raise NotImplementedError(
            "GraphProcessor.export_to_bin() should be implemented in subclasses."
        )

    def build_all(self, data_path: str):
        start_time = time.time()
        self.edges.clear()
        self.nodes.clear()
        self.read_edges_from_directory(data_path)
        self.export_to_bin()
        elapsed = time.time() - start_time
        print(f"[INFO] Graph build_all done in {elapsed:.2f}s. ")


class SSSProcessor(GraphProcessor):
    def export_to_bin(self):
        if not self.save_dir:
            raise ValueError("save_dir is None, cannot export.")

        adj, num_nodes = self._build_adj(make_undirected=True)
        csr_cols = []
        vertexs = []
        csr_offsets = [0]
        num_edges = 0

        for i in range(num_nodes):
            for v in adj[i]:
                csr_cols.append(v)
                vertexs.append(i)
            num_edges += len(adj[i])
            csr_offsets.append(len(csr_cols))

        os.makedirs(self.save_dir, exist_ok=True)
        # print(f"num_nodes:{num_nodes}, num_edges:{num_edges}")
        # print(f"csr_cols:{csr_cols}")
        # print(f"csr_offsets:{csr_offsets}")
        # print(f"vertexs:{vertexs}")

        write_arr_bin(os.path.join(self.save_dir, "csr_cols.bin"), csr_cols)
        write_arr_bin(os.path.join(self.save_dir, "csr_offsets.bin"), csr_offsets)
        write_arr_bin(os.path.join(self.save_dir, "vertexs.bin"), vertexs)
        write_int_bin(os.path.join(self.save_dir, "num_nodes.bin"), num_nodes)
        write_int_bin(os.path.join(self.save_dir, "num_edges.bin"), num_edges)


class TCProcessor(GraphProcessor):
    def export_to_bin(self):
        if not self.save_dir:
            raise ValueError("save_dir is None, cannot export.")
        adj, num_nodes = self._build_adj(make_undirected=True)
        csr_cols = []
        csr_offsets = [0]
        num_edges = 0
        for u in range(num_nodes):
            tri_neighbors = [v for v in adj[u] if v > u]
            csr_cols.extend(tri_neighbors)
            num_edges += len(tri_neighbors)
            csr_offsets.append(len(csr_cols))
        os.makedirs(self.save_dir, exist_ok=True)

        # print(f"num_nodes:{num_nodes}, num_edges:{num_edges}")
        # print(f"csr_cols:{csr_cols}")
        # print(f"csr_offsets:{csr_offsets}")

        write_arr_bin(os.path.join(self.save_dir, "csr_cols_tri.bin"), csr_cols)
        write_arr_bin(os.path.join(self.save_dir, "csr_offsets_tri.bin"), csr_offsets)
        write_int_bin(os.path.join(self.save_dir, "num_nodes_tri.bin"), num_nodes)
        write_int_bin(os.path.join(self.save_dir, "num_edges_tri.bin"), num_edges)


class IRProcessor:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.doc_count = 0

        self.inverted_index_offsets = []
        self.inverted_index = []
        self.inverted_index_num = 0

        self.query = []
        self.query_offsets = []
        self.query_num = 0

    @staticmethod
    def high_freq_filter(word_to_id, inverted_index_set, doc_count, strategy=3) -> tuple:
        """高频词过滤统一入口"""
        if strategy not in range(1, 5):
            raise ValueError("Invalid high frequency filter strategy")
        vocab_before = len(word_to_id)
        func = globals().get(f"high_freq_filter_v{strategy}")
        new_word_to_id, new_inverted_index_set = func(word_to_id, inverted_index_set, doc_count)
        vocab_after = len(new_word_to_id)
        ratio = (vocab_before - vocab_after) / vocab_before if vocab_before > 0 else 0.0
        print(
            f"[HighFreqFilter-v{strategy}] "
            f"filtered {vocab_before - vocab_after} / {vocab_before} words "
            f"({ratio:.2%})"
        )

        return new_word_to_id, new_inverted_index_set

    def _iter_jsonl(self, path: str):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def build_inverted_index(self, corpus_path: str):
        word_to_id = {}
        inverted_index_set = defaultdict(set)
        current_id = 0
        doc_id = 0

        for doc in tqdm(
            self._iter_jsonl(corpus_path),
            desc="Building inverted index",
            dynamic_ncols=True,
            file=sys.stderr,
            mininterval=0.2,
        ):
            text = doc.get("text", "")
            tokens = set(tokenize(text))
            for token in tokens:
                wid = word_to_id.get(token)
                if wid is None:
                    wid = current_id
                    word_to_id[token] = wid
                    current_id += 1
                inverted_index_set[wid].add(doc_id)
            doc_id += 1

        self.doc_count = doc_id

        max_wid = max(inverted_index_set.keys(), default=-1)
        dense_inverted_index_set = [set() for _ in range(max_wid + 1)]
        for wid, s in inverted_index_set.items():
            dense_inverted_index_set[wid] = s

        new_word_to_id, new_inverted_index_set = self.high_freq_filter(
            word_to_id, dense_inverted_index_set, self.doc_count
        )

        del inverted_index_set, dense_inverted_index_set

        self.inverted_index_num = len(new_inverted_index_set)
        self.inverted_index_offsets = [0]
        self.inverted_index.clear()

        for s in new_inverted_index_set:
            self.inverted_index_offsets.append(self.inverted_index_offsets[-1] + len(s))
            for doc in sorted(s):
                self.inverted_index.append(doc)

        return new_word_to_id, self.inverted_index_offsets

    def build_query(self, queries_path: str, word_to_id: dict, inverted_index_offsets: list):
        offset = 0

        self.query.clear()
        self.query_offsets.clear()
        self.query_num = 0

        for doc in tqdm(
            self._iter_jsonl(queries_path),
            desc="Building queries",
            dynamic_ncols=True,
            file=sys.stderr,
            mininterval=0.2,
        ):
            text = doc.get("text", "")
            tokens = tokenize(text)

            wset = set()
            for w in tokens:
                wid = word_to_id.get(w)
                if wid is not None:
                    wset.add(wid)

            def df(wid: int) -> int:
                return inverted_index_offsets[wid + 1] - inverted_index_offsets[wid]

            ordered = sorted(wset, key=lambda wid: (df(wid), wid))

            self.query_offsets.append(offset)
            self.query.extend(ordered)
            offset += len(ordered)
            self.query_num += 1

        self.query_offsets.append(offset)

    def revise_inverted_index(self, doc_count: int):
        """可逆重排倒排索引中的文档ID"""
        step = 10007
        while gcd(step, doc_count) != 1:
            step += 1
        step2 = (step * step) % doc_count
        new_ids = [(i * step2) % doc_count for i in range(doc_count)]
        self.inverted_index = [new_ids[x] for x in self.inverted_index]

    def save_all(self):
        """保存数据为二进制文件"""
        if not self.save_dir:
            raise ValueError("save_dir is None, cannot save.")
        out = lambda name: os.path.join(self.save_dir, name)
        write_arr_bin(out("inverted_index_offsets.bin"), self.inverted_index_offsets)
        write_arr_bin(out("inverted_index.bin"), self.inverted_index)
        write_int_bin(out("inverted_index_num.bin"), self.inverted_index_num)
        write_arr_bin(out("query.bin"), self.query)
        write_arr_bin(out("query_offsets.bin"), self.query_offsets)
        write_int_bin(out("query_num.bin"), self.query_num)

    def build_all(self, data_path):
        """类的主执行函数"""
        start = time.time()

        corpus_path = os.path.join(data_path, "corpus.jsonl")
        query_path = os.path.join(data_path, "queries.jsonl")
        word_to_id, offsets = self.build_inverted_index(corpus_path)
        self.build_query(query_path, word_to_id, offsets)
        self.revise_inverted_index(self.doc_count)
        self.save_all()
        elapsed = time.time() - start
        print(f"[INFO] IR build_all done in {elapsed:.2f}s . ")


def processor_factory(mode: str, save_dir=None) -> Processor:
    """预处理工厂函数"""
    mode = mode.lower()
    if mode == "ir":
        return IRProcessor(save_dir)
    elif mode == "sss":
        return SSSProcessor(save_dir)
    elif mode == "tc":
        return TCProcessor(save_dir)
    else:
        raise ValueError(f"Unknown graph mode: {mode}")
