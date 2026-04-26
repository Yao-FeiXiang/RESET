import os
import json
import time
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
from collections import defaultdict
from util import *
from math import gcd


class Processor(ABC):
    """预处理抽象基类"""

    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def build_all(self, data_path):
        """构建所有数据的抽象方法"""
        pass


class GraphProcessor(Processor):
    """图数据处理器基类"""

    def __init__(self, save_dir=None):
        super().__init__(save_dir)
        self.edges = []
        self.nodes = set()

    def add_edge(self, u, v):
        """添加一条边，跳过自环"""
        if u == v:
            return
        self.edges.append((u, v))
        self.nodes.update([u, v])

    def remap_nodes(self):
        """节点ID重映射，使用可逆排列确保打乱"""
        sorted_nodes = sorted(self.nodes)
        node_map = {node: idx for idx, node in enumerate(sorted_nodes)}
        num_nodes = len(sorted_nodes)

        # 寻找与节点数互质的步长
        step = 10007
        while gcd(step, num_nodes) != 1:
            step += 1

        # 生成排列并应用
        perm = [(i * step) % num_nodes for i in range(num_nodes)]
        perm_map = {i: perm[i] for i in range(num_nodes)}
        self.edges = [(perm_map[node_map[u]], perm_map[node_map[v]]) for u, v in self.edges]
        self.nodes = set(perm_map.values())

    def read_edges_from_file(self, file_path, remap=False):
        """从单个文件读取边 - 单次遍历优化版"""
        try:
            if os.path.getsize(file_path) == 0:
                return
        except OSError:
            return

        # 使用文件大小估算进度，避免两次读取
        file_size = os.path.getsize(file_path)
        use_tqdm = file_size >= 50_000_000  # 50MB以上显示进度条

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if use_tqdm:
                import io
                f = tqdm(f, total=file_size, unit='B', unit_scale=True, 
                        desc=f"Reading {os.path.basename(file_path)}",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for line in f:
                # 快速跳过注释和空行
                if not line or line[0] == '#':
                    continue
                
                # 快速解析：直接查找第一个空格和第二个空格
                try:
                    idx1 = line.index(' ')
                    idx2 = line.index(' ', idx1 + 1) if ' ' in line[idx1 + 1:] else len(line)
                    u = int(line[:idx1])
                    v = int(line[idx1 + 1:idx2])
                    if u != v:
                        self.edges.append((u, v))
                        self.nodes.add(u)
                        self.nodes.add(v)
                except (ValueError, IndexError):
                    # 降级到split方法处理异常格式
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            u = int(parts[0])
                            v = int(parts[1])
                            if u != v:
                                self.edges.append((u, v))
                                self.nodes.add(u)
                                self.nodes.add(v)
                        except ValueError:
                            pass

        if remap:
            self.remap_nodes()

    def read_edges_from_directory(self, directory_path, remap=True):
        """从目录读取所有边文件"""
        for filename in sorted(os.listdir(directory_path)):
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
        """构建邻接表，自动去重并排序"""
        num_nodes = len(self.nodes)
        adj = [[] for _ in range(num_nodes)]

        for u, v in self.edges:
            if u == v:
                continue
            adj[u].append(v)
            if make_undirected:
                adj[v].append(u)

        # 去重并排序邻居
        for i in range(num_nodes):
            adj[i] = sorted(set(adj[i]))

        return adj, num_nodes

    @abstractmethod
    def export_to_bin(self):
        """导出为二进制文件的抽象方法"""
        pass

    def build_all(self, data_path: str):
        """构建图数据的完整流程"""
        start_time = time.time()
        self.edges.clear()
        self.nodes.clear()
        self.read_edges_from_directory(data_path)
        self.export_to_bin()
        elapsed = time.time() - start_time
        print(f"[INFO] Graph build_all done in {elapsed:.2f}s. ")


class SSSProcessor(GraphProcessor):
    """集相似搜索图处理器"""

    def export_to_bin(self):
        """导出SSS格式的二进制文件"""
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

        write_arr_bin(os.path.join(self.save_dir, "csr_cols.bin"), csr_cols)
        write_arr_bin(os.path.join(self.save_dir, "csr_offsets.bin"), csr_offsets)
        write_arr_bin(os.path.join(self.save_dir, "vertexs.bin"), vertexs)
        write_int_bin(os.path.join(self.save_dir, "num_nodes.bin"), num_nodes)
        write_int_bin(os.path.join(self.save_dir, "num_edges.bin"), num_edges)


class TCProcessor(GraphProcessor):
    """三角计数图处理器"""

    def export_to_bin(self):
        """导出TC格式的二进制文件"""
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

        write_arr_bin(os.path.join(self.save_dir, "csr_cols_tri.bin"), csr_cols)
        write_arr_bin(os.path.join(self.save_dir, "csr_offsets_tri.bin"), csr_offsets)
        write_int_bin(os.path.join(self.save_dir, "num_nodes_tri.bin"), num_nodes)
        write_int_bin(os.path.join(self.save_dir, "num_edges_tri.bin"), num_edges)


class IRProcessor(Processor):
    """信息检索数据处理器"""

    def __init__(self, save_dir=None):
        super().__init__(save_dir)
        self.doc_count = 0

        # 倒排索引相关
        self.inverted_index_offsets = []
        self.inverted_index = []
        self.inverted_index_num = 0

        # 查询相关
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
        """迭代读取jsonl文件"""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _build_word_to_id(self, all_tokens: set) -> dict:
        """按字母顺序分配词ID，确保确定性"""
        word_to_id = {}
        current_id = 0
        for token in sorted(all_tokens):
            word_to_id[token] = current_id
            current_id += 1
        return word_to_id

    def build_inverted_index(self, corpus_path: str):
        """构建倒排索引 - 单次遍历优化版"""
        # 第一次遍历：收集所有token并缓存文档数据（使用list避免二次IO）
        all_tokens = set()
        doc_cache = []
        
        for doc in self._iter_jsonl(corpus_path):
            text = doc.get("text", "")
            tokens = tokenize(text)
            all_tokens.update(tokens)
            doc_cache.append(set(tokens))
        
        # 分配词ID
        word_to_id = self._build_word_to_id(all_tokens)

        # 构建倒排索引（从缓存中读取，避免二次IO）
        inverted_index_set = defaultdict(set)
        
        for doc_id, tokens in tqdm(
            enumerate(doc_cache),
            total=len(doc_cache),
            desc="Building inverted index",
            dynamic_ncols=True,
            file=sys.stderr,
            mininterval=0.2,
        ):
            for token in tokens:
                wid = word_to_id[token]
                inverted_index_set[wid].add(doc_id)

        self.doc_count = len(doc_cache)

        # 转换为稠密数组形式
        max_wid = max(inverted_index_set.keys(), default=-1)
        dense_inverted_index_set = [set() for _ in range(max_wid + 1)]
        for wid, s in inverted_index_set.items():
            dense_inverted_index_set[wid] = s

        # 应用高频词过滤
        new_word_to_id, new_inverted_index_set = self.high_freq_filter(
            word_to_id, dense_inverted_index_set, self.doc_count
        )

        # 构建最终的偏移数组和数据数组
        self.inverted_index_num = len(new_inverted_index_set)
        self.inverted_index_offsets = [0]
        self.inverted_index.clear()

        for s in new_inverted_index_set:
            self.inverted_index_offsets.append(self.inverted_index_offsets[-1] + len(s))
            self.inverted_index.extend(sorted(s))

        return new_word_to_id, self.inverted_index_offsets

    def build_query(self, queries_path: str, word_to_id: dict, inverted_index_offsets: list):
        """构建查询的token序列和偏移数组"""
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

            # 过滤不在词典中的词并去重
            wset = {word_to_id[w] for w in tokens if w in word_to_id}

            # 按文档频率(df)排序，确保确定性
            def df(wid: int) -> int:
                return inverted_index_offsets[wid + 1] - inverted_index_offsets[wid]

            ordered = sorted(wset, key=lambda wid: (df(wid), wid))

            self.query_offsets.append(offset)
            self.query.extend(ordered)
            offset += len(ordered)
            self.query_num += 1

        self.query_offsets.append(offset)

    def revise_inverted_index(self, doc_count: int):
        """可逆重排倒排索引中的文档ID，确保无偏性"""
        step = 10007
        while gcd(step, doc_count) != 1:
            step += 1
        step2 = (step * step) % doc_count
        new_ids = [(i * step2) % doc_count for i in range(doc_count)]
        self.inverted_index = [new_ids[x] for x in self.inverted_index]

    def save_all(self):
        """保存所有数据为二进制文件"""
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
        """IR预处理的完整执行流程"""
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
