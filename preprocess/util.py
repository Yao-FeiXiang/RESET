import struct
import os
import re
import sys

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def write_arr_bin(filename, vec):
    with open(filename, "wb") as f:
        f.write(struct.pack("Q", len(vec)))
        f.write(struct.pack(f"{len(vec)}i", *vec))


def write_int_bin(filename, val):
    with open(filename, "wb") as f:
        f.write(struct.pack("i", val))


def write_float_bin(filename, val):
    with open(filename, "wb") as f:
        f.write(struct.pack("f", val))


def get_hyperparameters(dataset_path: str, mode: str):
    from hyperparamenter_solver import get_hyperparamenters

    hyperparams = get_hyperparamenters(dataset_path, mode)
    alpha, b = hyperparams["alpha"], hyperparams["b"]
    write_float_bin(os.path.join(dataset_path, "load_factor.bin"), alpha)
    write_int_bin(os.path.join(dataset_path, "bucket_size.bin"), b)
    return alpha, b


def tokenize(text: str):
    """
    简单的分词函数，去除标点符号并转换为小写
    """
    raw = text.split()
    out = []
    for tok in raw:
        tok = re.sub(r"[^\w\s]", "", tok)
        tok = tok.lower()
        if tok:
            out.append(tok)
    return out


def remove_redundance(directory_path, more_types=None):
    """
    删除所有中间文件,如.bin文件
    :param directory_path: 目录路径
    :param more_types: 其他需要删除的文件类型列表
    """
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在！")
        return
    redundant_types = [".bin"] + (more_types if more_types else [])
    for filename in os.listdir(directory_path):
        if filename.endswith(tuple(redundant_types)):
            file_path = os.path.join(directory_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"无法删除文件 {file_path}:{e}")


def _rebuild_index(word_to_id, inverted_index_set, remove_ids):
    """公共部分：根据 remove_ids 重新映射并构建新索引"""
    old_to_new = {}
    new_word_to_id = {}
    new_id = 0

    for w, old_id in word_to_id.items():
        if old_id not in remove_ids:
            old_to_new[old_id] = new_id
            new_word_to_id[w] = new_id
            new_id += 1

    new_inverted_index_set = [set() for _ in range(new_id)]
    for old_id, s in enumerate(inverted_index_set):
        if old_id not in remove_ids:
            new_inverted_index_set[old_to_new[old_id]] = s

    return new_word_to_id, new_inverted_index_set


def high_freq_filter_v1(word_to_id, inverted_index_set, top_k=100):
    """策略1: 删除出现频率最高的 top_k 个词"""
    freq = sorted(
        [(i, len(s)) for i, s in enumerate(inverted_index_set)], key=lambda x: x[1], reverse=True
    )
    remove_ids = {f[0] for f in freq[:top_k]}
    __removed_words_print(word_to_id, inverted_index_set, remove_ids)
    return _rebuild_index(word_to_id, inverted_index_set, remove_ids)


def high_freq_filter_v2(word_to_id, inverted_index_set, doc_count=None, threshold=0.7):
    """策略2: 删除出现在超过 freq_ratio_threshold 比例文档中的词"""
    if doc_count is None or doc_count == 0:
        doc_count = max((max(s) if s else 0) for s in inverted_index_set) + 1

    remove_ids = {i for i, s in enumerate(inverted_index_set) if len(s) / doc_count > threshold}
    __removed_words_print(word_to_id, inverted_index_set, remove_ids)
    return _rebuild_index(word_to_id, inverted_index_set, remove_ids)


def high_freq_filter_v3(word_to_id, inverted_index_set, doc_count=None, ratio=0.0001):
    """策略3: 基于 TF-IDF 的低信息词过滤"""
    from math import log

    if doc_count is None:
        doc_count = max((max(s) if s else 0) for s in inverted_index_set) + 1

    idf_scores = [log((doc_count + 1) / (1 + len(s))) for s in inverted_index_set]
    sorted_ids = sorted(range(len(idf_scores)), key=lambda i: idf_scores[i])
    remove_count = int(len(sorted_ids) * ratio)
    remove_ids = set(sorted_ids[:remove_count])
    __removed_words_print(word_to_id, inverted_index_set, remove_ids)
    return _rebuild_index(word_to_id, inverted_index_set, remove_ids)


def high_freq_filter_v4(word_to_id, inverted_index_set):
    """策略4: 基于 spaCy 停用词表过滤"""
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS

    remove_ids = {word_to_id[w] for w in word_to_id if w.lower() in EN_STOP_WORDS}
    __removed_words_print(word_to_id, inverted_index_set, remove_ids)
    return _rebuild_index(word_to_id, inverted_index_set, remove_ids)


DEBUG = True


def debug(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def __removed_words_print(word_to_id, inverted_index_set, remove_ids, max_display=50, off=True):
    """[调试]打印被删除的高频词及其出现次数"""
    if not DEBUG or off:
        return

    word_counts = [
        (w, len(inverted_index_set[word_to_id[w]]))
        for w in word_to_id
        if word_to_id[w] in remove_ids
    ]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print(f"删除高频词共{len(remove_ids)}个,其中频率前{min(max_display, len(word_counts))}的词:")
    for w, cnt in word_counts[:max_display]:
        print(f"{w}: {cnt}")
