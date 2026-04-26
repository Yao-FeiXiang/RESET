import os
import requests
import zipfile
import tarfile
import gzip
import shutil
import json
import ir_datasets

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
graph_datasets_dir = os.path.join(workspace, 'graph_datasets')
ir_datasets_dir = os.path.join(workspace, 'ir_datasets')
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(graph_datasets_dir, exist_ok=True)
os.makedirs(ir_datasets_dir, exist_ok=True)

datasets = [
    {'name': 'ce', 'type': 'ir_datasets', 'full_name': 'beir/cqadupstack/english'},
    {'name': 'fe', 'type': 'ir_datasets', 'full_name': 'beir/fever'},
    {'name': 'hp', 'type': 'ir_datasets', 'full_name': 'beir/hotpotqa'},
    {'name': 'lt', 'type': 'ir_datasets', 'full_name': 'lotte/lifestyle/dev/forum'},
    {'name': 'ms', 'type': 'ir_datasets', 'full_name': 'beir/msmarco'},
    {'name': 'nq', 'type': 'ir_datasets', 'full_name': 'beir/nq'},
    {
        'name': 'bm',
        'type': 'graph_datasets',
        'url': 'https://nrvis.com/download/data/bio/bio-mouse-gene.zip',
    },
    {'name': 'gp', 'type': 'graph_datasets', 'url': 'https://snap.stanford.edu/data/gplus.tar.gz'},
    {
        'name': 'sc18',
        'type': 'graph_datasets',
        'url': 'https://nrvis.com/download/data/graph500/graph500-scale18-ef16_adj.zip',
    },
    {
        'name': 'sc19',
        'type': 'graph_datasets',
        'url': 'https://nrvis.com/download/data/graph500/graph500-scale19-ef16_adj.zip',
    },
    {
        'name': 'sc20',
        'type': 'graph_datasets',
        'url': 'https://nrvis.com/download/data/graph500/graph500-scale20-ef16_adj.zip',
    },
    {
        'name': 'wt',
        'type': 'graph_datasets',
        'url': 'https://snap.stanford.edu/data/wiki-Talk.txt.gz',
    },
]


def download_and_extract(url, extract_to):
    # 创建临时标记文件，支持断点续传检测
    marker_file = os.path.join(extract_to, '.download_in_progress')

    # 检查是否已有完成的数据集（有数据且无进度标记）
    if (
        os.path.exists(extract_to)
        and len(os.listdir(extract_to)) > 0
        and not os.path.exists(marker_file)
    ):
        print(f"Dataset at {extract_to} already completed, skipping")
        return

    # 创建进度标记，表示开始下载
    os.makedirs(extract_to, exist_ok=True)
    with open(marker_file, 'w') as f:
        f.write('')

    temp_file = os.path.join(tmp_dir, 'temp_' + os.path.basename(url))
    print(f"Downloading {url} to {temp_file}")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Extracting to {extract_to}")

        if url.endswith('.zip'):
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif url.endswith('.tar.gz'):
            with tarfile.open(temp_file, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif url.endswith('.txt.gz'):
            with gzip.open(temp_file, 'rb') as f_in:
                output_file = os.path.join(extract_to, os.path.basename(url)[:-3])
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        print(f"Extraction completed successfully")

    finally:
        # 确保临时文件被删除，即使下载/解压失败
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Removed temp file: {temp_file}")

    # 处理子目录情况：如果只有一个子目录，将其中的文件移动到上层
    items = os.listdir(extract_to)
    if len(items) == 1:
        subdir = os.path.join(extract_to, items[0])
        if os.path.isdir(subdir):
            print(f"Moving files from subdir {subdir} to {extract_to}")
            for f in os.listdir(subdir):
                src = os.path.join(subdir, f)
                dst = os.path.join(extract_to, f)
                shutil.move(src, dst)
            os.rmdir(subdir)

    # 移除进度标记，表示完成
    if os.path.exists(marker_file):
        os.remove(marker_file)


def export_ir_dataset(ds, out_dir):
    # 创建进度标记
    marker_file = os.path.join(out_dir, '.download_in_progress')

    # 检查是否已有完成的数据集
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0 and not os.path.exists(marker_file):
        print(f"Dataset at {out_dir} already completed, skipping")
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(marker_file, 'w') as f:
        f.write('')

    try:
        # corpus
        with open(os.path.join(out_dir, 'corpus.jsonl'), 'w', encoding='utf-8') as f:
            for doc in ds.docs_iter():
                json.dump(
                    {'_id': doc.doc_id, 'text': doc.text, 'title': getattr(doc, 'title', '')},
                    f,
                    ensure_ascii=False,
                )
                f.write('\n')
        # queries
        if hasattr(ds, 'queries_iter'):
            with open(os.path.join(out_dir, 'queries.jsonl'), 'w', encoding='utf-8') as f:
                for query in ds.queries_iter():
                    json.dump({'_id': query.query_id, 'text': query.text}, f, ensure_ascii=False)
                    f.write('\n')
        # qrels - 即使没有也创建空文件，避免预处理报错
        qrels_path = os.path.join(out_dir, 'qrels.tsv')
        if hasattr(ds, 'qrels_iter'):
            with open(qrels_path, 'w', encoding='utf-8') as f:
                try:
                    for qrel in ds.qrels_iter():
                        f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
                except Exception:
                    pass
        # 如果 qrels 文件不存在或为空，创建空文件
        if not os.path.exists(qrels_path) or os.path.getsize(qrels_path) == 0:
            with open(qrels_path, 'w') as f:
                pass
    finally:
        # 移除进度标记
        if os.path.exists(marker_file):
            os.remove(marker_file)


if __name__ == "__main__":
    for dataset in datasets:
        name = dataset['name']
        type_ = dataset['type']
        print(f"\nProcessing dataset: {name}")
        try:
            if type_ == 'graph_datasets':
                target_dir = os.path.join(graph_datasets_dir, name)
                url = dataset['url']
                download_and_extract(url, target_dir)
            elif type_ == 'ir_datasets':
                target_dir = os.path.join(ir_datasets_dir, name)
                full_name = dataset['full_name']
                ds = ir_datasets.load(full_name)
                export_ir_dataset(ds, target_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {str(e)}")
    print("\nAll downloads completed")
