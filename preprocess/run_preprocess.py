from preprocessor import processor_factory, Processor
from util import get_hyperparameters
import os
import argparse

GRAPH_DATASETS = "../graph_datasets"
IR_DATASETS = "../ir_datasets"


def has_bin_files(path: str) -> bool:
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".bin"):
                return True
    return False


def batch_process(mode: str, update=False):
    datasets_dir = IR_DATASETS if mode == "ir" else GRAPH_DATASETS
    if not os.path.exists(datasets_dir):
        print(f"Dataset path:{datasets_dir} does not exist!")
        exit(1)

    datasets = sorted(os.listdir(datasets_dir))
    total = len(datasets)
    failed = []

    for i, dataset in enumerate(datasets, 1):
        path = os.path.join(datasets_dir, dataset)
        print(f"\n[{i}/{total}] Processing {dataset}...")
        try:
            single_process(path, mode, update)
        except Exception as e:
            print(f"[ERROR] Failed to process {dataset}: {str(e)}")
            failed.append(dataset)

    if failed:
        print(f"\n=== Summary ===")
        print(f"Total: {total}, Success: {total - len(failed)}, Failed: {len(failed)}")
        print(f"Failed datasets: {', '.join(failed)}")
    else:
        print(f"\nAll {total} datasets processed successfully!")


def single_process(
    dataset_path: str, mode: str, update: bool = False, calculate_hyperparameters: bool = False
):
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist!")
        return

    marker_file = os.path.join(dataset_path, '.preprocess_in_progress')

    if not update and has_bin_files(dataset_path) and not os.path.exists(marker_file):
        print(f"[SKIP] dataset {dataset_path} already has been processed.")
        return

    # 创建进度标记
    with open(marker_file, 'w') as f:
        f.write('')

    try:
        print(f"[PROCESSING] dataset at {os.path.abspath(dataset_path)} in mode:{mode}")
        p: Processor = processor_factory(mode, save_dir=dataset_path)
        p.build_all(data_path=dataset_path)

        if calculate_hyperparameters:
            alpha, b = get_hyperparameters(dataset_path, mode)
            print(f"Calculated hyperparameters: alpha={alpha}, b={b}")
    finally:
        # 移除进度标记
        if os.path.exists(marker_file):
            os.remove(marker_file)


def main():
    parser = argparse.ArgumentParser(description='Run preprocessing on datasets')
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        choices=['sss', 'tc', 'ir'],
        help='Processing mode: sss, tc, or ir',
    )
    parser.add_argument(
        '-u', '--update', action='store_true', help='Force update even if already processed'
    )
    args = parser.parse_args()

    if args.mode:
        batch_process(mode=args.mode, update=args.update)
    else:
        # 默认行为
        batch_process(mode="ir", update=args.update)


if __name__ == "__main__":
    main()
