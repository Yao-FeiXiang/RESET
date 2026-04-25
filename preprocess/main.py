from preprocessor import processor_factory, Processor
from util import get_hyperparameters
import os

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
    for dataset in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, dataset)
        single_process(path, mode, update)


def single_process(
    dataset_path: str, mode: str, update: bool = False, calculate_hyperparameters: bool = False
):
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist!")
        exit(1)
    if not update and has_bin_files(dataset_path):
        print(f"[SKIP] dataset {dataset_path} already has been processed.")
        return
    print(f"[PROCESSING] dataset at {os.path.abspath(dataset_path)} in mode:{mode}")
    p: Processor = processor_factory(mode, save_dir=dataset_path)

    p.build_all(data_path=dataset_path)

    if calculate_hyperparameters:
        alpha, b = get_hyperparameters(dataset_path, mode)
        print(f"Calculated hyperparameters: alpha={alpha}, b={b}")


def test():
    name = "test"
    path_graph = os.path.join(GRAPH_DATASETS, name)
    try:
        single_process(path_graph, mode="sss", update=True, calculate_hyperparameters=True)
        # single_process(path_graph, mode="tc", update=True)
        # name = "climate-fever"
        # path_ir = os.path.join(IR_DATASETS, name)
        # single_process(path_ir, mode="ir", update=True)
    except Exception as e:
        print(f"Error processing dataset {name}: {e}")


def main():
    # for mode in ["sss", "tc", "ir"]:
    #     batch_process(mode=mode, update=False)
    # batch_process(mode="sss", update=False)
    batch_process(mode="ir", update=False)
    # batch_process(mode="tc", update=True)s


if __name__ == "__main__":
    main()
    # path = os.path.join(IR_DATASETS, "lt")
    # single_process(path, "ir", update=True, calculate_hyperparameters=False)
