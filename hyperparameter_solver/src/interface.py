import json
import os
from .solver import HyperparameterSolver


def get_hyperparamenters(dataset_path: str, mode: str = "sss"):
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    dataset_path = os.path.abspath(dataset_path)
    solver = HyperparameterSolver(config, mode, dataset_path)
    rows = solver.solve()
    best = min(rows, key=lambda r: r["cost"])
    return {"alpha": float(best["alpha"]), "b": int(best["b"])}
