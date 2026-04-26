#!/usr/bin/env python3
"""
预计算常见配置的超参数推荐值并缓存

使用方法:
    python precompute_cache.py           # 预计算所有常见配置
    python precompute_cache.py --show    # 仅显示缓存内容
    python precompute_cache.py --clear   # 清空缓存
"""

import argparse
from c_api import RecommendationCache, _cache, COMMON_CONFIGS, COMMON_DATA_SIZES


def show_cache():
    """显示缓存内容"""
    stats = _cache.get_stats()
    print(f"=== 缓存统计 ===")
    print(f"缓存条目数: {stats['total_entries']}")
    print(f"缓存文件: {stats['cache_file']}")
    print()

    if not _cache.cache:
        print("缓存为空")
        return

    print("=== 缓存内容 ===")
    print(
        f"{'序号':<4} {'数组大小':<12} {'W':<4} {'S_trans':<8} {'S_slot':<8} {'alpha':<8} {'b':<4} {'cost':<12}"
    )
    print("-" * 70)

    # 从键解析参数
    import json

    config_path = "config.json"
    with open(config_path, "r") as f:
        base_config = json.load(f)

    # 反向查找：遍历所有可能的配置组合
    count = 0
    for gpu_name, hw_params in COMMON_CONFIGS.items():
        for size_category, sizes in COMMON_DATA_SIZES.items():
            for arr_size in sizes:
                cached = _cache.get(
                    arr_size, hw_params["W"], hw_params["S_trans"], hw_params["S_slot"], "sss"
                )
                if cached is not None:
                    alpha, b, cost = cached
                    count += 1
                    print(
                        f"{count:<4} {arr_size:<12} {hw_params['W']:<4} {hw_params['S_trans']:<8} {hw_params['S_slot']:<8} {alpha:<8.2f} {b:<4} {cost:<12.4f}"
                    )

    print(f"\n总计: {count} 条缓存记录")


def main():
    parser = argparse.ArgumentParser(description="超参数缓存管理工具")
    parser.add_argument("--show", action="store_true", help="显示缓存内容")
    parser.add_argument("--clear", action="store_true", help="清空缓存")
    parser.add_argument(
        "--gpu", type=str, default="all", help="指定GPU型号 (A100, V100, 3090, 4090, H100)"
    )
    parser.add_argument(
        "--size", type=str, default="all", help="指定数据规模 (small, medium, large)"
    )

    args = parser.parse_args()

    if args.clear:
        print("清空缓存...")
        _cache.clear()
        print("完成")
        return

    if args.show:
        show_cache()
        return

    # 预计算
    print("=" * 60)
    print("预计算常见配置的超参数推荐值")
    print("=" * 60)

    if args.gpu != "all" or args.size != "all":
        # 自定义预计算
        import json

        config_path = "config.json"
        with open(config_path, "r") as f:
            base_config = json.load(f)

        gpus = [args.gpu] if args.gpu != "all" else COMMON_CONFIGS.keys()
        sizes = [args.size] if args.size != "all" else COMMON_DATA_SIZES.keys()

        total = 0
        for gpu_name in gpus:
            if gpu_name not in COMMON_CONFIGS:
                print(f"警告: 未知GPU型号 {gpu_name}")
                continue
            hw_params = COMMON_CONFIGS[gpu_name]

            for size_cat in sizes:
                if size_cat not in COMMON_DATA_SIZES:
                    print(f"警告: 未知规模类别 {size_cat}")
                    continue
                for arr_size in COMMON_DATA_SIZES[size_cat]:
                    cached = _cache.get(
                        arr_size, hw_params["W"], hw_params["S_trans"], hw_params["S_slot"], "sss"
                    )
                    if cached is not None:
                        print(f"跳过 GPU={gpu_name}, size={arr_size} (已缓存)")
                        continue

                    print(f"计算 GPU={gpu_name}, size={arr_size}...")
                    config = dict(base_config)
                    config.update(hw_params)

                    from solver import HyperparameterSolver

                    solver = HyperparameterSolver(config, "sss", arr_size=arr_size)
                    results = solver.solve()
                    best = min(results, key=lambda r: r["cost"])

                    _cache.put(
                        arr_size,
                        hw_params["W"],
                        hw_params["S_trans"],
                        hw_params["S_slot"],
                        float(best["alpha"]),
                        int(best["b"]),
                        float(best["cost"]),
                        "sss",
                    )
                    total += 1
                    print(f"  -> alpha={best['alpha']}, b={best['b']}, cost={best['cost']:.4f}")

        print(f"\n完成！新增 {total} 条缓存记录")
    else:
        # 完整预计算
        _cache.precompute_common_configs()

    show_cache()


if __name__ == "__main__":
    main()
