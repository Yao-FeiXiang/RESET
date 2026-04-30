#!/usr/bin/env python3
"""
预计算常见配置的超参数推荐值并缓存

使用方法:
    python precompute_cache.py           # 预计算所有常见配置
    python precompute_cache.py --show    # 仅显示缓存内容
    python precompute_cache.py --clear   # 清空缓存
"""

import argparse
from c_api import RecommendationCache, _cache, COMMON_CONFIGS, COMMON_DATA_SIZES, get_cache_config


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
        f"{'序号':<4} {'模式':<5} {'数组大小':<12} {'W':<4} {'S_trans':<8} {'S_slot':<8} {'alpha':<8} {'b':<4} {'cost':<12}"
    )
    print("-" * 90)

    # 遍历所有缓存条目
    count = 0
    # 按键名排序，使输出更有序
    sorted_keys = sorted(_cache.cache.keys())
    for key in sorted_keys:
        alpha, b, cost = _cache.cache[key]
        # 从键名解析参数（格式：mode_sizeX_WX_TX_SX）
        parts = key.split("_")
        mode = parts[0] if parts else "?"
        arr_size = parts[1].replace("size", "") if len(parts) > 1 else "?"
        W = parts[2].replace("W", "") if len(parts) > 2 else "?"
        S_trans = parts[3].replace("T", "") if len(parts) > 3 else "?"
        S_slot = parts[4].replace("S", "") if len(parts) > 4 else "?"

        count += 1
        print(
            f"{count:<4} {mode:<5} {arr_size:<12} {W:<4} {S_trans:<8} {S_slot:<8} {alpha:<8.2f} {b:<4} {cost:<12.4f}"
        )

    print(f"\n总计: {count} 条缓存记录")


def parse_list_arg(arg_str: str, item_type: type = int):
    """解析逗号分隔的列表参数"""
    if not arg_str:
        return None
    arg_str = arg_str.strip()
    return [item_type(x.strip()) for x in arg_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="超参数缓存管理工具 - 支持基于参数值列表的笛卡尔积预计算"
    )
    parser.add_argument("--show", action="store_true", help="显示缓存内容")
    parser.add_argument("--clear", action="store_true", help="清空缓存")

    # 新的笛卡尔积预计算参数
    parser.add_argument("--W", type=str, default=None, help="W值列表，逗号分隔 (例如: 32,64)")
    parser.add_argument(
        "--S-trans", type=str, default=None, help="S_trans值列表，逗号分隔 (例如: 32,64,128)"
    )
    parser.add_argument(
        "--S-slot", type=str, default=None, help="S_slot值列表，逗号分隔 (例如: 4,8,16)"
    )
    parser.add_argument(
        "--arr-size", type=str, default=None, help="数组大小列表，逗号分隔 (例如: 10000,100000)"
    )
    parser.add_argument(
        "--mode", type=str, default=None, help="应用模式列表，逗号分隔 (例如: sss,ir,tc)"
    )

    # 向后兼容的GPU参数（旧方式
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

    # 检查是否使用新的笛卡尔积方式
    use_new_method = any([args.W, args.S_trans, args.S_slot, args.arr_size, args.mode])

    if use_new_method:
        # 新的笛卡尔积预计算方式
        W_values = parse_list_arg(args.W) if args.W else None
        S_trans_values = parse_list_arg(args.S_trans) if args.S_trans else None
        S_slot_values = parse_list_arg(args.S_slot) if args.S_slot else None
        arr_size_values = parse_list_arg(args.arr_size) if args.arr_size else None
        mode_values = parse_list_arg(args.mode, str) if args.mode else None

        print("使用笛卡尔积预计算模式")
        _cache.precompute_common_configs(
            W_values=W_values,
            S_trans_values=S_trans_values,
            S_slot_values=S_slot_values,
            arr_size_values=arr_size_values,
            mode_values=mode_values,
        )
    elif args.gpu != "all" or args.size != "all":
        # 旧的GPU预计算方式（向后兼容）
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
        # 默认使用config.json中的配置进行笛卡尔积预计算
        print("使用config.json中的cache_config配置")
        _cache.precompute_common_configs()

    show_cache()


if __name__ == "__main__":
    main()
