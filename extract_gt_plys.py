#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
镜像子目录结构，并将每个场景的点云(来自 point_clouds/0 与 point_clouds/1)
按编号合并并体素下采样，只把结果写到镜像的对应子目录。

用法示例：
    python mirror_merge_downsample.py \
        --in_root /path/to/input_root \
        --out_root /path/to/output_root \
        --voxel 0.01 \
        --count 30 \
        --src0 0 --src1 1 \
        --pattern "{:d}.ply" \
        --clean

依赖：
    pip install open3d tqdm
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import open3d as o3d
from tqdm import tqdm
import shutil


def read_pcd(path: Path) -> Optional[o3d.geometry.PointCloud]:
    if not path.exists():
        return None
    try:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            return None
        return pcd
    except Exception as e:
        print(f"[WARN] 读取失败：{path} -> {e}", file=sys.stderr)
        return None


def merge_and_downsample(p0: o3d.geometry.PointCloud,
                         p1: o3d.geometry.PointCloud,
                         voxel: float) -> o3d.geometry.PointCloud:
    merged = p0 + p1
    if voxel and voxel > 0:
        merged = merged.voxel_down_sample(voxel_size=voxel)
    return merged


def process_scene(scene_in_dir: Path,
                  scene_out_dir: Path,
                  src0: str, src1: str,
                  count: int, voxel: float,
                  pattern: str) -> None:
    pc_root = scene_in_dir / "point_clouds"
    d0 = pc_root / src0
    d1 = pc_root / src1
    if not (d0.is_dir() and d1.is_dir()):
        print(f"[SKIP] 缺少 point_clouds/{src0} 或 {src1} ：{scene_in_dir}")
        return

    scene_out_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    done = 0
    for idx in range(count):
        f0 = d0 / pattern.format(idx)
        f1 = d1 / pattern.format(idx)
        p0 = read_pcd(f0)
        p1 = read_pcd(f1)
        if p0 is None or p1 is None:
            missing += 1
            print(f"[WARN] 缺失或空点云：{f0 if p0 is None else ''} {f1 if p1 is None else ''}")
            continue

        merged = merge_and_downsample(p0, p1, voxel)
        out_path = scene_out_dir / pattern.format(idx)
        ok = o3d.io.write_point_cloud(str(out_path), merged)
        if not ok:
            print(f"[WARN] 写入失败：{out_path}", file=sys.stderr)
        else:
            done += 1

    print(f"[INFO] {scene_in_dir.name}: 写出 {done} 个，缺失/跳过 {missing} 个。")


def main():
    ap = argparse.ArgumentParser(description="镜像目录 + 合并并下采样点云(0/1)到镜像目录")
    ap.add_argument("--in_root", type=Path, required=True, help="输入根目录（其下为多个场景子目录）")
    ap.add_argument("--out_root", type=Path, required=True, help="输出根目录（将镜像子目录结构）")
    ap.add_argument("--voxel", type=float, default=0.01, help="体素下采样大小，默认 0.01")
    ap.add_argument("--count", type=int, default=30, help="编号个数：30 表示 0..29")
    ap.add_argument("--src0", type=str, default="0", help="point_clouds 下第一个源子目录名，默认 '0'")
    ap.add_argument("--src1", type=str, default="1", help="point_clouds 下第二个源子目录名，默认 '1'")
    ap.add_argument("--pattern", type=str, default="{:d}.ply", help="文件名模式，默认 '{:d}.ply'")
    ap.add_argument("--clean", action="store_true",
                    help="若目标场景子目录已存在，先清空再写入")
    args = ap.parse_args()

    if not args.in_root.is_dir():
        print(f"[ERROR] 输入根目录不存在：{args.in_root}", file=sys.stderr)
        sys.exit(1)

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 仅镜像一级子目录（问题描述是：根目录下一堆文件夹）
    scene_in_dirs = sorted([p for p in args.in_root.iterdir() if p.is_dir()])
    if not scene_in_dirs:
        print(f"[WARN] {args.in_root} 下未发现子目录")
        sys.exit(0)

    for scene_in in tqdm(scene_in_dirs, desc="处理场景"):
        # 目标为 out_root/同名子目录
        scene_out = args.out_root / scene_in.name

        if args.clean and scene_out.exists():
            shutil.rmtree(scene_out, ignore_errors=True)

        # 只创建空壳，然后写合并后的 ply；不复制其它结构/文件
        scene_out.mkdir(parents=True, exist_ok=True)

        process_scene(scene_in, scene_out,
                      args.src0, args.src1,
                      args.count, args.voxel,
                      args.pattern)

    print("[DONE] 全部完成。")


if __name__ == "__main__":
    main()