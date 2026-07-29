#!/usr/bin/env python3
"""Create a frame-subsampled copy of a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import cv2
import numpy as np
import pandas as pd
import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
META_FILES_TO_REWRITE = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}


def stride_indices(length: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if length <= 0:
        return []
    return list(range(0, length, stride))


def subsample_frame_table(
    df: pd.DataFrame,
    *,
    stride: int,
    episode_index: int,
    global_start_index: int,
    fps: float,
) -> pd.DataFrame:
    indices = stride_indices(len(df), stride)
    out = df.iloc[indices].copy().reset_index(drop=True)
    length = len(out)
    out["timestamp"] = np.arange(length, dtype=np.float32) / fps
    out["frame_index"] = np.arange(length, dtype=np.int64)
    out["episode_index"] = np.full(length, episode_index, dtype=np.int64)
    out["index"] = np.arange(global_start_index, global_start_index + length, dtype=np.int64)
    return out


def update_info(info: dict, *, fps: float, episode_lengths: list[int]) -> dict:
    updated = dict(info)
    updated["fps"] = fps
    updated["total_frames"] = int(sum(episode_lengths))
    updated["total_episodes"] = len(episode_lengths)
    return updated


def _as_array(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray | list | tuple):
        return np.stack(values)
    return values.reshape(-1, 1)


def _stats_for_array(array: np.ndarray) -> dict:
    array = np.asarray(array, dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def compute_episode_stats(episode_index: int, df: pd.DataFrame) -> dict:
    skip_columns = {"episode_index", "index", "task_index"}
    stats = {}
    for column in df.columns:
        if column in skip_columns:
            continue
        array = _as_array(df[column])
        if not np.issubdtype(array.dtype, np.number):
            continue
        stats[column] = _stats_for_array(array)
    return {"episode_index": episode_index, "stats": stats}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_static_files(src: pathlib.Path, dst: pathlib.Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir():
            continue
        if rel.parts[0] in {"data", "videos"}:
            continue
        if rel.parts[0] == "meta" and rel.name in META_FILES_TO_REWRITE:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def parse_episode_index(path: pathlib.Path) -> int:
    stem = path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(stem.removeprefix("episode_"))


def subsample_parquets(src: pathlib.Path, dst: pathlib.Path, *, stride: int, fps: float) -> tuple[list[dict], list[dict]]:
    parquet_paths = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under {src / 'data'}")

    episodes = []
    episode_stats = []
    global_index = 0
    for parquet_path in tqdm.tqdm(parquet_paths, desc="Subsampling parquet"):
        episode_index = parse_episode_index(parquet_path)
        episode_table = pd.read_parquet(parquet_path)
        out = subsample_frame_table(
            episode_table,
            stride=stride,
            episode_index=episode_index,
            global_start_index=global_index,
            fps=fps,
        )
        global_index += len(out)

        rel = parquet_path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)

        episodes.append({"episode_index": episode_index, "tasks": _episode_tasks(src, episode_index), "length": len(out)})
        episode_stats.append(compute_episode_stats(episode_index, out))

    return episodes, episode_stats


def _episode_tasks(src: pathlib.Path, episode_index: int) -> list[str]:
    episodes_path = src / "meta" / "episodes.jsonl"
    for row in read_jsonl(episodes_path):
        if row["episode_index"] == episode_index:
            return row.get("tasks", [])
    return []


def subsample_video(src_video: pathlib.Path, dst_video: pathlib.Path, *, stride: int, fps: float) -> int:
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video: {dst_video}")

    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            writer.write(frame)
            written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    return written


def subsample_videos(src: pathlib.Path, dst: pathlib.Path, *, stride: int, fps: float) -> None:
    video_paths = sorted(path for path in (src / "videos").rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    for video_path in tqdm.tqdm(video_paths, desc="Subsampling videos"):
        subsample_video(video_path, dst / video_path.relative_to(src), stride=stride, fps=fps)


def write_meta(src: pathlib.Path, dst: pathlib.Path, *, fps: float, episodes: list[dict], episode_stats: list[dict]) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    info = update_info(info, fps=fps, episode_lengths=[row["length"] for row in episodes])
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n")
    write_jsonl(dst / "meta" / "episodes.jsonl", episodes)
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", episode_stats)


def subsample_dataset(src: pathlib.Path, dst: pathlib.Path, *, stride: int, overwrite: bool = False) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = float(info["fps"]) / stride
    copy_static_files(src, dst)
    episodes, episode_stats = subsample_parquets(src, dst, stride=stride, fps=fps)
    subsample_videos(src, dst, stride=stride, fps=fps)
    write_meta(src, dst, fps=fps, episodes=episodes, episode_stats=episode_stats)
    print(f"Done: {src} -> {dst}")
    print(f"stride={stride}, fps={info['fps']} -> {fps}, total_frames={sum(row['length'] for row in episodes)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="source LeRobot dataset directory")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="destination LeRobot dataset directory")
    parser.add_argument("--stride", type=int, default=4, help="keep one frame every N frames")
    parser.add_argument("--overwrite", action="store_true", help="replace destination if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsample_dataset(args.src, args.dst, stride=args.stride, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
