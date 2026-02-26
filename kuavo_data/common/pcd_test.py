#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd


VIDEO_KEYS = {
	"h": {
		"rgb": "observation.images.head_cam_h",
		"depth": "observation.depth_h",
		"pc": "observation.pc_h",
	},
	"l": {
		"rgb": "observation.images.wrist_cam_l",
		"depth": "observation.depth_l",
		"pc": "observation.pc_l",
	},
	"r": {
		"rgb": "observation.images.wrist_cam_r",
		"depth": "observation.depth_r",
		"pc": "observation.pc_r",
	},
}


def parse_chunk_file(path: Path) -> tuple[int, int]:
	chunk_match = re.search(r"chunk-(\d+)", str(path.parent))
	file_match = re.search(r"file-(\d+)\.parquet", path.name)
	if chunk_match is None or file_match is None:
		raise ValueError(f"无法从路径解析 chunk/file: {path}")
	return int(chunk_match.group(1)), int(file_match.group(1))


def load_metadata(lerobot_dir: Path) -> dict:
	info_path = lerobot_dir / "meta" / "info.json"
	if not info_path.exists():
		raise FileNotFoundError(f"未找到元信息文件: {info_path}")
	with info_path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_all_frames(lerobot_dir: Path) -> pd.DataFrame:
	parquet_files = sorted((lerobot_dir / "data").glob("chunk-*/file-*.parquet"))
	if not parquet_files:
		raise FileNotFoundError(f"未找到 parquet 文件: {lerobot_dir / 'data'}")

	dfs = []
	for parquet_path in parquet_files:
		chunk_idx, file_idx = parse_chunk_file(parquet_path)
		df = pd.read_parquet(parquet_path)
		df = df.copy()
		df["__chunk_idx"] = chunk_idx
		df["__file_idx"] = file_idx
		df["__local_frame_idx"] = np.arange(len(df), dtype=np.int64)
		dfs.append(df)

	merged = pd.concat(dfs, ignore_index=True)
	return merged


def pick_frame(df: pd.DataFrame, episode_index: int, target_sec: float) -> pd.Series:
	episode_df = df[df["episode_index"] == episode_index]
	if episode_df.empty:
		valid_episodes = sorted(df["episode_index"].unique().tolist())
		raise ValueError(f"episode_index={episode_index} 不存在，可选: {valid_episodes}")

	distance = np.abs(episode_df["timestamp"].to_numpy() - target_sec)
	best_idx_in_episode = int(np.argmin(distance))
	selected = episode_df.iloc[best_idx_in_episode]
	return selected


def save_point_cloud_ply(points_rgb: np.ndarray, out_path: Path) -> None:
	points_rgb = np.asarray(points_rgb)
	if points_rgb.ndim == 1:
		if points_rgb.size == 0:
			raise ValueError("点云为空")
		first_elem = points_rgb[0]
		if np.isscalar(first_elem):
			if points_rgb.size % 6 == 0:
				points_rgb = points_rgb.reshape(-1, 6)
			elif points_rgb.size % 3 == 0:
				points_rgb = points_rgb.reshape(-1, 3)
		else:
			points_rgb = np.asarray(list(points_rgb), dtype=np.float32)

	if points_rgb.ndim != 2 or points_rgb.shape[1] < 3:
		raise ValueError(f"点云数据形状异常: {points_rgb.shape}")

	points = points_rgb[:, :3].astype(np.float64)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)

	if points_rgb.shape[1] >= 6:
		colors = points_rgb[:, 3:6].astype(np.float64)
		colors = np.clip(colors, 0.0, 1.0)
		pcd.colors = o3d.utility.Vector3dVector(colors)

	ok = o3d.io.write_point_cloud(str(out_path), pcd)
	if not ok:
		raise RuntimeError(f"写入点云失败: {out_path}")


def _read_video_frame_with_ffmpeg(video_path: Path, frame_idx: int, fps: float) -> np.ndarray:
	ffmpeg_bin = shutil.which("ffmpeg")
	if ffmpeg_bin is None:
		raise RuntimeError("OpenCV 无法解码该视频，且系统未安装 ffmpeg")

	timestamp = float(frame_idx) / max(float(fps), 1e-6)
	cmd = [
		ffmpeg_bin,
		"-hide_banner",
		"-loglevel",
		"error",
		"-ss",
		f"{timestamp:.6f}",
		"-i",
		str(video_path),
		"-frames:v",
		"1",
		"-f",
		"image2pipe",
		"-vcodec",
		"png",
		"-",
	]
	result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
	if result.returncode != 0 or not result.stdout:
		raise RuntimeError(
			f"ffmpeg 读取视频帧失败: {video_path}, frame_idx={frame_idx}, stderr={result.stderr.decode('utf-8', errors='ignore')}"
		)

	png_buf = np.frombuffer(result.stdout, dtype=np.uint8)
	frame = cv2.imdecode(png_buf, cv2.IMREAD_COLOR)
	if frame is None:
		raise RuntimeError(f"ffmpeg 输出帧解码失败: {video_path}, frame_idx={frame_idx}")
	return frame


def read_video_frame(video_path: Path, frame_idx: int, fps: float, prefer_ffmpeg: bool = False) -> np.ndarray:
	if not video_path.exists():
		raise FileNotFoundError(f"未找到视频文件: {video_path}")

	if prefer_ffmpeg:
		return _read_video_frame_with_ffmpeg(video_path=video_path, frame_idx=frame_idx, fps=fps)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"无法打开视频: {video_path}")

	cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
	ok, frame = cap.read()
	cap.release()

	if ok and frame is not None:
		return frame

	return _read_video_frame_with_ffmpeg(video_path=video_path, frame_idx=frame_idx, fps=fps)


def export_one_frame(
	lerobot_dir: Path,
	out_root: Path,
	episode_index: int,
	target_sec: float,
) -> Path:
	meta = load_metadata(lerobot_dir)
	fps = float(meta.get("fps", 10))

	df = load_all_frames(lerobot_dir)
	selected = pick_frame(df, episode_index=episode_index, target_sec=target_sec)

	used_episode = int(selected["episode_index"])
	used_timestamp = float(selected["timestamp"])
	used_frame_index = int(selected["frame_index"])
	used_global_index = int(selected["index"])
	used_chunk = int(selected["__chunk_idx"])
	used_file = int(selected["__file_idx"])
	local_frame_idx = int(selected["__local_frame_idx"])

	folder_name = (
		f"episode_{used_episode}_t_{used_timestamp:.3f}s"
		f"_fidx_{used_frame_index}_gidx_{used_global_index}"
	)
	out_dir = out_root / folder_name
	out_dir.mkdir(parents=True, exist_ok=True)

	summary = {
		"request": {
			"episode_index": episode_index,
			"target_sec": target_sec,
		},
		"selected": {
			"episode_index": used_episode,
			"timestamp": used_timestamp,
			"frame_index": used_frame_index,
			"global_index": used_global_index,
			"fps": fps,
			"parquet_chunk": used_chunk,
			"parquet_file": used_file,
			"video_local_frame_idx": local_frame_idx,
		},
	}

	def is_av1(video_key: str) -> bool:
		feature = meta.get("features", {}).get(video_key, {})
		codec = feature.get("info", {}).get("video.codec", "")
		return str(codec).lower() == "av1"

	for cam, keys in VIDEO_KEYS.items():
		pc = selected[keys["pc"]]
		save_point_cloud_ply(pc, out_dir / f"pc_{cam}.ply")

		rgb_video = (
			lerobot_dir
			/ "videos"
			/ keys["rgb"]
			/ f"chunk-{used_chunk:03d}"
			/ f"file-{used_file:03d}.mp4"
		)
		depth_video = (
			lerobot_dir
			/ "videos"
			/ keys["depth"]
			/ f"chunk-{used_chunk:03d}"
			/ f"file-{used_file:03d}.mp4"
		)

		rgb = read_video_frame(
			rgb_video,
			local_frame_idx,
			fps=fps,
			prefer_ffmpeg=is_av1(keys["rgb"]),
		)
		depth = read_video_frame(
			depth_video,
			local_frame_idx,
			fps=fps,
			prefer_ffmpeg=is_av1(keys["depth"]),
		)

		cv2.imwrite(str(out_dir / f"rgb_{cam}.png"), rgb)
		cv2.imwrite(str(out_dir / f"depth_{cam}.png"), depth)

	with (out_dir / "frame_info.json").open("w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	return out_dir


def main() -> None:
	parser = argparse.ArgumentParser(description="导出 LeRobot 数据集中指定时刻的一帧点云与图像")
	parser.add_argument(
		"--lerobot-dir",
		type=Path,
		required=True,
		help="LeRobot 数据集根目录（包含 data/videos/meta）",
	)
	parser.add_argument(
		"--target-sec",
		type=float,
		default=5.0,
		help="目标时间（秒），在指定 episode 内选最近帧",
	)
	parser.add_argument(
		"--episode-index",
		type=int,
		default=0,
		help="episode 序号，默认 0",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		required=True,
		help="输出根目录，会在此目录下自动创建帧文件夹",
	)

	args = parser.parse_args()
	out_dir = export_one_frame(
		lerobot_dir=args.lerobot_dir,
		out_root=args.out_dir,
		episode_index=args.episode_index,
		target_sec=args.target_sec,
	)
	print(f"导出完成: {out_dir}")


if __name__ == "__main__":
	main()
