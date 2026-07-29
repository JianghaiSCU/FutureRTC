#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGBD 转彩色点云的小工具（部署侧使用）

功能概述：
- 对输入的 RGB、深度图进行基础预处理（裁剪、通道规范、深度去噪）；
- 基于相机内参将深度投影为三维坐标，并绑定对应像素的 RGB 颜色；
- 提供距离过滤、Z 轴范围过滤，以及固定点数的采样/补全，输出 (N, 6) 的 XYZRGB 点云。

注意：
- 本文件只封装流程，具体图像/点云步骤由同目录的 `_image_process.py` 与 `_pointcloud_process.py` 提供。
"""

import numpy as np
from typing import Optional, Tuple, List
import h5py
import argparse
import cv2
import json
import os
from _image_process import ImageProcessor
from _pointcloud_process import PointCloudProcessor
import sys
#sys.path.append("C:\\Users\\colonzyx\\AppData\\Roaming\\Python\\Python311\\site-packages")
sys.path.append('/vepfs-share/zouyixian/code/Robotwin_v2/policy/DP3/scripts/visualizer')
# from visualizer.pointcloud import Visualizer, visualize_pointcloud
import visualizer

class RGBDToPointCloudDeploy:
    """RGBD 转点云的部署类。

    参数说明：
    - num_points: 输出点云固定点数（不足会补全，超出会下采样）。
    - max_distance: 距离阈值（单位米），超过此距离的点会被过滤；为 None 或 <=0 则不启用。
    - z_min/z_max: Z 轴过滤范围（单位米），用于去除过近或过远的点；为 None 则不启用对应边界。
    """
    def __init__(
        self,
        num_points: int = 10000,
        use_table_segmentation: bool = False,   # 默认不开启去桌面，开启后需要有fixed_plane_params.npy文件，去除平面
        intrinsics_json: Optional[str] = None,  # 相机内参 JSON 文件路径
        crop_polygon: Optional[List[List[int]]] =None,
        work_space: Optional[list] = None,      # 工作空间限制 [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
        extrinsics: Optional[np.ndarray] = None,
    ) -> None:
        # 图像处理器：设置裁剪区域与深度去噪核大小
        self.img_processor = ImageProcessor(
            crop_polygon = crop_polygon,
            depth_denoise_ksize=5,  # 深度去噪核大小（中值滤波，>=3 生效）
        )
        # 部署参数
        self.extrinsics = (extrinsics if extrinsics is not None else np.eye(4, dtype=float))
        self.use_table_segmentation = use_table_segmentation
        self.num_points = num_points
        self.work_space = work_space

        # 如果提供了 intrinsics_json，则从文件读取对应字段并映射为 PointCloudProcessor 需要的键
        if intrinsics_json is not None:
            try:
                intrinsics_path = os.path.expanduser(os.path.expandvars(intrinsics_json))
                with open(intrinsics_path, 'r') as f:
                    j = json.load(f)
                fx = float( j.get('fx', 0.0))
                fy = float(j.get('fy', 0.0))
                cx = float(j.get('ppx', j.get('cx', 0.0)))
                cy = float(j.get('ppy', j.get('cy', 0.0)))
                depth_scale = float(j.get('depth_scale', 1.0))
                intrinsics = dict(fx=fx, fy=fy, cx=cx, cy=cy)
            except Exception as e:
                raise(f"读取相机内参 JSON 失败 ({intrinsics_json}): {e}")

        # 点云处理器：相机内参与深度缩放因子（mm -> m）
        self.pc_processor = PointCloudProcessor(
            intrinsics = intrinsics,
            depth_scale= depth_scale,
            extrinsics=extrinsics,
        )

    def _grid_sample(self, points: np.ndarray, grid_size: float) -> np.ndarray:
        """对点云利用均匀采样做下采样。

            参数:
                points: (N,C) 数组，前3列为 XYZ，其余列为属性（例如 RGB）
                grid_size: 网格格子大小（米）。若传入 <=0，则根据期望采样数自适应计算一个合适的 grid_size。

            返回:
                采样后的点数组（从每个非空格子中保留一个代表点，返回顺序依赖于原始点顺序）。
        """
        xyz = points[:, :3]
        if xyz.size == 0:
            return points
        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        xyz_range = np.maximum(xyz_max - xyz_min, 1e-9)
        volume = float(xyz_range[0] * xyz_range[1] * xyz_range[2])
        vol_per_pt = volume / max(1, self.num_points)
        if grid_size <= 0:
            grid_size = max(np.power(max(vol_per_pt, 1e-12), 1.0 / 3.0), 0.001)

        scaled = xyz / grid_size
        grid = np.floor(scaled).astype(np.int64)
        off = np.min(grid, axis=0)
        grid -= off
        max_dim = np.max(grid, axis=0) + 1
        keys = grid[:, 0] + grid[:, 1] * max_dim[0] + grid[:, 2] * max_dim[0] * max_dim[1]
        _, idx = np.unique(keys, return_index=True)
        return points[idx]
    
    def grid_downsample(self, points: np.ndarray) -> np.ndarray:
        """对 points 应用网格/体素下采样的封装函数。

        参数:
            points: 输入点云数组，形状 (N, C)，前 3 列为 XYZ。
        返回:
            网格下采样后的点云数组。
        """
        spread = np.ptp(points[:, :3], axis=0)
        grid_size = max(float(np.max(spread)) / 100.0 if spread.size else 0.0, 0.001)

        out = self._grid_sample(points, grid_size)

        # fix to num_points
        if out.shape[0] > self.num_points:
            sel = np.random.choice(out.shape[0], self.num_points, replace=False)
            out = out[sel]
        elif out.shape[0] < self.num_points:
            if out.shape[0] > 0:
                rep = np.random.choice(out.shape[0], self.num_points - out.shape[0], replace=True)
                out = np.vstack([out, out[rep]])
            else:
                out = np.zeros((self.num_points, points.shape[1]), dtype=np.float32)

        return out.astype(np.float32)


    def process_single_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """处理单帧 RGBD，输出彩色点云。

        参数：
        - rgb_frame: (H, W, 3) 的 RGB 图像（uint8/float32）。
        - depth_frame: (H, W) 的深度图（uint16/float32），单位与相机深度缩放对应。

        返回：
        - (num_points, 6) 的 float32 数组，列含义为 [X, Y, Z, R, G, B]。
        """
        # --------------------------
        # 步骤1：图像预处理
        # --------------------------
        # RGB 预处理：裁剪（固定区域）、通道规范（BGR->RGB）
        processed_rgb = self.img_processor.process_rgb(rgb_frame)
        # Depth 预处理：裁剪、去噪、保持与 RGB 对齐
        processed_depth = self.img_processor.process_depth(depth_frame)

        # --------------------------
        # 步骤2：原始点云生成，包括标定坐标转换
        # --------------------------
        try:
            # 期望采样点数（用于网格采样密度估计）
            self.pc_processor.expected_samples = self.num_points
            # RGBD 生成彩色点云（启用网格采样 + 距离过滤）
            pcd = self.pc_processor.rgbd_to_colored_point_cloud(
                rgb=processed_rgb,
                depth=processed_depth,
                use_table_segmentation=self.use_table_segmentation,
            )

        # --------------------------
        # 步骤3：工作空间裁剪
        # --------------------------     
            # 工作空间裁剪（可选）
            if self.work_space is not None and len(self.work_space) == 3:
                pcd = pcd[np.where((pcd[..., 0] > self.work_space[0][0]) & (pcd[..., 0] < self.work_space[0][1]) &
                                (pcd[..., 1] > self.work_space[1][0]) & (pcd[..., 1] < self.work_space[1][1]) &
                                (pcd[..., 2] > self.work_space[2][0]) & (pcd[..., 2] < self.work_space[2][1]))]

        except Exception as e:
            # 异常回退：返回全零点云（部署容错）
            print(f"单帧点云生成异常: {str(e)}，返回全零点云")
            pcd = np.zeros((self.num_points, 6), dtype=np.float32)
        
        return pcd.astype(np.float32)  # 统一输出float32，减少部署内存占用


    def process_hand_camera_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, position:np.ndarray) -> np.ndarray:  
        """对于眼在手上的相机，处理单帧 RGBD，输出彩色点云。"""
        # --------------------------
        # 步骤1：图像预处理
        # --------------------------
        # RGB 预处理：裁剪（固定区域）、通道规范（BGR->RGB）
        processed_rgb = self.img_processor.process_rgb(rgb_frame)
        # Depth 预处理：裁剪、去噪、保持与 RGB 对齐
        processed_depth = self.img_processor.process_depth(depth_frame)

        matrix_one = np.array(position).reshape(4, 4)
        # print("matrix_one:\n", matrix_one)

        # --------------------------
        # 步骤2：点云生成
        # --------------------------
        try:
            # 期望采样点数（用于网格采样密度估计）
            self.pc_processor.expected_samples = self.num_points
            # RGBD 生成彩色点云（启用网格采样 + 距离过滤）
            pcd = self.pc_processor.rgbd_to_colored_point_cloud(
                rgb=processed_rgb,
                depth=processed_depth,
                use_table_segmentation=self.use_table_segmentation,
            )

            # 眼在手上生成点云，多一步利用机器人末端位姿转换到基座坐标系
            pcd = self.pc_processor.transform_points(pcd, matrix_one)

        # --------------------------
        # 步骤3：工作空间裁剪
        # --------------------------     
            # 工作空间裁剪（可选）
            if self.work_space is not None and len(self.work_space) == 3:
                pcd = pcd[np.where((pcd[..., 0] > self.work_space[0][0]) & (pcd[..., 0] < self.work_space[0][1]) &
                                (pcd[..., 1] > self.work_space[1][0]) & (pcd[..., 1] < self.work_space[1][1]) &
                                (pcd[..., 2] > self.work_space[2][0]) & (pcd[..., 2] < self.work_space[2][1]))]
        
        except Exception as e:
            # 异常回退：返回全零点云（部署容错）
            print(f"单帧点云生成异常: {str(e)}，返回全零点云")
            pcd = np.zeros((self.num_points, 6), dtype=np.float32)

        return pcd.astype(np.float32)  # 统一输出float32，减少部署内存占用
    

# --------------------------
# 简单示例（实际接入时请替换为真实相机/文件输入）
# --------------------------
def visualize_point_cloud(
    pcd: np.ndarray,
    title: str = "Point Cloud",
    show_axes: bool = False,
    axes_poses: Optional[list] = None,
    axis_size: float = 0.1,
) -> None:
    """可视化 (N,6) XYZRGB 点云。

    支持显示坐标轴：
    - 如果 `show_axes` 为 True 且 `axes_poses` 为 None，则在原点显示一个坐标系。
    - 如果 `axes_poses` 为 list	of 4x4 numpy 矩阵，则会为每个 pose 绘制一个坐标系（Open3D）或箭头（Matplotlib）。

    优先使用 Open3D；若不可用则回退到 Matplotlib 3D 散点。
    """
    try:
        import open3d as o3d
        pts = pcd[:, :3].astype(np.float64)
        cols = pcd[:, 3:6].astype(np.float64)
        # 颜色归一化到 [0,1]
        if np.nanmax(cols) > 1.5:
            cols = np.clip(cols / 255.0, 0.0, 1.0)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(cols)

        geometries = [pc]
        if show_axes:
            # 如果没有传入 axes_poses，则在原点创建一个坐标系
            if axes_poses is None:
                axes_poses = [np.eye(4, dtype=np.float64)]
            for pose in axes_poses:
                try:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
                    frame.transform(np.asarray(pose, dtype=np.float64))
                    geometries.append(frame)
                except Exception:
                    # 若 pose 不是 numpy 数组或维度不对，跳过
                    continue

        o3d.visualization.draw_geometries(geometries, window_name=title)
        return
    except Exception as e:
        print(f"Open3D 可视化失败，改用 Matplotlib：{e}")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        pts = pcd[:, :3]
        cols = pcd[:, 3:6]
        # 颜色归一化到 [0,1]
        if np.nanmax(cols) > 1.5:
            cols = np.clip(cols / 255.0, 0.0, 1.0)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        # 点大小根据数量自适应，避免过密
        s = max(0.5, 40000.0 / max(pts.shape[0], 1))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=s, marker='.', linewidths=0)

        # 绘制坐标轴（Matplotlib）
        if show_axes:
            if axes_poses is None:
                axes_poses = [np.eye(4, dtype=np.float64)]
            for pose in axes_poses:
                try:
                    pose = np.asarray(pose, dtype=np.float64)
                    origin = pose[:3, 3]
                    # x, y, z 方向向量（列）
                    x_dir = pose[:3, 0]
                    y_dir = pose[:3, 1]
                    z_dir = pose[:3, 2]
                    ax.quiver(
                        origin[0], origin[1], origin[2],
                        x_dir[0], x_dir[1], x_dir[2],
                        length=axis_size, color='r', normalize=True
                    )
                    ax.quiver(
                        origin[0], origin[1], origin[2],
                        y_dir[0], y_dir[1], y_dir[2],
                        length=axis_size, color='g', normalize=True
                    )
                    ax.quiver(
                        origin[0], origin[1], origin[2],
                        z_dir[0], z_dir[1], z_dir[2],
                        length=axis_size, color='b', normalize=True
                    )
                except Exception:
                    continue

        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.tight_layout()
        plt.show()
    except Exception as e2:
        print(f"Matplotlib 可视化失败：{e2}")

        
def _ensure_depth_2d(depth: np.ndarray) -> np.ndarray:
    """将深度数据规范为 (H, W) 形状。支持 (H, W) 或 (H, W, 1)。"""
    if depth.ndim == 2:
        return depth
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return depth[..., 0]
    raise ValueError(f"不支持的 depth 形状: {depth.shape}，期望 (H,W) 或 (H,W,1)")


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """将 RGB 规范为 uint8 [0,255]。若为 float 且范围[0,1]，则缩放。"""
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating):
        maxv = float(np.nanmax(rgb)) if rgb.size > 0 else 1.0
        if maxv <= 1.5:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        return rgb
    return rgb.astype(np.uint8)


def _decode_img_bytes_to_bgr(data: bytes) -> np.ndarray:
    """将 JPEG/PNG 字节解码为 BGR 图像 (H,W,3)。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图像字节解码失败：返回 None")
    return img


def _decode_depth_bytes_to_2d(data: bytes) -> np.ndarray:
    """将深度 PNG/JPEG 字节解码为 (H,W) 深度图，保持原通道或取单通道。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    dep = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise ValueError("深度字节解码失败：返回 None")
    if dep.ndim == 3:
        # 若解码得到多通道（少见），取第一通道
        dep = dep[..., 0]
    return dep

def _load_h5_size(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        dataset = f["robot"]["matrix"]
        return dataset.shape[0]

def load_h5_position(
    h5_path: str,
    frame_idx: int = 0,
) -> np.ndarray:

    with h5py.File(h5_path, 'r') as f:
        if 'robot' not in f:
            raise KeyError("HDF5 中未找到 'robot' 数据集")
        pos_ds = f["robot"]["matrix"][frame_idx]
        # pos_ds = f["robot"]["state_Q"][frame_idx][:6]
        

        return pos_ds.astype(np.float32)

def load_h5_rgb_depth(
    h5_path: str,
    camera_name: Optional[str] = None,
    frame_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """从 HDF5 读取一帧 RGB 与 Depth。

    约定结构（推测）：/camera/<camera_name>/{rgb,depth}
    - rgb: (H,W,3) 或 (T,H,W,3)
    - depth: (H,W) / (H,W,1) 或 (T,H,W)/(T,H,W,1)

    返回: (rgb, depth, resolved_camera_name)
    """
    with h5py.File(h5_path, 'r') as f:
        if 'camera' not in f:
            raise KeyError("HDF5 中未找到 'camera' 组")
        cam_group = f['camera']

        # 自动选择相机名称
        if camera_name is None:
            keys = list(cam_group.keys())
            if not keys:
                raise KeyError("'camera' 组下没有任何相机子组")
            camera_name = keys[0]

        if camera_name not in cam_group:
            raise KeyError(f"指定相机 '{camera_name}' 不存在。可用相机: {list(cam_group.keys())}")

        g = cam_group[camera_name]
        if 'rgb' not in g or 'depth' not in g:
            raise KeyError(f"/camera/{camera_name} 下未找到 'rgb' 或 'depth' 数据集")

        rgb_ds = g['rgb']
        dep_ds = g['depth']

        # 读取第 frame_idx 帧或单帧
        if rgb_ds.ndim == 4:
            rgb = rgb_ds[frame_idx]
        elif rgb_ds.ndim == 1:
            # (T,) - 每个元素可能是变长字节
            item = rgb_ds[min(frame_idx, rgb_ds.shape[0]-1)]
            if isinstance(item, (bytes, bytearray, np.void)):
                rgb = _decode_img_bytes_to_bgr(bytes(item))
            else:
                rgb = np.asarray(item)
        else:
            # 可能是单帧 (H,W,3) 或标量字节
            data = rgb_ds[...]
            if isinstance(data, (bytes, bytearray, np.void)):
                rgb = _decode_img_bytes_to_bgr(bytes(data))
            else:
                rgb = np.asarray(data)
        if dep_ds.ndim == 4:
            # (T,H,W,C)
            depth = dep_ds[frame_idx]
        elif dep_ds.ndim == 3:
            # (T,H,W) 或 (H,W,1)
            if dep_ds.shape[-1] in (1, 3):
                # (T,H,W,1) 压成了 3 维（h5py 某些存法），仍按时间维读取
                depth = dep_ds[frame_idx]
            else:
                depth = dep_ds[frame_idx] if dep_ds.shape[0] > 1 else dep_ds[...]
        elif dep_ds.ndim == 1:
            # (T,) - 字节序列
            item = dep_ds[min(frame_idx, dep_ds.shape[0]-1)]
            if isinstance(item, (bytes, bytearray, np.void)):
                depth = _decode_depth_bytes_to_2d(bytes(item))
            else:
                depth = np.asarray(item)
        else:
            # 标量或单帧
            data = dep_ds[...]
            if isinstance(data, (bytes, bytearray, np.void)):
                depth = _decode_depth_bytes_to_2d(bytes(data))
            else:
                depth = np.asarray(data)

        rgb = _as_uint8_rgb(np.asarray(rgb))
        depth = _ensure_depth_2d(np.asarray(depth))
        return rgb, depth, camera_name


def load_h5_rgb_depth_batch(
    h5_path: str,
    camera_name: Optional[str] = None,
    start_idx: int = 0,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """批量读取多帧 RGB 与 Depth。若数据集为单帧，将重复该帧至 batch。"""
    with h5py.File(h5_path, 'r') as f:
        if 'camera' not in f:
            raise KeyError("HDF5 中未找到 'camera' 组")
        cam_group = f['camera']
        if camera_name is None:
            keys = list(cam_group.keys())
            if not keys:
                raise KeyError("'camera' 组下没有任何相机子组")
            camera_name = keys[0]
        if camera_name not in cam_group:
            raise KeyError(f"指定相机 '{camera_name}' 不存在。可用相机: {list(cam_group.keys())}")

        g = cam_group[camera_name]
        if 'rgb' not in g or 'depth' not in g:
            raise KeyError(f"/camera/{camera_name} 下未找到 'rgb' 或 'depth' 数据集")

        rgb_ds = g['rgb']
        dep_ds = g['depth']

        # 判断是否有时间维
        has_time_rgb = (rgb_ds.ndim == 4)
        has_time_dep = (dep_ds.ndim == 4) or (dep_ds.ndim == 3 and dep_ds.shape[-1] == 1)

        rgb_list: List[np.ndarray] = []
        dep_list: List[np.ndarray] = []
        for i in range(batch_size):
            t = start_idx + i
            # 读取 RGB
            if has_time_rgb:
                item = rgb_ds[min(t, rgb_ds.shape[0]-1)]
            else:
                item = rgb_ds[...]
            if isinstance(item, (bytes, bytearray, np.void)):
                rgb = _decode_img_bytes_to_bgr(bytes(item))
            else:
                rgb = np.asarray(item)

            # 读取 Depth
            if dep_ds.ndim == 4:
                ditem = dep_ds[min(t, dep_ds.shape[0]-1)]
            elif dep_ds.ndim == 3 and (dep_ds.shape[-1] == 1 or dep_ds.shape[-1] == 3):
                ditem = dep_ds[min(t, dep_ds.shape[0]-1)]
            elif dep_ds.ndim == 3:
                ditem = dep_ds[min(t, dep_ds.shape[0]-1)]
            else:
                ditem = dep_ds[...]
            if isinstance(ditem, (bytes, bytearray, np.void)):
                depth = _decode_depth_bytes_to_2d(bytes(ditem))
            else:
                depth = np.asarray(ditem)

            rgb_list.append(_as_uint8_rgb(rgb))
            dep_list.append(_ensure_depth_2d(depth))

        rgb_batch = np.stack(rgb_list, axis=0)
        depth_batch = np.stack(dep_list, axis=0)
        return rgb_batch, depth_batch, camera_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 HDF5 读取 RGBD 并生成点云")
    parser.add_argument('--h5', type=str, help='HDF5 文件路径',  default="/vepfs-share/zouyixian/data/z1_xipan/box_one_new/episode9.hdf5")
    parser.add_argument('--frame', type=int, default=270, help='单帧索引（默认0）')
    parser.add_argument('--num_points', type=int, default=8092, help='输出点云点数')

    args = parser.parse_args()

    # 1a) 初始化 - 左手相机
    extrinsics_hand =    np.array([
            [0, 0, 1, -0.06],
            [-1, 0, 0, 0.06],
            [0, -1, 0, 0.065],
            [0, 0, 0, 1]
        ], dtype=float)

    # extrinsics_hand =    np.array([
    #         [0, 0, 1, -0.0241-0.035],
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0.081-0.025],
    #         [0, 0, 0, 1]
    #     ], dtype=float)
    rgbd2pcd_hand  = RGBDToPointCloudDeploy(
        num_points=args.num_points,
        intrinsics_json = '/vepfs-share/zouyixian/data/z1_xipan/box_one_new/intrinsics/left_camera_intr_depth_d455.json',
        extrinsics= extrinsics_hand,
        # work_space = [[0, 0.55], [-0.4, 0.6],  [-1, 0.6]],
        work_space = [[0, 0.55], [-0.4, 0.6],  [0.053, 0.6]],
    )

    # 1b) 初始化 - 第三方相机
    extrinsics_head =    np.array(
         [[ 0.07560813 ,-0.79225732  ,0.6054854,  -0.01274354],
            [-0.99713381, -0.05841193 , 0.04808437, -0.18953245],
            [-0.00272764, -0.60738551, -0.79440299 , 0.6540201 ],
            [ 0.    ,      0.    ,      0.   ,       1.        ]]
                                , dtype=float)

    rgbd2pcd_head = RGBDToPointCloudDeploy(
        num_points=args.num_points,
        intrinsics_json='/vepfs-share/zouyixian/data/z1_xipan/box_one_new/intrinsics/head_camera_intr_depth_d455.json',
        work_space = [[0, 0.55], [-0.4, 0.6],  [0.053, 0.6]],
        # work_space = [[0, 0.55], [-0.4, 0.6],  [0.0, 0.6]],
        extrinsics = extrinsics_head,
    )

    num = _load_h5_size(args.h5)
    print(f"num = {num}")
    step = 8

    # for i in range(0, num - step, step): 
    for i in range(1): 
        args.frame = i


        # 2a) 单帧读取与处理
        rgb, depth, cam_used = load_h5_rgb_depth(
            h5_path=args.h5,
            camera_name="head_camera",
            frame_idx=args.frame,
        )

        single_pcd_head = rgbd2pcd_head.process_single_frame(rgb_frame=rgb, depth_frame=depth)
        print(f"相机: {cam_used} | 单帧点云形状: {single_pcd_head.shape} (格式: [num_points, XYZRGB])")

        # 2a) 单帧读取与处理
        rgb, depth, cam_used = load_h5_rgb_depth(
            h5_path=args.h5,
            camera_name='left_camera',
            frame_idx=args.frame,
        )
        position = load_h5_position(
            h5_path=args.h5,
            frame_idx=args.frame,
        )

        single_pcd_hand = rgbd2pcd_hand.process_hand_camera_frame(rgb_frame=rgb, depth_frame=depth, position=position)
        if single_pcd_hand.shape[0] > 5000:
            print(f"相机: {cam_used} | 单帧点云形状: {single_pcd_hand.shape} (格式: [num_points, XYZRGB])")
            single_pcd = np.vstack([single_pcd_hand, single_pcd_head]) 
        else:
            single_pcd = single_pcd_head

        # 统一下采样
        single_pcd = rgbd2pcd_hand.grid_downsample(single_pcd)
        print(f"统一下采样后点云形状: {single_pcd.shape} (格式: [num_points, XYZRGB])")

        # 可视化单帧点云 — 同时显示基座坐标系（origin）和相机坐标系（在基座下的位置）
        try:
            # 添加一个红色原点标记
            origin = np.array([[0.0, 0.0, 0.0, 255, 0, 0]], dtype=single_pcd.dtype)
            points_with_origin = np.vstack([single_pcd, origin])
            visualizer.visualize_pointcloud(points_with_origin)
            # visualizer.visualize_pointcloud_and_save(pointcloud=points_with_origin, save_path=f"./pcd/{args.frame}.png")
        except Exception as e:
            print(f"可视化失败（单帧）：{e}")

        
