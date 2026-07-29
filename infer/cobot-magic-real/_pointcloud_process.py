      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云处理工具

功能：
- 将 RGBD 反投影为三维坐标 XYZ
- 为每个点附加 RGB 颜色，得到 XYZRGB 格式
- 可选的预处理：坐标变换、工作空间裁剪、桌面分割、孤立点去除

Usage:
    from pointcloud_processing import PointCloudProcessor
    pcp = PointCloudProcessor(
        intrinsics=dict(fx=381.153, fy=380.674, cx=310.863, cy=244.217),
        depth_scale=0.0010000000474974513,
    )
    pc6 = pcp.rgbd_to_colored_point_cloud(rgb, depth, use_advanced=True, num_points=4000)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
import os
try:
    from termcolor import cprint
except Exception:
    def cprint(msg, color=None):
        print(msg)

class PointCloudProcessor:
    def __init__(
        self,
        intrinsics: Dict[str, float],
        depth_scale: float = 1.0,
        extrinsics: Optional[np.ndarray] = None,
    ) -> None:
        self.fx = float(intrinsics.get('fx'))
        self.fy = float(intrinsics.get('fy'))
        self.cx = float(intrinsics.get('cx'))
        self.cy = float(intrinsics.get('cy'))
        self.depth_scale = float(depth_scale)
        self.extrinsics = (
            extrinsics.astype(np.float32) if extrinsics is not None else np.eye(4, dtype=np.float32)
        )

    def transform_points(self, pc: np.ndarray, extrinsics) -> np.ndarray:
        """将点云的前三列 XYZ 通过 self.extrinsics 进行坐标变换并返回新的数组。

        说明：该函数不会修改输入数组（会先复制），返回值与输入列数相同，
        只对前 3 列（XYZ）进行变换，保留后续的属性列（例如 RGB）。

        参数:
            pc: 输入点云，形状 (N, C)。若为空则直接返回空数组副本。
        返回:
            经过外参变换后的点云数组（dtype float32）。
        """
        if pc.size == 0:
            return pc.astype(np.float32).copy()
        out = pc.astype(np.float32).copy()
        xyz = out[:, :3]
        ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_h = np.hstack([xyz, ones])
        # 注意矩阵乘法顺序：extrinsics (4x4) 与 xyz_h.T (4xN)
        xyz_w = (extrinsics @ xyz_h.T).T
        out[:, :3] = xyz_w[:, :3]

        return out


    
    def rgbd_to_colored_point_cloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,                                
        use_table_segmentation: bool = False,       # 默认不开启去桌面，开启后需要有fixed_plane_params.npy文件，去除平面
        crop_offset: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """将深度反投影为 XYZ 并附加 RGB 颜色 -> (N,6) float32。
        输入：rgb 为 HxWx3，depth 为 HxW；假定 rgb 值范围为 0..255。
        参数：
            use_table_segmentation: 若为 True（默认），会尝试分割并移除桌面平面，
                然后再进行其他预处理步骤。设置为 False 可跳过平面分割。
        """
        # 生成原始点云
        ys, xs = np.where(depth > 0)
        z = depth[ys, xs].astype(np.float32) * float(self.depth_scale)
        if crop_offset is None:
            ox, oy = 0.0, 0.0
        else:
            ox, oy = float(crop_offset[0]), float(crop_offset[1])
        x3 = ((xs.astype(np.float32) + ox) - self.cx) * z / self.fx
        y3 = ((ys.astype(np.float32) + oy) - self.cy) * z / self.fy
        xyz = np.stack([x3, y3, z], axis=1).astype(np.float32)
        if rgb.ndim == 3 and rgb.shape[2] >= 3:
            colors = rgb[ys, xs, :3].astype(np.float32)
        else:
            colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        pc = np.hstack([xyz, colors]).astype(np.float32)

        # 坐标变换   
        pc = self.transform_points(pc, self.extrinsics)

        # 利用桌面的平面方程去除桌面（仅在需要时启用）
        if use_table_segmentation:
            table_points, non_table_points = self.segment_table_point(pc)
            pc = non_table_points

        # 去除孤立点
        pc = self.remove_outliers_by_neighbors(pc, k=10, threshold=0.05)

        return pc

    def load_plane_params(self, load_path: Optional[str] = None):
        """加载预保存的固定平面方程参数 (a,b,c,d).

        If load_path is None, default to `fixed_plane_params.npy` located in the
        same directory as this module.
        """
        try:
            if load_path is None:
                module_dir = os.path.dirname(os.path.abspath(__file__))
                load_path = os.path.join(module_dir, 'fixed_plane_params.npy')
            load_path = os.path.expanduser(os.path.expandvars(load_path))
            return np.load(load_path)
        except Exception as e:
            cprint(f"加载平面参数失败 (path={load_path}): {e}", "red")
            raise

    def segment_plane_with_fixed_model(self, o3d_pcd, plane_params, distance_threshold=0.01):
        """
        用固定平面方程分割Open3D点云，返回桌面和非桌面点云（Open3D对象）
        参数：
            o3d_pcd: Open3D点云对象（输入的原始点云）
            plane_params: 固定平面方程参数 (a,b,c,d)，满足 ax + by + cz + d = 0
            distance_threshold: 距离平面小于该阈值的点视为桌面点
        返回：
            table_o3d: 桌面点云（Open3D对象）
            non_table_o3d: 非桌面点云（Open3D对象）
        """
        # 将Open3D点云转换为numpy数组 (N,3)
        pcd_points = np.asarray(o3d_pcd.points)
        
        # 解析平面方程参数
        a, b, c, d = plane_params
        
        # 计算每个点到平面的距离：|ax + by + cz + d| / sqrt(a² + b² + c²)
        numerator = np.abs(a * pcd_points[:, 0] + b * pcd_points[:, 1] + c * pcd_points[:, 2] + d)
        denominator = np.sqrt(a**2 + b**2 + c**2)
        distances = numerator / denominator  # 形状 (N,)
        
        # 筛选桌面点的索引（距离小于等于阈值）
        table_indices = np.where(distances <= distance_threshold)[0]
        
        # 分割桌面和非桌面点云（Open3D对象）
        table_o3d = o3d_pcd.select_by_index(table_indices)
        non_table_o3d = o3d_pcd.select_by_index(table_indices, invert=True)
        
        return table_o3d, non_table_o3d

    # 使用RANSAC算法进行桌面点分割
    def segment_table_point(self, pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割桌面点云和非桌面点云（输入输出均为NumPy数组）
        pc: 输入点云，shape (N,6)，格式 [X,Y,Z,R,G,B]
        返回: (桌面点云数组, 非桌面点云数组)，均为 (M,6) 格式
        """
        import open3d as o3d
        # 1. 先将NumPy数组转换为Open3D的PointCloud对象
        o3d_pcd = o3d.geometry.PointCloud()
        # 提取XYZ坐标（前3列）
        o3d_pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        # 提取RGB颜色（后3列，归一化到0-1）
        if pc.shape[1] >= 6:
            rgb = pc[:, 3:6].astype(np.float32) / 255.0  # 假设原始RGB是0-255整数
            o3d_pcd.colors = o3d.utility.Vector3dVector(rgb)

        # 2. 加载预保存的固定平面方程（之前生成的桌面平面）
        # 默认从本模块同目录下的 fixed_plane_params.npy 读取
        plane_params = self.load_plane_params()

        # 3. 用固定平面方程分割点云
        table_o3d, non_table_o3d = self.segment_plane_with_fixed_model(
            o3d_pcd=o3d_pcd,
            plane_params=plane_params,
            distance_threshold=0.01  # 与拟合时保持一致的阈值
        )

        # 4. 将Open3D对象转回NumPy数组（保留XYZRGB格式）
        table_np = PointCloudProcessor.o3d_to_numpy(table_o3d)
        non_table_np = PointCloudProcessor.o3d_to_numpy(non_table_o3d)

        return table_np, non_table_np

    def remove_outliers_by_neighbors(self, pc: np.ndarray, k=10, threshold=0.05) -> np.ndarray:
        """
        基于近邻距离过滤孤立点
        pc: 输入点云 (N,6) [X,Y,Z,R,G,B]
        k: 参考的近邻点数（如10个最近点）
        threshold: 距离阈值，超过此值的点视为噪声
        """
        if pc.shape[0] < k:  # 点太少时直接返回
            return pc
        
        # 提取XYZ坐标
        xyz = pc[:, :3]
        # 构建KDTree加速近邻搜索
        from scipy.spatial import KDTree
        tree = KDTree(xyz)
        # 查找每个点的k个最近邻（不包括自身）
        distances, _ = tree.query(xyz, k=k+1)  # k+1是因为第一个是自身
        avg_distances = np.mean(distances[:, 1:], axis=1)  # 计算平均距离（排除自身）
        
        # 保留平均距离小于阈值的点
        mask = avg_distances < threshold
        return pc[mask]
    
    from sklearn.cluster import DBSCAN

    def remove_outliers_by_dbscan(self, pc: np.ndarray, eps=0.03, min_samples=10) -> np.ndarray:
        """
        基于DBSCAN聚类过滤孤立点
        pc: 输入点云 (N,6)
        eps: 聚类的距离阈值（两个点视为同一簇的最大距离）
        min_samples: 形成密集簇的最小点数
        """
        from sklearn.cluster import DBSCAN
        if pc.shape[0] < min_samples:
            return pc
        
        xyz = pc[:, :3]
        # DBSCAN聚类：-1表示噪声点
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
        labels = db.labels_
        
        # 统计每个簇的点数，保留最大的簇
        if len(set(labels)) <= 1:  # 只有一个簇或全是噪声
            return pc
        max_cluster = np.argmax(np.bincount(labels[labels != -1]))  # 排除噪声点后找最大簇
        mask = labels == max_cluster
        return pc[mask]

    @staticmethod
    def o3d_to_numpy(o3d_pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """将Open3D点云转换为NumPy数组（XYZRGB格式）"""
        xyz = np.asarray(o3d_pcd.points)  # (M,3)
        rgb = np.asarray(o3d_pcd.colors)  # (M,3)，范围0-1
        rgb = (rgb * 255).astype(np.uint8)  # 转回0-255整数
        return np.hstack([xyz, rgb])  # 拼接为(M,6)

    @staticmethod
    def filter_z_range(pc: np.ndarray, z_min: Optional[float] = None, z_max: Optional[float] = None) -> np.ndarray:
        """Filter points by Z value in camera/world coordinates depending on preprocessing.
        pc shape: (N,C) with C>=3.
        """
        if pc.size == 0:
            return pc
        z = pc[:, 2]
        mask = np.ones(pc.shape[0], dtype=bool)
        if z_min is not None:
            mask &= (z >= float(z_min))
        if z_max is not None:
            mask &= (z <= float(z_max))
        return pc[mask]

    @staticmethod
    def filter_distance(pc: np.ndarray, max_distance: float) -> np.ndarray:
        """Filter points with Euclidean distance greater than max_distance from origin.
        pc shape: (N,C) with C>=3.
        """
        if pc.size == 0 or max_distance is None or max_distance <= 0:
            return pc
        d = np.linalg.norm(pc[:, :3], axis=1)
        return pc[d <= float(max_distance)]
    

    