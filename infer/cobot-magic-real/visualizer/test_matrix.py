import numpy as np
import math

# 用户提供 / 已存在变量
# points_orig: (N,3) 点云（在相机系或转换后看起来与 base 有偏差）
# extrinsics_matrix: 4x4 (你当前用来变换的矩阵，例如 T_base2cam 或 T_cam2base)
# 我们假设你现在用的是 left-mul 形式: pts_base = (extrinsics_matrix @ pts_cam_h.T).T[:, :3]

# 例：若你已经变换得到 points_base = (extrinsics_matrix @ H.T).T[:, :3]
# 那么 R_transform 就是 extrinsics_matrix[:3,:3]
extrinsics_matrix = np.array(
                                [[0.081265,   -0.66492733,  0.74247401, -0.01365506],
                                [-0.9950363,  -0.01119717,  0.09888066, -0.26896289],
                                [-0.05743485, -0.74682412, -0.66253677,  0.67637774],
                                [ 0.,          0.,          0.,          1.        ]]
                            )
R = extrinsics_matrix[:3, :3]
t = extrinsics_matrix[:3, 3]

def is_rotation_matrix(R, tol=1e-6):
    should_be_I = R @ R.T
    I = np.eye(3)
    err = np.linalg.norm(should_be_I - I)
    return err, should_be_I

err, should_be_I = is_rotation_matrix(R)
print("R 正交性误差 (||R*R^T - I||):", err)
if err > 1e-3:
    print("注意：R 看起来不够正交，可能标定出错或有数值问题。")

# 计算 R 的轴-角与角度大小
def rotation_matrix_to_axis_angle(R):
    # 使用 Rodrigues/角-轴提取
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2.0)))
    if abs(angle) < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz])
    axis = axis / np.linalg.norm(axis)
    return axis, angle

axis, angle = rotation_matrix_to_axis_angle(R)
print("R 角轴表示 axis:", axis, " angle (rad):", angle, " angle (deg):", np.degrees(angle))

# 如果你有 ground-truth 对应点 (ref_points) 与转换后点 (src_points)，用 Kabsch 求最佳刚体变换
# 示例：假如你有 M 对点对 ref_points (M,3) in base系，src_points (M,3) in transformed cloud (当前结果)
# 则可计算最佳旋转 R_kabsch 和平移 t_kabsch，使 R_kabsch * src + t_kabsch ≈ ref

def kabsch(src, dst):
    """
    src, dst: (M,3) corresponding points
    returns R, t that minimize ||R src + t - dst||
    """
    assert src.shape == dst.shape
    M = src.shape[0]
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R_k = Vt.T @ U.T
    # fix reflection
    if np.linalg.det(R_k) < 0:
        Vt[2,:] *= -1
        R_k = Vt.T @ U.T
    t_k = centroid_dst - R_k @ centroid_src
    return R_k, t_k

# === 如果你有对应点对，请放到 ref_points, src_points 中并开启以下代码 ===
# ref_points = np.array([...])   # 在 base/world 坐标系中测量得到的真实点 (M,3)
# src_points = np.array([...])   # 变换后的点云中对应的点 (M,3)（手动挑点或用角点检测）
# R_k, t_k = kabsch(src_points, ref_points)
# print("Kabsch 得到的修正旋转 R_k:\n", R_k)
# print("Kabsch 得到的修正平移 t_k:", t_k)
# # 把修正作用到原始 extrinsic 上（如果 extrinsics 表示先旋转再平移，即 p' = R*p + t）
# # 要让 p_base_corrected = R_k @ (R @ p_cam + t) + t_k = (R_k @ R) @ p_cam + (R_k @ t + t_k)
# R_corrected = R_k @ R
# t_corrected = R_k @ t + t_k
# print("修正后 R_corrected 的轴角：", rotation_matrix_to_axis_angle(R_corrected))
# # 将修正后的 transform 写回矩阵
# extrinsics_matrix_corrected = extrinsics_matrix.copy()
# extrinsics_matrix_corrected[:3,:3] = R_corrected
# extrinsics_matrix_corrected[:3,3] = t_corrected

# === 如果没有对应点，你可以做近似的微小旋转修正来试验（比如绕某轴转 -2deg）
def small_rotation_correction(axis, deg):
    rad = np.deg2rad(deg)
    ux, uy, uz = axis / np.linalg.norm(axis)
    c = math.cos(rad)
    s = math.sin(rad)
    R_corr = np.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])
    return R_corr

# 试验：如果你视觉上发现点云绕 Z 轴顺时针偏 5 度，你可能需要：
R_try = small_rotation_correction(np.array([0,0,1]), -5.0)  # 注意方向符号（根据观察调整）
R_test = R_try @ R  # left-multiply 修正
print("尝试用小角度修正后的 R_test 角轴：", rotation_matrix_to_axis_angle(R_test))

# 最后：把修正结果写回变换矩阵并用它来变换点云，观察可视化差异
# extrinsics_matrix_corrected = extrinsics_matrix.copy()
# extrinsics_matrix_corrected[:3,:3] = R_test
# # (平移一般保持，或可同时修正)
# H = np.hstack([points_orig, np.ones((points_orig.shape[0],1))])
# pts_corrected = (extrinsics_matrix_corrected @ H.T).T[:, :3]
