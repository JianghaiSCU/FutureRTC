import sys
#sys.path.append("C:\\Users\\colonzyx\\AppData\\Roaming\\Python\\Python311\\site-packages")
sys.path.append('/vepfs-share/zouyixian/code/myutils/visualizer')
import visualizer
import zarr
import numpy as np




# 打开 Zarr 存储
path ='/vepfs-share/zouyixian/code/Robotwin_v2/policy/DP3/data/box_one_new_select_zyx_extraction_8_multi-D455-1-2048.zarr'
store = zarr.DirectoryStore(path)
# 打开点云数据集
point_cloud = zarr.open(store, mode='r')
#group = 1
#for i in range (length):
first_frame_cloud = point_cloud["data/point_cloud"][0]
#print(first_frame_cloud.shape)
points = first_frame_cloud


# extrinsics_matrix = np.array(
#                                 [[0.081265,   -0.66492733,  0.74247401, -0.01365506],
#                                 [-0.9950363,  -0.01119717,  0.09888066, -0.26896289],
#                                 [-0.05743485, -0.74682412, -0.66253677,  0.67637774],
#                                 [ 0.,          0.,          0.,          1.        ]]
#                             )

# T_cam2base = np.linalg.inv(extrinsics_matrix)
# # print("extrinsics_matrix t =", extrinsics_matrix[:3, 3])
# # print("T_cam2base t =", T_cam2base[:3, 3])



# WORK_SPACE = [
#     [-1, 1],
#     [-1, 1],
#     [-1, 1]
# ]

# #     # crop



# # # 标定结果（基座 -> 相机）
# # # 现在你会直接提供 base->cam 的旋转与平移（R_base2cam, t_base2cam）
# # R_base2cam = np.array([
# #     [ 0.081265,   -0.66492733,  0.74247401],
# #     [-0.9950363,  -0.01119717,  0.09888066],
# #     [-0.05743485, -0.74682412, -0.66253677]
# # ], dtype=np.float32)
# # t_base2cam = np.array([-0.01365506,-0.26896289,0.67637774], dtype=np.float32)
# # # 构造齐次矩阵 T_base2cam
# # T_base2cam = np.eye(4, dtype=np.float32)
# # T_base2cam[:3, :3] = R_base2cam
# # T_base2cam[:3, 3] = t_base2cam
# # # 求逆：得到 camera -> base
# # T_cam2base = np.linalg.inv(T_base2cam)
# # # 保存以便外部使用（例如可视化时显示相机坐标系）
# # # PointCloudProcessor 期望行向量右乘 extrinsics（将点从相机系变换到基座系），
# # # 所以传入 T_cam2base 的转置（row-vector 右乘的约定）
# # extrinsics_for_processor = T_cam2base.T
# # print("相机到基座的外参（用于点云处理器，row-vector 右乘形式）：\n", extrinsics_for_processor)

# point_xyz = points[..., :3]
# point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
# # point_homogeneous = point_homogeneous @ extrinsics_for_processor # np.dot(point_homogeneous, extrinsics_matrix)
# point_homogeneous = (extrinsics_matrix @ point_homogeneous.T).T

# point_xyz = point_homogeneous[..., :-1]
# points[..., :3] = point_xyz

# points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
#                             (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
#                             (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

# your_pointcloud = points


# 添加一个红色原点标记
origin = np.array([[0.0, 0.0, 0.0, 255, 0, 0]], dtype=points.dtype)
points_with_origin = np.vstack([points, origin])
visualizer.visualize_pointcloud(points_with_origin)


# origin_base = np.array([0,0,0,1])
# origin_in_cam = extrinsics_matrix @ origin_base
# print("base origin in cam :", origin_in_cam[:3])
# print("cam pos in base :", T_cam2base[:3,3])

'''
# 打开 Zarr 存储
path ='/vepfs-share/zouyixian/code/Robotwin_v2/policy/DP3/data/box_one_new_select_extraction_z_8-D455-36.zarr'
store = zarr.DirectoryStore(path)
# 打开点云数据集
point_cloud = zarr.open(store, mode='r')
#group = 1
#for i in range (length):
first_frame_cloud = point_cloud["data/point_cloud"][0]
#print(first_frame_cloud.shape)
your_pointcloud = first_frame_cloud # your point cloud data, numpy array with shape (N, 3) or (N, 6)
visualizer.visualize_pointcloud(your_pointcloud)
'''

'''
# 打开 Zarr 存储
path = "/data/code/cfm_dp3/data/metaworld_drawer-open_expert_10.zarr"
store = zarr.DirectoryStore(path)
data_ = zarr.open(store, mode='r')
action = data_['data/action'][0]
print(action)

f = open('/data/code/3D-Diffusion-Policy/roll_data/roll_action.txt', 'w')
for i in range(32):
    action = data_['data/action'][i]
    f.write(('action {}: '.format(i))+ str(action) + '\n')
    print('write action {}'.format(i))
f.flush()
import matplotlib
#matplotlib.use('TkAgg')
'''
'''
import matplotlib.pyplot as plt
path = '/data/code/flowpolicy/data/metaworld_shelf-place_expert_10_1.zarr'
store = zarr.DirectoryStore(path)
data_ = zarr.open(store, mode='r')
length = 200
img_size = 224  # 设置目标图片尺寸为 224x224

for i in range(length):
    img = data_['data/img'][i]

    # 去除白边并调整图片尺寸
    if img.shape[:2] != (img_size, img_size):
        img = plt.imread(img)  # 读取图像
        img = plt.imread(plt.imread(img))  # 这行代码实际上没有效果，只是为了示例
        img = plt.imresize(img, (img_size, img_size), interpolation='nearest')  # 调整图像尺寸

    plt.axis('off')  # 关闭坐标轴的显示
    plt.imshow(img, cmap='gray')  # 如果是灰度图像，使用灰度色彩映射
    plt.savefig("/data/code/flowpolicy/shelf-place-exp/img/{:04d}.png".format(i+1), bbox_inches='tight', pad_inches=0)
    print('Saved the {}th image'.format(i+1))
'''

