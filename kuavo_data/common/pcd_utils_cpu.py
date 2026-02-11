# 26.2.5.11
import open3d as o3d
import numpy as np
import warnings
import cv2

# 随机选择点云采样方法
def choose_method():
    methods = ['fps', 'random', 'voxel', 'normals', 'curvature']
    return np.random.choice(methods)

# 点云增强函数，输入numpy，输出numpy
def augmentation(point_cloud, task_id, camera_id, target_points=4096):
    pcd = convert_numpy_to_o3d(np_pcd=point_cloud)

    if task_id == 1:
        if camera_id == 'cam_h':
            max_dist = np.random.uniform(0.6, 0.7)
        elif camera_id in ['cam_l', 'cam_r']:
            max_dist = np.random.uniform(0.3, 0.4)
        else:
            raise ValueError(f"Unknown camera ID: {camera_id}")
    else:
        raise ValueError(f"Unknown task ID: {task_id}")

    pcd = clean_point_cloud(pcd=pcd,
                            task_id=task_id,
                            camera_id=camera_id,
                            max_dist=max_dist,
                            remove_outliers=True)
    
    pcd = resize_point_cloud_o3d(pcd=pcd,
                                 target_points=target_points,
                                 method=choose_method())

    return convert_o3d_to_numpy(pcd=pcd)

# 图像调整函数
def resize_images(rgb_img, depth_img, target_width, target_height):
    rgb_resized = cv2.resize(rgb_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return rgb_resized, depth_resized

# 点云生成函数 (输入图像 -> 输出Open3d)
def generate_point_cloud(rgb_img, depth_img, intrinsic_matrix, depth_scale):
    o3d_rgb = o3d.geometry.Image(rgb_img)
    o3d_depth = o3d.geometry.Image(depth_img)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=100.0,
        convert_rgb_to_intensity=False
    )

    h, w = depth_img.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd

# 点云清洗函数
def clean_point_cloud(pcd, task_id, camera_id, max_dist, remove_outliers):

    if not pcd.has_points():
        warnings.warn("Point cloud has no points.")
        return pcd

    if task_id == 1:
        # 任务1：保留以 (0,0,0) 为球心，半径为 max_dist 的球内点
        if max_dist is None:
            if camera_id == 'cam_h':
                max_dist = 0.7
            elif camera_id in ['cam_l', 'cam_r']:
                max_dist = 0.4
            else:
                raise ValueError(f"Unknown camera ID: {camera_id}")
        points = np.asarray(pcd.points)
        dist_sq = np.sum(points**2, axis=1)
        mask = dist_sq <= (max_dist ** 2)
        pcd = pcd.select_by_index(np.where(mask)[0])
        
    else:
        warnings.warn(f"未知任务ID {task_id}，未执行点云清洗。")

    if remove_outliers and len(pcd.points) > 0:
        # 统计学离群点去除
        # nb_neighbors: 考虑相邻点的数量，数量越多计算越慢但越平滑
        # std_ratio: 标准差倍数。值越小，滤除越严格；通常在 1.0 到 3.0 之间
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

# 点云数量统一化函数 (输入O3D -> 输出O3D)
def resize_point_cloud_o3d(pcd, target_points=4096, method='fps'):
    # method 可选 'fps', 'random', 'voxel', 'normals', 'curvature'
    if not pcd.has_points():
        warnings.warn("Point cloud has no points.")
        return pcd
    
    n_input = len(pcd.points)

    if n_input == target_points:
        return pcd
    if n_input < target_points:
        n_resample = target_points - n_input
        resample_indices = np.random.choice(n_input, n_resample, replace=True)
        points = np.asarray(pcd.points)
        new_points = points[resample_indices].copy()
        # 添加微小抖动 (Jittering)，防止点完全重合导致神经网络梯度消失
        # 0.001 代表 1mm 的标准差，可根据你的相机精度调整
        jitter = np.random.normal(0, 0.001, size=new_points.shape)
        new_points += jitter
        
        final_points = np.concatenate([points, new_points], axis=0)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(final_points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            final_colors = np.concatenate([colors, colors[resample_indices]], axis=0)
            new_pcd.colors = o3d.utility.Vector3dVector(final_colors)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            final_normals = np.concatenate([normals, normals[resample_indices]], axis=0)
            new_pcd.normals = o3d.utility.Vector3dVector(final_normals)
        return new_pcd

    if method == 'fps':
        pcd = pcd.farthest_point_down_sample(target_points)

    elif method == 'random':
        indices = np.random.choice(n_input, target_points, replace=False)
        pcd = pcd.select_by_index(indices)

    elif method == 'voxel':
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        bbox_volume = np.prod(max_bound - min_bound + 1e-6) 
        estimated_voxel_size = (bbox_volume / (target_points * 1.5)) ** (1/3) 
        pcd_down = pcd
        for _ in range(5): # 最多尝试5次，防止死循环
            pcd_down = pcd.voxel_down_sample(voxel_size=estimated_voxel_size)
            if len(pcd_down.points) > target_points * 0.8: # 只要达到了目标的80%以上就可以接受
                break
            # 如果点太少，将 voxel_size 缩小为原来的 0.618 (黄金分割) 或 0.5
            estimated_voxel_size *= 0.6
        pcd = pcd_down
    
    elif method == 'normals':
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        normals = np.asarray(pcd.normals)
        bins_per_axis = 5 # 5x5x5 = 125 个桶
        bins = np.floor((normals + 1) / 2 * (bins_per_axis - 1)).astype(int) # 归一化到 [0, bins-1]
        bin_indices = bins[:, 0] * bins_per_axis**2 + bins[:, 1] * bins_per_axis + bins[:, 2]
        
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        
        selected_indices = []
        points_per_bin = target_points // len(unique_bins)
        remainder = target_points % len(unique_bins)
        
        for i, bin_idx in enumerate(unique_bins):
            indices_in_bin = np.where(bin_indices == bin_idx)[0]
            n_sample = points_per_bin + (1 if i < remainder else 0)
            if len(indices_in_bin) <= n_sample:
                selected_indices.extend(indices_in_bin)
            else:
                selected_indices.extend(np.random.choice(indices_in_bin, n_sample, replace=False))
        
        pcd = pcd.select_by_index(selected_indices)

    elif method == 'curvature':
        if not pcd.has_normals(): 
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        covariances = np.asarray(pcd.covariances)
        
        eigenvalues, _ = np.linalg.eigh(covariances)
        eigenvalues = np.abs(eigenvalues) 
        
        sum_eigen = np.sum(eigenvalues, axis=1)
        sum_eigen[sum_eigen == 0] = 1e-6 
        
        curvature = eigenvalues[:, 0] / sum_eigen
        
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        curvature[curvature < 0] = 0
        
        curvature_sum = np.sum(curvature)
        
        if curvature_sum == 0:
            indices = np.random.choice(n_input, target_points, replace=False)
        else:
            prob = curvature / curvature_sum
            prob /= np.sum(prob) 
            
            indices = np.random.choice(n_input, target_points, replace=False, p=prob)
            
        pcd = pcd.select_by_index(indices)
    
    else:
        raise ValueError(f"Unknown resizing method: {method}")
    
    if len(pcd.points) != target_points:
        pcd = resize_point_cloud_o3d(pcd, target_points, method='random')
    return pcd

# 转换函数 (输入Open3d -> 输出Numpy)
def convert_o3d_to_numpy(pcd):
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        warnings.warn("Point cloud has no points.")
        return np.zeros((0, 6), dtype=np.float32)  # 返回空点云
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        warnings.warn("Point cloud has no colors. Filling with zeros.")
        colors = np.zeros_like(points)

    return np.hstack([points, colors]).astype(np.float32)

# 转换函数 (输入Numpy -> 输出Open3d)
def convert_numpy_to_o3d(np_pcd):
    pcd = o3d.geometry.PointCloud()

    if np_pcd is None or np_pcd.shape[0] == 0:
        return pcd

    points = np_pcd[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)

    if np_pcd.shape[1] >= 6:
        colors = np_pcd[:, 3:6]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# 任务1 点云处理流水线
def process_pcd_task1(rgb_img, depth_img, camera_id, method, target_points=4096, fov_deg=60.0):
    # 修改图像大小为原来1/2
    rgb_img, depth_img = resize_images(rgb_img, depth_img, rgb_img.shape[1]//2, rgb_img.shape[0]//2)
    # fov_deg: 水平视场角，默认为 60 度
    h, w = depth_img.shape
    fov_rad = np.deg2rad(fov_deg)
    f = w / (2 * np.tan(fov_rad / 2))
    cx = w / 2.0
    cy = h / 2.0
    intrinsic_matrix = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0]
    ])
    # 生成点云
    pcd = generate_point_cloud(rgb_img=rgb_img, 
                               depth_img=depth_img, 
                               intrinsic_matrix=intrinsic_matrix, 
                               depth_scale=1000.0)
    # 清洗点云
    pcd = clean_point_cloud(pcd=pcd, 
                            task_id=1, 
                            camera_id=camera_id,
                            max_dist=None,
                            remove_outliers=True)
    # 统一点云数量
    pcd = resize_point_cloud_o3d(pcd=pcd, 
                                 target_points=target_points, 
                                 method=method)
    # 返回处理后的点云
    return convert_o3d_to_numpy(pcd=pcd)