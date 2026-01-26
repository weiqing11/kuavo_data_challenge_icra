import open3d as o3d
import numpy as np
import cv2
import sys
import os
import rosbag
import rospy
import matplotlib.pyplot as plt

# 点云生成函数 (输入图像 -> 输出Open3d)
def generate_point_cloud(rgb_img, depth_img, intrinsic_matrix, depth_scale=1000.0):
    """
    根据 RGB 和 Depth 生成 Open3D 点云对象。
    
    Args:
        depth_scale (float): 深度比例因子，默认1000.0 (将mm转换为m)
    Returns:
        o3d.geometry.PointCloud: 生成的点云对象
    """
    # 转换为 Open3D 图像格式
    o3d_rgb = o3d.geometry.Image(rgb_img)
    o3d_depth = o3d.geometry.Image(depth_img)

    # 创建 RGBD 图像
    # depth_trunc=100.0 表示 100米内都保留，相当于在这一步不截断，交给后续清洗步骤处理
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=100.0,
        convert_rgb_to_intensity=False
    )

    # 创建相机内参对象
    h, w = depth_img.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd

# 点云清洗函数
def clean_point_cloud(pcd, task_id, max_dist=0.65, remove_outliers=False):
    points = np.asarray(pcd.points)
    
    has_colors = pcd.has_colors()
    if has_colors:
        colors = np.asarray(pcd.colors)
    
    if task_id == 1:
        # 任务1：保留以 (0,0,0) 为球心，半径为 max_dist 的球内点
        # max_dist 推荐为1
        dist_sq = np.sum(points**2, axis=1)
        mask = dist_sq <= (max_dist ** 2)
        
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(points[mask])
        
        if has_colors:
            pcd_clean.colors = o3d.utility.Vector3dVector(colors[mask])

        pcd = pcd_clean
        
    else:
        print(f"[Warning] 未知任务ID {task_id}，未执行点云清洗。")

    if remove_outliers and len(pcd.points) > 0:
        # 统计学离群点去除
        # nb_neighbors: 考虑相邻点的数量，数量越多计算越慢但越平滑
        # std_ratio: 标准差倍数。值越小，滤除越严格；通常在 1.0 到 3.0 之间
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        pcd = pcd.select_by_index(ind)

    return pcd

# 点云数量统一化函数 (输入O3D -> 输出O3D)
def resize_point_cloud_o3d(pcd, target_points=4096, method='fps'):
    """
    点云采样集成函数
    
    Args:
        pcd: o3d.geometry.PointCloud 对象
        target_points: 目标点数 (4096)
        method: 采样方法可选: 'fps', 'random', 'uniform', 'voxel'
    """
    n_input = len(pcd.points)
    
    if n_input < target_points:
        # 如果原始点数不足，进行上采样（重复采样）
        indices = np.random.choice(n_input, target_points, replace=True)
        return pcd.select_by_index(indices)

    if method == 'fps':
        # 1. 最远点采样 (Farthest Point Sampling)
        # 优点：几何覆盖面最广，最适合深度学习
        # 缺点：计算最慢
        return pcd.farthest_point_down_sample(target_points)

    elif method == 'random':
        # 2. 随机采样 (Random Sampling)
        # 优点：极速
        # 缺点：分布不均，可能会丢失稀疏区域特征
        indices = np.random.choice(n_input, target_points, replace=False)
        return pcd.select_by_index(indices)

    elif method == 'uniform':
        # 3. 均匀采样 (Uniform Sampling)
        # 逻辑：按固定的步长提取点
        every_k_points = n_input // target_points
        subset_pcd = pcd.uniform_down_sample(every_k_points)
        # 由于取整原因，点数可能略多于或少于4096，再做一次微调
        return resize_point_cloud_o3d(subset_pcd, target_points, method='random')

    elif method == 'voxel':
        # 4. 体素采样 (Voxel Downsampling)
        # 逻辑：先用格点降采样，再随机缩放到固定规模
        # 这里的 voxel_size 需要根据点云范围调整，暂设一个经验值
        avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        voxel_size = avg_dist * 2 
        voxel_pcd = pcd.voxel_down_sample(voxel_size)
        
        # 体素采样无法控制精确数量，因此递归调用自身进行微调
        return resize_point_cloud_o3d(voxel_pcd, target_points, method='random')

    else:
        raise ValueError("Method must be 'fps', 'random', 'uniform', or 'voxel'")

# 保存函数 (输入Open3d -> 保存文件)
def save_point_cloud_o3d(pcd, save_path):
    """
    将 Open3D 格式的点云保存到指定路径。
    """
    if not isinstance(pcd, o3d.geometry.PointCloud):
        print("❌ [Error] 保存失败：输入数据不是 Open3D 点云格式。")
        return

    # 确保目录存在
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存
    success = o3d.io.write_point_cloud(save_path, pcd)
    if success:
        print(f"✅ 点云已保存至: {save_path}")
    else:
        print(f"❌ 点云保存失败: {save_path}")

# 转换函数 (输入Open3d -> 输出Numpy)
def convert_o3d_to_numpy(pcd):
    """
    将 Open3D 点云转换为 Numpy 数组 (N, 6)。
    格式: [x, y, z, r, g, b]
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
        
    # 拼接 XYZ 和 RGB
    return np.hstack([points, colors]).astype(np.float32)

# 任务1 点云处理流水线
def process_pcd_task1(rgb_img, depth_img):
    intrinsic_matrix = np.array([
        [554.25, 0.0, 320.0],
        [0.0, 554.25, 240.0],
        [0.0, 0.0, 1.0]
    ])
    # 生成点云
    pcd = generate_point_cloud(rgb_img, depth_img, intrinsic_matrix, depth_scale=1000.0)
    # 清洗点云
    pcd = clean_point_cloud(pcd, task_id=1, max_dist=0.65, remove_outliers=True)
    # 统一点云数量
    pcd = resize_point_cloud_o3d(pcd, target_points=4096, method='fps')
    # 返回处理后的点云
    return convert_o3d_to_numpy(pcd)

# 图像格式检查函数
def test_pipeline_from_rosbag(bag_path, target_sec):
    print("================ 启动 ROS Bag 点云生成测试 ================")

    # --- 配置区域 (请修改这里) ---
    BAG_PATH = bag_path  # ROS Bag 文件路径
    TARGET_SEC = target_sec  # 跳转到指定秒数后开始读取数据

    RGB_TOPIC = "/cam_h/color/image_raw/compressed"
    DEPTH_TOPIC = "/cam_h/depth/image_raw/compressedDepth"
    SAVE_PATH = "./debug_output/test_cloud.ply"

    if not os.path.exists(BAG_PATH):
        print(f"❌ 文件不存在: {BAG_PATH}")
        return

    bag = rosbag.Bag(BAG_PATH, 'r')
    seek_time = bag.get_start_time() + TARGET_SEC
    print(f"📂 读取 Bag: {BAG_PATH}")
    print(f"⏱️ 跳转时间: {TARGET_SEC}s")

    rgb_img = None
    depth_img = None

    # 使用 rospy.Time 而不是 rosbag.rostime.Time
    # 读取消息
    for topic, msg, t in bag.read_messages(topics=[RGB_TOPIC, DEPTH_TOPIC], start_time=rospy.Time.from_sec(seek_time)):
        
        # 解码 RGB
        if topic == RGB_TOPIC and rgb_img is None:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        
        # 解码 Depth (CompressedDepth)
        elif topic == DEPTH_TOPIC and depth_img is None:
            np_arr = np.frombuffer(msg.data, np.uint8)
            # CompressedDepth 格式通常有 12 字节的 Header，需要跳过
            depth_img = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                # 备用方案：如果不去头能解码，则直接解码
                depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if rgb_img is not None and depth_img is not None:
            print("✅ 成功提取到一帧 RGB 和 Depth 数据")
            break
    
    bag.close()

    # 如果没找到数据，为了防止后面报错，加个检查
    if rgb_img is None or depth_img is None:
        print("❌ 未能在指定时间点之后找到完整的一对 RGB/Depth 数据。")
        return

    # --- 开始测试流程 ---

    debug_dir = "./debug_output"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        print(f"📂 已创建文件夹: {debug_dir}")

    # 保存一下图像
    plt.imsave(f"{debug_dir}/test_rgb.png", rgb_img)
    plt.imsave(f"{debug_dir}/test_depth.png", depth_img, cmap='plasma')
    cv2.imwrite(f"{debug_dir}/test_depth_raw.png", depth_img)
    print("📷 已保存 RGB 和 Depth 图像用于调试。")

    result = process_pcd_task1(rgb_img, depth_img)
    print(f"📊 处理后点云数据形状: {result.shape}")

    print("================ 测试完成 ================")

if __name__ == "__main__":
    bag_path = "/home/robot/zhuyihui/my_data/kuavo_data_challenge_icra/sim/TASK1-ToySorting/task1_0001.bag"
    target_sec = 15.0
    test_pipeline_from_rosbag(bag_path, target_sec)