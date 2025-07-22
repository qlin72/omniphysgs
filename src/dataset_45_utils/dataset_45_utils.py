import os
import numpy as np
from plyfile import PlyData

def load_2objs_point_cloud_info(folder_path):
    """
    输入: folder_path（str）- 包含 point_cloud 文件夹的路径
    输出: (num_pts1, num_pts2, velocity1, velocity2)
          每个为: int, int, np.ndarray(3,), np.ndarray(3,)
    """
    pc_folder = os.path.join(folder_path, "point_clouds")
    file1 = os.path.join(pc_folder, "sampled_obj1_with_velocity.ply")
    file2 = os.path.join(pc_folder, "sampled_obj2_with_velocity.ply")

    # 定义速度坐标变换矩阵
    # 将 [x, y, z] → [x, z, -y]
    V_transform = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)

    def read_ply_velocity(path):
        ply = PlyData.read(path)
        vertex = ply['vertex']
        
        # 检查 vx/vy/vz 是否存在
        required_fields = ('vx', 'vy', 'vz')
        for field in required_fields:
            if field not in vertex[0].dtype.names:
                raise ValueError(f"Missing '{field}' in vertex properties of {path}")
        
        vx = vertex['vx']
        vy = vertex['vy']
        vz = vertex['vz']
        raw_velocity = np.array([vx[0], vy[0], vz[0]], dtype=np.float32)

        # 应用坐标变换
        velocity = V_transform @ raw_velocity

        num_points = len(vertex)

        return num_points, velocity

    num1, vel1 = read_ply_velocity(file1)
    num2, vel2 = read_ply_velocity(file2)

    return num1, num2, vel1, vel2