import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# === 路径配置（请改成你自己的） ===

json_path = "/home/ubuntu/Documents/code/LXY/terra_ply_island/terra_ply/query/interval1_AMvalley03/sampleinfos_interpolated.json"
txt_path = "/home/ubuntu/Documents/code/LXY/terra_ply_island/terra_ply/query/interval1_AMvalley03/sampleinfos_interpolated.txt"


# === 主函数 ===
def json_to_txt(json_path, txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(txt_path, 'w') as f_out:
        for item in data:
            name = item["OriginalImageName"]
            T4x4 = np.array(item["T4x4"])
            
            # 提取平移向量（假设是 [lon, lat, alt]）
            lon, lat, alt = T4x4[0, 3], T4x4[1, 3], T4x4[2, 3]

            # 提取旋转矩阵并转欧拉角 (xyz 顺序)
            R_c2w = T4x4[:3, :3]
            euler = R.from_matrix(R_c2w).as_euler('xyz', degrees=True)
            roll, pitch, yaw = euler

            # 写入文本文件
            f_out.write(f"{name} {lon} {lat} {alt} {roll} {pitch} {yaw}\n")

    print(f"✅ 已生成：{txt_path}")

# === 自动执行 ===
if __name__ == "__main__":
    json_to_txt(json_path, txt_path)


