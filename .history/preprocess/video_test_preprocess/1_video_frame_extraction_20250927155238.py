import cv2
import os
import math
# ==== 配置 ====
video_path = '/mnt/sda/MapScape/sup/DJI_20250926150103_0001_V.mp4'  # 输入视频路径
output_dir = '/mnt/sda/MapScape/query/images/DJI_20250926150103_0001_V'       # 输出图像文件夹
frame_interval = 1         # 每隔多少帧保存一次（1 表示逐帧）
target_width = 3840
target_height = 2160
print(output_dir)
# ==== 裁切秒数（可选）====
start_sec = 0              # None 表示从头开始
end_sec = 888               # None 表示直到结束

# ==== 最大输出帧数（可选）====
max_output_frames = None    # None 表示不限制数量

# ==== 创建输出目录 ====
os.makedirs(output_dir, exist_ok=True)

# ==== 打开视频 ====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 无法打开视频文件:", video_path)
    exit()

# ==== 获取视频属性 ====
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps == 0:
    print("❌ 无法获取视频帧率")
    cap.release()
    exit()

# ==== 计算裁切范围 ====
start_frame = math.floor(start_sec * fps) if start_sec is not None else 0
end_frame = math.ceil(end_sec * fps) if end_sec is not None else total_frames

# 防止越界
start_frame = max(0, min(start_frame, total_frames - 1))
end_frame = max(start_frame + 1, min(end_frame, total_frames))

# ==== 定位起始帧 ====
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_id = start_frame
saved_id = 0

while frame_id < end_frame:
    if max_output_frames is not None and saved_id >= max_output_frames:
        break

    success, frame = cap.read()
    if not success:
        print(f"⚠️ 读取第 {frame_id} 帧失败，提前结束")
        break

    if (frame_id - start_frame) % frame_interval == 0:
        # resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frame_filename = os.path.join(output_dir, f"{saved_id}_0.png")
        cv2.imwrite(frame_filename, frame)
        saved_id += 1

    frame_id += 1

cap.release()
print(f"✅ 完成：共保存 {saved_id} 帧图像到 {output_dir}")
