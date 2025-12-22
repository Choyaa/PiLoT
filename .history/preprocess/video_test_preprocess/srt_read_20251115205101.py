import re
import math

# ==== 配置 ====
# srt_file   = '/mnt/sda/MapScape/sup/long_term_sikong/foggy/DJI_20250927082946_0001_V.srt'
srt_file = '/mnt/sda/MapScape/sup/long_term_sikong/cloudy/DJI_20250926150103_0001_V.srt'

output_txt = '/mnt/sda/MapScape/query/poses/DJI_20250926150103_0001_V.txt'

# ==== 时间段和采样 ====
start_sec      = 0          # 起始秒
end_sec        = 201       # 结束秒；若想直到文件结束，可设为很大或用 None
fps            = 29.97        # 帧率
frame_interval = 2            # 抽帧间隔（从 start_frame 开始每隔 N 帧取一帧）

start_frame = math.floor(start_sec * fps)
end_frame   = math.floor(end_sec * fps) if end_sec is not None else None

# ==== 解析 SRT 并收集位姿 ====
pose_by_fid = {}  # key: frame_id, value: [lon, lat, alt, roll, pitch, yaw]

with open(srt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i]
    if not line.startswith("FrameCnt:"):
        continue

    m = re.search(r"FrameCnt:\s*(\d+)", line)
    if not m:
        continue
    frame_id = int(m.group(1))

    # 时间窗口过滤
    if frame_id < start_frame:
        continue
    if end_frame is not None and frame_id > end_frame:
        continue

    # 抽帧（以 start_frame 为相位起点）
    if frame_interval is not None and frame_interval > 1:
        if (frame_id - start_frame) % frame_interval != 0:
            continue

    info_line = lines[i + 1] if i + 1 < len(lines) else ""

    lat_match   = re.search(r"latitude:\s*([-\d.]+)",  info_line)
    lon_match   = re.search(r"longitude:\s*([-\d.]+)", info_line)
    alt_match   = re.search(r"abs_alt:\s*([-\d.]+)",   info_line)
    roll_match  = re.search(r"gb_roll:\s*([-\d.]+)",   info_line)
    pitch_match = re.search(r"gb_pitch:\s*([-\d.]+)",  info_line)
    yaw_match   = re.search(r"gb_yaw:\s*([-\d.]+)",    info_line)

    if not all([lat_match, lon_match, alt_match, roll_match, pitch_match, yaw_match]):
        continue

    lat   = float(lat_match.group(1))
    lon   = float(lon_match.group(1))
    alt   = float(alt_match.group(1))
    roll  = float(roll_match.group(1))
    pitch = float(pitch_match.group(1)) + 90.0  # 你的原始修正
    yaw   = -float(yaw_match.group(1))          # 你的原始修正（取反）

    pose_by_fid[frame_id] = [lon, lat, alt, roll, pitch, yaw]

# ==== 排序并写出 ====
sorted_fids = sorted(pose_by_fid.keys())

with open(output_txt, 'w') as f:
    for i, fid in enumerate(sorted_fids):
        name = f"{i}_0.png"  # 连续命名
        values = pose_by_fid[fid]
        f.write(f"{name} {' '.join(map(str, values))}\n")

print(f"✅ 成功提取 {len(sorted_fids)} 帧（范围：frame {start_frame}"
      f"{'' if end_frame is None else f' ~ {end_frame}'}，间隔：{frame_interval}），保存至 {output_txt}")

