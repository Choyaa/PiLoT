# UAVScenes 测试步骤
# 1. 下载官方ply模型，使用meshlab转换为obj格式
# 2. OSGBlab,obj转OSGB，导入obj路径（包括xml所在的路径）； OSGB转3dtiles，输出模型将包含地理坐标
# 3. 转换官方提供的pose.json文件，转为地理坐标的，使用/home/ubuntu/Documents/code/github/FPV/PiLoT_uavscenes/blender_renderer/json_to_cgcs2000.py转换
# 4. 可以测试了
# Blender方案
# 1. Meshlab转换为obj模型
# 2. 打开Blender，导入obj模型，模型使用相对坐标系，因为json格式的变换矩阵是相对的
# 3. 执行/home/ubuntu/Documents/code/github/FPV/PiLoT_uavscenes/blender_renderer/blender_engine.py可渲染
#!/usr/bin/env bash
# run_by_names.sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ==== 所有配置（names 作为 key） ====
names=(

  "interval1_HKairport_GNSS01"
  "interval1_HKisland_GNSS01"
  "interval1_HKisland_GNSS03"
  "interval1_AMvalley01"

)


# ==== 你想运行哪些 name？====
target_names=(
  # "interval1_HKairport_GNSS01"
#   "interval1_HKisland_GNSS03"
  "interval1_HKisland_GNSS01"
  # "interval1_AMvalley01"


)

# ==== 从 txt 中读取 init_euler 和 init_trans ====
read_pose_from_file() {
  local name="$1"
  local pose_file="/mnt/sda/MapScape/query/poses/${name}.txt"

  if [[ ! -f "$pose_file" ]]; then
    echo "❌ 找不到 pose 文件: $pose_file"
    return 1
  fi

  local first_line
  first_line=$(head -n 1 "$pose_file")

  # 解析：name lon lat alt roll pitch yaw
  read -r _ lon lat alt roll pitch yaw <<< "$first_line"

  # 构造 init_euler 和 init_trans
  init_euler="[$pitch, $roll, $yaw]"
  init_trans="[$lon, $lat, $alt]"

  echo "$init_euler|$init_trans"
  return 0
}  # ✅ 这一行必须存在，否则后续 for/if 会错乱

# ==== 遍历 target_names ====
for target_name in "${target_names[@]}"; do
  index=-1
  for i in "${!names[@]}"; do
    if [[ "${names[$i]}" == "$target_name" ]]; then
      index=$i
      break
    fi
  done

  if [[ $index -ge 0 ]]; then
    result=$(read_pose_from_file "$target_name")
    if [[ $? -ne 0 ]]; then
      echo "❌ 无法读取 $target_name 的位姿，跳过"
      continue
    fi

    IFS='|' read -r euler trans <<< "$result"

    echo "==== 正在运行 $target_name ===="
    echo "euler : $euler"
    echo "trans : $trans"

    echo "--- fpvloc"
    python /home/ubuntu/Documents/code/github/FPV/PiLoT/main.py \
      --config "/home/ubuntu/Documents/code/github/FPV/PiLoT/configs/uav_scenes.yaml" \
      --init_euler "$euler" \
      --init_trans "$trans" \
      --name "$target_name"

    echo -e "==== 运行 $target_name 结束 ====\n"

    # ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    # ps aux | grep multiprocessing.resource_tracker | grep -v grep | awk '{print $2}' | xargs kill -9 || true
  else
    echo "❌ 未找到 name=$target_name 对应的配置"
  fi
done
