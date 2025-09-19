import json, shutil
from pathlib import Path

# input_dir = Path("dataset_45_new")       # 源目录
# output_dir = Path("dataset_45_new_json") # 目标目录

input_dir = Path('/data/dataset_45_new')      # 源目录
output_dir = Path('/data/dataset_45_new_json') # 目标目录

output_dir.mkdir(parents=True, exist_ok=True)

def transform(v):
    """[x,y,z] → [x,z,-y]"""
    x, y, z = v
    return [float(x), float(z), float(-y)]

count = 0
for src_json in input_dir.glob("*/metadata.json"):  # 只遍历一层子目录
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 修改 initial_velocity
    data["obj1"]["initial_velocity"] = transform(data["obj1"]["initial_velocity"])
    data["obj2"]["initial_velocity"] = transform(data["obj2"]["initial_velocity"])

    # 覆盖写回源文件
    with open(src_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 构造目标路径并写入
    rel_path = src_json.parent.relative_to(input_dir)
    dst_dir = output_dir / rel_path
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_json = dst_dir / "metadata.json"

    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    count += 1
    print(f"[OK] 修改并复制: {src_json} -> {dst_json}")

print(f"\n处理完成，共 {count} 个 metadata.json")