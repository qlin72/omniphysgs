from ruamel.yaml import YAML
import re, os

def update_yaml_with_order(source_path, target_path):
    yaml = YAML()
    yaml.preserve_quotes = True

    # 读取源文件（保留顺序）
    with open(source_path, 'r') as f:
        data = yaml.load(f)

    # 从文件名提取 ID
    filename = os.path.basename(target_path)
    match = re.search(r'dataset_45_(\d+)\.yaml', filename)
    if not match:
        raise ValueError(f"无法从文件名中提取 ID: {filename}")
    id_str = match.group(1)

    # 修改 model_path
    if 'train' not in data:
        data['train'] = {}
    data['train']['model_path'] = f'dataset/dataset_45_new/{id_str}'

    # 清空目标文件
    with open(target_path, 'w') as f:
        f.write('')

    # 保留顺序写入
    with open(target_path, 'w') as f:
        yaml.dump(data, f)

    print(f"[INFO] 使用 ruamel.yaml 保留顺序写入 {target_path}")

# 示例使用
source_yaml = 'configs/dataset_45_01.yaml'
target_yaml = 'configs/dataset_45_89.yaml'
update_yaml_with_order(source_yaml, target_yaml)