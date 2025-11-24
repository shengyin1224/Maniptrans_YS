import os
import xml.etree.ElementTree as ET

# --- 1. 配置路径 (请根据你的实际情况检查) ---
# (这些路径都是相对于你运行脚本的位置)
BASE_DIR = "/media/magic-4090/DATA1/shengyin/ManipTrans/ManipTrans/"
VISUAL_MESH_DIR = os.path.join(BASE_DIR, "data/OakInk-v2/object_raw/align_ds")
COLLISION_MESH_DIR = os.path.join(BASE_DIR, "data/OakInk-v2/coacd_object_preview_new/align_ds")
# 生成的URDF文件将与碰撞模型放在一起
OUTPUT_URDF_DIR = COLLISION_MESH_DIR 

# --- 2. 加载你的URDF模板内容 ---
# 确保你的模板文件路径正确
URDF_TEMPLATE_PATH = os.path.join(BASE_DIR, "assets/obj_urdf_example.urdf")

try:
    with open(URDF_TEMPLATE_PATH, 'r') as f:
        urdf_template_string = f.read()
except FileNotFoundError:
    print(f"❌ 错误: URDF模板文件未找到: '{URDF_TEMPLATE_PATH}'")
    exit()

# --- 3. 主逻辑 ---
def main():
    """遍历所有碰撞模型并生成对应的URDF文件。"""
    print("🚀 开始生成URDF文件...")

    if not os.path.isdir(COLLISION_MESH_DIR):
        print(f"❌ 错误: 碰撞模型目录不存在: '{COLLISION_MESH_DIR}'")
        return

    # 使用 os.walk 遍历所有子目录和文件
    for root, _, files in os.walk(COLLISION_MESH_DIR):
        for filename in files:
            # 只处理 .obj 或 .ply 结尾的碰撞模型文件
            if not (filename.endswith('.obj') or filename.endswith('.ply')):
                continue

            # --- 构建各种绝对路径 ---
            collision_mesh_full_path = os.path.join(root, filename)
            relative_path_from_root = os.path.relpath(collision_mesh_full_path, COLLISION_MESH_DIR)
            visual_mesh_full_path = os.path.join(VISUAL_MESH_DIR, relative_path_from_root)
            output_urdf_path = os.path.join(OUTPUT_URDF_DIR, os.path.splitext(relative_path_from_root)[0] + '.urdf')

            if not os.path.exists(visual_mesh_full_path):
                print(f"⚠️ 警告: 未找到对应的视觉模型 '{visual_mesh_full_path}', 已跳过。")
                continue

            # --- 计算URDF内部引用的相对路径 ---
            urdf_dir = os.path.dirname(output_urdf_path)
            visual_rel_path = os.path.relpath(visual_mesh_full_path, urdf_dir)
            collision_rel_path = os.path.relpath(collision_mesh_full_path, urdf_dir)

            # --- 使用XML解析器安全地修改模板 ---
            tree = ET.ElementTree(ET.fromstring(urdf_template_string))
            xml_root = tree.getroot()
            
            # 查找并修改visual和collision标签
            visual_mesh_tag = xml_root.find('.//link/visual/geometry/mesh')
            collision_mesh_tag = xml_root.find('.//link/collision/geometry/mesh')
            
            if visual_mesh_tag is not None and collision_mesh_tag is not None:
                visual_mesh_tag.set('filename', visual_rel_path)
                collision_mesh_tag.set('filename', collision_rel_path)
            else:
                print(f"❌ 错误: 无法在模板中找到 'visual/geometry/mesh' 或 'collision/geometry/mesh' 标签。")
                continue

            # --- 写入新的URDF文件 ---
            tree.write(output_urdf_path, encoding='UTF-8', xml_declaration=True)
            print(f"✅ 已创建: {output_urdf_path}")

    print("\n🎉 --- 所有URDF文件已生成完毕！ ---")

if __name__ == "__main__":
    main()