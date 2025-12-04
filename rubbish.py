import pickle
import numpy as np

# 请替换为你实际的文件路径
path_rh = "/home/zxlei/data/humanoid/Maniptrans_YS/data/retargeting/Humoto/mano2inspire_rh/baking_with_spatula_mixing_bowl_and_scooping_to_tray-244.pkl"
path_lh = "/home/zxlei/data/humanoid/Maniptrans_YS/data/retargeting/Humoto/mano2inspire_lh/baking_with_spatula_mixing_bowl_and_scooping_to_tray-244.pkl" # <--- 检查这个文件

with open(path_rh, 'rb') as f: data_rh = pickle.load(f)
with open(path_lh, 'rb') as f: data_lh = pickle.load(f)

pos_rh = data_rh['opt_wrist_pos'][0]
pos_lh = data_lh['opt_wrist_pos'][0]

print(f"Right Hand Base: {pos_rh}")
print(f"Left  Hand Base: {pos_lh}")
print(f"Distance: {np.linalg.norm(pos_rh - pos_lh)}")