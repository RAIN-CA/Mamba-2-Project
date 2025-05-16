import os
import tarfile
import pandas as pd
from tqdm import tqdm
from os.path import join
from PIL import Image
import io
import lmdb
import torch
import numpy as np
from transformers import AutoFeatureExtractor, ViTForImageClassification

# 读取 FPS 信息的 CSV 文件
fps_csv_path = '/root/autodl-tmp/ek100/EPIC_100_video_info.csv'
fps_info = pd.read_csv(fps_csv_path, dtype={'video_id': str})

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ViT 模型路径
model_path = "/root/autodl-tmp/huggingface/deit-tiny-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()

# LMDB 数据库
env = lmdb.open('/root/autodl-tmp/ek100/lmdb', map_size=900 * 1024**3)

def get_fps(video_id):
    """获取视频的 FPS 值"""
    row = fps_info[fps_info['video_id'] == video_id]
    if not row.empty:
        return float(row.iloc[0]['fps'])
    return None

def extract_vit_features_batch(images):
    """
    从给定的一批图像中提取 ViT 特征
    Args:
        images (list of PIL.Image): 输入的 RGB 图像列表
    Returns:
        list of numpy.ndarray: 每个图像的特征数据列表，存储为 float16 格式
    """
    inputs = feature_extractor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 移动到 GPU 或 CPU

    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    features_batch = []
    for state in hidden_states[-1]:
        if state.dim() == 3 and state.size(1) > 1:
            features = state[:, 1:, :].squeeze().detach().cpu().to(torch.float16).numpy()
        else:
            features = state.squeeze().detach().cpu().to(torch.float16).numpy()
        features_batch.append(features)
    
    return features_batch

def create_keys_from_frames(data_path, batch_size=700):
    for split in ['train', 'test']:
        split_path = join(data_path, split, 'EPIC-KITCHENS')
        for group_id in sorted(os.listdir(split_path)):
            group_path = join(split_path, group_id, 'rgb_frames')
            if not os.path.isdir(group_path):
                continue
            
            for video_tar in sorted(os.listdir(group_path)):
                if not video_tar.endswith('.tar'):
                    continue
                video_id = video_tar.replace('.tar', '')
                tar_path = join(group_path, video_tar)
                
                fps = get_fps(video_id)
                if fps is None:
                    print(f"警告: 未找到 {video_id} 的 FPS 信息，跳过")
                    continue
                
                try:
                    with tarfile.open(tar_path, 'r') as tar:
                        members = sorted([m for m in tar.getmembers() if m.isfile()], key=lambda x: x.name)
                        total_frames = len(members)
                        
                        if fps >= 59:
                            selected_indices = range(0, total_frames, 2)  # 60Hz → 30Hz，每2帧取1帧
                        elif fps == 50:
                            selected_indices = np.linspace(0, total_frames - 1, int(total_frames * 30 / 50)).astype(int)  # 50Hz → 30Hz，均匀采样
                        else:
                            print(f"未知 FPS: {fps}，跳过 {video_id}")
                            continue
                        
                        print(f"处理视频 {video_id} | FPS: {fps} | 选取 {len(selected_indices)} 帧")

                        batch_images, batch_keys = [], []
                        frame_counter = 0
                        
                        for count, member in enumerate(tqdm(members, desc=f"处理 {video_id}")):
                            if count not in selected_indices:
                                continue
                            
                            with tar.extractfile(member) as file:
                                img = Image.open(io.BytesIO(file.read())).convert('RGB')
                                frame_num = str(frame_counter).zfill(10)
                                key = f"{video_id}_frame_{frame_num}.jpg"
                                batch_images.append(img)
                                batch_keys.append(key)
                                frame_counter += 1

                                if len(batch_images) == batch_size:
                                    features_batch = extract_vit_features_batch(batch_images)
                                    with env.begin(write=True) as txn:
                                        for key, features in zip(batch_keys, features_batch):
                                            txn.put(key.encode(), features.tobytes())
                                    batch_images, batch_keys = [], []

                        if batch_images:
                            features_batch = extract_vit_features_batch(batch_images)
                            with env.begin(write=True) as txn:
                                for key, features in zip(batch_keys, features_batch):
                                    txn.put(key.encode(), features.tobytes())
                            
                        print(f"{video_id} 处理完毕\n")
                except tarfile.ReadError:
                    print(f"文件 {video_tar} 读取错误，跳过")
                    continue

# 运行数据集创建
data_path = '/root/autodl-tmp/data'
create_keys_from_frames(data_path)
