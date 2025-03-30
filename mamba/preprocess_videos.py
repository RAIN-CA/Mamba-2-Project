import os
import cv2
import torch
import lmdb
import numpy as np
from PIL import Image
import csv
from torch.utils.data import Dataset
from transformers import AutoModel, CLIPImageProcessor

class VideoActionDataset(Dataset):
    def __init__(self, train_txt_path, root_path, csv_file_path, frame_interval=5, model=None, image_processor=None, batch_size=32):
        """
        初始化 Dataset 对象，提前确认每个视频是否在 CSV 文件中有标签的记录，并按帧间隔读取。
        
        参数:
        - frame_interval: 每隔多少帧提取一次。
        - model: 用于提取视频帧特征的预训练模型。
        - image_processor: 用于处理视频帧的图像处理器。
        - batch_size: 每批次提取特征时使用的帧数量，减少显存压力。
        """
        self.video_paths, self.video_ids = self.get_video_paths_and_ids(train_txt_path, root_path)
        self.frame_interval = frame_interval
        self.csv_data = self.read_csv_file(csv_file_path)
        self.model = model
        self.image_processor = image_processor
        self.batch_size = batch_size

        # 过滤掉没有标签的视频，并记录每个视频的总帧数
        self.valid_videos = self.filter_videos_with_labels()

    def get_video_paths_and_ids(self, train_txt_path, root_path):
        """ 获取视频路径和对应的ID """
        video_paths = []
        video_ids = []
        with open(train_txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                relative_path = line.strip()
                full_path = os.path.join(root_path, relative_path)
                video_paths.append(full_path)
                video_id = os.path.splitext(os.path.basename(full_path))[0]
                video_ids.append(video_id)
        return video_paths, video_ids

    def read_csv_file(self, csv_file_path):
        """ 读取 CSV 文件并解析标签 """
        data = {}
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                video_id = row[1].strip()
                start_frame = int(row[2].strip())
                end_frame = int(row[3].strip())
                verb_class = int(row[4].strip())
                noun_class = int(row[5].strip())
                activity_class = int(row[6].strip())
                
                if video_id not in data:
                    data[video_id] = []
                
                data[video_id].append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'verb_class': verb_class,
                    'noun_class': noun_class,
                    'activity_class': activity_class
                })
        return data

    def filter_videos_with_labels(self):
        """ 过滤掉没有有效标签的视频，并记录视频的总帧数 """
        valid_videos = []
        
        for video_path, video_id in zip(self.video_paths, self.video_ids):
            if video_id not in self.csv_data:
                continue  # 跳过没有标签的数据
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue  # 无法打开视频，跳过该视频

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                continue  # 如果视频没有帧，跳过

            # 找到视频的最后一类标签的结束帧
            max_label_frame = max(entry['end_frame'] for entry in self.csv_data[video_id])

            # 如果视频的帧数大于最后一类标签的结束帧，则截断到该标签帧
            if total_frames > max_label_frame:
                total_frames = max_label_frame

            # 保存有效视频信息
            if total_frames > 0 and max_label_frame > 0:
                video_info = {
                    'video_path': video_path,
                    'video_id': video_id,
                    'total_frames': total_frames,
                    'max_frame': max_label_frame  # 记录标签范围内的最大帧
                }
                valid_videos.append(video_info)

            cap.release()  # 释放视频资源

        return valid_videos

    def extract_features_batch(self, frames):
        """ 批量提取视频帧的特征，减少显存压力 """
        batch_features = []
        with torch.no_grad():
            processed_frames = self.image_processor(images=frames, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
            outputs = self.model(processed_frames)
            
            # 转换输出为 float32
            batch_features = outputs.last_hidden_state.to(torch.float32)  # 转换为 float32 类型
            
        return batch_features.cpu().numpy()  # 转为 numpy 格式

    def __getitem__(self, idx):
        """
        动态加载视频帧并按批次返回设定的帧间隔。
        """
        if idx >= len(self.valid_videos):
            raise IndexError(f"Index {idx} 超出数据集范围")

        video_info = self.valid_videos[idx]
        video_path = video_info['video_path']
        total_frames = video_info['total_frames']
        max_frame = video_info['max_frame']
        video_id = video_info['video_id']

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return None

        frames_features = []
        frames_labels = []
        batch_frames = []
        frame_idx = 0

        while frame_idx <= max_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"帧读取失败，帧索引: {frame_idx}，跳过")
                frame_idx += self.frame_interval
                continue

            # 提取帧并加入批处理缓存
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(Image.fromarray(frame_rgb))

            # 获取该帧的未来标签
            label = self.get_future_frame_label(video_id, frame_idx)
            if label is not None:
                frames_labels.append(label)
            else:
                print(f"帧 {frame_idx} 没有标签，停止读取")
                break

            # 当达到batch_size时，提取特征并存储
            if len(batch_frames) >= self.batch_size:
                batch_features = self.extract_features_batch(batch_frames)
                frames_features.extend(batch_features)
                batch_frames = []  # 清空批处理缓存

            frame_idx += self.frame_interval

        # 处理剩余不足 batch_size 的帧
        if batch_frames:
            batch_features = self.extract_features_batch(batch_frames)
            frames_features.extend(batch_features)

        cap.release()

        if not frames_features or not frames_labels:
            print(f"视频 {video_id} 没有有效数据，返回 None")
            return None

        return frames_features, frames_labels

    def get_future_frame_label(self, video_id, frame_idx):
        """ 获取帧的未来标签 """
        if video_id not in self.csv_data:
            return None

        for i, entry in enumerate(self.csv_data[video_id]):
            if entry['start_frame'] <= frame_idx <= entry['end_frame']:
                if i + 1 >= len(self.csv_data[video_id]):
                    return entry  # 返回最后一类标签
                else:
                    return self.csv_data[video_id][i + 1]

        for entry in self.csv_data[video_id]:
            if frame_idx < entry['start_frame']:
                return entry

        return None

    def __len__(self):
        return len(self.valid_videos)


class LMDBHandler:
    def __init__(self, lmdb_path, map_size=int(6e11)):
        self.env = lmdb.open(lmdb_path, map_size=map_size)

    def store_data(self, video_id, frames_features, frames_labels):
        with self.env.begin(write=True) as txn:
            for idx, (features, label) in enumerate(zip(frames_features, frames_labels)):
                key = f"{video_id}_{idx}".encode('ascii')
                value = {
                    'features': features,
                    'verb_class': label['verb_class'],
                    'noun_class': label['noun_class'],
                    'activity_class': label['activity_class']
                }
                txn.put(key, np.array(value, dtype=object))

    def close(self):
        self.env.close()


def main():
    # 初始化模型和图像处理器
    path = "/root/autodl-tmp/huggingface/InternViT-6B-448px-V1-2"
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    image_processor = CLIPImageProcessor.from_pretrained(path)

    # 初始化视频数据集
    dataset = VideoActionDataset(
        train_txt_path="/root/autodl-tmp/EPIC_KITCHENS_2018/video/train_test.txt", 
        root_path='/root/autodl-tmp/EPIC_KITCHENS_2018/video', 
        csv_file_path='/root/autodl-tmp/EPIC55/training.csv', 
        frame_interval=30, 
        model=model, 
        image_processor=image_processor,
        batch_size=32  # 批量大小为32，控制显存占用
    )

    # 初始化 LMDB 存储
    lmdb_handler = LMDBHandler(lmdb_path="/root/autodl-tmp/lmdb/train_video_features.lmdb")

    # 开始处理并存储特征
    total_videos = len(dataset)
    for idx in range(total_videos):
        video_info = dataset.valid_videos[idx]
        video_id = video_info['video_id']
        try:
            # 获取视频帧特征和标签
            frames_features, frames_labels = dataset[idx]

            # 将提取的特征和标签存储到 LMDB 中
            lmdb_handler.store_data(video_id, frames_features, frames_labels)
            print(f"视频 {video_id} 的特征已存储，帧数: {len(frames_features)}")

        except Exception as e:
            print(f"处理视频 {video_id} 时发生错误: {e}")

    # 关闭 LMDB 句柄
    lmdb_handler.close()
    print("所有视频处理完成，数据已存储到 LMDB 中")

if __name__ == "__main__":
    main()