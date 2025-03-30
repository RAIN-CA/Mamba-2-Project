import os
import cv2
import torch
import torch.nn.functional as F

import csv
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, CLIPImageProcessor
from PIL import Image
import gc

import torch.nn as nn
from mamba_ssm import Mamba2

class VideoActionDataset(Dataset):
    def __init__(self, train_txt_path, root_path, csv_file_path, frame_interval=5, frame_segment_length=200, sequence_length=10, model=None, image_processor=None):
        """
        初始化 Dataset 对象，提前确认每个视频是否在 CSV 文件中有标签的记录，并进行帧切割。
        
        参数:
        - frame_interval: 每隔多少帧提取一次。
        - frame_segment_length: 每次返回的帧片段长度（用于控制显存）。
        - sequence_length: 训练时使用的序列长度。
        - model: 用于提取视频帧特征的预训练模型。
        - image_processor: 用于处理视频帧的图像处理器。
        """
        self.video_paths, self.video_ids = self.get_video_paths_and_ids(train_txt_path, root_path)
        self.frame_interval = frame_interval
        self.frame_segment_length = frame_segment_length
        self.sequence_length = sequence_length  # 加入 sequence_length 参数
        self.csv_data = self.read_csv_file(csv_file_path)
        self.model = model
        self.image_processor = image_processor

        # 过滤掉没有标签的视频，并记录每个视频的总帧数
        self.valid_videos = self.filter_videos_with_labels()

        # 预先计算每个视频的帧片段并保存
        self.segmented_frames = self.segment_videos()
        
        # 打印每个视频的切割信息以便检查
        self.print_segment_info()

    def segment_videos(self):
        """ 对每个视频进行切割，返回每个片段的起始帧和结束帧信息 """
        segmented_frames = []
        for video_info in self.valid_videos:
            video_segments = []
            total_frames = video_info['total_frames']
            max_frame = video_info['max_frame']
            for start_frame in range(0, max_frame, self.frame_interval * self.frame_segment_length):
                end_frame = min(start_frame + self.frame_segment_length * self.frame_interval, max_frame)
                video_segments.append((start_frame, end_frame))
            segmented_frames.append({
                'video_path': video_info['video_path'],
                'video_id': video_info['video_id'],
                'segments': video_segments
            })
        return segmented_frames

    def print_segment_info(self):
        """ 打印每个视频的切割片段信息 """
        for video_info in self.segmented_frames:
            print(f"视频ID: {video_info['video_id']}")
            for segment in video_info['segments']:
                print(f"片段: 起始帧 {segment[0]}, 结束帧 {segment[1]}")

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
                verb_class = row[4].strip()
                noun_class = row[5].strip()
                activity_class = row[6].strip()
                
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
        """ 过滤掉没有标签的视频，并记录每个视频的总帧数 """
        valid_videos = []

        for video_path, video_id in zip(self.video_paths, self.video_ids):
            if video_id not in self.csv_data:
                continue  # 没有标签记录，跳过该视频

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue  # 无法打开视频，跳过该视频

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame = max(entry['end_frame'] for entry in self.csv_data[video_id])

            if total_frames > 0 and max_frame > 0:
                video_info = {
                    'video_path': video_path,
                    'video_id': video_id,
                    'total_frames': total_frames,
                    'max_frame': max_frame
                }
                valid_videos.append(video_info)

            cap.release()

        return valid_videos

    def extract_features_from_frame(self, frame):
        """ 提取视频帧的特征 """
        with torch.no_grad():
            processed_frame = self.image_processor(images=Image.fromarray(frame), return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
            outputs = self.model(processed_frame)
            return outputs.last_hidden_state

    def __getitem__(self, idx):
        """
        动态加载视频帧并返回设定长度的帧片段。
        使用初始化时切割好的帧片段信息。
        """
        # 确保 idx 在合法范围内
        if idx >= len(self):
            raise IndexError(f"Index {idx} 超出数据集范围")

        # 计算当前 idx 对应的视频和片段索引
        total_segments_per_video = [len(video_info['segments']) for video_info in self.segmented_frames]
        cumulative_segments = [sum(total_segments_per_video[:i+1]) for i in range(len(total_segments_per_video))]

        # 找到对应的视频和片段
        video_idx = next(i for i, cum_seg in enumerate(cumulative_segments) if cum_seg > idx)
        segment_idx = idx - (cumulative_segments[video_idx - 1] if video_idx > 0 else 0)

        video_info = self.segmented_frames[video_idx]
        start_frame, end_frame = video_info['segments'][segment_idx]

        # print(f"处理视频: {video_info['video_id']}, 起始帧: {start_frame}, 结束帧: {end_frame}")

        # 获取视频路径并打开视频
        video_path = video_info['video_path']
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return None

        frames = []
        frame_labels = []
        for frame_idx in range(start_frame, end_frame, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"帧读取失败，帧索引: {frame_idx}，继续读取下一帧")
                continue  # 跳过该帧并继续

            # 提取帧特征并存储
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feature = self.extract_features_from_frame(frame_rgb)
            frames.append(feature)

            # 获取标签并存储
            label = self.get_frame_label(video_info['video_id'], frame_idx)
            frame_labels.append(label)

            # print(f"提取帧索引: {frame_idx} 的特征, 标签: {label}")

        cap.release()

        if not frames or not frame_labels:
            print(f"视频 {video_info['video_id']} 没有有效数据，返回 None")
            return None

        return frames, frame_labels

    def get_frame_label(self, video_id, frame_idx):
        """ 获取帧的标签，查找下一个标签范围的动作类别 """
        if video_id not in self.csv_data:
            return None

        future_label = None

        # 遍历当前视频的所有标签范围
        for i, entry in enumerate(self.csv_data[video_id]):
            # 当前帧位于当前标签范围内，尝试获取下一个标签范围
            if entry['start_frame'] <= frame_idx <= entry['end_frame']:
                # 查找下一个标签范围
                if i + 1 < len(self.csv_data[video_id]):
                    future_label = self.csv_data[video_id][i + 1]  # 获取下一个标签
                    return {
                        'verb_class': future_label['verb_class'],
                        'noun_class': future_label['noun_class'],
                        'activity_class': future_label['activity_class']
                    }
                else:
                    # 如果没有下一个标签，返回当前标签
                    return {
                        'verb_class': entry['verb_class'],
                        'noun_class': entry['noun_class'],
                        'activity_class': entry['activity_class']
                    }

            # 如果当前帧还没有进入任何标签范围，则查找最近的未来标签
            if frame_idx < entry['start_frame']:
                return {
                    'verb_class': entry['verb_class'],
                    'noun_class': entry['noun_class'],
                    'activity_class': entry['activity_class']
                }

        return None
    
    def __len__(self):
        """ 返回视频帧片段的总数 """
        return sum(len(video['segments']) for video in self.segmented_frames)

def simple_collate_fn(batch, window_size=10):
    """
    自定义 collate_fn，直接根据 Dataset 返回的帧进行窗口分割。
    """
    sequences = []
    labels = []

    for data in batch:
        if data is None:
            continue  # 跳过无效数据

        frame_features, frame_labels = data
        if len(frame_features) < window_size:
            print("跳过特征数量不足的样本")
            continue  # 跳过特征数量不足的样本
        
        # 直接分割特征，按顺序读取
        for i in range(0, len(frame_features) - window_size + 1):
            sequence = frame_features[i:i + window_size]
            sequence_labels = frame_labels[i:i + window_size]
            
            # 检查最后一帧的标签是否为 None
            if sequence_labels[-1] is None:
                continue  # 跳过该窗口
            
            # 直接使用已经提取好的特征
            sequences.append(torch.stack(sequence))
            
            # 将最后一帧的标签作为序列标签
            labels.append(sequence_labels[-1])
    
    if len(sequences) == 0:
        print("无有效序列")
        return torch.empty(0), []

    sequences = torch.stack(sequences)
    return sequences, labels

import torch
import torch.nn as nn

class Mamba2WithDETRDecoder(nn.Module):
    def __init__(self, mamba_model, input_feature_dim=3200, hidden_dim=512, num_heads=8, 
                 num_decoder_layers=6, num_classes_activity=2513, 
                 num_classes_noun=352, num_classes_verb=125):
        super(Mamba2WithDETRDecoder, self).__init__()
        self.mamba_model = mamba_model

        # CNN 降维 (将特征从 3200 降维到 512)
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(in_channels=input_feature_dim, out_channels=1024, kernel_size=1),
            nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1)
        )

        # Transformer decoder部分 (DETR风格)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Object queries: learnable embeddings, 设置query数量等于 activity 类别数量
        self.object_queries = nn.Embedding(num_classes_activity, hidden_dim)
        
        # 分类头 (用于 activity, noun, verb 的分类)
        self.class_embed_activity = nn.Linear(hidden_dim, num_classes_activity)
        self.class_embed_noun = nn.Linear(hidden_dim, num_classes_noun)
        self.class_embed_verb = nn.Linear(hidden_dim, num_classes_verb)

    def forward(self, x, next_frame_features=None):
        # 将数据类型转换为 float32 以匹配 CNN 的参数类型
        x = x.float()
        
        # CNN 降维: Conv1d expects [batch_size, channels, sequence_length] -> [batch_size, 3200, 2560]
        x = x.permute(0, 2, 1)  # 交换维度以匹配 Conv1d 输入要求
        reduced_feature = self.feature_reduction(x)  # 输出形状: [batch_size, 512, 2560]

        # 调整维度并传入 Mamba2 模型
        reduced_feature = reduced_feature.permute(0, 2, 1)  # 重新调整维度为 [batch_size, 2560, 512]
        mamba_out, final_state = self.mamba_model(reduced_feature)  # 获取 Mamba 模型的输出

        # 如果 next_frame_features 不为 None，则处理未来帧
        if next_frame_features is not None:
            # 将 next_frame_features 转换为 float32
            next_frame_features = next_frame_features.float()

            # 未来帧的处理：对 next_frame_features 使用同样的 CNN 缩放
            next_frame_features = next_frame_features.permute(0, 2, 1)  # [batch_size, 3200, 256] -> [batch_size, seq_len, 3200]
            next_frame_features_scaled = self.feature_reduction(next_frame_features)  # CNN 降维, 得到 [batch_size, 512, 256]

            # 获取 `next_frame_features` 的帧长度 (256) 来确定删除的帧数量
            frame_length = next_frame_features.size(-1)

            # 删除输入序列的第一个视频帧数据
            reduced_feature = reduced_feature[:, frame_length:, :]  # 删除第一个视频帧 (256维度的帧)

            # 拼接缩放后的未来帧特征到输入序列的末尾
            next_frame_features_scaled = torch.cat((reduced_feature, next_frame_features_scaled.permute(0, 2, 1)), dim=1)  # [batch_size, 2560, 512]
        else:
            next_frame_features_scaled = None  # 如果没有未来帧，返回 None

        # 检查 final_state 的形状并降维
        if final_state.dim() == 4:
            # 假设你想将 [batch_size, 32, 64, 128] 转换为 [batch_size, 2048, 128]
            batch_size, c1, c2, c3 = final_state.shape
            final_state = final_state.view(batch_size, -1, c3)  # 调整为 [batch_size, 2048, 128]

        # 使用线性层将 128 维度转换为 512
        # final_state = nn.Linear(128, 512).to(final_state.device)(final_state)  # [batch_size, 2048, 512]

        # 调整为 [sequence_length, batch_size, 512]
        final_state = final_state.permute(1, 0, 2)  # [2048, batch_size, 512]

        # Object queries 创建并重复
        batch_size = final_state.shape[1]
        queries = self.object_queries.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 通过 transformer decoder
        tgt = torch.zeros_like(queries)  # decoder的初始输入
        decoder_output = self.transformer_decoder(tgt, final_state)  # 输出 [num_queries, batch_size, hidden_dim]

        # 对 decoder 的输出进行分类预测
        pred_activity = self.class_embed_activity(decoder_output)  # [num_queries, batch_size, num_classes_activity]
        pred_noun = self.class_embed_noun(decoder_output)          # [num_queries, batch_size, num_classes_noun]
        pred_verb = self.class_embed_verb(decoder_output)          # [num_queries, batch_size, num_classes_verb]

        return pred_activity, pred_noun, pred_verb, mamba_out, next_frame_features_scaled

# 创建 Mamba2 模型
dim = 512  # 输入特征维度
mamba_model = Mamba2(
    d_model=dim,
    d_state=128,
    d_conv=4,
    expand=4,
).to("cuda")

# 创建带有 CNN 降维逻辑的模型
mamba_2_query_model = Mamba2WithDETRDecoder(mamba_model, input_feature_dim=3200, hidden_dim=512).to("cuda")

import pandas as pd

# 读取 action.csv 文件
def load_action_csv(file_path):
    """
    读取包含 action、verb、noun 的 CSV 文件，并返回一个 DataFrame。
    
    参数:
    - file_path: str - action.csv 文件的路径
    
    返回:
    - action_df: pd.DataFrame - 包含 action、verb、noun 映射关系的 DataFrame
    """
    action_df = pd.read_csv(file_path)
    return action_df

# 示例使用
file_path = '/root/autodl-tmp/EPIC55/actions.csv'
action_df = load_action_csv(file_path)

def query_action_by_id(action_df, action_id):
    """
    查询给定的 action_id 对应的 verb_id 和 noun_id。
    
    参数:
    - action_df: pd.DataFrame - 包含 action、verb、noun 映射关系的 DataFrame
    - action_id: int - 查询的 action 的 id
    
    返回:
    - action_info: dict - 包含 action、verb_id 和 noun_id 的字典
    """
    result = action_df[action_df['id'] == action_id]
    if result.empty:
        return None  # 如果没有找到对应的 action_id
    
    action_info = {
        'action': result['action'].values[0],
        'verb_id': result['verb'].values[0],
        'noun_id': result['noun'].values[0]
    }
    return action_info

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# 定义优化器
optimizer = optim.Adam(mamba_2_query_model.parameters(), lr=1e-4)

# 定义处理批次数据的函数
def process_batch_data(batch_data):
    """
    处理输入的批次数据，去除不必要的维度和分类特征，并缩放 patches。
    
    参数:
    - batch_data: 输入批次数据，形状为 [batch_size, 10, 1, 1025, 3200]

    返回:
    - processed_data: 处理后的数据，形状为 [batch_size, 2560, 3200]
    """
    # 1. 移除中间的 1 维度，形状变为 [batch_size, 10, 1025, 3200]
    batch_data = batch_data.squeeze(2)

    # 2. 移除分类特征，保留图像 patches 的特征。形状变为 [batch_size, 10, 1024, 3200]
    patch_data = batch_data[:, :, 1:, :]  # 移除第一个分类特征（位于第 1 个位置）

    # 3. 将 32x32 的图像 patches 特征缩放为 16x16
    # 首先 reshape 成 [batch_size*10, 3200, 32, 32]，然后进行缩放
    patch_data = patch_data.view(patch_data.size(0) * patch_data.size(1), 3200, 32, 32)
    
    # 使用 interpolate 进行缩放
    patch_data = F.interpolate(patch_data, size=(16, 16), mode='bilinear', align_corners=False)

    # 重新 reshape 回 [batch_size, 10, 16*16, 3200]
    patch_data = patch_data.view(batch_data.size(0), batch_data.size(1), 16*16, 3200)

    # 4. 将 256 在序列长度维度上展开，得到形状 [batch_size, 2560, 3200]
    processed_data = patch_data.view(patch_data.size(0), -1, patch_data.size(-1))

    return processed_data

# 定义计算损失的函数
def compute_loss_with_noun_verb_match(action_df, pred_activity, pred_noun, pred_verb, 
                                     activity_labels, noun_labels, verb_labels):
    """
    计算 activity、noun 和 verb 三个分类任务的损失，并分别考虑 noun 和 verb 自身的匹配。
    
    参数:
    - action_df: pd.DataFrame - 包含 action、verb、noun 映射关系的 DataFrame
    - pred_activity: 模型对 activity 分类的预测，形状为 [num_queries, batch_size, num_classes]
    - pred_noun: 模型对 noun 分类的预测，形状为 [num_queries, batch_size, num_classes]
    - pred_verb: 模型对 verb 分类的预测，形状为 [num_queries, batch_size, num_classes]
    - activity_labels: 真实的 activity 标签，形状为 [batch_size]
    - noun_labels: 真实的 noun 标签，形状为 [batch_size]
    - verb_labels: 真实的 verb 标签，形状为 [batch_size]
    
    返回:
    - total_loss: 总损失
    """
    # 选择最高得分的 query 对应的预测
    pred_activity_max = pred_activity.max(dim=0)[0]  # [batch_size, num_classes]
    pred_noun_max = pred_noun.max(dim=0)[0]  # [batch_size, num_classes]
    pred_verb_max = pred_verb.max(dim=0)[0]  # [batch_size, num_classes]

    # 定义标准的交叉熵损失
    criterion_activity = torch.nn.CrossEntropyLoss()
    criterion_noun = torch.nn.CrossEntropyLoss()
    criterion_verb = torch.nn.CrossEntropyLoss()
    
    # 计算 activity、noun、verb 的交叉熵损失
    loss_activity = criterion_activity(pred_activity_max, activity_labels)
    loss_noun = criterion_noun(pred_noun_max, noun_labels)
    loss_verb = criterion_verb(pred_verb_max, verb_labels)
    
    # 额外的损失项，基于 noun 和 verb 与 activity 之间的组合
    additional_loss = 0
    for i in range(len(activity_labels)):
        predicted_noun_id = pred_noun_max[i].argmax().item()
        predicted_verb_id = pred_verb_max[i].argmax().item()
        true_activity_id = activity_labels[i].item()
        
        # 根据 activity id 获取对应的 noun 和 verb
        action_info = query_action_by_id(action_df, true_activity_id)
        if action_info is None:
            continue
        
        true_noun_id = action_info['noun_id']
        true_verb_id = action_info['verb_id']
        
        # 如果 noun 和 verb 都与 activity 匹配，不增加损失
        if predicted_noun_id == true_noun_id and predicted_verb_id == true_verb_id:
            continue
        
        # 如果只有 noun 或 verb 匹配，增加适中的损失
        elif predicted_noun_id == true_noun_id or predicted_verb_id == true_verb_id:
            additional_loss += 0.5
        
        # 如果 noun 和 verb 都不匹配，增加较大的损失
        else:
            additional_loss += 1.0
    
    # 总损失 = activity 损失 + noun 损失 + verb 损失 + 额外的组合损失
    total_loss = loss_activity + loss_noun + loss_verb + additional_loss
    
    return total_loss, loss_activity, loss_noun, loss_verb

# 分组训练函数
def grouped_training_with_model(dataloader, model, batch_size=10, device='cuda'):
    # 用于保存剩余未分组的数据
    remaining_data = None
    remaining_labels = None
    
    model.train()  # 将模型设置为训练模式

    for sequences, labels in dataloader:
        if sequences.nelement() == 0:
            print("无有效数据")
            continue
        
        print(f"从 DataLoader 中获取的数据形状: {sequences.shape}, 标签数量: {len(labels)}")
        
        # 如果有剩余数据，将其与当前数据拼接
        if remaining_data is not None:
            sequences = torch.cat((remaining_data, sequences), dim=0)
            labels = remaining_labels + labels
            remaining_data = None
            remaining_labels = None
        
        # 分批次处理数据
        num_batches = sequences.shape[0] // batch_size
        # print("start train by batch")
        for batch_idx in range(num_batches):
            start_time = time.time()  # 记录开始时间
            
            # 取出当前批次的数据和标签
            batch_data = sequences[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            batch_labels = labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # 数据处理：去除不必要的维度、缩放 patches 等操作
            processed_data = process_batch_data(batch_data)
            # print("data processed")

            # 将 batch_labels 中的分类标签分离
            activity_labels = torch.tensor([int(label['activity_class']) for label in batch_labels]).to(device)
            noun_labels = torch.tensor([int(label['noun_class']) for label in batch_labels]).to(device)
            verb_labels = torch.tensor([int(label['verb_class']) for label in batch_labels]).to(device)
            # print("labels loaded")
            # 数据传入模型进行前向传播
            
            print(f"processed data shape: {processed_data.shape}")
            pred_activity, pred_noun, pred_verb = model(processed_data)
            print(f"pre shape: {pred_activity.shape}")

            # 计算损失
            total_loss, loss_activity, loss_noun, loss_verb = compute_loss_with_noun_verb_match(action_df, pred_activity, pred_noun, pred_verb, activity_labels, noun_labels, verb_labels)
            
            # 反向传播和优化器更新
            optimizer.zero_grad()  # 清空梯度
            total_loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            
            # 计算处理时间
            elapsed_time = time.time() - start_time
            
            print(f"训练了 {batch_size} 组数据，数据形状: {processed_data.shape}")
            print(f"标签: {batch_labels}")
            print(f"总损失: {total_loss.item():.4f}，活动损失: {loss_activity.item():.4f}，名词损失: {loss_noun.item():.4f}，动词损失: {loss_verb.item():.4f}")
            print(f"单个数据组处理时间: {elapsed_time:.4f} 秒")

            # 释放显存
            del batch_data, processed_data, pred_activity, pred_noun, pred_verb
            torch.cuda.empty_cache()
        
        # 处理剩余数据
        remaining_samples = sequences.shape[0] % batch_size
        if remaining_samples > 0:
            remaining_data = sequences[-remaining_samples:]
            remaining_labels = labels[-remaining_samples:]

        # 释放显存
        del sequences, labels
        torch.cuda.empty_cache()

    # 输出剩余未处理的数量
    if remaining_data is not None:
        print(f"保留了 {remaining_data.shape[0]} 组数据供下次训练")

# 训练多个 epoch 的主循环
def train_epochs(num_epochs, dataloader, model, batch_size=10, device='cuda'):
    save_dir = '/root/autodl-tmp/mamba_results/pth'
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，创建目录

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        grouped_training_with_model(dataloader, model, batch_size, device)
        
        # 保存模型权重到指定路径
        model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型保存至: {model_path}")
        print(f"Epoch {epoch + 1} 完成")
    
# 加载模型和 image processor，只加载一次
path = "/root/autodl-tmp/huggingface/InternViT-6B-448px-V1-2"
internvl_model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).cuda().eval()

image_processor = CLIPImageProcessor.from_pretrained(path)

# # 创建Dataset，并将加载好的模型和处理器传入
# dataset = VideoActionDataset(
#     train_txt_path='/root/autodl-tmp/EPIC_KITCHENS_2018/video/train_test.txt',
#     root_path='/root/autodl-tmp/EPIC_KITCHENS_2018/video',
#     csv_file_path='/root/autodl-tmp/EPIC55/training.csv',
#     frame_interval=30,         # 每隔30帧提取一帧
#     frame_segment_length=50,   # 每次返回的帧片段长度为10
#     sequence_length=10,
#     model=internvl_model,
#     image_processor=image_processor
# )

eval_dataset = VideoActionDataset(
    train_txt_path='/root/autodl-tmp/EPIC_KITCHENS_2018/video/test_test_2.txt',
    root_path='/root/autodl-tmp/EPIC_KITCHENS_2018/video',
    csv_file_path='/root/autodl-tmp/EPIC55/video_label.csv',
    frame_interval=30,         # 每隔30帧提取一帧
    frame_segment_length=50,   # 每次返回的帧片段长度为10
    sequence_length=10,
    model=internvl_model,
    image_processor=image_processor
)

# 使用时创建 DataLoader
window_size = 10  # 每个时间序列窗口的长度

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: simple_collate_fn(x, window_size))
eval_dataloader = DataLoader(
    eval_dataset,  # 评估用的数据集
    batch_size=1,  # 评估时每次处理一个样本
    shuffle=False,  # 不打乱数据
    collate_fn=lambda x: simple_collate_fn(x, window_size=10)  # 保留之前的过滤逻辑
)


def load_model(model, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model

# 模型评估函数
def grouped_evaluation_with_model(dataloader, model, batch_size=10, device='cuda'):
    # 设置日志文件保存路径
    log_dir = '/root/autodl-tmp/mamba_results/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'evaluation_log.txt')

    # 打开文件准备写入
    with open(log_file_path, 'w') as log_file:
        total_correct_activity = 0
        total_correct_noun = 0
        total_correct_verb = 0
        total_samples = 0

        model.eval()  # 设置模型为评估模式

        with torch.no_grad():  # 在评估过程中不需要计算梯度
            for sequences, labels in dataloader:
                if sequences.nelement() == 0:
                    log_file.write("无有效数据\n")
                    continue

                # 分批次处理数据
                num_batches = sequences.shape[0] // batch_size
                remaining_size = sequences.shape[0] % batch_size

                for batch_idx in range(num_batches):
                    # 记录开始时间
                    start_time = time.time()

                    # 取出当前批次的数据和标签
                    batch_data = sequences[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                    batch_labels = labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                    # 写入批次信息
                    log_file.write(f"\n处理第 {batch_idx + 1} 个批次，数据形状: {batch_data.shape}\n")

                    # 数据处理
                    processed_data = process_batch_data(batch_data)
                    log_file.write(f"处理后的数据形状: {processed_data.shape}\n")

                    # 将 batch_labels 中的分类标签分离
                    activity_labels = torch.tensor([int(label['activity_class']) for label in batch_labels]).to(device)
                    noun_labels = torch.tensor([int(label['noun_class']) for label in batch_labels]).to(device)
                    verb_labels = torch.tensor([int(label['verb_class']) for label in batch_labels]).to(device)

                    # 写入标签信息
                    log_file.write(f"Batch {batch_idx + 1} 的真实 Activity 标签: {activity_labels.tolist()}\n")
                    log_file.write(f"Batch {batch_idx + 1} 的真实 Noun 标签: {noun_labels.tolist()}\n")
                    log_file.write(f"Batch {batch_idx + 1} 的真实 Verb 标签: {verb_labels.tolist()}\n")

                    # 前向传播，获得预测结果
                    pred_activity, pred_noun, pred_verb, _, _ = model(processed_data)

                    # 获取预测的类别：选择每个 batch 中得分最高的 query 对应的预测
                    pred_activity_labels = pred_activity.max(dim=0)[0].argmax(dim=-1)  # 取最大得分query的activity预测结果
                    pred_noun_labels = pred_noun.max(dim=0)[0].argmax(dim=-1)          # 取最大得分query的noun预测结果
                    pred_verb_labels = pred_verb.max(dim=0)[0].argmax(dim=-1)          # 取最大得分query的verb预测结果

                    # 写入预测结果信息
                    log_file.write(f"Batch {batch_idx + 1} 的预测 Activity 标签: {pred_activity_labels.tolist()}\n")
                    log_file.write(f"Batch {batch_idx + 1} 的预测 Noun 标签: {pred_noun_labels.tolist()}\n")
                    log_file.write(f"Batch {batch_idx + 1} 的预测 Verb 标签: {pred_verb_labels.tolist()}\n")

                    # 对比真实标签与预测标签
                    log_file.write(f"Batch {batch_idx + 1} 的 Activity 对比 (预测 vs 真实): {list(zip(pred_activity_labels.tolist(), activity_labels.tolist()))}\n")
                    log_file.write(f"Batch {batch_idx + 1} 的 Noun 对比 (预测 vs 真实): {list(zip(pred_noun_labels.tolist(), noun_labels.tolist()))}\n")
                    log_file.write(f"Batch {batch_idx + 1} 的 Verb 对比 (预测 vs 真实): {list(zip(pred_verb_labels.tolist(), verb_labels.tolist()))}\n")

                    # 统计每一类的正确预测数量
                    activity_correct = (pred_activity_labels == activity_labels).sum().item()
                    noun_correct = (pred_noun_labels == noun_labels).sum().item()
                    verb_correct = (pred_verb_labels == verb_labels).sum().item()

                    # 写入每个批次的正确预测数量
                    log_file.write(f"第 {batch_idx + 1} 批次正确的 Activity 数量: {activity_correct}\n")
                    log_file.write(f"第 {batch_idx + 1} 批次正确的 Noun 数量: {noun_correct}\n")
                    log_file.write(f"第 {batch_idx + 1} 批次正确的 Verb 数量: {verb_correct}\n")

                    total_correct_activity += activity_correct
                    total_correct_noun += noun_correct
                    total_correct_verb += verb_correct
                    total_samples += batch_data.shape[0]

                    # 记录每个批次的处理时间
                    end_time = time.time()
                    batch_time = end_time - start_time
                    log_file.write(f"第 {batch_idx + 1} 批次处理时间: {batch_time:.4f} 秒\n")

            # 计算并写入正确率
            if total_samples > 0:
                activity_accuracy = total_correct_activity / total_samples * 100
                noun_accuracy = total_correct_noun / total_samples * 100
                verb_accuracy = total_correct_verb / total_samples * 100
            else:
                activity_accuracy = noun_accuracy = verb_accuracy = 0

            log_file.write("\n==================================================\n")
            log_file.write(f"总样本数量: {total_samples}\n")
            log_file.write(f"Activity 准确率: {activity_accuracy:.2f}%\n")
            log_file.write(f"Noun 准确率: {noun_accuracy:.2f}%\n")
            log_file.write(f"Verb 准确率: {verb_accuracy:.2f}%\n")
            log_file.write("==================================================\n")

    print(f"评估日志已保存至: {log_file_path}")

# 运行评估
def run_evaluation(model, model_path, eval_dataloader, device='cuda', result_dir='/root/autodl-tmp/mamba_results/eval_result'):
    # 确保结果保存的目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 加载模型
    model = load_model(model, model_path, device)

    # 评估模型
    activity_acc, noun_acc, verb_acc = grouped_evaluation_with_model(eval_dataloader, model, 10,  device)

    # 保存评估结果
    result_file = os.path.join(result_dir, 'evaluation_results.txt')
    with open(result_file, 'a') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Activity Accuracy: {activity_acc * 100:.2f}%\n")
        f.write(f"Noun Accuracy: {noun_acc * 100:.2f}%\n")
        f.write(f"Verb Accuracy: {verb_acc * 100:.2f}%\n")
        f.write("="*50 + "\n")

    print(f"评估结果已保存到: {result_file}")

# 示例使用
run_evaluation(mamba_2_query_model, '/root/autodl-tmp/mamba_results/pth/model_epoch_5.pth', eval_dataloader)

# 释放模型占用的显存
del internvl_model
del mamba_2_query_model
torch.cuda.empty_cache()
gc.collect()
