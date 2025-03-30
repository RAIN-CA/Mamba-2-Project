from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F
# from mamba_ssm import Mamba2

class OpenLSTM(nn.Module):
    """"An LSTM implementation that returns the intermediate hidden and cell states.
    The original implementation of PyTorch only returns the last cell vector.
    For RULSTM, we want all cell vectors computed at intermediate steps"""
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        """
            feat_in: input feature size
            feat_out: output feature size
            num_layers: number of layers
            dropout: dropout probability
        """
        super(OpenLSTM, self).__init__()

        # simply create an LSTM with the given parameters
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        # manually iterate over each input to save the individual cell vectors
        last_cell=None
        last_hid=None
        hid = []
        cell = []
        for i in range(seq.shape[0]):
            el = seq[i,...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid,last_cell))
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)

        return torch.stack(hid, 0),  torch.stack(cell, 0)

class RULSTM(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1, 
            sequence_completion=False, return_context=False):
        """
            num_class: number of classes
            feat_in: number of input features
            hidden: number of hidden units
            dropout: dropout probability
            depth: number of LSTM layers
            sequence_completion: if the network should be arranged for sequence completion pre-training
            return_context: whether to return the Rolling LSTM hidden and cell state (useful for MATT) during forward
        """
        super(RULSTM, self).__init__()
        self.feat_in = feat_in
        self.dropout = nn.Dropout(dropout)
        self.hidden=hidden
        self.rolling_lstm = OpenLSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.unrolling_lstm = nn.LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_class))
        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        # permute the inputs for compatibility with the LSTM
        inputs=inputs.permute(1,0,2)

        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        x, c = self.rolling_lstm(self.dropout(inputs))
        x = x.contiguous() # batchsize x timesteps x hidden
        c = c.contiguous() # batchsize x timesteps x hidden

        # accumulate the predictions in a list
        predictions = [] # accumulate the predictions in a list
        
        # for each time-step
        for t in range(x.shape[0]):
            # get the hidden and cell states at current time-step
            hid = x[t,...]
            cel = c[t,...]

            if self.sequence_completion:
                # take current + future inputs (looks into the future)
                ins = inputs[t:,...]
            else:
                # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                ins = inputs[t,...].unsqueeze(0).expand(inputs.shape[0]-t+1,inputs.shape[1],inputs.shape[2]).to(inputs.device)
            
            # initialize the LSTM and iterate over the inputs
            h_t, (_,_) = self.unrolling_lstm(self.dropout(ins), (hid.contiguous(), cel.contiguous()))
            # get last hidden state
            h_n = h_t[-1,...]

            # append the last hidden state to the list
            predictions.append(h_n)
        
        # obtain the final prediction tensor by concatenating along dimension 1
        x = torch.stack(predictions,1)

        # apply the classifier to each output feature vector (independently)
        y = self.classifier(x.view(-1,x.size(2))).view(x.size(0), x.size(1), -1)
        
        if self.return_context:
            # return y and the concatenation of hidden and cell states 
            c=c.squeeze().permute(1,0,2)
            return y, torch.cat([x, c],2)
        else:
            return y

class RULSTMFusion(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
        """
        super(RULSTMFusion, self).__init__()
        self.branches = nn.ModuleList(branches)

        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        in_size = 2*len(self.branches)*hidden
        
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), len(self.branches)))


    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        scores, contexts = [], []

        # for each branch
        for i in range(len(inputs)):
            # feed the inputs to the LSTM and get the scores and context vectors
            s, c = self.branches[i](inputs[i])
            scores.append(s)
            contexts.append(c)

        context = torch.cat(contexts, 2)
        context = context.view(-1, context.shape[-1])

        # Apply the MATT network to the context vectors
        # and normalize the outputs using softmax
        a = F.softmax(self.MATT(context),1)

        # array to contain the fused scores
        sc = torch.zeros_like(scores[0])

        # fuse all scores multiplying by the weights
        for i in range(len(inputs)):
            s = (scores[i].view(-1,scores[i].shape[-1])*a[:,i].unsqueeze(1)).view(sc.shape)
            sc += s

        # return the fused scores
        return sc


# from transformers import Mamba2Config, Mamba2Model
from mamba_ssm import Mamba2Simple, Mamba2
import torch
import torch.nn as nn

class RUMambaTimeSeries(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, sequence_completion=False, return_context=False):
        super(RUMambaTimeSeries, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        # print(f"hidden = {hidden}")

        self.mamba = Mamba2Simple(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=feat_in, # Model dimension d_model
            d_state=128,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            headdim=64,
            chunk_size=1,
            use_mem_eff_path=False
        ).to("cuda")

        # 分类器：直接线性层，将展平后的特征映射到类别
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_class))

        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        # 输入 Mamba2 模型时将其传递为 inputs_embeds
        seq_len = inputs.shape[1]
        # outputs, hidden_states = self.mamba(self.dropout(inputs))

        # 从 Mamba2 的输出中提取最后的 hidden states
        # hidden_states = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        # print(f"hidden_states shape = {hidden_states.shape}") # hidden_states shape = torch.Size([128, 14, 1024])

        # print(f"hidden_states shape: {hidden_states.shape}")

        predictions = []  # 用于存储每个时间步的预测

        for t in range(seq_len):  # 遍历时间步
            if self.sequence_completion:
                # 获取当前及未来的输入
                future_inputs = inputs[:, t:, :]  # 从时间步 t 到最后
            else:
                # 将当前时间步的输入扩展为剩余时间步
                future_inputs = inputs[:, t, :].unsqueeze(1).expand(-1, inputs.shape[1] - t, -1)

            # 将未来的输入传递给 Mamba 模型的 unrolling 部分
            future_outputs, hidden_states = self.mamba(self.dropout(future_inputs))
            
            # 取最后一个时间步的 hidden state 作为预测
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

            # 将最后一个 hidden state 传递给分类器
            y_t = self.classifier(last_hidden)  # [batch_size, num_class]
            predictions.append(y_t)

        # 堆叠所有时间步的预测
        y = torch.stack(predictions, dim=1)  # [batch_size, sequence_length, num_class]

        # print(f"y shape: {y.shape}")

        if self.return_context:
            return y, hidden_states
        else:
            return y

class RUMambaTimeSeriesDirect(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, sequence_completion=False, return_context=False):
        super(RUMambaTimeSeriesDirect, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        # print(f"hidden = {hidden}")

        self.mamba = Mamba2Simple(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=feat_in, # Model dimension d_model
            d_state=128,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            headdim=64,
            chunk_size=1,
            use_mem_eff_path=False
        ).to("cuda")

        # 分类器：直接线性层，将展平后的特征映射到类别
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_class))

        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        # 输入 Mamba2 模型时将其传递为 inputs_embeds
        outputs, hidden_states = self.mamba(self.dropout(inputs)) # [batch_size, sequence_length, hidden_size]

        # print(f"hidden_states shape: {hidden_states.shape}")

        predictions = []  # 用于存储每个时间步的预测

        for t in range(hidden_states.shape[1]):  # 遍历时间步
            last_hidden = hidden_states[:, t, :]  # [batch_size, hidden_size]

            # 将最后一个 hidden state 传递给分类器
            y_t = self.classifier(last_hidden)  # [batch_size, num_class]
            predictions.append(y_t)

        # 堆叠所有时间步的预测
        y = torch.stack(predictions, dim=1)  # [batch_size, sequence_length, num_class]

        # print(f"y shape: {y.shape}")

        if self.return_context:
            return y, hidden_states
        else:
            return y

class SingleLSTMForComparison(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1, sequence_completion=False, return_context=False):
        super(SingleLSTMForComparison, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        # 单一LSTM模型
        self.lstm = nn.LSTM(input_size=feat_in, hidden_size=hidden, num_layers=depth, dropout=dropout if depth > 1 else 0, batch_first=True)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_class)
        )
        
        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        inputs = self.dropout(inputs)  # 添加 dropout

        # LSTM 计算，返回输出和最后的 hidden state
        lstm_out, (hidden, cell) = self.lstm(inputs)  # lstm_out: [batch_size, sequence_length, hidden]

        # 将 view 替换为 reshape，防止内存非连续性问题
        y = self.classifier(lstm_out.reshape(-1, lstm_out.size(2)))  # [batch_size * sequence_length, hidden]
        y = y.reshape(lstm_out.size(0), lstm_out.size(1), -1)  # [batch_size, sequence_length, num_classes]

        if self.return_context:
            return y, lstm_out  # 返回分类结果和 LSTM 输出
        else:
            return y


from mamba_ssm import Mamba2Simple, Mamba2

import torch
import torch.nn as nn

class Mamba2Model(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, sequence_completion=False, return_context=False):
        super(Mamba2Model, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        self.mamba = Mamba2Simple(
            d_model=192, 
            d_state=128,  
            d_conv=4,    
            expand=1,    
            headdim=8,
            chunk_size=196,
            use_mem_eff_path=False
        ).to("cuda")

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1536, num_class))

        self.sequence_completion = sequence_completion
        self.return_context = return_context

        # 初始化 mamba_loss 属性
        self.mamba_loss = 0

    def forward(self, inputs):
        # print(f"inputs: {inputs}")
        seq_len = inputs.shape[1]
        inputs = inputs[:, :, 1:, :]  # 去除 cls 部分
        
        predictions = []  

        for t in range(seq_len):  
            if self.sequence_completion:
                future_inputs = inputs[:, t:, :, :]  # 从时间步 t 到最后
            else:
                future_inputs = inputs[:, t, :, :].unsqueeze(1).expand(-1, inputs.shape[1] - t, -1, -1)

            future_inputs = inputs.reshape(future_inputs.size(0), -1, future_inputs.size(-1))
            
            future_outputs, hidden_states = self.mamba(self.dropout(future_inputs))
            
            last_hidden = hidden_states[:, -1, :]  

            y_t = self.classifier(last_hidden)  
            predictions.append(y_t)

            # 在第一个时间步更新 Mamba loss
            if t == 0:
                self.update_mamba_loss(future_outputs, inputs)

        y = torch.stack(predictions, dim=1) 

        if self.return_context:
            return y, hidden_states
        else:
            return y

    def update_mamba_loss(self, future_outputs, inputs):
        # 使用 chunksize 去除最后一个块的输出数据，以对齐时间步
        chunksize = self.mamba.chunk_size  # 假设 chunksize 是 Mamba 模型的属性
        aligned_outputs = future_outputs[:, :-chunksize, :]  # 去掉最后一个 chunksize 的输出
        
        # 从第 2 个时间步开始的输入，展平并对齐到相同的维度
        target_data = inputs[:, 1:, :, :].reshape(aligned_outputs.size(0), -1, aligned_outputs.size(-1))  
        
        # 调试信息
        # print(f"aligned_outputs shape: {aligned_outputs.shape}")
        # print(f"target_data shape: {target_data.shape}")
        
        # 检查对齐后的形状
        if aligned_outputs.shape != target_data.shape:
            raise ValueError(f"Shape mismatch between aligned_outputs {aligned_outputs.shape} and target_data {target_data.shape}")
        
        # 更新 Mamba loss
        self.mamba_loss = self.mamba_loss_fn(aligned_outputs, target_data)

    def mamba_loss_fn(self, aligned_outputs, target_data):
        return nn.MSELoss()(aligned_outputs, target_data)
    
    def get_mamba_loss(self):
        """返回当前的 mamba_loss 值"""
        return self.mamba_loss

import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        # 创建位置编码矩阵，针对每个时间步生成相同编码用于该时间步下的所有 patches
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为模型的一个 buffer，避免更新
        self.register_buffer('pe', pe.unsqueeze(0).unsqueeze(2))  # [1, max_len, 1, d_model]

    def forward(self, x):
        # 检查位置编码是否匹配输入大小
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Position encoding length {self.pe.size(1)} is less than input sequence length {x.size(1)}.")
        if x.size(3) != self.pe.size(3):
            raise ValueError(f"Position encoding feature size {self.pe.size(3)} does not match input feature size {x.size(3)}.")
        
        # 为每个时间步的所有 patches 添加相同的时间位置编码
        x = x + self.pe[:, :x.size(1), :, :]  # 广播位置编码到 [batch, seqlen, patches, d_model]
        return x

class RMamba2TimeSeriesDirect(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, sequence_completion=False, return_context=False):
        super(RMamba2TimeSeriesDirect, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        # 定义 Mamba 模型
        self.mamba = Mamba2Simple(
            d_model=192, 
            d_state=128,  
            d_conv=4,    
            expand=8,    
            headdim=64,
            chunk_size=256,  # 设置为 256
            use_mem_eff_path=False
        ).to("cuda")

        # 时间位置编码
        self.temporal_position_encoding = TemporalPositionalEncoding(192)

        # 分类器：直接线性层，将展平后的特征映射到类别
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1536, num_class))

        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        # 输入形状：[batch, seqlen, 197, 192]
        inputs = inputs[:, :, 1:, :]  # 去除 cls 部分，得到 [batch, seqlen, 196, 192]
        
        # 填充每帧的 patch 数量至 256
        pad_size = 256 - inputs.size(2)
        if pad_size > 0:
            padding = torch.zeros(inputs.size(0), inputs.size(1), pad_size, inputs.size(3), device=inputs.device)
            inputs = torch.cat([inputs, padding], dim=2)  # [batch, seqlen, 256, 192]

        # 添加时间位置编码，将同一时间步的编码应用于该帧的所有 patches
        inputs = self.temporal_position_encoding(inputs)

        # 合并时间和空间维度以输入到 Mamba 模型
        inputs = inputs.reshape(inputs.size(0), -1, inputs.size(-1))  # [batch, seqlen * 256, 192]

        # 输入 Mamba2 模型并获取隐藏状态
        outputs, hidden_states = self.mamba(self.dropout(inputs))  # [batch_size, sequence_length, hidden_size]
        
        # 获取最后一个时间步的 hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # 将最后一个 hidden state 传递给分类器
        final_output = self.classifier(last_hidden)  # [batch_size, num_class]

        # 将最后一个时间步的结果扩展到每个时间步
        y = final_output.unsqueeze(1).expand(-1, hidden_states.size(1), -1)  # [batch_size, sequence_length, num_class]

        if self.return_context:
            return y, hidden_states
        else:
            return y
        
import torch
import torch.nn as nn

class MambaTimeSeriesClassifier(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, return_context=False):
        super(MambaTimeSeriesClassifier, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        self.mamba = Mamba2Simple(
            d_model=192, 
            d_state=128,
            d_conv=4,    
            expand=4,    
            headdim=32,
            chunk_size=256,
            use_mem_eff_path=False
        ).to("cuda")

        self.temporal_encoding = TemporalPositionalEncoding(192)
        
        # 分类器输入维度基于 final_state 的处理方式调整
        nheads = 8  # 假设 mamba 的多头数量为 8
        headdim = 32  # 假设每头的维度为 64
        dstate = 128  # 假设 d_state 为 128
        # feature_dim = nheads * headdim * dstate  # 如果展平 final_state
        feature_dim = headdim * dstate  # 如果聚合 final_state

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_class)
        )
        self.return_context = return_context

    def forward(self, inputs):
        # 输入形状：[batch, seqlen, 197, 192]
        inputs = inputs[:, :, 1:, :]  # 去除 CLS token，得到 [batch, seqlen, 196, 192], 14*14 patches

        # 填充每帧的 patch 数量至 256
        pad_size = 256 - inputs.size(2)
        if pad_size > 0:
            padding = torch.zeros(inputs.size(0), inputs.size(1), pad_size, inputs.size(3), device=inputs.device)
            inputs = torch.cat([inputs, padding], dim=2)  # [batch, seqlen, 256, 192]

        # 添加时间位置编码
        inputs = self.temporal_encoding(inputs)

        # 存储每个时间步的分类结果
        predictions = []

        # 遍历每个时间步，使用不同长度的子序列输入 Mamba 模型
        for t in range(inputs.size(1)):
            # 截取从时间步 t 开始的子序列
            x = inputs[:, t:, :, :]  # [batch, seqlen - t, 256, 192]

            # 合并时间和空间维度以输入到 Mamba 模型
            reshaped_x = x.reshape(x.size(0), -1, x.size(-1))  # [batch, (seqlen - t) * 256, 192]
            _, final_state = self.mamba(self.dropout(reshaped_x))  # [batch, nheads, headdim, dstate]

            # 处理 final_state
            # aggregated_state = final_state.view(final_state.size(0), -1)  # 展平成一维
            aggregated_state = final_state.sum(dim=1)  # 或者按 nheads 累加
            # 展平成二维张量
            aggregated_state = aggregated_state.view(aggregated_state.size(0), -1)  # [batch, feature_dim1 * feature_dim2]
            # print(f"aggregated_state shape: {aggregated_state.shape}")

            # 分类器输出预测
            pred = self.classifier(aggregated_state)  # [batch_size, num_class]
            predictions.append(pred.unsqueeze(1))  # 收集每个时间步的预测结果

            # 删除不再使用的张量并释放显存
            del x, reshaped_x, final_state
            torch.cuda.empty_cache()

        # 将所有时间步的预测拼接成最终输出
        y = torch.cat(predictions, dim=1)  # [batch, seqlen, num_class]

        # 删除 predictions 列表并释放显存
        del predictions
        torch.cuda.empty_cache()

        if self.return_context:
            return y, aggregated_state
        else:
            return y
        
class MambaTimeSeriesClassifier_V2(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, return_context=False):
        super(MambaTimeSeriesClassifier_V2, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        
        # 图像特征提取的 4 个 Mamba 模型
        self.mamba_region1 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.mamba_region2 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.mamba_region3 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.mamba_region4 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        
        # 时序特征提取的 Mamba 模型
        self.mamba_sequence = Mamba2Simple(d_model=128, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")

        self.temporal_encoding = TemporalPositionalEncoding(192)
        
        # 分类器
        headdim = 32
        dstate = 128
        feature_dim = headdim * dstate
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_class)
        )
        self.return_context = return_context

    @staticmethod
    def process_final_state(final_state):
        """
        按 nheads 维度累加，并保留 [batch, headdim, dstate] 的形状。
        Args:
            final_state (torch.Tensor): [batch, nheads, headdim, dstate]
        Returns:
            torch.Tensor: [batch, headdim, dstate]
        """
        # 按 nheads 累加
        aggregated_state = final_state.sum(dim=1)  # [batch, headdim, dstate]
        return aggregated_state

    def forward(self, inputs):
        inputs = inputs[:, :, 1:, :]  # 去除 CLS token，得到 [batch, seqlen, 196, 192], 14x14 patches
        inputs = inputs.view(inputs.size(0), inputs.size(1), 14, 14, inputs.size(-1))

        # 分割为四个区域
        region1 = inputs[:, :, :7, :7, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))
        region2 = inputs[:, :, :7, 7:, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))
        region3 = inputs[:, :, 7:, :7, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))
        region4 = inputs[:, :, 7:, 7:, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))

        frame_features = []
        for t in range(inputs.size(1)):
            # 获取每帧的四个区域
            r1, r2, r3, r4 = region1[:, t, :, :], region2[:, t, :, :], region3[:, t, :, :], region4[:, t, :, :]

            # 提取区域特征
            _, f1 = self.mamba_region1(r1)
            _, f2 = self.mamba_region2(r2)
            _, f3 = self.mamba_region3(r3)
            _, f4 = self.mamba_region4(r4)

            # 处理特征，保留 [batch, headdim, dstate]
            f1_processed = self.process_final_state(f1)
            f2_processed = self.process_final_state(f2)
            f3_processed = self.process_final_state(f3)
            f4_processed = self.process_final_state(f4)

            # 将四个区域的特征按 headdim 拼接
            frame_feature = torch.cat([f1_processed, f2_processed, f3_processed, f4_processed], dim=1)  # [batch, 4*headdim, dstate]
            frame_features.append(frame_feature)

        # 将所有帧的特征堆叠为时序数据
        sequence_features = torch.stack(frame_features, dim=1)  # [batch, seqlen, 4*headdim, dstate]

        # 合并 seqlen 和 headdim 维度，控制特征维度大小
        sequence_features = sequence_features.view(sequence_features.size(0), sequence_features.size(1) * sequence_features.size(2), -1)  # [batch, seqlen*4*headdim, dstate]

        # print(f"sequence_features shape: {sequence_features.shape}")
        
        # 时序模型处理
        _, final_state = self.mamba_sequence(sequence_features)
        # print(f"final_state shape:{final_state.shape}")

        # 分类器输入
        aggregated_state = final_state.sum(dim=1)  # 按 nheads 累加
        aggregated_state = aggregated_state.view(aggregated_state.size(0), -1)  # 展平成二维张量 [batch, feature_dim]
        # print(f"aggregated_state shape: {aggregated_state.shape}")

        y = self.classifier(aggregated_state)  # 分类预测
        # 将分类结果扩展到时间维度，与输入 seqlen 对齐
        seqlen = inputs.size(1)  # 输入的时间步长
        y = y.unsqueeze(1).repeat(1, seqlen, 1)  # [batch, seqlen, num_class]
        return y
    
# import torch
# import torch.nn as nn
# import torch_geometric.nn as pyg_nn  # 使用 PyTorch Geometric 的 GCNConv

# class MambaTimeSeriesClassifier_GNN(nn.Module):
#     def __init__(self, num_class, feat_in, hidden, dropout=0.8, return_context=False):
#         super(MambaTimeSeriesClassifier_GNN, self).__init__()
#         self.feat_in = feat_in
#         self.hidden = hidden
#         self.dropout = nn.Dropout(dropout)

#         # 图神经网络（GNN）
#         self.gnn1 = pyg_nn.GCNConv(feat_in, hidden)
#         self.gnn2 = pyg_nn.GCNConv(hidden, hidden)

#         # 时序特征提取的 Mamba 模块
#         self.mamba_sequence = Mamba2Simple(d_model=hidden, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")

#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(hidden, num_class)
#         )
#         self.return_context = return_context

#     def forward(self, inputs):
#         """
#         Args:
#             inputs: [batch, seqlen, 197, feat_in] (197 = 1 (CLS token) + 14*14 patches)
#         """
#         # 去除 CLS token，得到 [batch, seqlen, 196, feat_in]
#         inputs = inputs[:, :, 1:, :]  # [batch, seqlen, 196, feat_in]

#         batch_size, seqlen, num_patches, feat_dim = inputs.shape
#         patch_dim = int(num_patches**0.5)  # 假设输入为 14x14 patches
#         assert patch_dim * patch_dim == num_patches, "Patch size must be square."

#         # 构建边关系（以 2D 网格连接为例）
#         edge_index = self.create_grid_edges(patch_dim).to(inputs.device)  # [2, num_edges]

#         # 存储每帧的图特征
#         frame_features = []
#         for t in range(seqlen):
#             frame_patches = inputs[:, t, :, :]  # [batch, 196, feat_in]

#             # 将每帧的 patch 特征送入 GNN
#             x = frame_patches.view(-1, feat_dim)  # 展平为 [batch*num_patches, feat_in]
#             x = self.gnn1(x, edge_index).relu()
#             x = self.gnn2(x, edge_index)  # [batch*num_patches, hidden]

#             # 汇总图特征（可以使用池化或展平）
#             frame_feature = x.view(batch_size, num_patches, -1).mean(dim=1)  # [batch, hidden]
#             frame_features.append(frame_feature)

#         # 将所有帧的特征堆叠为时序特征
#         sequence_features = torch.stack(frame_features, dim=1)  # [batch, seqlen, hidden]

#         # 时序建模
#         _, final_state = self.mamba_sequence(sequence_features)  # [batch, nheads, headdim, dstate]

#         # 分类器输入
#         aggregated_state = final_state.sum(dim=1)  # 按 nheads 累加
#         aggregated_state = aggregated_state.view(aggregated_state.size(0), -1)  # [batch, hidden]

#         y = self.classifier(aggregated_state)  # [batch, num_class]

#         # 将分类结果扩展到时间维度，与输入 seqlen 对齐
#         y = y.unsqueeze(1).repeat(1, seqlen, 1)  # [batch, seqlen, num_class]
#         return y

#     @staticmethod
#     def create_grid_edges(patch_dim):
#         """
#         创建 2D 网格的边关系，用于 GNN 的输入。
#         Args:
#             patch_dim: int, 网格的边长（如 14 表示 14x14 patches）。
#         Returns:
#             edge_index: [2, num_edges]，图的边。
#         """
#         edges = []
#         for i in range(patch_dim):
#             for j in range(patch_dim):
#                 node = i * patch_dim + j
#                 # 右邻居
#                 if j + 1 < patch_dim:
#                     edges.append((node, node + 1))
#                 # 下邻居
#                 if i + 1 < patch_dim:
#                     edges.append((node, node + patch_dim))
#         edge_index = torch.tensor(edges + [(j, i) for i, j in edges]).t()  # 双向边
#         return edge_index



class MambaTimeSeriesClassifier_V3(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, return_context=False):
        super(MambaTimeSeriesClassifier_V3, self).__init__()
        self.feat_in = feat_in
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)

        # 图像特征提取的 4 个 Mamba 模型
        self.mamba_region1 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.mamba_region2 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.mamba_region3 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.mamba_region4 = Mamba2Simple(d_model=192, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")

        # 时序特征提取 Mamba 模型
        self.mamba_sequence = Mamba2Simple(d_model=128, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")

        self.temporal_encoding = TemporalPositionalEncoding(192)

        headdim = 32
        dstate = 128
        feature_dim = headdim * dstate

        # V3: 改进后的 decoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_class)
        )

        self.return_context = return_context

    @staticmethod
    def process_final_state(final_state):
        aggregated_state = final_state.sum(dim=1)
        return aggregated_state

    def forward(self, inputs):
        inputs = inputs[:, :, 1:, :]
        inputs = inputs.view(inputs.size(0), inputs.size(1), 14, 14, inputs.size(-1))

        region1 = inputs[:, :, :7, :7, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))
        region2 = inputs[:, :, :7, 7:, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))
        region3 = inputs[:, :, 7:, :7, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))
        region4 = inputs[:, :, 7:, 7:, :].reshape(inputs.size(0), inputs.size(1), -1, inputs.size(-1))

        frame_features = []
        for t in range(inputs.size(1)):
            r1, r2, r3, r4 = region1[:, t, :, :], region2[:, t, :, :], region3[:, t, :, :], region4[:, t, :, :]

            _, f1 = self.mamba_region1(r1)
            _, f2 = self.mamba_region2(r2)
            _, f3 = self.mamba_region3(r3)
            _, f4 = self.mamba_region4(r4)

            f1_processed = self.process_final_state(f1)
            f2_processed = self.process_final_state(f2)
            f3_processed = self.process_final_state(f3)
            f4_processed = self.process_final_state(f4)

            frame_feature = torch.cat([f1_processed, f2_processed, f3_processed, f4_processed], dim=1)
            frame_features.append(frame_feature)

        sequence_features = torch.stack(frame_features, dim=1)
        sequence_features = sequence_features.view(sequence_features.size(0), sequence_features.size(1) * sequence_features.size(2), -1)

        _, final_state = self.mamba_sequence(sequence_features)

        aggregated_state = final_state.sum(dim=1)
        aggregated_state = aggregated_state.view(aggregated_state.size(0), -1)

        y = self.classifier(aggregated_state)
        seqlen = inputs.size(1)
        y = y.unsqueeze(1).repeat(1, seqlen, 1)
        return y


class OpenMamba(nn.Module):
    """
    An SSM implementation that returns the intermediate states for each time step.
    This is similar to OpenLSTM but uses a Mamba model for state-space modeling.
    """
    def __init__(self, d_model, d_state, d_conv, expand, headdim, chunk_size, use_mem_eff_path, dropout=0.0):
        """
        Args:
            d_model(feat_in): Input feature size
            feat_out: Output feature size
            d_state: State dimension of Mamba
            d_conv: Convolution dimension of Mamba
            expand: Expansion factor of Mamba
            headdim: Head dimension of Mamba
            chunk_size: Chunk size for Mamba
            dropout: Dropout probability
        """
        super(OpenMamba, self).__init__()

        # Initialize the Mamba model
        self.mamba = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path
        ).to("cuda")
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        """
        Forward pass through the OpenMamba model.
        Args:
            seq: Input sequence of shape [seq_len, batch_size, feat_in]
        Returns:
            outputs: Hidden states at each time step [seq_len, batch_size, feat_out]
            states: Final states at each time step [seq_len, batch_size, d_state]
        """
        last_state = None  # Placeholder for the previous step's state
        outputs = []
        states = []

        for t in range(seq.size(0)):
            # Extract the t-th time step
            x_t = seq[t, ...].unsqueeze(0)  # [1, batch_size, feat_in]

            if last_state is not None:
                # Pass the input and the previous state
                output, final_state = self.mamba(self.dropout(x_t), last_state)
            else:
                # Pass the input only
                output, final_state = self.mamba(self.dropout(x_t))

            # Store the output and final state
            outputs.append(output)
            states.append(final_state)

            # Update the last state
            last_state = final_state

        # Stack outputs and states along the time dimension
        outputs = torch.cat(outputs, dim=0)  # [seq_len, batch_size, feat_out]
        states = torch.stack(states, dim=0)  # [seq_len, batch_size, d_state]

        return outputs, states

class RUMamba(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1, 
            sequence_completion=False, return_context=False):
        """
            num_class: number of classes
            feat_in: number of input features
            hidden: number of hidden units
            dropout: dropout probability
            depth: number of LSTM layers
            sequence_completion: if the network should be arranged for sequence completion pre-training
            return_context: whether to return the Rolling LSTM hidden and cell state (useful for MATT) during forward
        """
        super(RUMamba, self).__init__()
        self.feat_in = feat_in
        self.dropout = nn.Dropout(dropout)
        self.hidden=hidden
        self.rolling_lstm = OpenMamba(d_model=hidden, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        self.unrolling_lstm = Mamba2Simple(d_model=hidden, d_state=128, d_conv=4, expand=4, headdim=32, chunk_size=256, use_mem_eff_path=False).to("cuda")
        
        # 分类器
        headdim = 32
        dstate = 128
        feature_dim = headdim * dstate
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_class)
        )
        
        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        # permute the inputs for compatibility with the LSTM
        inputs=inputs.permute(1,0,2)

        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        _, x = self.rolling_lstm(self.dropout(inputs))
        x = x.contiguous() # batchsize x timesteps x hidden

        # accumulate the predictions in a list
        predictions = [] # accumulate the predictions in a list
        
        # for each time-step
        for t in range(x.shape[0]):
            # get the hidden and cell states at current time-step
            hid = x[t,...]

            if self.sequence_completion:
                # take current + future inputs (looks into the future)
                ins = inputs[t:,...]
            else:
                # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                ins = inputs[t,...].unsqueeze(0).expand(inputs.shape[0]-t+1,inputs.shape[1],inputs.shape[2]).to(inputs.device)
            
            # initialize the LSTM and iterate over the inputs
            _, h_n = self.unrolling_lstm(self.dropout(ins), initial_states=hid.contiguous())

            # append the last hidden state to the list
            predictions.append(h_n)
        
        # obtain the final prediction tensor by concatenating along dimension 1
        x = torch.stack(predictions,1)

        # apply the classifier to each output feature vector (independently)
        y = self.classifier(x.view(-1,x.size(2))).view(x.size(0), x.size(1), -1)
        
        if self.return_context:
            # return y and the concatenation of hidden and cell states 
            c=c.squeeze().permute(1,0,2)
            return y, torch.cat([x, c],2)
        else:
            return y