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
            seq: Input sequence of shape # [batch, seqlen, dim]
        Returns:
            outputs: Hidden states at each time step [seq_len, batch_size, feat_out]
            states: Final states at each time step [seq_len, batch_size, d_state]
        """
        last_state = None  # Placeholder for the previous step's state
        outputs = []
        states = []

        # print(f"seq shape: {seq.shape}")

        for t in range(seq.size(1)):
            # Extract the t-th time step
            x_t = seq[:, t, :].unsqueeze(1)  # [batch, 1, dim]

            if last_state is not None:
                # Pass the input and the previous state
                output, final_state = self.mamba(self.dropout(x_t), initial_states=last_state)
            else:
                # Pass the input only
                output, final_state = self.mamba(self.dropout(x_t))

            # Store the output and final state
            outputs.append(output)
            states.append(final_state)
            # print(f"final state shape: {final_state.shape}")

            # Update the last state
            last_state = final_state

        # Stack outputs and states along the time dimension
        outputs = torch.cat(outputs, dim=0)  # [seq_len, batch_size, feat_out]
        states = torch.stack(states, dim=0)  # [seq_len, batch_size, d_state]
        # print(f"states shape: {states.shape}")

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
        self.rolling_mamba = OpenMamba(d_model=hidden, d_state=128, d_conv=4, expand=2, headdim=32, chunk_size=1, use_mem_eff_path=False).to("cuda")
        self.unrolling_mamba = Mamba2Simple(d_model=hidden, d_state=128, d_conv=4, expand=2, headdim=32, chunk_size=1, use_mem_eff_path=False).to("cuda")
        
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
        # print(f"inputs shape: {inputs.shape}")
        # inputs=inputs.permute(1,0,2)

        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        _, x = self.rolling_mamba(self.dropout(inputs))
        x = x.contiguous() # batchsize x timesteps x hidden

        # accumulate the predictions in a list
        predictions = [] # accumulate the predictions in a list
        
        # for each time-step
        for t in range(x.shape[0]):
            # get the hidden and cell states at current time-step
            hid = x[t]
            # print(f"hid shape: {hid.shape}")

            if self.sequence_completion:
                # take current + future inputs (looks into the future)
                ins = inputs[:,t:,:]
            else:
                # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                ins = inputs[:,t,:].unsqueeze(0).expand(inputs.shape[0]-t+1,inputs.shape[1],inputs.shape[2]).to(inputs.device)
            # print(f"ins shape: {ins.shape}")
            # initialize the LSTM and iterate over the inputs
            _, h_n = self.unrolling_mamba(self.dropout(ins), initial_states=hid.contiguous())

            # append the last hidden state to the list
            predictions.append(h_n)
        
        # obtain the final prediction tensor by concatenating along dimension 1
        x = torch.stack(predictions,1)
        # print(f"x shape: {x.shape}")

        # Step 1: 沿 nhead 求和
        x = x.sum(dim=2)  # [batchsize, seqlen, headdim, dstate]

        # Step 2: 合并 headdim 和 dstate
        batchsize, seqlen, headdim, dstate = x.shape
        x = x.view(batchsize, seqlen, headdim * dstate)  # [batchsize, seqlen, dim]

        # Step 3: 传递给分类器
        y = self.classifier(x)  # [batchsize, seqlen, num_class]
        
        if self.return_context:
            # return y and the concatenation of hidden and cell states 
            c=c.squeeze().permute(1,0,2)
            return y, torch.cat([x, c],2)
        else:
            return y