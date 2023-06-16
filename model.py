import torch 
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as bp
from __init__ import EEG_SHHS_Dataset
from torch.utils.data import  DataLoader
from model_others.Flowformer import FlowformerClassiregressor


INPUT_SIZE = 16 * 60 * 60 * 64 #hours to 64hz


# class DumpRNN(nn.Module):
#     def __init__(self, net):
#         super(DumpRNN, self).__init__()
#         self.net = net

#     def forward(self, x):
#         return self.net(x), None


class DumpRNN(nn.Module):
    def __init__(self, net):
        super(DumpRNN, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x), None

    # returns the attention and output from flowformer
    def forward_with_visualization(self, x):
        return self.net.forward_with_visualization(x)

def get_rnn(Scale,
            Capacity,
            rnn,
            rnn_layers):
    if rnn == 'lstm':
        lstm = nn.LSTM(
            int(Scale * Capacity),
            int(Scale * Capacity // 2),
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
    elif rnn == 'sru':
        from sru import SRU

        lstm = SRU(
            int(Scale * Capacity),
            int(Scale * Capacity // 2),
            num_layers=rnn_layers,
            bidirectional=True,
        )
    elif rnn == 'no':
        lstm = DumpRNN(nn.Identity())

    elif rnn == 'flow':
        

        dim = int(Capacity * Scale)
        flowfromer = FlowformerClassiregressor(
            feat_dim=dim, max_len=1800, d_model=dim, d_ff=dim, num_layers=rnn_layers,
        )
        lstm = DumpRNN(flowfromer)
    else:
        assert False

    return lstm


class EEG_Encoder(nn.Module):
    def __init__(self, arch_scale=0.5, arch_capacity=512, arch_rnn='S.flow.lstm.flow.lstm', arch_rnn_layers=1):
        Scale = arch_scale
        Capacity = arch_capacity
        rnn = arch_rnn
        rnn_layers = arch_rnn_layers
        # print(f'===> EEG Encoder using {rnn} {rnn_layers}...')
        super(EEG_Encoder, self).__init__()

        self.conv1 = nn.Conv1d(1, int(Scale * 64), kernel_size=13, stride=1, padding=6)
        self.block1 = BottleNeck1d(int(Scale * 64), int(Scale * 64), 1, 9, Scale)
        self.block2 = BottleNeck1d(int(Scale * 64), int(Scale * 128), 2, 9, Scale)

        self.block3 = BottleNeck1d(int(Scale * 128), int(Scale * 128), 2, 9, Scale)
        self.block4 = BottleNeck1d(int(Scale * 128), int(Scale * 256), 2, 7, Scale)
        self.block5 = BottleNeck1d(int(Scale * 256), int(Scale * 256), 4, 9, Scale)

        self.block6 = BottleNeck1d(int(Scale * 256), int(Scale * 512), 2, 5, Scale)
        self.block7 = BottleNeck1d(int(Scale * 512), int(Scale * 512), 2, 5, Scale)
        self.block8 = BottleNeck1d(int(Scale * 512), int(Scale * Capacity), 1, 5, Scale)
        self.block9 = BottleNeck1d(int(Scale * Capacity), int(Scale * Capacity), 5, 9, Scale)

        self.BN1 = nn.BatchNorm1d(int(Scale * 256), track_running_stats=False)
        self.BN2 = nn.BatchNorm1d(int(Scale * Capacity), track_running_stats=False)

        self.block10 = BottleNeck1d(int(Scale * Capacity), int(Scale * Capacity), 1, 3, Scale)
        self.block11 = BottleNeck1d(int(Scale * Capacity), int(Scale * Capacity), 1, 5, Scale)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3)

        self.BN3 = nn.BatchNorm1d(int(Scale * Capacity), track_running_stats=False)

        if rnn.startswith('P'):  # parallel rnn
            self.lstm = StackParallel([
                get_rnn(Scale, Capacity, r, rnn_layers)
                for r in rnn.split('.')[1:]
            ])
        elif rnn.startswith('S'):  # sequantial rnn
            self.lstm = StackSequential([
                get_rnn(Scale, Capacity, r, rnn_layers)
                for r in rnn.split('.')[1:]
            ])
        else:
            self.lstm = get_rnn(Scale, Capacity, rnn, rnn_layers)

    def forward(self, x, return_last_att_layer=False):
        att = None
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.block10(x)
        x = self.block11(x)
        x = self.avgpool3(x)

        x = F.relu(self.BN3(x))
        x, atts = self.lstm(x.transpose(1, 2), return_last_att_layer)
        x = x.transpose(1, 2)

        return x, atts

class BottleNeck1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, Scale):
        super(BottleNeck1d, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Scale = Scale

        self.pre_act = nn.BatchNorm1d(in_channels, track_running_stats=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=False)

        if in_channels >= int(Scale * 128) or out_channels >= int(Scale * 128):
            p = 0.05
            if in_channels >= int(Scale * 256) or out_channels >= int(Scale * 256):
                p = 0.1
            if in_channels >= int(Scale * 512) or out_channels >= int(Scale * 512):
                p = 0.25
            self.dropout = nn.Dropout(p=p)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        x = F.relu(self.pre_act(x))
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = F.relu(self.bn(self.conv1(x)))
        if self.in_channels >= int(self.Scale * 128) or self.out_channels >= int(self.Scale * 128):
            x = self.dropout(x)
        x = self.conv2(x)
        x = x + y
        return x


class StackParallel(nn.Module):
    def __init__(self, models):
        super(StackParallel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x)[0] for model in self.models]
        return torch.stack(outputs).sum(0), None


class StackSequential(nn.Module):
    def __init__(self, models):
        super(StackSequential, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, return_last_att_layer):
        atts = []
        # print("x shape 2", x.shape)
        # print("return_last_att_layer", return_last_att_layer)
        for i, model in enumerate(self.models):
            if return_last_att_layer and (i % 2 == 0):
                x, att = model.forward_with_visualization(x)
                atts.append(att)
            else:
                x = model(x)[0]

        return x, atts



class BranchVarEncoder(nn.Module):

    def __init__(self):
        super(BranchVarEncoder, self).__init__()

        num_channel = 256 #1024 #if args.model_type != 'nas2' else 2016
        self.num_var = 1 #args.num_var
        self.num_fold = 2 #args.num_fold
        self.vis_attention = False #args.vis_attention
        self.multi_head_attn = False #args.multi_head_attn

        self.attention_fc1 = Conv_BN(num_channel, 256, 1, 1, 0, False, 1, False)
        self.attention_fc2 = nn.Conv1d(256, self.num_fold, 1, 1, 0, 1, 1, False)
        self.attention_pool = nn.AdaptiveAvgPool1d(1)

        # if args.model_type == 'type34':
        #     self.attention_fc1 = Conv_BN(2048, 512, 1, 1, 0, False, 1, False)
        #     self.attention_fc2 = nn.Conv1d(512, self.num_fold, 1, 1, 0, 1, 1, False)

    def forward(self, x):
        x, _ = x
        attn_mask = None ## custom because we don't need the attention here
        attn_mask = torch.ones([x.size(0), x.size(-1)]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) if attn_mask is None else attn_mask[:, ::12]
        x_att = self.attention_fc1(x)
        x_att_logit = self.attention_fc2(x_att)
        x_att = torch.exp(x_att_logit)
        x_att = x_att * attn_mask
        x_att = x_att / torch.sum(x_att, 2, keepdim=True)
        pred = torch.bmm(x, x_att.transpose(1, 2))
        encoding = pred.view(pred.size(0), -1, 1) if self.multi_head_attn else self.attention_pool(pred)

        # not predict total power when predict variable
        if not self.vis_attention:
            return encoding
        else:
            return encoding, x_att, x_att_logit


class BranchVarPredictor(nn.Module):

    def __init__(self):
        super(BranchVarPredictor, self).__init__()

        num_channel = 256 #1024 #if args.model_type != 'nas2' else 2016
        self.num_var = 2 #args.num_var
        self.multi_head_attn = False #args.multi_head_attn
        self.vis_attention = False #args.vis_attention
        self.var_fc1 = Conv_BN(num_channel * self.num_fold, 256, 1, 1, 0, False) if self.multi_head_attn \
            else Conv_BN(num_channel, 256, 1, 1, 0, False)
        self.var_dropout1 = nn.Dropout(0.5)
        self.var_fc2 = Conv_BN(256, 64, 1, 1, 0, False)
        self.var_dropout2 = nn.Dropout(0.5)
        self.var_fc3 = nn.Conv1d(64, self.num_var, 1, 1, 0)

        self.softmax = nn.Softmax(dim=1)
        # if args.model_type == 'type34':
        #     self.var_fc1 = Conv_BN(2048 * self.num_fold, 512, 1, 1, 0, False) if self.multi_head_attn \
        #         else Conv_BN(2048, 512, 1, 1, 0, False)
        #     self.var_dropout1 = nn.Dropout(0.5)
        #     self.var_fc2 = Conv_BN(512, 128, 1, 1, 0, False)
        #     self.var_dropout2 = nn.Dropout(0.5)
        #     self.var_fc3 = nn.Conv1d(128, self.num_var, 1, 1, 0)

    def forward(self, encoding):
        pred = self.var_dropout1(self.var_fc1(encoding))
        encoding_pd_last = self.var_dropout2(self.var_fc2(pred))
        fc3_output = self.var_fc3(encoding_pd_last)
        # pred = F.sigmoid(self.var_fc3(encoding_pd_last))
        pred = self.softmax(fc3_output)
        pred = pred.squeeze(2)

        if not self.vis_attention:
            return pred
        else:
            return pred, encoding_pd_last



class Conv_BN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_BN=True, groups=1, bias=True):
        super(Conv_BN, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.BN = nn.BatchNorm1d(out_channels)
        self.is_BN = is_BN

    def forward(self, x):
        x = self.conv(x)
        if self.is_BN:
            x = self.BN(x)
        x = F.relu(x)
        return x



# class BottleNeck1d_3(nn.Module):

#     def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, group_num=1):
#         super(BottleNeck1d_3, self).__init__()
#         self.stride = stride
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

#         self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm1d(hidden_channels)

#         self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
#                                padding=(kernel_size - 1) // 2, groups=group_num)
#         self.bn2 = nn.BatchNorm1d(hidden_channels)

#         self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         if self.stride != 1 or self.in_channels != self.out_channels:
#             y = self.shortcut(x)
#         else:
#             y = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))

#         x = F.relu(x + y)
#         return x
    
    
# class EEG_Enc(nn.Module):
#     """ EEG encoder
#     Input shape: B x 1 x (#second * 64) (EEG time)
#     Output shape: B x #channel_enc x (#second / 12) (EEG Spec)
#     Output shape: B x #channel_enc x (#second / 3) (EEG time)
#     """

#     def __init__(self, rnn_layer=1, rnn_drop_between=0.0):
#         super(EEG_Enc, self).__init__()
#         self.block1_0 = BottleNeck1d_3(1, 128, 128, 1, 5)

#         self.block2_0 = BottleNeck1d_3(128, 64, 128, 1, 5)
#         self.block2_1 = BottleNeck1d_3(128, 64, 128, 1, 5)

#         self.block2_3 = BottleNeck1d_3(128, 64, 256, 3, 5)
        
#         self.sru3 = nn.LSTM(256, 128, num_layers=rnn_layer, batch_first=True, bidirectional=True, dropout=rnn_drop_between)
#         # self.sru3 = SRU(256, 128, num_layers=args.rnn_layer, dropout=args.rnn_drop_between,rnn_dropout=args.rnn_drop_in, use_tanh=1, bidirectional=True)
#         self.block_sru3 = BottleNeck1d_3(256, 128, 256, 2, 3)
        
#         self.sru6 = nn.LSTM(256, 128, num_layers=rnn_layer, batch_first=True, bidirectional=True, dropout=rnn_drop_between)
#         # self.sru6 = SRU(256, 128, num_layers=args.rnn_layer, dropout=args.rnn_drop_between,rnn_dropout=args.rnn_drop_in, use_tanh=1, bidirectional=True)
#         self.block_sru6 = BottleNeck1d_3(256, 128, 512, 2, 3)
        
#         self.sru12 = nn.LSTM(512, 256, num_layers= rnn_layer, batch_first=True, bidirectional=True, dropout=rnn_drop_between)
#         # self.sru12 = SRU(512, 256, num_layers=args.rnn_layer, dropout=args.rnn_drop_between,rnn_dropout=args.rnn_drop_in, use_tanh=1, bidirectional=True)
    
#     def forward(self, x):
#         x = self.block1_0.forward(x)

#         x = self.block2_0.forward(x)
#         x = self.block2_1.forward(x)
#         x = self.block2_3.forward(x)  # 256

#         x = x.transpose(0, 1)
#         x, _ = self.sru3(x.transpose(0, 2))
#         x = x.transpose(0, 2)
#         x = x.transpose(0, 1).contiguous()  # 512
#         x = self.block_sru3.forward(x)

#         x = x.transpose(0, 1)
#         x, _ = self.sru6(x.transpose(0, 2))
#         x = x.transpose(0, 2)
#         x = x.transpose(0, 1).contiguous()  # 1024
#         x = self.block_sru6(x)

#         x = x.transpose(0, 1)
#         x, _ = self.sru12(x.transpose(0, 2))  # 1024
#         x = x.transpose(0, 2)
#         x = x.transpose(0, 1).contiguous()
#         return x
    
if __name__ == '__main__':
    dataset = EEG_SHHS_Dataset()
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    eeg_encoder = EEG_Encoder().cuda()
    pd_encoder = BranchVarEncoder().cuda()
    pd_predictor = BranchVarPredictor().cuda()
    
    model = nn.Sequential(eeg_encoder, pd_encoder, pd_predictor)

    model = nn.DataParallel(model)
    
    aa = iter(dataloader)
    
    
    
    data, label = next(aa)
    
    
    data = data.cuda()
    
    output = model(data)
    # eeg_encode_output = eeg_encoder(data)
    # pd_encode_output = pd_encoder(eeg_encode_output)
    # output = pd_predictor(pd_encode_output)
    
    bp()


