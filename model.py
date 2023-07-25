import torch 
import torch.nn as nn
from torch.nn import functional as F
from ipdb import set_trace as bp
from dataset import EEG_SHHS_Dataset
from torch.utils.data import  DataLoader
from model_others.Flowformer import FlowformerClassiregressor

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

def fc_block(in_channels: int, out_channels: int, is_bn=True, dropout=0.5) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout),
    ) if is_bn else nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

class DumpRNN(nn.Module):
    def __init__(self, net):
        super(DumpRNN, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x), None

    # returns the attention and output from flowformer
    def forward_with_visualization(self, x):
        return self.net.forward_with_visualization(x)

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
    
class StagePredictorModel(nn.Module):
    def __init__(self, arch_scale=0.5, arch_capacity=512, input_dim = None):
        Scale = arch_scale
        Capacity = arch_capacity
        num_classes = 5
        # if getattr(args, 'dual_level', False):
        #     num_classes = 6  # [A,R,L,D] + [N1,N2]

        super(StagePredictorModel, self).__init__()
        self.BN0 = nn.BatchNorm1d(int(Scale * Capacity), track_running_stats=False)
        self.BN1 = nn.BatchNorm1d(int(Scale * Capacity * 2), track_running_stats=False)
        self.BN2 = nn.BatchNorm1d(int(Scale * Capacity * 2), track_running_stats=False)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)

        if input_dim == None:
            input_dim = int(Scale * Capacity)

        self.fc1 = nn.Conv1d(input_dim, int(Scale * Capacity * 2), kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv1d(int(Scale * Capacity * 2), int(Scale * Capacity * 2), kernel_size=1, stride=1, padding=0)
        self.fc3 = nn.Conv1d(int(Scale * Capacity * 2), num_classes, kernel_size=1, stride=1, padding=0)

        self.Scale = Scale
        self.Capacity = Capacity
        self.num_classes = num_classes

    def forward(self, x):
        x, _ = x
        x = self.fc1(x)
        x = self.dropout1(F.relu(self.BN1(x)))
        x = self.fc2(x)
        x = self.dropout2(F.relu(self.BN2(x)))
        x = self.fc3(x)
        return x

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

    def __init__(self, args):
        super(BranchVarEncoder, self).__init__()

        num_channel = args.num_channel #256 #1024 #if args.model_type != 'nas2' else 2016
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
        if len(x) == 2:
            x, _ = x
        elif len(x) == 4:
            x, _, _, _ = x
        
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

    def __init__(self, args):
        super(BranchVarPredictor, self).__init__()

        num_channel = args.num_channel #256 #1024 #if args.model_type != 'nas2' else 2016
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
 
class BranchHYPredictor(nn.Module):

    # EncodingWidth, Capacity = 256, 512

    def __init__(self, args):
        super(BranchHYPredictor, self).__init__()
        self.EncodingWidth = args.num_channel 
        self.Capacity = 512 
        
        self.vis_attention = False #args.vis_attention
        # self.hy_type = args.hy_type
        self.output_dim = 2 #args.hy_outdim if args.hy_type == 'cls' else args.hy_outdim - 1
        self.fc1 = fc_block(self.EncodingWidth, self.Capacity, is_bn=False)
        self.fc2 = fc_block(self.Capacity, self.Capacity, is_bn=False)
        self.fc3 = fc_block(self.Capacity, self.Capacity, is_bn=False)
        self.fc_final = nn.Linear(self.Capacity, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        encoding = data.view(data.size(0), -1)
        pred = self.fc1(encoding)
        pred = self.fc2(pred)
        encoding_hy_last = self.fc3(pred)
        pred = self.relu(self.fc_final(encoding_hy_last)) #i added the relu #F.sigmoid(self.fc_final(encoding_hy_last)) if self.hy_type.startswith('ordinal') else \
        if not self.vis_attention:
            return pred
        else:
            return pred, encoding_hy_last

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

class BottleNeck1d_IRLAS_3(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, group_num=1):
        super(BottleNeck1d_IRLAS_3, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.comb_channels = out_channels // 3

        self.conv_left = nn.Conv1d(in_channels, self.comb_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_left = nn.BatchNorm1d(self.comb_channels)

        self.conv_mid_3x3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_mid_1x1 = nn.Conv1d(in_channels, self.comb_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_mid_1 = nn.BatchNorm1d(in_channels)
        self.bn_mid_2 = nn.BatchNorm1d(self.comb_channels)

        self.conv_right = nn.Conv1d(in_channels, self.comb_channels, kernel_size=5, stride=stride, padding=2)
        self.bn_right = nn.BatchNorm1d(self.comb_channels)

    def forward(self, x):
        x_left = F.relu(self.bn_left(self.conv_left(x)))

        x_mid = self.bn_mid_1(self.conv_mid_3x3(x))
        x_mid = F.relu(x_mid + x)
        x_mid = F.relu(self.bn_mid_2(self.conv_mid_1x1(x_mid)))

        x_right = F.relu(self.bn_right(self.conv_right(x)))

        x = torch.cat([x_left, x_mid, x_right], 1)
        return x

class BBEncoder(nn.Module):
    """ BB Encoder
    Input shape: B x 3 x (#second * 10)
    Output shape: B x #channel_enc x (#second / 12)
    """

    def __init__(self):
        super(BBEncoder, self).__init__()

        # self.model_type = args.model_type
        # condense model when memory usage is limited
        # self.updrs_consistency = args.updrs_consistency
        self.conv0 = nn.Conv1d(3, 64, kernel_size=11, stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64, eps=0.001, momentum=0.1, affine=True)
        capacity = 84
        self.block1_0 = BottleNeck1d_IRLAS_3(64, capacity*3, 1, 5)
        self.block1_1 = BottleNeck1d_IRLAS_3(capacity*3, capacity*3, 1, 5)
        self.block1_2 = BottleNeck1d_IRLAS_3(capacity*3, capacity*3, 1, 5)
        self.block1_3 = BottleNeck1d_IRLAS_3(capacity*3, capacity*6, 2, 5)

        self.block2_0 = BottleNeck1d_IRLAS_3(capacity*6, capacity*6, 1, 5)
        self.block2_1 = BottleNeck1d_IRLAS_3(capacity*6, capacity*6, 1, 5)
        self.block2_2 = BottleNeck1d_IRLAS_3(capacity*6, capacity*6, 1, 5)
        self.block2_3 = BottleNeck1d_IRLAS_3(capacity*6, capacity*24, 3, 5)
        # self.sru3 = SRU(capacity*24, capacity*12, num_layers=args.rnn_layer, dropout=args.rnn_drop_between,
        #                 rnn_dropout=args.rnn_drop_in, use_tanh=1, bidirectional=True)
        rnn_layer = 1
        rnn_drop_between = 0.0
        
        self.sru3 = nn.LSTM(capacity*24, capacity*12, num_layers=rnn_layer, batch_first=True, bidirectional=True, dropout=rnn_drop_between)
        self.block_sru3 = BottleNeck1d_IRLAS_3(capacity*24, capacity*24, 2, 3)
        # self.sru6 = SRU(capacity*24, capacity*12, num_layers=args.rnn_layer, dropout=args.rnn_drop_between,
        #                 rnn_dropout=args.rnn_drop_in, use_tanh=1, bidirectional=True)
        self.sru6 = nn.LSTM(capacity*24, capacity*12, num_layers=rnn_layer, batch_first=True, bidirectional=True, dropout=rnn_drop_between)
        self.block_sru6 = BottleNeck1d_IRLAS_3(capacity*24, capacity*24, 2, 3)
        self.sru12 = nn.LSTM(capacity*24, capacity*12, num_layers=rnn_layer, batch_first=True, bidirectional=True, dropout=rnn_drop_between)
        # self.sru12 = SRU(capacity*24, capacity*12, num_layers=args.rnn_layer, dropout=args.rnn_drop_between,
        #                     rnn_dropout=args.rnn_drop_in, use_tanh=1, bidirectional=True)
        # lstm = nn.LSTM(
        #     int(Scale * Capacity),
        #     int(Scale * Capacity // 2),
        #     num_layers=rnn_layers,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        
    def forward(self, x):
        # only bb as input

        x = self.conv0(x.repeat(1,3,1))

        x = self.block1_0.forward(x)
        x = self.block1_1.forward(x)
        x = self.block1_2.forward(x)
        x = self.block1_3.forward(x)

        x = self.block2_0.forward(x)
        x = self.block2_1.forward(x) # if self.updrs_consistency != 'triplet' else x
        skip1 = self.block2_2.forward(x)  # 256
        x = self.block2_3.forward(skip1)

        x = x.transpose(0, 1)
        x, _ = self.sru3(x.transpose(0, 2))
        x = x.transpose(0, 2)
        skip2 = x.transpose(0, 1).contiguous()  # 512
        x = self.block_sru3.forward(skip2)

        x = x.transpose(0, 1)
        x, _ = self.sru6(x.transpose(0, 2))
        x = x.transpose(0, 2)
        skip3 = x.transpose(0, 1).contiguous()  # 1024
        x = self.block_sru6(skip3)

        x = x.transpose(0, 1)
        x, _ = self.sru12(x.transpose(0, 2))  # 1024
        x = x.transpose(0, 2)
        x = x.transpose(0, 1).contiguous()

        return x, skip1, skip2, skip3

class SimplePredictor(nn.Module):
    def __init__(self, layer_dims=[256,64,16], encoding_width=768, output_dim=2):
        super(SimplePredictor, self).__init__() # so this subclass can inhert nn.Module functionality
        self.output_dim = output_dim
        self.fc1 = fc_block(encoding_width, layer_dims[0], is_bn=False)
        self.fc2 = fc_block(layer_dims[0], layer_dims[1], is_bn=False)
        self.fc3 = fc_block(layer_dims[1], layer_dims[2], is_bn=False)
        #self.fc4 = fc_block(layer_dims[2], layer_dims[3], is_bn=False)
        self.fc_final = nn.Linear(layer_dims[2], output_dim)


    def forward(self, x):
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #x = self.fc4(x)
        pred = self.fc_final(x)
        if(self.output_dim==1):
            pred = pred.squeeze(1)
        return pred

class SimpleAttentionPredictor(nn.Module):
    def __init__(self, args, encoding_width=768):
        super(SimpleAttentionPredictor, self).__init__()
        self.num_classes = args.num_classes
        bns = args.batch_norms
        layer_dims = args.layer_dims
        self.attention = AttentionCondensation(input_size=encoding_width, hidden_size=layer_dims[0], num_heads=args.num_heads)
        #self.fc1 = fc_block(layer_dims[0]*args.num_heads, layer_dims[0], is_bn=bns[0], dropout=args.dropout)
        #self.fc2 = fc_block(layer_dims[0], layer_dims[1], is_bn=bns[1], dropout=args.dropout)
        self.fc2 = fc_block(layer_dims[0]*args.num_heads, layer_dims[1], is_bn=bns[1], dropout=args.dropout) # rename to fc2
        self.fc3 = fc_block(layer_dims[1], layer_dims[2], is_bn=bns[2], dropout=args.dropout) # rename to fc3
        self.fc_final = nn.Linear(layer_dims[2], self.num_classes)

    def forward(self, x):
        #bp()
        x = self.attention(x)
        #x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        pred = self.fc_final(x)
        if(self.num_classes == 1):
            pred = pred.squeeze(1)
        return pred

class AttentionCondensation(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(AttentionCondensation, self).__init__()
        self.value_layer = nn.ModuleList([nn.Linear(input_size, hidden_size) for i in range(num_heads)])
        self.query_layer = nn.ModuleList([nn.Linear(input_size, 1) for i in range(num_heads)])
        self.softmax = nn.Softmax(dim=1)
        self.num_heads = num_heads
        
    def forward(self, x):
        query = [self.softmax(self.query_layer[i](x)) for i in range(self.num_heads)]
        values = [self.value_layer[i](x) for i in range(self.num_heads)]
        attended_output = [torch.matmul(query[i].permute(0, 2, 1), values[i]).permute(0,2,1).squeeze(2) for i in range(self.num_heads)]
        total_output = torch.cat(attended_output, dim=1)
        return total_output
    
class SimonModel(nn.Module):
    def __init__(self, args):
        super(SimonModel, self).__init__()
        self.initial_fc_size = args.num_heads * args.hidden_size if args.num_heads > 0 else args.hidden_size
        self.num_heads = args.num_heads
        
        if args.num_heads > 0:
            self.encoder = AttentionCondensation(768, args.hidden_size, args.num_heads)
        else:
            self.encoder = nn.Sequential(nn.Linear(768, self.initial_fc_size), nn.LayerNorm(self.initial_fc_size))
            
        
        
        self.fc1 = nn.Sequential(nn.Linear(self.initial_fc_size, args.hidden_size), nn.LayerNorm(args.hidden_size))
        self.fc2 = nn.Sequential(nn.Linear(args.hidden_size, args.fc2_size), nn.LayerNorm(args.fc2_size))
        self.fc3 = nn.Linear(args.fc2_size, args.num_classes)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)
        
    def forward(self, x):
        if self.num_heads > 0:
            return self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.encoder(x))))))
        else:
            return self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.relu(self.encoder(torch.mean(x,axis=1))))))))

# class AttentionCondensation(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(AttentionCondensation, self).__init__()
#         self.value_layer = nn.Linear(input_size, hidden_size)
#         self.query_layer = nn.Linear(input_size, 1)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         query = self.softmax(self.query_layer(x))
#         values = self.value_layer(x)
#         #bp()
#         attended_output = torch.matmul(query.permute(0, 2, 1), values)
#         return attended_output.permute(0,2,1).squeeze(2)

# class SimpleRegressor(nn.Module):
#     def __init__(self, layer_dims=[256,64,16], encoding_width=768):
#         super(SimpleRegressor, self).__init__() 
#         self.fc1 = fc_block(encoding_width, layer_dims[0], is_bn=False)
#         self.fc2 = fc_block(layer_dims[0], layer_dims[1], is_bn=False)
#         self.fc3 = fc_block(layer_dims[1], layer_dims[2], is_bn=False)
#         self.fc_final = nn.Linear(layer_dims[2], 1)


#     def forward(self, x):
#         #print(x.shape)
#         x = self.fc1(x)
#         #print(x.shape)
#         x = self.fc2(x)
#         #print(x.shape)
#         x = self.fc3(x)
#         pred = self.fc_final(x)
#         return pred

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
