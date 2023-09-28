import torch 
import torch.nn as nn
from torch.nn import functional as F
from ipdb import set_trace as bp
# from BrEEG.task_spec import ResidualBlock1D
from dataset import EEG_Encoding_SHHS2_Dataset
from torch.utils.data import  DataLoader
import math 
import argparse

def get_rnn(Scale, Capacity, rnn, rnn_layers):
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

'''
EEG_Encoder Model
'''
class EEG_Encoder(nn.Module):
    '''
    Intialize EEG_Encoder
    '''
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





if __name__ == '__main__':
    print('models')
