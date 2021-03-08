# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:03:38 2021

@author: PathakS
"""

import torch
import utils
import torch.nn as nn

#Initialization
time_period_sample=3750
channel_dic={'0':'eeg','1':'eog','2':'emg'}
channel_count={'eeg':2,'eog':2,'emg':1}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSTM_ReturnAllChunk_residual_connection(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True, seq_length=8):
        super(LSTM_ReturnAllChunk_residual_connection, self).__init__()
        #spatial filtering
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=bidirectional)
        self.residual_connection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim*2),
            nn.BatchNorm1d(self.hidden_dim*2),
            nn.ReLU()
        )
        self.apply(self.init_weights)
    
    def init_weights(self,x):
        if isinstance(x, nn.LSTM):
            for param in x.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        if type(x) == nn.Linear or type(x) == nn.BatchNorm1d:
            x.weight.data.normal_(0, 0.02) 
            x.bias.data.fill_(0.01)
    
    def forward(self, x):
        input_data, hidden_state = x
        x_seq=utils.to_sequence_LSTM_ReturnAllChunk(input_data,self.seq_length)
        bs=x_seq.shape[0]
        seq_size=x_seq.shape[1]
        lstm_out=torch.zeros((bs,self.seq_length,self.hidden_dim*2)).to(device)
        self.lstm.flatten_parameters()
        for elem in range(bs):
            #print(hidden_state.shape)
            lstm_out1, hidden_state = self.lstm(x_seq[elem,:,:].view(1,self.seq_length,-1),hidden_state)
            #print(lstm_out1.shape)
            lstm_out[elem]=lstm_out1[0]
            #print(lstm_out.shape)
        
        print("lstm shape 1:",lstm_out.shape)
        out = lstm_out[:,:,:].contiguous().view(bs*seq_size,-1)  
        out_res=self.residual_connection(input_data)
        out=out+out_res
        print("final in lstm:",out.shape)        
        return out, hidden_state

class LSTM_ReturnAllChunk(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True, seq_length=8):
        super(LSTM_ReturnAllChunk, self).__init__()
        #spatial filtering
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=bidirectional)
        self.apply(self.init_weights)
    
    def init_weights(self,x):
        if isinstance(x, nn.LSTM):
            for param in x.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
    
    def forward(self, x):
        input_data, hidden_state = x
        x_seq=utils.to_sequence_LSTM_ReturnAllChunk(input_data,self.seq_length)
        bs=x_seq.shape[0]
        seq_size=x_seq.shape[1]
        lstm_out=torch.zeros((bs,self.seq_length,self.hidden_dim*2)).to(device)
        #print(bs)
        #print(x_seq.shape)
        self.lstm.flatten_parameters()
        for elem in range(bs):
            #print(hidden_state.shape)
            lstm_out1, hidden_state = self.lstm(x_seq[elem,:,:].view(1,self.seq_length,-1),hidden_state)
            #print(lstm_out1.shape)
            lstm_out[elem]=lstm_out1[0]
            #print(lstm_out.shape)
        
        print("lstm shape 1:",lstm_out.shape)
        out = lstm_out[:,:,:].contiguous().view(bs*seq_size,-1)  
        print("final in lstm:",out.shape)        
        return out, hidden_state

class ChannelFeaturesBlock(nn.Module):
    def __init__(self, subchannel):
        super(ChannelFeaturesBlock, self).__init__()
        #spatial filtering
        self.sub_channel=subchannel
        if subchannel==1:
            self.layer2=nn.Sequential(
                nn.Conv1d(1,8,64,groups=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(16)
            )
            self.layer3=nn.Sequential(
                nn.Conv1d(8,8,64,groups=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(16)
            )
        else:
            self.layer1=nn.Sequential(
                nn.Conv2d(1,subchannel,(subchannel,1),groups=1),
                nn.BatchNorm2d(subchannel),
                nn.ReLU()
            )
            #Two blocks of temporal filtering: layer2 and layer 3
            self.layer2=nn.Sequential(
                nn.Conv2d(1,8,(1,64),groups=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d((1,16))
            )
            self.layer3=nn.Sequential(
                nn.Conv2d(8,8,(1,64),groups=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d((1,16))
            )
        self.layer4=nn.Dropout(p=0.5)
        self.apply(self.init_weights)

    def init_weights(self,x):
        if type(x) == nn.Conv2d or type(x) == nn.Linear or type(x) == nn.BatchNorm2d or type(x) == nn.Conv1d or type(x) == nn.BatchNorm1d:
            x.weight.data.normal_(0, 0.02) 
            x.bias.data.fill_(0.01)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        if self.sub_channel!=1:
            x=self.layer1(x)
            x=x.view(-1,1,self.sub_channel,3750) #batch_size,no.of channels=1,no.of rows=2,timesteps=3750 (CNN 2d (x=no.of rows=the no.of channels in psg,y=timesteps))
        x=self.layer2(x)
        x=self.layer3(x)
        x = x.view(-1, self.num_flat_features(x))
        x=self.layer4(x)
        return x
        
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        in_dimen=time_period_sample*n_channels
        out_dimen1=10000
        out_dimen2=5000
        out_dimen3=1000
        out_dimen4=5
        
        self.linear1=nn.Linear(in_dimen,out_dimen1)
        self.batchNorm1=nn.BatchNorm1d(out_dimen1)
        self.relu1=nn.ReLU()
        
        self.linear2=nn.Linear(out_dimen1,out_dimen2)
        self.batchNorm2=nn.BatchNorm1d(out_dimen2)
        self.relu2=nn.ReLU()
        
        self.linear3=nn.Linear(out_dimen2,out_dimen3)
        self.batchNorm3=nn.BatchNorm1d(out_dimen3)
        self.relu3=nn.ReLU()
        
        self.linear4=nn.Linear(out_dimen3,out_dimen4)
        
        self.apply(self.init_weights)
        
    def init_weights(self,x):
        if type(x) == nn.Linear or type(x) == nn.BatchNorm1d:
            x.weight.data.normal_(0, 0.02) 
            x.bias.data.fill_(0.01)
    
    def forward(self, x):
        if x.shape[0]==1:
            x=self.linear1(x)
            x=self.relu1(x)
            x=self.linear2(x)
            x=self.relu2(x)
            x=self.linear3(x)
            x=self.relu3(x)
        else:
            x=self.linear1(x)
            x=self.batchNorm1(x)
            x=self.relu1(x)
            x=self.linear2(x)
            x=self.batchNorm2(x)
            x=self.relu2(x)
            x=self.linear3(x)
            x=self.batchNorm3(x)
            x=self.relu3(x)
        
        x=self.linear4(x)
        return x
    
class DeepSleepSpatialTemporalNet(nn.Module):
    def __init__(self, mainchannels, lstm_option, rc_option, multiple_gpu):
        super(DeepSleepSpatialTemporalNet, self).__init__()
        self.channelBlocks=nn.ModuleDict()
        self.multiple_gpu=multiple_gpu
        self.lstm_option=lstm_option
        self.rc_option=rc_option
        in_dimen_after_cnn=400 #have to calculate this that what is the in_dimen of linear layer
        out_dimen=5
        for ch_no in range(mainchannels):
            self.channelBlocks[channel_dic[str(ch_no)]]=ChannelFeaturesBlock(channel_count[channel_dic[str(ch_no)]])#.to(device)
        if self.lstm_option==True:
            hidden_dimen=20
            in_dimen=hidden_dimen*2 #biLSTM, i.e. hidden dimen*2
            if self.rc_option==True:
                self.lstm = LSTM_ReturnAllChunk_residual_connection(input_dim=in_dimen_after_cnn, hidden_dim=hidden_dimen)
            else:
                self.lstm = LSTM_ReturnAllChunk(input_dim=in_dimen_after_cnn, hidden_dim=hidden_dimen)         
        else:
            in_dimen=in_dimen_after_cnn #have to calculate this that what is the in_dimen of linear layer
        self.linear=nn.Linear(in_dimen,out_dimen)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.02)
        self.linear.bias.data.fill_(0.01)
    
    def input_splitter(self,x):
        eeg_input=x[:,:,0:2,:]
        eog_input=x[:,:,2:4,:]
        emg_input=x[:,:,4:5,:].view(-1,1,time_period_sample)
        return eeg_input,eog_input,emg_input
    
    def forward(self,x):
        if self.lstm_option==True:
            input_data, hidden_state=x
        else:
            input_data=x
        if self.multiple_gpu==True and self.lstm_option==True:
            input_data=input_data.view(input_data.shape[0]*input_data.shape[1],input_data.shape[2],input_data.shape[3],input_data.shape[4])
        print("In the model:",input_data.shape[0])
        eeg_channel,eog_channel,emg_channel=self.input_splitter(input_data)
        out_eeg=self.channelBlocks['eeg'](eeg_channel)
        out_eog=self.channelBlocks['eog'](eog_channel)
        out_emg=self.channelBlocks['emg'](emg_channel)
        out_allchannels=torch.cat((out_eeg,out_eog,out_emg),1)
        #print(out_allchannels.shape) #Print this the first time for a new architecture to know the dimension of linear layer
        if self.lstm_option==True:
            out_allchannels, hidden_state=self.lstm([out_allchannels,hidden_state])
            hidden_state=(hidden_state[0].detach(),hidden_state[1].detach())  
        out=self.linear(out_allchannels)
        if self.lstm_option==True:
            return out, hidden_state
        else:
            return out

class DeepSleepDNNNet(nn.Module):
    def __init__(self, mainchannels, lstm_option, rc_option, multiple_gpu):
        super(DeepSleepDNNNet, self).__init__()
        self.multiple_gpu=multiple_gpu
        self.lstm_option=lstm_option
        self.dnn=DNN()
    
    def input_splitter_concat(self,x):
        print("x",x.shape)
        input_concat=torch.cat((x[:,:,0,:],x[:,:,1,:],x[:,:,2,:],x[:,:,3,:],x[:,:,4,:].view(-1,1,time_period_sample)),2)
        print("concat",input_concat.shape)
        input_concat=torch.squeeze(input_concat,1)
        print("concat",input_concat.shape)
        return input_concat
    
    def forward(self,x):
        if self.multiple_gpu==True and self.lstm_option==True:
            x=x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        print("In the model:",x.shape[0])
        x=self.input_splitter_concat(x)
        x=self.dnn(x)
        return x