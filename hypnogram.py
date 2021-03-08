# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:25:54 2019

@author: PathakS
"""

import models
import torch
import numpy as np
import math
import utils
import pickle
from collections import OrderedDict
#import matplotlib.pyplot as plt
from visbrain.io import write_fig_hyp
from torch.utils.data import DataLoader, SequentialSampler
#from matplotlib import colors as mcolors

#Initialization
main_channels=3
batch_size=128
time_steps=3750
n_channels=5
seq_length=8
n_workers=3
best_hyp_sim=5
worst_hyp_sim=-1
best_predict_label=[]
best_true_label=[]
worst_predict_label=[]
worst_true_label=[]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def hypnogram(data_iterator_Test,path_to_trained_model):
    i=0
    tot_hyp_sim=0
    global best_predict_label
    global best_true_label
    global worst_predict_label
    global worst_true_label
    
    for idx, data, label in data_iterator_Test:
        output_class=predict(path_to_trained_model, data)
        label_crop=label[int(seq_length/2)-1:len(label)-int(seq_length/2)]
        tot_hyp_sim=measure_hypnograms(label_crop.to(device), output_class.to(device), tot_hyp_sim)
        print("i:",i)
        i=i+1
    avg_hyp_sim=tot_hyp_sim/i
    print("avg hyp sim on test set", avg_hyp_sim)
    print("best hyp sim:", best_hyp_sim)
    print("worst hyp sim", worst_hyp_sim)
    
    #Predicted Best Hypnogram
    pred_data_plot=np.repeat(best_predict_label.cpu().numpy(),30)
    write_fig_hyp(pred_data_plot, sf=1, title='Predicted Hypnogram', file="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/hypnograms/SHHS_cnnlstm_allchunk_oversampling_hyp_predicted_best.pdf", ascolor=True)
    
    #True Best Hypnogram
    true_data_plot=np.repeat(best_true_label.cpu().numpy(),30)
    write_fig_hyp(true_data_plot, sf=1, title='True Hypnogram', file="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/hypnograms/SHHS_cnnlstm_allchunk_oversampling_hyp_true_best.pdf", ascolor=True)
    
    #Predicted Worst Hypnogram
    pred_data_plot=np.repeat(worst_predict_label.cpu().numpy(),30)
    write_fig_hyp(pred_data_plot, sf=1, title='Predicted Hypnogram', file="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/hypnograms/SHHS_cnnlstm_allchunk_oversampling_hyp_predicted_worst.pdf", ascolor=True)
    
    #True Worst Hypnogram
    true_data_plot=np.repeat(worst_true_label.cpu().numpy(),30)
    write_fig_hyp(true_data_plot, sf=1, title='True Hypnogram', file="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/hypnograms/SHHS_cnnlstm_allchunk_oversampling_hyp_true_worst.pdf", ascolor=True)

def measure_hypnograms(true, predicted, tot_hyp_sim):
    global best_hyp_sim
    global worst_hyp_sim
    global best_predict_label
    global best_true_label
    global worst_predict_label
    global worst_true_label
    
    hyp_sim=float(sum(abs(true-predicted)).item())/len(true)
    
    if hyp_sim<best_hyp_sim:
        best_hyp_sim=hyp_sim
        best_predict_label=predicted
        best_true_label=true
    elif hyp_sim>worst_hyp_sim:
        worst_hyp_sim=hyp_sim
        worst_predict_label=predicted
        worst_true_label=true
    tot_hyp_sim+=hyp_sim
    print("hyp sim", hyp_sim)
    return tot_hyp_sim

def data_normalizer(allData_30secs):
    data_shape=allData_30secs.shape
    x=np.empty(data_shape)
    m=0
    for data in allData_30secs:
        data_ch_norm_list=[]
        for ch in range(data.shape[1]):
            data_ch=np.array(data[:,ch])
            data_ch_mean=(np.mean(data_ch,axis=1)).reshape(-1,1)
            data_ch_std=(np.std(data_ch,axis=1)).reshape(-1,1)
            if math.isnan(data_ch_std) or not data_ch_std:
                data_ch_norm=(data_ch-data_ch_mean)
            else:
                data_ch_norm=(data_ch-data_ch_mean)/data_ch_std
            data_ch_norm_list.append(data_ch_norm[0])
        x[m]=np.array([data_ch_norm_list])
        m=m+1
    x=torch.from_numpy(x).float()
    return x

def load_model(path):
    model_parameters=torch.load(path, map_location=device)
    model_weights=model_parameters['state_dict']
    model=models.DeepSleepSpatialTemporalNet(main_channels, False, False, False)
    state_dict_remove_module = OrderedDict()
    for k, v in model_weights.items():
        if k[:7]=='module.':
            if device=="cuda:0":
                state_dict_remove_module[k] = v
            else:
                name = k[7:] # remove `module.`
                state_dict_remove_module[name] = v
        else:
            if device=="cuda:0":
                name = 'module.' + k
                state_dict_remove_module[name] = v
            else:
                state_dict_remove_module[k] = v
            
    model.load_state_dict(state_dict_remove_module)
    return model.to(device)

def predict(path,data):
    model=load_model(path)
    model.eval()
    data=data_normalizer(data).to(device)
    hidden_state=torch.zeros(2,1,20).to(device)
    cell_state=torch.zeros(2,1,20).to(device)
    initial_hidden_states=(hidden_state,cell_state)
    combined_input = [data,initial_hidden_states]
    output, hidden_state_last = model(combined_input)
    predict_class=output.argmax(dim=1, keepdim=True).view(-1).cpu()
    return predict_class

if __name__ == '__main__': 
    #data should be in [batch size, 1, rows, columns] format, where rows= 5 (5 signals - eeg1,eeg2,eogL,eogR, emg) 
    #and columns=3750 (125*30secs) for this model. Batch size means number of inout 30 secs signal chunks
    #batch size can be 1 or more
    #data should be in torch.tensor data type.
    
    model_name=input('Enter the name of the model (without the extension .tar):')
    path_to_trained_model="D:/DEEPSLEEP/test_for_github/saved_models/shhs/"+model_name+".tar"
    
    #Single test file
    #path_to_hdf5_file_test='D:/DEEPSLEEP/codes and results/SHHS data codes/hdf5_file_test_eeg_annotation_shhs1-203371.hdf5'
    #path_to_file_length_test='D:/DEEPSLEEP/codes and results/SHHS data codes/test30secChunk_eeg_annotation_shhs1-203371.pkl'
    
    #Multiple patient data
    path_to_hdf5_file_test='D:/DEEPSLEEP/test_for_github/datasets/shhs/test/hdf5_file_test_all_chunking_shhs1.hdf5'
    path_to_file_length_test='D:/DEEPSLEEP/test_for_github/datasets/shhs/test/testFilesNum30secEpochs_all_shhs1.pkl'
    
    f_file_length_test=open(path_to_file_length_test,'rb')
    file_length_dic_test=pickle.load(f_file_length_test)
    
    print("start generator test")
    data_gen_test=utils.my_generator1(path_to_hdf5_file_test)
    print("start dataloader test")
    
    sampler_test=SequentialSampler(data_gen_test)
    data_iterator_Test=DataLoader(data_gen_test,batch_size=1,num_workers=n_workers,batch_sampler=utils.CustomSequentialLSTMBatchSampler_ReturnAllChunks(sampler_test,0,file_length_dic_test,seq_length))
    
    hypnogram(data_iterator_Test,path_to_trained_model)