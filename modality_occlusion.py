# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:33:07 2020

@author: PathakS
"""

import os

import torch
import math
import utils
import pickle
import numpy as np
import openpyxl as op
from openpyxl import Workbook
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
#from visbrain.io import write_fig_hyp
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix

import models

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
    output=model(data.to(device))
    predict_class=output.argmax(dim=1, keepdim=True)
    return predict_class

def modality_occlusion(data_iterator_test,batches_test,modality_to_occlude):
    conf_mat=np.zeros((5,5))
    batch_no=0
    for idx, data, label in data_iterator_test:
        batch_no=batch_no+1
        print(str(batch_no)+'/'+str(batches_test))
        temp=data.clone().detach()
        if modality_to_occlude=='EEG':
            temp[:,0,0,:]=data[:,0,0,:]*0 #EEG
            temp[:,0,1,:]=temp[:,0,1,:]*0 #EEG
        elif modality_to_occlude=='EOG':
            temp[:,0,2,:]=data[:,0,2,:]*0 #EOG
            temp[:,0,3,:]=temp[:,0,3,:]*0 #EOG
        elif modality_to_occlude=='EMG':
            temp[:,0,4,:]=data[:,0,4,:]*0 #EMG
        elif modality_to_occlude=='All':
            temp[:,0,0,:]=data[:,0,0,:]*0 #EEG
            temp[:,0,1,:]=temp[:,0,1,:]*0 #EEG
            temp[:,0,2,:]=temp[:,0,2,:]*0 #EOG
            temp[:,0,3,:]=temp[:,0,3,:]*0 #EOG
            temp[:,0,4,:]=temp[:,0,4,:]*0 #EMG
        
        output_class=predict(path_to_trained_model,temp)
        conf_mat=confusion_matrix_channel_occlusion(label,output_class.view(-1),conf_mat)            
        #print(conf_mat)
        
    sheet.append([modality_to_occlude])
    for row in conf_mat.tolist():
        sheet.append(row)
    fig_filename='./channel occlusion/'+'conf_mat_'+modality_to_occlude+"_occlusion.pdf"
    utils.confusion_matrix_norm_func(conf_mat,fig_name=fig_filename)
    
    metrics_class,metrics_model=utils.metrics_confusion_matrix_per_class(conf_mat) 
    sheet.append(['Class','Prec','Rec','F1'])
    for i,res in enumerate(metrics_class):
        sheet.append([classes_name[i]]+res)
    sheet.append(['Acc','Bal_Acc','Macro_F1','Cohens Kappa','Wt_Macro_F1'])
    sheet.append(metrics_model)
    print(metrics_class)
    print(metrics_model)
    print(conf_mat)    
    
def confusion_matrix_channel_occlusion(true,pred,conf_mat):
    true=true.cpu()
    pred=pred.cpu()
    conf_mat=conf_mat+confusion_matrix(true, pred, [0,1,2,3,4])
    return conf_mat

if __name__ == '__main__':
    #data should be in [batch size, 1, rows, columns] format, where rows= 5 (5 signals - eeg1,eeg2,eogL,eogR, emg) 
    #and columns=3750 (125*30secs) for this model. Batch size means number of inout 30 secs signal chunks
    #batch size can be 1 or more
    #data should be in torch.tensor data type.
    batch_size=1000
    time_steps=3750
    n_channels=5
    seq_length=8
    workers=3
    main_channels=3
    classes_name=['W','N1','N2','N3','REM']
    modality_list=['EEG','EOG','EMG','All']
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    ##trained model cnn
    model_name=input('Enter the name of the model (without the extension .tar):')
    path_to_trained_model="D:/DEEPSLEEP/test_for_github/saved_models/shhs/"+model_name+".tar"
        
    ##data for whole test set
    path_to_hdf5_file_test='D:/DEEPSLEEP/test_for_github/datasets/shhs/test/hdf5_file_test_all_chunking_shhs1.hdf5'
    path_to_file_length_test='D:/DEEPSLEEP/test_for_github/datasets/shhs/test/testFilesNum30secEpochs_all_shhs1.pkl'
    
    path_to_results='./channel occlusion/confusion_matrix_channel_occlusion.xlsx'
    if os.path.isfile(path_to_results):
        wb = op.load_workbook(path_to_results)
        sheet=wb.get_sheet_by_name('Sheet')
    else:
        wb=Workbook()
        sheet=wb.active
    
    f_file_length_test=open(path_to_file_length_test,'rb')
    file_length_dic_test=pickle.load(f_file_length_test)
    batches_test=np.sum(np.ceil(np.array(list(file_length_dic_test.values()))/batch_size),dtype='int32')
    f_file_length_test.close()
    
    print("start generator test")
    data_gen_test=utils.my_generator1(path_to_hdf5_file_test)
    print("start dataloader test")
    sampler_test=SequentialSampler(data_gen_test)
    data_iterator_Test=DataLoader(data_gen_test,batch_size=1,num_workers=workers,batch_sampler=utils.CustomSequentialBatchSampler(sampler_test,batch_size,file_length_dic_test))
    
    for modality in modality_list:
        modality_to_occlude=modality
        modality_occlusion(data_iterator_Test,batches_test,modality)
    
    #save file
    wb.save(path_to_results)