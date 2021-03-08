# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:58:31 2019

@author: PathakS
"""

import os
import torch
import utils
import pickle
import models
import numpy as np
from scipy import signal
import openpyxl as op
import matplotlib.pyplot as plt
from openpyxl import Workbook
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler

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

def confusion_matrix_func(true,pred,conf_mat):
    conf_mat=conf_mat+confusion_matrix(true.cpu(), pred.cpu(), [0,1,2,3,4])
    return conf_mat

def metrics(conf_mat,total_metric_all_occlu):
    prec=[]
    recall=[]
    fscore=[]
    for i in range(conf_mat.shape[0]):
        print(conf_mat[i,i])
        print(np.sum(conf_mat[:,i]))
        prec_val=conf_mat[i,i]/np.sum(conf_mat[:,i])
        recall_val=conf_mat[i,i]/np.sum(conf_mat[i,:])
        prec.append(prec_val)
        recall.append(recall_val)
        fscore.append(2*prec_val*recall_val)
    total_metric_all_occlu.append(list(zip(prec,recall,fscore)))
    return total_metric_all_occlu

def diff_real_scores_occluded_scores(real_score,occluded_score):
    diff_score=np.empty((8,5,3))
    for i,row in enumerate(total_metric_all_occlu):
        for j,occlu_scores in enumerate(row):
            if 'nan' in occlu_scores:
                for k,val in enumerate(occlu_scores):
                    if val!='nan':
                        diff_score[i,j,k]=real_score[j][k]-occlu_scores[k]
                    else:
                        diff_score[i,j,k]=str(val)
            else:
                diff_score[i,j,:]=real_score[j]-np.array(occlu_scores)
    return diff_score

def frequency_occlusion(data_iterator_test,occluding_band,band_type,total_metric_all_occlu,batches):
    conf_mat=np.zeros((5,5))
    batch_no=0
    for idx, data, label_true in data_iterator_test:
        batch_no+=1
        print(data.shape)
        temp=data.clone().detach()
        if band_type==['bandpass']:
            temp[:,0,0,:]=torch.from_numpy(np.ascontiguousarray(utils.butter_bandpass_filter(occluding_band,band_type[0],sfreq,4,temp[:,0,0,:])))
            temp[:,0,1,:]=torch.from_numpy(np.ascontiguousarray(utils.butter_bandpass_filter(occluding_band,band_type[0],sfreq,4,temp[:,0,1,:])))
        else:
            temp[:,0,0,:]=torch.from_numpy(np.ascontiguousarray(utils.butter_bandpass_filter(occluding_band,band_type[0],sfreq,4,temp[:,0,0,:]))) 
            temp[:,0,1,:]=torch.from_numpy(np.ascontiguousarray(utils.butter_bandpass_filter(occluding_band,band_type[0],sfreq,4,temp[:,0,1,:])))
        #------------------------Frequency plotting----------------------------
        #win = 4 * sfreq
        #freqs, psd = signal.welch(temp[0,0,0,:], sfreq, nperseg=win)
        #plt.plot(freqs, psd)
        #plt.show()
        #-----------------------------------------------------------------------------
            
        output_class=predict(path_to_trained_model,temp)
        conf_mat=confusion_matrix_func(label_true,output_class,conf_mat)
        print("Batch no:{}/{}".format(batch_no,batches))
    print(conf_mat)
    sheet.append([str(occluding_band)+" "+str(band_type)])
    sheet.append([0,1,2,3,4])
    for row in conf_mat.tolist():
        sheet.append(row)
    total_metric_all_occlu=metrics(conf_mat,total_metric_all_occlu)
    if not os.path.isdir('./frequency domain/'):
        os.mkdir('./frequency domain/')
    fig_filename='./frequency domain/'+'conf_mat_'+str(occluding_band)+"_"+str(band_type)+"_occlusion.pdf"
    utils.confusion_matrix_norm_func(conf_mat,fig_name=fig_filename)
    return total_metric_all_occlu


if __name__ == '__main__':
    #data should be in [batch size, 1, rows, columns] format, where rows= 5 (5 signals - eeg1,eeg2,eogL,eogR, emg) 
    #and columns=3750 (125*30secs) for this model. Batch size means number of input 30 secs signal chunks
    #batch size can be 1 or more
    #data should be in torch.tensor data type.
    batch_size=1000
    time_steps=3750
    n_channels=5
    sfreq=125
    seq_length=8
    workers=3
    main_channels=3
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    occlu_freq_list=[[0.16,3.99],[4,7.99],[8,11.99],[12,15.99],[16,30]]
    total_metric_all_occlu=[]
    
    ##trained model cnn
    model_name=input('Enter the name of the model (without the extension .tar):')
    path_to_trained_model='D:/DEEPSLEEP/test_for_github/saved_models/shhs/'+model_name+'.tar'
    
    ##data for whole test set
    path_to_hdf5_file_test='D:/DEEPSLEEP/test_for_github/datasets/shhs/test/hdf5_file_test_all_chunking_shhs1.hdf5'
    path_to_file_length_test='D:/DEEPSLEEP/test_for_github/datasets/shhs/test/testFilesNum30secEpochs_all_shhs1.pkl'
    
    #path_to_result_conf_mat
    path_to_results='./frequency domain/confusion_matrix_freq_occlusion.xlsx'
    if os.path.isfile(path_to_results):
        wb = op.load_workbook(path_to_results)
    else:
        wb=Workbook()
        sheet=wb.active
    
    f_file_length_test=open(path_to_file_length_test,'rb')
    file_length_dic_test=pickle.load(f_file_length_test)
    batches_test=np.sum(np.ceil(np.array(list(file_length_dic_test.values()))/batch_size),dtype='int32')
    f_file_length_test.close()
    #batch_size=file_length_dic_test['0'] #batch size when you want to test for one patient at a time
    
    print("start generator test")
    data_gen_test=utils.my_generator1(path_to_hdf5_file_test)
    print("start dataloader test")
    
    sampler_test=SequentialSampler(data_gen_test)
    data_iterator_Test=DataLoader(data_gen_test,batch_size=1,num_workers=workers,batch_sampler=utils.CustomSequentialBatchSampler(sampler_test,batch_size,file_length_dic_test))
    
    #frequency occlusion function call
    for i, occlu_freq in enumerate(occlu_freq_list):
        if i==2 or i==3 or i==4:
            print("Starting band "+str(occlu_freq))
            total_metric_all_occlu=frequency_occlusion(data_iterator_Test, occlu_freq, ['bandpass'], total_metric_all_occlu, batches_test)
            print(total_metric_all_occlu)
        print("Starting low high "+str(occlu_freq))
        total_metric_all_occlu=frequency_occlusion(data_iterator_Test, occlu_freq, ['bandstop'], total_metric_all_occlu, batches_test)
        print(total_metric_all_occlu)
    
    print(total_metric_all_occlu)
    
    #save file
    wb.save(path_to_results)