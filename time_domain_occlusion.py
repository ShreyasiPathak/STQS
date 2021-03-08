# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:25:54 2019

@author: PathakS
"""

import models
import torch
import operator
import numpy as np
import math
import utils
import pickle
import itertools
from scipy import signal
from collections import OrderedDict, Counter
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import colors as mcolors
import matplotlib.cm as cm
#import matplotlib.patches as patches
#from visbrain.io import write_fig_hyp
from torch.utils.data import DataLoader, SequentialSampler

def frequency_signal(eeg):
    sfreq=125
    f1,pxx1=signal.periodogram(eeg,sfreq)
    plt.plot(f1,pxx1)
    plt.show()

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
    predict_class=output.argmax(dim=1, keepdim=True).item()
    return predict_class

def time_domain_occlusion_CNN(data_iterator_test):
    conf_mat=np.zeros((25,5))
    for idx, data, label in data_iterator_test:
        #print(idx)
        occl_chunk_sec=5
        slide=1
        no_of_batches=len(data)
        chunk_no=388
        #for chunk_no in range(0,no_of_batches):
        #if label[chunk_no]==2:
        #print(chunk_no)
        
        
        print("Chunk no:{}/{}".format(chunk_no,no_of_batches))
        start=0
        end=start+occl_chunk_sec*125
        #chunk_no=74#75#73#81
        dic_prediction_change={}
        dic_slide_count={}
        dic_prediction_change_changedClass={}
        pred_occl=[]
        true_class=label[chunk_no].item()
        output_class_wo_occlusion=predict(path_to_trained_model,data[chunk_no].clone().detach().unsqueeze(0))
        #print(output_class_wo_occlusion)
        #if output_class_wo_occlusion==label[chunk_no]:
        #--------------------Plotting the raw signal-------------
        #print("Time:", chunk_no/2)
        #fig = plt.figure(figsize=(15,3))
        #plt.xlim(-0.5, 30.5)
        #plt.plot(np.linspace(0,29.992,3750),data[chunk_no,0,0,:].numpy()) #already data normalized in generator function in utils
        #fig.savefig('C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/occlusion/time domain/final/chunk83_stage2_raw.png', bbox_inches='tight')
        #--------------------End of plotting---------------------
        #print("true class",true_class)
        #print("output w/o occlusion:",output_class_wo_occlusion)
        while end<=len(data[chunk_no,0,0,:]):
            temp=data[chunk_no].clone().detach()
            temp=temp.unsqueeze(0)
            #print(temp.shape)
            temp[0,0,0,start:end]=temp[0,0,0,start:end]*0
            temp[0,0,1,start:end]=temp[0,0,1,start:end]*0
            output_class=predict(path_to_trained_model,temp)
            pred_occl.append(output_class)
            for slide_no in range(int(occl_chunk_sec/slide)):
                end_sec_slide_no=(start/125+(slide_no*slide))+slide # end second of one occlusion patch
                dic_slide_count[end_sec_slide_no]=dic_slide_count.get(end_sec_slide_no,0)+1 #how many times one occlusion patch can be seen, in all the shifting window of occlusion
                if end_sec_slide_no not in dic_prediction_change_changedClass.keys():
                    dic_prediction_change_changedClass[end_sec_slide_no]=[]
                if output_class_wo_occlusion!=output_class:
                    dic_prediction_change[end_sec_slide_no]=dic_prediction_change.get(end_sec_slide_no,0)+1
                    dic_prediction_change_changedClass[end_sec_slide_no].append(output_class)
                else:
                    dic_prediction_change[end_sec_slide_no]=dic_prediction_change.get(end_sec_slide_no,0)
                
            start=start+slide*125
            end=end+slide*125
        print("predicted class occlusion:",pred_occl)
        conf_mat=confusion_matrix_time_domain(true_class,output_class_wo_occlusion,pred_occl,conf_mat)
        plotting_occlusion_graph(data,true_class,chunk_no,dic_prediction_change,dic_prediction_change_changedClass,dic_slide_count)
        print(conf_mat)
    print(conf_mat)

def confusion_matrix_time_domain(true_class,pred_ori,pred_occl,conf_mat):
    dic_pred_occl=Counter(pred_occl)
    for item in dic_pred_occl.items():
        index_x=dic_indexing_time_domain_occl[str(pred_ori)+str(item[0])]
        conf_mat[index_x,true_class]=conf_mat[index_x,true_class]+item[1]
    return conf_mat

def plotting_occlusion_graph(data,true_class,chunk_no,dic_prediction_change,dic_prediction_change_changedClass, dic_slide_count):
    my_cmap = cm.get_cmap('viridis') # or any other one
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1) # the color maps work for [0, 1]
    fig, ax1 = plt.subplots(nrows=1,figsize=(20, 5),sharex=True)
    i=0
    true_label=true_class
    for item in dic_prediction_change.items():
        if i==0:
            dic_item_previous=0
        if dic_prediction_change_changedClass[item[0]]==[]:
            occluded_prediction=str(true_label)
        else:
            occluded_prediction=str(max(Counter(dic_prediction_change_changedClass[item[0]]).items(), key=operator.itemgetter(1))[0])
        ax1.plot(np.linspace(dic_item_previous,item[0],(item[0]-dic_item_previous)*125),data[chunk_no,0,0,int(dic_item_previous)*125:int(item[0])*125].numpy(),color=my_cmap(norm(item[1]/dic_slide_count[item[0]])))
        ax1.annotate('', xy=(dic_item_previous, 5.9), xytext=(item[0], 5.9), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '|-|'})
        ax1.annotate(str("{0:.2f}".format(item[1]/dic_slide_count[item[0]])), xy=(dic_item_previous+(item[0]-dic_item_previous)/2, 6.1), ha='center', va='center')
        ax1.annotate('-->'+occluded_prediction, xy=(dic_item_previous+(item[0]-dic_item_previous)/2, 6.5), ha='center', va='center')
        dic_item_previous=item[0]
        i=i+1
    
    ax1.set(ylim=(-5,7))
    ax1.set(xlim=(-0.5,30.5))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (microvolt)')
    fig.suptitle('Sleep stage '+str(true_label))
    cmmapable = cm.ScalarMappable(norm, my_cmap)
    cmmapable.set_array(range(0, 1))
    plt.colorbar(cmmapable)
    fig.savefig('C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/occlusion/time domain/final/chunk388_stage4_1.pdf', format='pdf', bbox_inches='tight')
    #plt.figtext(0.6, 0.825, "(predicted class w/o occlusion -> predicted on occlusion)")
    plt.show()

if __name__ == '__main__':
    #data should be in [batch size, 1, rows, columns] format, where rows= 5 (5 signals - eeg1,eeg2,eogL,eogR, emg) 
    #and columns=3750 (125*30secs) for this model. Batch size means number of inout 30 secs signal chunks
    #batch size can be 1 or more
    #data should be in torch.tensor data type.
    
    #batch_size=1000
    time_steps=3750
    n_channels=5
    seq_length=8
    workers=3
    main_channels=3
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dic_indexing_time_domain_occl={'00':0,'01':1,'02':2,'03':3,'04':4,'10':5,'11':6,'12':7,'13':8,'14':9,'20':10,'21':11,'22':12,'23':13,'24':14,'30':15,'31':16,'32':17,'33':18,'34':19,'40':20,'41':21,'42':22,'43':23,'44':24}
    
    ##trained model cnn
    model_name=input('Enter the name of the model (without the extension .tar):')
    path_to_trained_model="D:/DEEPSLEEP/test_for_github/saved_models/shhs/"+model_name+".tar"
        
    ##Data for one patient
    path_to_hdf5_file_test='D:/DEEPSLEEP/codes and results/SHHS data codes/occlusion dataset/hdf5_file_test_eeg_annotation_shhs1-203371.hdf5'
    path_to_file_length_test='D:/DEEPSLEEP/codes and results/SHHS data codes/occlusion dataset/test30secChunk_eeg_annotation_shhs1-203371.pkl'
    
    ##data for whole test set
    #path_to_hdf5_file_test='D:/DEEPSLEEP/codes and results/SHHS data codes/occlusion dataset/hdf5_file_test_10files_chunking_SHHS.hdf5'
    #path_to_file_length_test='D:/DEEPSLEEP/codes and results/SHHS data codes/occlusion dataset/testFilesNum30secEpochs_10files.pkl'
    
    f_file_length_test=open(path_to_file_length_test,'rb')
    file_length_dic_test=pickle.load(f_file_length_test)
    f_file_length_test.close()
    batch_size=file_length_dic_test['0'] #batch size when you want to test for one patient at a time
    
    print("start generator test")
    data_gen_test=utils.my_generator1(path_to_hdf5_file_test)
    print("start dataloader test")
    sampler_test=SequentialSampler(data_gen_test)
    data_iterator_Test=DataLoader(data_gen_test,batch_size=1,num_workers=workers,batch_sampler=utils.CustomSequentialBatchSampler(sampler_test,batch_size,file_length_dic_test))
    
    time_domain_occlusion_CNN(data_iterator_Test)