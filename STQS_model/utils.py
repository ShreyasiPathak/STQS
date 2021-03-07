# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:08:25 2019

@author: PathakS
"""

import os
import h5py
import math
import torch
import pickle
import random
import sklearn
import operator
import numpy as np
import seaborn as sns
import scipy.io as sio
from random import shuffle
from collections import Counter
from torch.utils.data import Dataset, Sampler
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, butter, filtfilt#,periodogram

n_classes=5 
n_channels=5
chunk_secs=30
batch_size=192
time_period_sample=3750 # 30secs * 125
mat_files_name_of_data_stored='signals'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
path_to_mat_folder='D:/DEEPSLEEP/codes and results/eeg_annotations_all_channels_filtered_30_0.16/'

class my_generator1(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        f.close()
        print("Total Length instances in generator:",len_instances)
        return len_instances

    def __getitem__(self, idx):
        f = h5py.File(self.hdf5_file, 'r')
        x_30sec_epoch=f['data'][idx]
        y_30sec_epoch=f['label'][idx][0]
        x_30sec_epoch=data_normalizer(x_30sec_epoch)
        f.close()
        return idx, x_30sec_epoch, y_30sec_epoch

class CustomSequentialBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
        #print("file dic with length:",self.file_length_dic)
    
    def __iter__(self):
        batch = []
        len_tillCurrent=0
        file_current=0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent=len_tillCurrent+1
            if len(batch) == self.batch_size or len_tillCurrent==self.file_length_dic[str(file_current)]:
                yield batch
                batch = []
            if len_tillCurrent==self.file_length_dic[str(file_current)]:
                len_tillCurrent = 0
                file_current=file_current+1
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.file_length_dic.values()))/self.batch_size),dtype='int32')
        print("Length in custom batch:",length)
        return length

class CustomRandomBatchSamplerSlicedShuffled(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
    
    def __iter__(self):
        batch = []
        len_tillCurrent=0
        file_current=0
        batch_no=0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent=len_tillCurrent+1
            if len(batch) == self.batch_size or len_tillCurrent==self.file_length_dic[str(file_current)]:
                batch_no=batch_no+1
                print("batch:",batch_no)
                #batch.sort()
                yield batch
                batch = []
            if len_tillCurrent==self.file_length_dic[str(file_current)]:
                len_tillCurrent = 0
                file_current=file_current+1
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.file_length_dic.values()))/self.batch_size),dtype='int32')
        print("length in batch sampler:",length)
        return length

class CustomRandomSamplerSlicedShuffled(Sampler):
    def __init__(self, hdf5_file, dic_length):
        self.hdf5_file = hdf5_file
        self.dic_length=dic_length
    
    def __iter__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        OriginalIndices=list(range(0,len_instances))
        #print(self.dic_length.items())
        for item in self.dic_length.items():
            if item[0]==0:
                begin=0
            else:
                begin=self.dic_length[item[0]-1]
            end=item[1]
            slicedIndices=MutableSlice(OriginalIndices,begin,end)
            random.shuffle(slicedIndices)
        iter_shuffledIndex=iter(OriginalIndices)
        f.close()
        return iter_shuffledIndex
    
    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        f.close()
        return len_instances

class CustomWeightedRandomBatchSamplerSlicedShuffled(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
        #print("file dic with length:",self.file_length_dic)
    
    def __iter__(self):
        batch = []
        file_current=0
        batch_no=0
        for idx in self.sampler:
            if idx>=self.file_length_dic[file_current]:
                if batch!=[]:
                    batch_no=batch_no+1
                    print("batch:",batch_no)
                    yield batch
                    batch = []
                file_current=file_current+1
                print("file no __iter__:",file_current)
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch_no=batch_no+1
                print("batch:",batch_no)
                #batch.sort()
                yield batch
                batch = []    
        if len(batch)>0:
            batch_no=batch_no+1
            print("batch:",batch_no)
            yield batch
    
    def __len__(self):
        iterator=self.sampler
        sampler_array=np.array(list(iterator))
        totalBatches=0
        for len_item in self.file_length_dic.items():
            if len_item[0]==0:
                previous_file_len=0
            else:
                previous_file_len=self.file_length_dic[len_item[0]-1]
            count_oversampled_oneFile=((previous_file_len<sampler_array) & (sampler_array<len_item[1])).sum()
            totalBatches=totalBatches+np.ceil(count_oversampled_oneFile/self.batch_size)
            #print(count_oversampled_oneFile)
        print("batch sampler length:",totalBatches)
        return totalBatches

class CustomWeightedRandomSamplerSlicedShuffled(Sampler):
    def __init__(self, hdf5_file, dic_length):
        self.hdf5_file = hdf5_file
        self.dic_length = dic_length
    
    def __iter__(self):
        #np.set_printoptions(threshold=sys.maxsize)
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count=class_distribution_oversampling(self.hdf5_file)
        max_class=max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1],class_count[2],class_count[3],class_count[4]])
        labels=f['label'][:len_instances].reshape(-1)
        #print(np.append(OriginalIndices,np.random.choice(np.where(labels==0)[0],5),axis=0))
        for i in range(0,5):
            if i!=max_class[0]:
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],diff_count_class[i]),axis=0)
        #print(self.dic_length.items())
        OriginalIndices=np.sort(OriginalIndices)
        for item in self.dic_length.items():
            if item[0]==0:
                #begin_value=0
                begin_index=0
            else:
                begin_value=self.dic_length[item[0]-1]
                begin_index=np.where(OriginalIndices==begin_value)[0][0]
            end_index=np.where(OriginalIndices==item[1]-1)[0][-1]
            #print(begin_index)
            #print(end_index)
            slicedIndices=MutableSlice(OriginalIndices,begin_index,end_index+1)
            random.shuffle(slicedIndices)
        #print("shuffledIndex:",OriginalIndices[:3022])
        #input("halt")
        print("Sampler __iter__:",OriginalIndices.shape)
        print("Sampler iter len:",max_class[1]*5)
        iter_shuffledIndex=iter(OriginalIndices)
        #iter_index=iter(torch.randperm(len_instances).tolist())
        f.close()
        return iter_shuffledIndex
    
    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        class_count=class_distribution_oversampling(self.hdf5_file)
        max_class=max(class_count.items(), key=operator.itemgetter(1))
        f.close()
        len_instances_oversample=max_class[1]*5
        print("Sampler __len__:",len_instances_oversample)
        return len_instances_oversample

class CustomSequentialLSTMBatchSampler_ReturnAllChunks(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic,seq_length):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
        #print("file dic with length:",self.file_length_dic)
    
    def __iter__(self):
        batch = []
        len_tillCurrent=0
        file_current=0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent=len_tillCurrent+1
            if len(batch) == self.batch_size or len_tillCurrent==self.file_length_dic[str(file_current)]:
                if len_tillCurrent==self.file_length_dic[str(file_current)]:
                    if len(batch)%self.seq_length!=0:
                        quoteint=int(len(batch)/self.seq_length)
                        remainder=self.seq_length*(quoteint+1)-len(batch)
                        batch.extend(range(idx-len_tillCurrent+1,idx-len_tillCurrent+1+remainder))
                    len_tillCurrent = 0
                    file_current=file_current+1
                yield batch
                batch = []
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.file_length_dic.values()))/self.batch_size),dtype='int32')
        print(length)
        return length

class MutableSlice(object):
    def __init__(self, baselist, begin, end=None):
        self._base = baselist
        self._begin = begin
        self._end = len(baselist) if end is None else end

    def __len__(self):
        return self._end - self._begin

    def __getitem__(self, i):
        return self._base[self._begin + i]

    def __setitem__(self, i, val):
        self._base[i + self._begin] = val

def data_normalizer(data_30secs):
    data_ch_norm_list=[]
    num_of_channels=data_30secs.shape[1]
    for ch in range(num_of_channels):
        data_ch=np.array(data_30secs[:,ch,:])
        data_ch_mean=(np.mean(data_ch,axis=1)).reshape(-1,1)
        data_ch_std=(np.std(data_ch,axis=1)).reshape(-1,1)
        if math.isnan(data_ch_std) or not data_ch_std:
            data_ch_norm=(data_ch-data_ch_mean)
        else:
            data_ch_norm=(data_ch-data_ch_mean)/data_ch_std
        data_ch_norm_list.append(data_ch_norm[0])
    x=np.array([data_ch_norm_list])
    return x 

def calculate_num_samples(files,filename_pkl):
    #calculate the length of .mat files in terms of no.of 30 sec epoch batches
    #total_length_samples=0
    dic_files={}
    f=open(filename_pkl,'wb')
    for k,filename in enumerate(files):
        print(k,filename)
        file_data = sio.loadmat(path_to_mat_folder+"/"+filename,struct_as_record=False)
        len_signal=int(np.floor(len(file_data[mat_files_name_of_data_stored][0,0].eeg1[0])/time_period_sample))
        dic_files[str(k)]=len_signal
    pickle.dump(dic_files, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def calculate_num_samples_cumulative(files,filename_pkl):
    #calculate the length of .mat files in terms of no.of 30 sec epoch batches
    #total_length_samples=0
    dic_files={}
    s=0
    f=open(filename_pkl,'wb')
    for k,filename in enumerate(files):
        print(k,filename)
        file_data = sio.loadmat(path_to_mat_folder+"/"+filename,struct_as_record=False)
        len_signal=int(np.floor(len(file_data[mat_files_name_of_data_stored][0,0].eeg1[0])/time_period_sample))
        s=s+len_signal
        dic_files[k]=s
    pickle.dump(dic_files, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def butter_bandpass(freq_cut, filter_type, fs, order):
       nyq = 0.5 * fs
       norm_freq_cut = freq_cut / nyq
       b, a = butter(order, norm_freq_cut, btype=filter_type)
       return b, a
   
def butter_bandpass_filter(freq_cut, filter_type, fs, order, data):
       b, a = butter_bandpass(freq_cut, filter_type, fs, order)
       y = filtfilt(b, a, data)
       return y

def class_distribution_weightedloss1(hdf5_file):
    f = h5py.File(hdf5_file, 'r')
    dic_class=dict(Counter(np.reshape(f['label'],(-1))))
    f.close()
    print("dic_class:",dic_class)
    sum_class_dist=dic_class[0]+dic_class[1]+dic_class[2]+dic_class[3]+dic_class[4]
    weight=torch.tensor([1-(dic_class[0]/sum_class_dist),1-(dic_class[1]/sum_class_dist),1-(dic_class[2]/sum_class_dist),1-(dic_class[3]/sum_class_dist),1-(dic_class[4]/sum_class_dist)])
    print("weight:",weight)
    return weight

def class_distribution_weightedloss2(hdf5_file):
    #calculate the length of .mat files in terms of no.of 30 sec epoch batches
    f = h5py.File(hdf5_file, 'r')
    labels=np.reshape(f['label'],(-1))
    class_weight=sklearn.utils.class_weight.compute_class_weight('balanced', [0,1,2,3,4], labels)
    print(dict(Counter(labels)))
    print(class_weight)
    f.close()
    return torch.tensor(class_weight,dtype=torch.float32)
 
def class_distribution_oversampling(hdf5_file):
    f = h5py.File(hdf5_file, 'r')
    dic_class=dict(Counter(np.reshape(f['label'],(-1))))
    f.close()
    print("dic_class:",dic_class)
    return dic_class
 
def hdf5_creation1(mat_files,file_name):
    f = h5py.File(file_name, "w")
    dset_data = f.create_dataset("data", (1,1,n_channels,time_period_sample), maxshape=(None,1,n_channels,time_period_sample), chunks=(1000,1,n_channels,time_period_sample))
    dset_label = f.create_dataset("label", (1,1), maxshape=(None,1), dtype='i8',chunks=(1000,1))
    samples_in_all_files=0
    for k,file_name in enumerate(mat_files):
        print(k,file_name)
        
        #reading file data
        file_data = sio.loadmat(path_to_mat_folder+"/"+mat_files[k],struct_as_record=False)
        eeg1=file_data[mat_files_name_of_data_stored][0,0].eeg1[0]
        eeg2=file_data[mat_files_name_of_data_stored][0,0].eeg2[0]
        eogL=file_data[mat_files_name_of_data_stored][0,0].eogL[0]
        eogR=file_data[mat_files_name_of_data_stored][0,0].eogR[0]
        emg=file_data[mat_files_name_of_data_stored][0,0].emg[0]
        y=file_data[mat_files_name_of_data_stored][0,0].annotation[0]
        
        #resizing hdf5 file
        samples_in_one_file=int(np.floor(len(eeg1)/time_period_sample))
        samples_in_all_files+=samples_in_one_file
        dset_data.resize((samples_in_all_files,1,n_channels,time_period_sample))
        dset_label.resize((samples_in_all_files,1))
        
        #reshaping the signal into 30sec chunks
        eeg1_f=np.reshape(eeg1,((-1,time_period_sample)))
        eeg2_f=np.reshape(eeg2,((-1,time_period_sample)))
        eogL_f=np.reshape(eogL,((-1,time_period_sample)))
        eogR_f=np.reshape(eogR,((-1,time_period_sample)))
        emg_f=np.reshape(emg,((-1,time_period_sample)))
        y=np.reshape(y,((-1,time_period_sample)))
        
        
        #concatenating channels and creating [batch_size,1,n_channels,time_period_sample] dimensional data  
        x_30sec_epochs=np.concatenate((eeg1_f,eeg2_f,eogL_f,eogR_f,emg_f),axis=1).reshape((-1,1,n_channels,time_period_sample))
        y_30sec_epochs=np.unique(y,axis=1)
        print(x_30sec_epochs.shape)
        print(y_30sec_epochs.shape)
        
        #saving in hdf5 file
        dset_data[samples_in_all_files-samples_in_one_file:samples_in_all_files]=x_30sec_epochs
        dset_label[samples_in_all_files-samples_in_one_file:samples_in_all_files]=y_30sec_epochs
        
    print("dset data shape outside:",dset_data.shape)
    print("dset label shape outside:",dset_label.shape)
    f.close()

def to_sequence_for_gpu_LSTM_ReturnAllChunk(data,seq):
    #data shape=(samples, flatten dimension-output of cnn linear layer)
    print("sequence before gpu parallel:",data.shape)
    if len(data.shape)==4:
        channel_dimen=data.shape[2]
        time_points=data.shape[3]
        seq_samples=data.view(-1,seq,1,channel_dimen,time_points)
        print("sequence before gpu parallel reshaped:",seq_samples.shape)
    #seq_samples=seq_samples.to(device)
    elif len(data.shape)==2 or len(data.shape)==1:
        seq_samples=data.view(-1,seq,1)
        print("sequence before gpu parallel reshaped:",seq_samples.shape)
    return seq_samples


def to_sequence_LSTM_ReturnAllChunk(data,seq):
    #data shape=(samples, flatten dimension-output of cnn linear layer)
    print(data.shape)
    feature_dimen=data.shape[1]
    seq_samples=data.view(-1,seq,feature_dimen)
    print(seq_samples.shape)
    return seq_samples

def freeze_layers(model):
    for name,param in model.named_parameters():
        if "channelBlocks" in name:
            param.requires_grad=False
        #print(name,param.requires_grad)
    return model

def metrics_confusion_matrix_per_class(conf_mat):
    each_stage_metrics=[]
    each_model_metrics=[]
    wt_macro_f1=0
    macro_f1=0
    tp=0
    pe=0
    bal_acc=0
    for i in range(conf_mat.shape[0]):
        prec_val=conf_mat[i,i]/np.sum(conf_mat[:,i])
        recall_val=conf_mat[i,i]/np.sum(conf_mat[i,:])
        f_score=(2*prec_val*recall_val)/(prec_val+recall_val)
        each_stage_metrics.append([prec_val,recall_val,f_score])
        tp+=conf_mat[i,i]
        wt_macro_f1+=f_score*(np.sum(conf_mat[i,:])/np.sum(conf_mat))
        macro_f1+=f_score
        bal_acc+=recall_val
        pe+=(np.sum(conf_mat[i,:])/np.sum(conf_mat))*(np.sum(conf_mat[:,i])/np.sum(conf_mat))
    acc=tp/np.sum(conf_mat)
    bal_acc=bal_acc/5
    po=tp/np.sum(conf_mat)
    cohen_kappa=(po-pe)/(1-pe)
    macro_f1=macro_f1/5
    each_model_metrics=[acc,bal_acc,macro_f1,cohen_kappa,wt_macro_f1]
    return each_stage_metrics,each_model_metrics

def confusion_matrix_norm_func(conf_mat,fig_name):
    class_name=['W','N1','N2','N3','REM']
    conf_mat_norm=np.empty((5,5))
    #conf_mat=confusion_matrix(y_true, y_pred)
    for i in range(conf_mat.shape[0]):
        conf_mat_norm[i,:]=conf_mat[i,:]/sum(conf_mat[i,:])
    #print(conf_mat_norm)
    print_confusion_matrix(conf_mat_norm,class_name,fig_name)
    
def print_confusion_matrix(conf_mat_norm, class_names, fig_name, figsize = (2,2), fontsize=5):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    #sns.set()
    #grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    fig, ax = plt.subplots(figsize=figsize)
    #cbar_ax = fig.add_axes([.93, 0.1, 0.05, 0.77])
    #fig = plt.figure(figsize=figsize)
    heatmap=sns.heatmap(
        yticklabels=class_names,
        xticklabels=class_names,
        data=conf_mat_norm,
        ax=ax,
        cmap='YlGnBu',
        cbar=False,
        #cbar_ax=cbar_ax,
        annot=True,
        annot_kws={'size':fontsize},
        fmt=".2f",
        square=True
        #linewidths=0.75
        )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('True label',labelpad=0,fontsize=fontsize)
    ax.set_xlabel('Predicted label',labelpad=0,fontsize=fontsize)
    #cbar_ax.tick_params(labelsize=fontsize) 
    #ax.get_yaxis().set_visible(False)
    #plt.tight_layout()
    #plt.show()
    fig.savefig(fig_name, format='pdf', bbox_inches='tight')