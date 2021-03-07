# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:14:09 2018

@author: PathakS
"""

import mne
import scipy.io as sio
import xml.etree.cElementTree as Elem
import os
import utils

def edf_to_mat(path_to_annot_folder,path_to_edf_folder):
    for i in os.listdir(path_to_edf_folder):
        edf_files.append(i)
    
    for i in os.listdir(path_to_annot_folder):
        annotation_files.append(i)
    
    for i in range(0,len(edf_files)):
        #all the signals resampled to 125Hz, EEG and EMG already has a sampling rate of 125Hz
        #EOG has a sampling rate of 50Hz, which was resampled to 125Hz by the read_raw_edf command
        raw = mne.io.read_raw_edf(path_to_edf_folder+edf_files[i], preload=False)
        print(edf_files[i])
        sfreq=raw.info['sfreq']
        #print(raw.info)
        #print(sfreq)
        test_info=raw.info['ch_names']
        test_info_loc=raw.info['chs']
        print(test_info)
        print(test_info_loc)
        print(len(raw.times)) #Length of one eeg signal in terms of samples
        test_data=raw.copy()
        tree = Elem.parse(path_to_annot_folder+annotation_files[i])
        root=tree.getroot()
        annotation=[]
        onset=[]
        duration=[]
        annot=[]
        eeg1_2=[]
        eeg2_2=[]
        eogL_2=[]
        eogR_2=[]
        emg_2=[]
        
        #data,times=test_data[16,:]
        channel_indices = mne.pick_channels(test_data.info['ch_names'], ['EEG 2', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG'])
        print(channel_indices)
        #print(data.shape)
        #print(len(times))
        #print(test_data.pick())
        for event in root.findall(".//ScoredEvent/[EventType='Stages|Stages']"):
            for sublevel in event:
                if sublevel.tag=='EventConcept':
                    if int(sublevel.text.split('|')[1])==4:
                        annotation.append(3)
                    else:
                        annotation.append(int(sublevel.text.split('|')[1]))
                if sublevel.tag=='Start':
                    onset.append(float(sublevel.text))
                if sublevel.tag=='Duration':
                    duration.append(float(sublevel.text))
    
        
        eeg1,times=test_data[2,:] #C3/A2
        eeg2,times=test_data[7,:] #C4/A1
        eogL,times=test_data[5,:] #EOG(L)/PG1 (PG1 is the electrode at the center of the forehead - used as a reference electrode)
        eogR,times=test_data[6,:] #EOG(R)/PG1
        emg,times=test_data[4,:]
        eeg1_1=eeg1[0]
        eeg2_1=eeg2[0]
        eogL_1=eogL[0]
        eogR_1=eogR[0]
        emg_1=emg[0]
        #print(len(times_eogR),len(eogR[0]))
        for j in range(len(onset)):
            #if annotation[j] not in [0,1,2,3,4,5]:
            #    out.write(str(edf_files[i])+"\n")
            if annotation[j] in [0,1,2,3,4,5]:
                if onset[j] in times:
                    start_ind=times.tolist().index(onset[j])
                    #print(times[start_ind])
                    timesteps=int((duration[j])*125)
                    end_ind=start_ind+timesteps
                    eeg1_2.extend(eeg1_1[start_ind:end_ind])
                    eeg2_2.extend(eeg2_1[start_ind:end_ind])
                    eogL_2.extend(eogL_1[start_ind:end_ind])
                    eogR_2.extend(eogR_1[start_ind:end_ind])
                    emg_2.extend(emg_1[start_ind:end_ind])
                    #print(type(annotation[j]))
                    annot.extend([sleep_stage_dic[str(annotation[j])]]*int(duration[j])*125)
            #print(annotation[i])
        
        
        
        ###### Filtering #######
        #Lowpass filter 
        eeg1_filt=utils.butter_bandpass_filter(30,'low',sfreq,4,eeg1_2)
        eeg2_filt=utils.butter_bandpass_filter(30,'low',sfreq,4,eeg2_2)
        eogL_filt=utils.butter_bandpass_filter(30,'low',sfreq,4,eogL_2)
        eogR_filt=utils.butter_bandpass_filter(30,'low',sfreq,4,eogR_2)
        emg_filt=utils.butter_bandpass_filter(30,'low',sfreq,4,emg_2)
        #Highpass filter
        eeg1_filt=utils.butter_bandpass_filter(0.16,'high',sfreq,4,eeg1_filt)
        eeg2_filt=utils.butter_bandpass_filter(0.16,'high',sfreq,4,eeg2_filt)
        eogL_filt=utils.butter_bandpass_filter(0.16,'high',sfreq,4,eogL_filt)
        eogR_filt=utils.butter_bandpass_filter(0.16,'high',sfreq,4,eogR_filt)
        emg_filt=utils.butter_bandpass_filter(10,'high',sfreq,4,emg_filt)
        
        #f1,pxx1=signal.periodogram(eogL_2,sfreq)
        #plt.plot(f1,pxx1)
        #plt.show()
        dic={'signals': {'eeg1':eeg1_filt,'eeg2':eeg2_filt,'eogL':eogL_filt,'eogR':eogR_filt,'emg':emg_filt,'annotation':annot}}
        print(edf_files[i])
        sio.savemat('D:/DEEPSLEEP/test_for_github/datasets/shhs/mat files/eeg_annotation_'+str(edf_files[i]).strip('.edf'),dic)

#Initialization
sleep_stage_dic={'0':0,'1':1,'2':2,'3':3,'4':3,'5':4}
dic={}
edf_files = []
annotation_files=[]
annotations_all=[]
data_all=[]
times_all=[]
#out=open('files_with_unknown_sleep_scoring.txt','w')
path_to_annot_folder="D:/DEEPSLEEP/shhs/shhs/polysomnography/annotations-events-nsrr/shhs1/"
path_to_edf_folder="D:/DEEPSLEEP/shhs/shhs/polysomnography/edfs/shhs1/"
edf_to_mat(path_to_annot_folder, path_to_edf_folder)