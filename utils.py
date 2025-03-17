#!/usr/bin/env python

import numpy as np
import pandas as pd
from os.path import dirname
import tkinter as tk
from tkinter import filedialog as fd
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from tqdm import tqdm
from pyriemann.utils.test import is_sym_pos_def
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
from sklearn.model_selection import train_test_split
import joblib
from sklearn.decomposition import PCA
import matplotlib as mpl
import mne
import os

    
def saveto_npy(arr, pth):
    extension = 'npy'
    with open(pth, 'wb+') as fh:
        np.np_save(fh, arr, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())

def load_npy(pth):
    return np.np_load(pth)


def get_all_online_offline_files(path):
    directories = get_immediate_subdirectories(path)
    filepaths = []
    for dir in directories:
        filenames = get_filesNames_from_folder(path + '/' + dir)
        filenames = [path + '/' + dir + '/' + k for k in filenames if '.gdf' in k and ('online' in k or 'offline' in k)]
        if len(filenames)>0:
            filepaths += filenames
    signal, events_dataFrame = load_gdf_files(filepaths)
    return signal, events_dataFrame

def get_filesNames_from_folder(mypath):
    filenames = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return sorted(filenames)

def get_immediate_subdirectories(path_dir):
    subdirectories = [name for name in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, name))]
    return sorted(subdirectories)

def get_files(path):
    root = tk.Tk()

    filenames = ()
    while True:
        chosen_files = fd.askopenfilenames(initialdir=path)
        if len(chosen_files)==0:
            break
        filenames += chosen_files
        
    if filenames[0,-4:]=='.mat':
        signal, events_dataFrame, fs = load_mat_files(filenames)
    elif filenames[0,-4:]=='.gdf':
        signal, events_dataFrame = load_gdf_files(filenames)
        fs = None
    
    directory_path = dirname(filenames[0])
    root.destroy()
    return [signal, events_dataFrame, fs, directory_path]

def load_mat_files(filenames):
    signal = []
    eeg_dim_tot = 0
    d = dict()
    d['dur']=[]
    d['pos']=[]
    d['typ']=[]
    d['run']=[]
    d['day']=[]
    d['prt']=[]

    n_ses = 1
    last_day = ''
    for n,file in enumerate(filenames):
        print(' - Loading file: ' + file)
        data = loadmat(file_name=file)
        h = fix_mat(data['h'])
        d['dur'].append(h['EVENT']['DUR'])
        d['typ'].append(h['EVENT']['TYP'])
        d['pos'].append(h['EVENT']['POS']+eeg_dim_tot-1)
        d['run'].append([n]*len(h['EVENT']['DUR']))
        if file.split('/')[-2] == last_day or last_day=='':
            d['day'].append([n_ses]*len(h['EVENT']['DUR']))
        else:
            n_ses += 1
            d['day'].append([n_ses]*len(h['EVENT']['DUR']))
        if 'calibration' in file:
            d['prt'].append([0]*len(h['EVENT']['DUR']))
        elif 'evaluation' in file:
            d['prt'].append([1]*len(h['EVENT']['DUR']))
        elif 'control' in file:
            d['prt'].append([2]*len(h['EVENT']['DUR']))
        else:
            d['prt'].append([-1]*len(h['EVENT']['DUR']))
        last_day = file.split('/')[-2]
        signal.append(data['s'][:,:-1])
        eeg_dim_tot += data['s'].shape[0]


    signal = np.concatenate(signal, axis=0)
    d['dur'] = [ int(x) for x in np.concatenate(d['dur']) ]
    d['typ'] = [ int(x) for x in np.concatenate(d['typ']) ]
    d['pos'] = [ int(x) for x in np.concatenate(d['pos']) ]
    d['run'] = [ int(x) for x in np.concatenate(d['run']) ]
    d['day'] = [ int(x) for x in np.concatenate(d['day']) ]
    d['prt'] = [ int(x) for x in np.concatenate(d['prt']) ]
    #d['ses_vector'] = [ int(x) for x in np.concatenate(d['ses_vector']) ]

    fs = int(h['SampleRate'])
    
    events_dataFrame = pd.DataFrame(data=d)
    return signal, events_dataFrame, fs


def load_gdf_files(filenames):
    signal = []
    eeg_dim_tot = 0
    d = dict()
    d['pos']=[]
    d['typ']=[]
    d['run']=[]
    d['ses']=[]
    n_ses = 0
    last_day = ''
    for n,file in enumerate(filenames):
        print(' - Loading file: ' + file)
        eeg,h = read_gdf(file)
        d['typ'].append(h['EVENT']['TYP'])
        d['pos'].append(h['EVENT']['POS']+eeg_dim_tot-1)
        d['run'].append([n]*len(h['EVENT']['TYP']))
        if file.split('/')[-2] == last_day or last_day=='':
            d['ses'].append([n_ses]*len(h['EVENT']['TYP']))
        else:
            n_ses += 1
            d['ses'].append([n_ses]*len(h['EVENT']['TYP']))
        last_day = file.split('/')[-2]
        signal.append(eeg[:,:-1])
        eeg_dim_tot += eeg.shape[0]

    signal = np.concatenate(signal, axis=0)
    d['typ'] = [ int(x) for x in np.concatenate(d['typ']) ]
    d['pos'] = [ int(x) for x in np.concatenate(d['pos']) ]
    d['run'] = [ int(x) for x in np.concatenate(d['run']) ]
    d['ses'] = [ int(x) for x in np.concatenate(d['ses']) ]
    #d['ses_vector'] = [ int(x) for x in np.concatenate(d['ses_vector']) ]
    
    events_dataFrame = pd.DataFrame(data=d)
    return signal, events_dataFrame


def fix_mat(data):
    if data.dtype.names:
        new_data = dict()
        for name in data.dtype.names:
            new_data[name]=data[0][name][0]
        for k,v in new_data.items():
            if v.dtype.names:
                new_data[k] = fix_mat(v)
            else:
                new_data[k] = np.squeeze(v)
        return new_data
    else:
        return data

def get_channels_bool(wantedChannels, actualChannels):
    return [item in wantedChannels for item in actualChannels]


def select_channels(signal, wantedChannels, actualChannels=[]):
    if signal.shape[1] == 32 and len(actualChannels)==0:
        actualChannels = np.array(['FP1', 'FP2', 'FZ', 'FC5', 'FC1', 'FC2', 'FC6',  'C3',  'CZ',  
            'C4', 'CP5', 'CP1', 'CP2', 'CP6',  'P3', 'Pz',  'P4',  'F1',  'F2', 'FC3', 
            'FCZ', 'FC4', 'C5',  'C1',  'C2',  'C6', 'CP3', 'CP4',  'P5',  'P1', 'P2',  'P6'])
        
    elif signal.shape[1] == 16:
        actualChannels =  np.array(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3','C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz','CP2', 'CP4'])
    
    boolean_list = get_channels_bool(wantedChannels, actualChannels)
    return [signal[:,boolean_list], actualChannels[boolean_list]]


def get_bandranges(signal, bandranges, fs, filter_order):
    filt_signal = np.empty(tuple([len(bandranges)]) + signal.shape)
    for i,band in enumerate(bandranges):
        [b,a] = butter(filter_order,np.array(band)/(fs/2),'bandpass')
        filt_signal[i, :, :] = lfilter(b,a,signal,axis=0)
    return filt_signal


def get_trNorm_covariance_matrix(data, events, windowsLength, windowsShift, fc, substractWindowMean=True, dispProgress=True):
    cov_events = events.copy()
    if isinstance(events, pd.DataFrame):
        cov_events['pos'] = proc_pos2win(cov_events['pos'], windowsShift*fc, 'backward', windowsLength*fc)
        cov_events['dur'] = [ int(x) for x in cov_events['dur']/(windowsShift*fc)+1 ]

    n_bandranges, nsamples, nchannels = data.shape

    nwindows = int((nsamples-windowsLength*fc)/(windowsShift*fc))+1

    Cov = np.empty((n_bandranges, nwindows, nchannels, nchannels))

    if dispProgress:
        print(' - Computing covariance matrices on the band ranges')
    for bId in range(n_bandranges):
        ccov = get_sliding_covariance_trace_normalized(data[bId],  windowsLength*fc, windowsShift*fc, substractWindowMean, dispProgress=dispProgress)    # covariances matrix
        is_sym_pos_def(ccov)
        Cov[bId] = ccov
    return  [Cov,cov_events]


def proc_pos2win(POS, wshift, direction, wlength=-1):
    backward = False
    if direction=='forward':
        wlength = []
    elif direction=='backward':
        backward = True
        if wlength == -1:
            raise Exception("chk:arg', 'backward direction option requires to provide wlength")
    else:
        raise Exception('chk:arg', 'Direction not recognized: only forward and backward are allowed')
    wPOS = np.floor(POS/wshift) + 1
    if backward == True:
        wPOS = wPOS - np.floor(wlength/wshift)
    wPOS = [ int(x) for x in wPOS ]
    return wPOS


def get_sliding_covariance_trace_normalized(data, wlenght, wshift, substractWindowMean=True, dispProgress=True):
    nsamples, nchannels = data.shape
    
    wstart = np.arange(0, nsamples - wlenght + 1, wshift)
    wstop = wstart + wlenght
    
    nwins = len(wstart)
    
    C = np.empty((nwins, nchannels, nchannels))
    for wId in tqdm (range (nwins), bar_format='{l_bar}{bar:40}{r_bar}', disable=not dispProgress):        
        cstart = int(wstart[wId])
        cstop = int(wstop[wId])
        t_data = data[cstart:cstop, :]
        C[wId] = get_covariance_matrix_traceNorm(t_data, substractWindowMean)  
    return C


def matrix_to_maxRank(matrix):
    nchannels = matrix.shape[-1]
    data = matrix - np.mean(matrix, axis=0)
    actual_rank = getrank(data)
    if actual_rank < nchannels:
        for n_components in range(nchannels-1,int(nchannels/2),-1):
            pca = PCA(n_components=n_components)
            data_reduced = pca.fit_transform(data)
            data_reconstructed = pca.inverse_transform(data_reduced)
            new_rank = getrank(data_reconstructed)
            if new_rank == nchannels:
                print('N_components: ' + str(n_components))
                break         


def getrank(data):
    tmprank = np.linalg.matrix_rank(data)
    covarianceMatrix = np.cov(data.T)
    D, E = np.linalg.eig(covarianceMatrix)
    rankTolerance = 1e-7
    tmprank2 = np.sum(D > rankTolerance)
    if tmprank != tmprank2:
        # print(f'Warning: fixing rank computation inconsistency ({tmprank} vs {tmprank2})')
        tmprank2 = min(tmprank, tmprank2)
    return tmprank2


def get_covariance_matrix_traceNorm(data, substractWindowMean=True):
    if substractWindowMean:
        data -= np.mean(data, axis=0)
    t_cov = data.T @ data
    cov =  t_cov  / np.trace(t_cov)
    return cov

# def get_vector_onEvent(df_events, lengthVector, events, clmn_name='typ', onEvent=781):    
#     vector = np.full(lengthVector, np.nan)
#     idx_onEvent = df_events[df_events[clmn_name] == onEvent].index
#     start = df_events.loc[idx_onEvent, 'pos'].values
#     duration = df_events.loc[idx_onEvent, 'dur'].values
#     ev_idx = []
#     for ev in events:
#         idx = df_events[df_events[clmn_name] == ev].index
#         ev_idx += idx.tolist()
#     ev_idx.sort()
#     ev_type = df_events.loc[ev_idx, clmn_name].values
#     if len(ev_idx) != len(idx_onEvent):
#         result1 = np.setdiff1d(ev_idx, idx_onEvent)
#         print(ev_idx )
#         print(idx_onEvent)
#         print(result1)
#     # for [t_start, t_duration, t_ev_type] in zip(start, duration, ev_type):
#     #     vector[t_start:(t_start + t_duration)] = t_ev_type
#     return None#vector

def get_EventsVector_onFeedback(events, lengthVector,  events_typ, column_name='typ', use_eog=True):    
    vector = np.full(lengthVector, np.nan)
    idx_events = np.array(events[np.isin(events['typ'], events_typ)].index)
    ev_type = events.loc[idx_events, column_name].values

    is_feedback = events.loc[idx_events+1, 'typ'] == 781
    is_feedback = is_feedback.values
    idx_events[is_feedback] += 1
    start = events.loc[idx_events, 'pos'].values
    duration = events.loc[idx_events, 'dur'].values

    for [t_start, t_duration, t_ev_type] in zip(start, duration, ev_type):
        vector[t_start:(t_start + t_duration)] = t_ev_type
    if use_eog:
        idx = events[events['typ'] == 1024].index
        start = events.loc[idx, 'pos'].values
        duration = events.loc[idx, 'dur'].values
        ev_type = events.loc[idx, column_name].values
        for [t_start, t_duration, t_ev_type] in zip(start, duration, ev_type):
            vector[t_start:(t_start + t_duration)] = np.nan
    return vector

def get_vector_onEvent(df_events, lengthVector, clmn_name):
    vector = np.full(lengthVector, np.nan)
    for [t_start, t_duration, t_ev_type] in zip(df_events['pos'], df_events['dur'], df_events[clmn_name]):
        vector[t_start:(t_start + t_duration)] = t_ev_type
    return vector


def get_indices_train_validation(events, classes, percVal = 0.2):
    idx_train = []
    idx_val = []
    for cl in classes:
        info_cues = events.loc[events['typ']==cl]
        train, val = train_test_split(info_cues, test_size=percVal)
        for idx_cue in train.index:
            feedback = events.loc[idx_cue+1]
            idx_train += list(( feedback['pos'] + range(feedback['dur'])) )
        for idx_cue in val.index:
            feedback = events.loc[idx_cue+1]
            idx_val += list(( feedback['pos'] + range(feedback['dur'])) )
    idx_train.sort()
    idx_val.sort()  
    return [idx_train, idx_val]


def get_riemann_mean_covariance(cov, cueOnFeedbackVector=[], n_iter_max = 300, print_print=True, show_progess=True):
    if len(cueOnFeedbackVector)==0:
        cueOnFeedbackVector = np.full((cov.shape[1], 1), True)

    bool_fdbk = ~np.isnan(cueOnFeedbackVector)
    idx_fdbk = np.where(bool_fdbk)[0]

    if print_print:
        print(' - Extracting mean covariance matrix')

    n_bandranges, _, nchannels, _ = cov.shape
    mean_cov = np.empty((n_bandranges, nchannels, nchannels))
    for bId in range(mean_cov.shape[0]):        
        iter_max = min(int(np.floor(np.sum(idx_fdbk) / 2)), n_iter_max)
        t_ref = mean_riemann(cov[bId, idx_fdbk], maxiter=iter_max, show_progess=show_progess)
        is_sym_pos_def(t_ref)

        mean_cov[bId, :, :] = t_ref

    return mean_cov, idx_fdbk


def center_covariances(covariances, reference_matrix, inv_sqrt_ref=[]):

    cov_centered = np.empty(covariances.shape)

    if len(inv_sqrt_ref)==0:
        inv_sqrt_ref = invsqrtm(reference_matrix)

    cov_centered = inv_sqrt_ref @ covariances @ inv_sqrt_ref
    return cov_centered


def apply_ROI_over_channels(data, channels, channelGroups, returnMean=True):
    if len(channelGroups)==0:
        return data
    if returnMean:
        newData = np.empty((data.shape[0], len(channelGroups)))
    else:
        newData = np.empty(len(channelGroups), dtype=object)
    for nchs,chs in enumerate(channelGroups):
        _,idx,_ = np.intersect1d(channels, chs, return_indices=True)
        if returnMean:
            newData[:,nchs] = np.mean(data[:,idx], axis=1)
        else:
            newData[nchs] = data[:,idx]
    return newData


def get_nd_position(data, n_components=3, suppress_output=False):
    pca = PCA(n_components=n_components)
    projected_data = np.empty((data.shape[0], data.shape[1], n_components))
    expl_var= np.empty((data.shape[0]))
    for bId in range(data.shape[0]):
        projected_data[bId] = pca.fit_transform(data[bId])
        expl_var[bId] = sum(pca.explained_variance_ratio_)
        if not suppress_output:
            print('Explained variance of PCA components : ' + str(expl_var[bId] ))
    return projected_data,expl_var


def fromEvents2vector(length_vector, events_name, events):
    vector = np.full(length_vector, np.nan)
    for idx in range(len(events[events_name])):
        start = events.loc[idx, 'pos']
        duration = events.loc[idx, 'dur']
        vector[start:(start + duration)] = events.loc[idx, events_name]
    return vector

def get_trials_gradient(indexes, length_max):
    vector = np.empty((length_max))
    vector.fill(np.nan)

    count = 0
    for n_idx, idx in enumerate(indexes):
        if abs(idx-indexes[n_idx-1])>1:
            count = 0
        vector[idx] = count
        count += 1
    return vector

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def load(filename):
    return joblib.load(filename)

def save(filename, variable):
    joblib.dump(variable, filename)


def read_gdf(spath):
    raw = mne.io.read_raw_gdf(spath,verbose='error')
    # raw = mne.io.read_raw_gdf(spath)
    events,names = mne.events_from_annotations(raw,verbose='error')
    # events,names = mne.events_from_annotations(raw)
    names = {v:int(k) for k,v in names.items()}
    events_pos = events[:,0]
    events_typ = events[:,2]
    events_typ = [names[e] for  e in events_typ]
    eeg = raw.get_data().T
    header = {'SampleRate':raw.info['sfreq'],
                'EVENT':{'POS':np.array(events_pos),'TYP':np.array(events_typ)}
                }
    #print((np.array(events_typ)==781).sum(),'events found')
    return eeg,header

def get_canonical_bases(vector):
    n = len(vector)
    basis = np.eye(n)
    return basis


# # ---------------------- ONLINE ----------------------

def get_covariance_matrix_traceNorm_online(data):
    data -= np.mean(data, axis=(0,2), keepdims=True)
    cov = data.transpose((0,2,1)) @ data
    cov =  cov  / np.trace(cov, axis1=1, axis2=2).reshape(-1,1,1)
    cov = np.expand_dims(cov, axis=1)
    return cov

def center_covariance_online(covariance, inv_sqrt_mean_cov):
    cov_centered = inv_sqrt_mean_cov @ covariance @ inv_sqrt_mean_cov
    is_sym_pos_def(cov_centered)
    return cov_centered

