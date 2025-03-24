#!/usr/bin/env python

import numpy as np
import matplotlib as mpl

from pyriemann.utils.test import is_sym_pos_def
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm

from sklearn.decomposition import PCA
from tqdm import tqdm



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

