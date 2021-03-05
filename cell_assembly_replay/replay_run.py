import matplotlib.pyplot as plt
import numpy as np
import struct

import nelpy as nel
import nelpy.io

import os
import sys
from cell_assembly_replay import functions

import pandas as pd
import itertools
import statistics 
import math
from scipy import stats
from nelpy.analysis import replay

import multiprocessing
from joblib import Parallel, delayed

import statsmodels.api as sm
import pickle


def rescale(x,new_min,new_max):
    return ((x - min(x)) / (max(x) - min(x))) * ((new_max-new_min) + new_min)

def rescale_coords(df,session_epochs,maze_size_cm):
    for i,val in enumerate(session_epochs.data):
        temp_df = df[df['ts'].between(val[0],val[1])]
        
        x_range = max(temp_df.x) - min(temp_df.x)
        y_range = max(temp_df.y) - min(temp_df.y)
        x_y_ratio = x_range/y_range
        # if the ratio of x to y is > 5, it is probably a linear track
        if x_y_ratio > 5:
            df.loc[df['ts'].between(val[0],val[1]),'x'] = rescale(temp_df.x,0,maze_size_cm[i])
            df.loc[df['ts'].between(val[0],val[1]),'y'] = rescale(temp_df.y,0,maze_size_cm[i]/x_y_ratio)
        else:
            df.loc[df['ts'].between(val[0],val[1]),'x'] = rescale(temp_df.x,0,maze_size_cm[i])
            df.loc[df['ts'].between(val[0],val[1]),'y'] = rescale(temp_df.y,0,maze_size_cm[i])
    return df

def get_base_data(data_path,spike_path,session):
     # get data session path from mat file
    path = functions.get_session_path(os.path.join(data_path,session)+'.mat')

    # load position data from .mat file
    df = functions.load_position(os.path.join(data_path,session)+'.mat')
    # get the size of each maze
    maze_size_cm = functions.get_maze_size_cm(os.path.join(data_path,session)+'.mat')
    # get session epochs
    session_epochs = nel.EpochArray(functions.get_epochs(os.path.join(data_path,session)+'.mat'))
    # rescale epoch coordinates into cm
    df = rescale_coords(df,session_epochs,maze_size_cm)
    # put position into object
    pos = nel.AnalogSignalArray(timestamps=df.ts,
                            data=[df.x],
                            fs=1/statistics.mode(np.diff(df.ts)),
                            support=(session_epochs))

    # load spikes & add to object
    spikes = np.load(os.path.join(spike_path,session)+'.npy',allow_pickle=True)
    spikes_ = list(itertools.chain(*spikes))
    session_bounds = nel.EpochArray([min(spikes_),max(spikes_)])
    st = nel.SpikeTrainArray(timestamps=spikes,support=session_bounds, fs=32000)
    
    return maze_size_cm,pos,st

def score_array(posterior):
    nan_loc = np.isnan(posterior).any(axis=0)

    rows, cols = posterior.shape

    x = np.arange(cols)
    y = posterior.argmax(axis=0)
    w = posterior.max(axis=0)

    x = x[~nan_loc]
    y = y[~nan_loc]
    w = w[~nan_loc]
    
    X = sm.add_constant(x)
    wls_model = sm.WLS(y,X, weights=w)
    results = wls_model.fit()
    
    slope = results.params[1]
    intercept = results.params[0]
    log_like = wls_model.loglike(results.params)

    return results.rsquared,slope,intercept,log_like

def get_features(bst_placecells, posteriors, bdries, mode_pth, pos, ep_type,figs=False):

    traj_dist = []
    traj_speed = []
    traj_step = []
    replay_type = []
    dist_rat_start = []
    dist_rat_end = []
    for idx in range(bst_placecells.n_epochs):
        posterior_array = posteriors[:, bdries[idx]:bdries[idx+1]]

        nan_loc = np.isnan(posterior_array).any(axis=0)

        x = bst_placecells[idx].bin_centers
        # x = np.linspace(0,cols*.02,cols)
        y = mode_pth[bdries[idx]:bdries[idx+1]]

        # get spatial difference between bins
        dy = np.abs(np.diff(y));
        # get cumulative distance 
        traj_dist.append(np.nansum(dy))
        #  calculate avg speed of trajectory (dist(cm) / time(sec))
        traj_speed.append(np.nansum(dy) / (np.nanmax(x) - np.nanmin(x)))
        #  get mean step size 
        traj_step.append(np.nanmean(dy))

        rsquared,slope,intercept,log_like = score_array(posterior_array)
        
        rat_event_pos = np.interp(x,pos.abscissa_vals,pos.data[0])
        rat_x_position = np.nanmean(rat_event_pos)
        
        dist_rat_start.append(np.abs(rat_x_position - y[0]))
        dist_rat_end.append(np.abs(rat_x_position - y[-1]))

        if ep_type[idx] != "track":
            replay_type.append(np.nan)
            continue
            
        # what side of the track is the rat on ? 
        side = np.argmin(np.abs([0,120] - rat_x_position))
        if (side == 1) & (slope < 0):
            replay_type.append('forward')
        elif (side == 1) & (slope > 0):
            replay_type.append('reverse')
        elif (side == 0) & (slope < 0):
            replay_type.append('reverse')
        elif (side == 0) & (slope > 0):
            replay_type.append('forward')
           
        if figs:
            fig = plt.figure(figsize=(4,3))
            ax = plt.gca()
    #         npl.plot(pos,ax=ax)
            npl.plot(x,rat_event_pos,"^",color='brown',linewidth=10,ax=ax)
            ax.plot(x,y,'k',linewidth=2)
    #         ax.scatter(x,y,c=x,linewidth=2,cmap='viridis')
            ax.scatter(x[0],y[0],color='g')
            ax.scatter(x[-1],y[-1],color='r')
            ax.set_title(replay_type[idx])

    return traj_dist,traj_speed,traj_step,replay_type,dist_rat_start,dist_rat_end

def run_all(session,data_path,spike_path,save_path,mua_df,df_cell_class):
    
    maze_size_cm,pos,st = get_base_data(data_path,spike_path,session)

    # to make everything more simple, lets restrict to just the linear track
    pos = pos[0]
    st = st[0]
    maze_size_cm = maze_size_cm[0]

    # compute and smooth speed
    speed1 = nel.utils.ddt_asa(pos, smooth=True, sigma=0.1, norm=True)

    # find epochs where the animal ran > 4cm/sec
    run_epochs = nel.utils.get_run_epochs(speed1, v1=4, v2=4)

    st_run = st[run_epochs] # restrict spike trains to those epochs during which the animal was running
    ds_run = 0.5 
    ds_50ms = 0.05
    # smooth and re-bin:
    #     sigma = 0.3 # 300 ms spike smoothing
    bst_run = st_run.bin(ds=ds_50ms).smooth(sigma=0.3 , inplace=True).rebin(w=ds_run/ds_50ms)

    sigma = 0.2 # smoothing std dev in cm
    tc = nel.TuningCurve1D(bst=bst_run, extern=pos, n_extern=40, extmin=0, extmax=maze_size_cm, sigma=sigma, min_duration=1)

    # locate pyr cells that have at least 100 spikes and a peak rate at least 1 Hz
    temp_df = df_cell_class[df_cell_class.session == session]
    unit_ids_to_keep = (np.where(((temp_df.cell_type == "pyr")) &
                                 (temp_df.n_spikes >=100) &
                                 (tc.ratemap.max(axis=1) >=1) )[0]+1).squeeze().tolist()

    sta_placecells = st._unit_subset(unit_ids_to_keep)
    tc = tc._unit_subset(unit_ids_to_keep)
    # tc.reorder_units(inplace=True)
    
    # access decoding accuracy on behavioral time scale 
    posteriors, lengths, mode_pth, mean_pth = nel.decoding.decode1D(bst_run.loc[:,unit_ids_to_keep], tc, xmin=0, xmax=maze_size_cm)
    actual_pos = pos(bst_run.bin_centers)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(actual_pos, mode_pth)
    mean_error = np.mean(np.abs(actual_pos - mode_pth))
    
    # create intervals for PBEs epochs
    # first restrict to current session and to track + pre/post intervals
    temp_df = mua_df[(mua_df.session == session) & ((mua_df.ep_type == "pedestal_1") | (mua_df.ep_type == "track") | (mua_df.ep_type == "pedestal_2"))]
    # restrict to events at least 100ms
    temp_df = temp_df[temp_df.ripple_duration >= 0.1]
    # make epoch object
    PBEs = nel.EpochArray([np.array([temp_df.start_time,temp_df.end_time]).T])
    
    # bin data into 20ms 
    bst_placecells = sta_placecells[PBEs].bin(ds=0.02)
    # decode each event
    posteriors, bdries, mode_pth, mean_pth = nel.decoding.decode1D(bst_placecells, tc, xmin=0, xmax=maze_size_cm)
    # score each event
    scores, scores_shuffled, percentile = replay.score_Davidson_final_bst_fast(bst_placecells,tc,w=0,n_shuffles=500,n_samples=1000)
    # find sig events
    sig_event_idx,pvalues = replay.get_significant_events(scores, scores_shuffled.T, q=95)

    traj_dist,traj_speed,traj_step,replay_type,dist_rat_start,dist_rat_end = get_features(bst_placecells, posteriors, bdries, mode_pth, pos, list(temp_df.ep_type))
    
    # package data
    results = {}
    results['df'] = temp_df
    results['scores'] = scores
    results['sig_event_idx'] = sig_event_idx
    results['pvalues'] = pvalues
    results['traj_dist'] = traj_dist
    results['traj_speed'] = traj_speed
    results['traj_step'] = traj_step
    results['replay_type'] = replay_type
    results['dist_rat_start'] = dist_rat_start
    results['dist_rat_end'] = dist_rat_end
    results['decoding_r2'] = rvalue
    results['decoding_mean_error'] = mean_error

    return results

def main_loop(session,data_path,spike_path,save_path,mua_df,df_cell_class):
    '''
    main_loop: file management 
    '''
    
    base = os.path.basename(session)
    os.path.splitext(base)
    save_file = save_path + os.path.splitext(base)[0] + '.pkl'
    
    # check if saved file exists
    if os.path.exists(save_file):
        return
        
    # calc some features
    results = run_all(session,data_path,spike_path,save_path,mua_df,df_cell_class)
    # save file
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
        
def replay_run(data_path,spike_path,save_path,mua_df,df_cell_class,parallel=True):
    # find sessions to run
    sessions = pd.unique(mua_df.session)

    if parallel:
        num_cores = multiprocessing.cpu_count()         
        processed_list = Parallel(n_jobs=num_cores)(delayed(main_loop)(session,data_path,spike_path,save_path,mua_df,df_cell_class) for session in sessions)
    else:    
        for session in sessions:
            print(session)
            main_loop(session,data_path,spike_path,save_path,mua_df,df_cell_class)