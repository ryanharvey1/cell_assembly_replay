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
from nelpy.decoding import get_mode_pth_from_array

import multiprocessing
from joblib import Parallel, delayed

import statsmodels.api as sm
import pickle
import copy
from sklearn.isotonic import IsotonicRegression
from scipy.special import cotdg
from skimage.transform import radon
from scipy.stats import multivariate_normal, rv_histogram
import warnings

def rescale(x,new_min,new_max):
    """
    simple function to rescale vector x by new min and max
    """
    return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))) * ((new_max-new_min) + new_min)

def rescale_coords(df,session_epochs,maze_size_cm):
    """
    rescale xy coordinates of each epoch into cm
    note: automatically detects linear track by x to y ratio
    
    input:
        df: [ts,x,y] pandas data frame
        session_epochs: nelpy epoch class with epoch times
        mazesize: list with size of maze in cm for each epoch
    output:
        df: rescaled df
    """
    for i,val in enumerate(session_epochs.data):
        temp_df = df[df['ts'].between(val[0],val[1])]
        
        x_range = np.nanmax(temp_df.x) - np.nanmin(temp_df.x)
        y_range = np.nanmax(temp_df.y) - np.nanmin(temp_df.y)
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
    """
    Load and format data for replay analysis
    """
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

def decode_and_score(bst, tc, pos):
    # access decoding accuracy on behavioral time scale 
    posteriors, lengths, mode_pth, mean_pth = nel.decoding.decode1D(bst,
                                                                    tc,
                                                                    xmin=0,
                                                                    xmax=np.nanmax(pos.data))
    actual_pos = pos(bst.bin_centers)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(actual_pos, mode_pth)
    median_error = np.nanmedian(np.abs(actual_pos - mode_pth))
    
    return rvalue,median_error

def pooled_incoherent_shuffle_bst(bst):
    out = copy.deepcopy(bst)
    data = out._data

    for uu in range(bst.n_units):
        segment = np.atleast_1d(np.squeeze(data[uu, :]))
        segment = np.roll(segment, np.random.randint(len(segment)))
        data[uu, :] = segment
    return out

def decode_and_shuff(bst, tc, pos, n_shuffles=500):
    """
    """
    
    rvalue, median_error = decode_and_score(bst, tc, pos)

    scores = np.zeros(bst.n_epochs)
    if n_shuffles > 0:
        rvalue_time_swap = np.zeros((n_shuffles,1))
        median_error_time_swap = np.zeros((n_shuffles,1))
        
    for shflidx in range(n_shuffles):
        bst_shuff = pooled_incoherent_shuffle_bst(bst)
        rvalue_time_swap[shflidx], median_error_time_swap[shflidx] = decode_and_score(bst_shuff, tc, pos)
            
    return rvalue, median_error, rvalue_time_swap, median_error_time_swap

def _m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def _cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - _m(x, w)) * (y - _m(y, w))) / np.sum(w)


def _corr(x, y, w):
    """Weighted Correlation"""
    return _cov(x, y, w) / np.sqrt(_cov(x, x, w) * _cov(y, y, w))

def weighted_correlation(posterior, time, place_bin_centers):
    """ From Eric Denovellis """
    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    return _corr(time[:, np.newaxis],
                 place_bin_centers[np.newaxis, :], posterior)

def convert_polar_to_slope_intercept(
    n_pixels_from_center, projection_angle, center_pixel
):
    """ From Eric Denovellis """
    velocity = -cotdg(-projection_angle)
    start_position = (
        n_pixels_from_center / np.sin(-np.deg2rad(projection_angle))
        - velocity * center_pixel[0]
        + center_pixel[1]
    )
    return start_position, velocity

def detect_line_with_radon(
    posterior,
    dt,  # s
    dp,  # cm
    projection_angles=np.arange(-90, 90, 0.5),  # degrees
    filter_invalid_positions=True,
    incorporate_nearby_positions=True,
    nearby_positions_max=15,  # cm
):

    if incorporate_nearby_positions:
        n_nearby_bins = int(nearby_positions_max / 2 // dp)
        filt = np.ones(2 * n_nearby_bins + 1)
        posterior = np.apply_along_axis(
            lambda time_bin: np.convolve(time_bin, filt, mode="same"),
            axis=1, arr=posterior
        )
    else:
        n_nearby_bins = 1
    # Sinogram is shape (pixels_from_center, projection_angles)
    sinogram = radon(
        posterior.T, theta=projection_angles, circle=False,
        preserve_range=False
    )
    n_time, n_position_bins = posterior.shape
    center_pixel = np.asarray((n_time // 2, n_position_bins // 2))
    pixels_from_center = np.arange(
        -sinogram.shape[0] // 2 + 1, sinogram.shape[0] // 2 + 1)

    if filter_invalid_positions:
        start_positions, velocities = convert_polar_to_slope_intercept(
            pixels_from_center[:, np.newaxis],
            projection_angles[np.newaxis, :],
            center_pixel,
        )
        end_positions = start_positions + velocities * (n_time - 1)
        sinogram[(start_positions < 0) |
                 (start_positions > n_position_bins - 1)] = 0.0
        sinogram[(end_positions < 0) |
                 (end_positions > n_position_bins - 1)] = 0.0
        sinogram[:, np.isinf(velocities.squeeze())] = 0.0

    # Find the maximum of the sinogram
    n_pixels_from_center_ind, projection_angle_ind = np.unravel_index(
        indices=np.argmax(sinogram), shape=sinogram.shape
    )
    projection_angle = projection_angles[projection_angle_ind]
    n_pixels_from_center = pixels_from_center[n_pixels_from_center_ind]

    # Normalized score based on the integrated projection
    score = np.max(sinogram) / (n_time * n_nearby_bins)

    # Convert from polar form to slope-intercept form
    start_position, velocity = convert_polar_to_slope_intercept(
        n_pixels_from_center, projection_angle, center_pixel
    )

    # Convert from pixels to position units
    start_position *= dp
    velocity *= dp / dt

    # Estimate position for the posterior
    time = np.arange(n_time) * dt
    radon_position = start_position + velocity * time

    return start_position, velocity, radon_position, score

def score_array(posterior):
    """
    takes in posterior matrix (distance by time) and conducts
    weighted least squares
    """
    nan_loc = np.isnan(posterior).any(axis=0)

    rows, cols = posterior.shape

    x = np.arange(cols)
    y = posterior.argmax(axis=0)
    w = posterior.max(axis=0)

    x = x[~nan_loc]
    y = y[~nan_loc]
    w = w[~nan_loc]
    
    # if only one time bin is active
    if len(x)==1:
        return np.nan,np.nan,np.nan,np.nan
    
    X = sm.add_constant(x)
    wls_model = sm.WLS(y,X, weights=w)
    results = wls_model.fit()
    
    slope = results.params[1]
    intercept = results.params[0]
    log_like = wls_model.loglike(results.params)

    return results.rsquared,slope,intercept,log_like

def get_score_coef(bst,bdries,posterior):
    """
    runs score_array on each event epoch in bst (binned spike train)
    """
    scores = np.zeros(bst.n_epochs)
    slope = np.zeros(bst.n_epochs)
    intercept = np.zeros(bst.n_epochs)
    log_like = np.zeros(bst.n_epochs)

    for idx in range(bst.n_epochs):
        posterior_array = posterior[:, bdries[idx]:bdries[idx+1]]
        scores[idx],slope[idx],intercept[idx],log_like[idx] = score_array(posterior_array)
    return scores,slope,intercept,log_like

# def get_scores(bst, posterior, bdries, n_shuffles=500,dt=0.02,dp=3,max_position=120):
#     """
#     runs score_array on observed data and then conducts a shuffle analysis using
#     two types of procedures (time swap and column cycle).
    
#     Will run through each epoch in your binned spike train
#     """
# #     posterior, bdries, mode_pth, mean_pth = nel.decoding.decode1D(bst, tuningcurve, xmin=0, xmax=120)

#     place_bin_edges = np.arange(0, max_position + dp, dp)
#     place_bin_centers = place_bin_edges[:-1] + np.diff(place_bin_edges) / 2
        
#     wls_scores = np.zeros(bst.n_epochs)
#     radon_scores = np.zeros(bst.n_epochs)
#     w_corr_scores = np.zeros(bst.n_epochs)
    
#     if n_shuffles > 0:
#         wls_time_swap = np.zeros((n_shuffles, bst.n_epochs))
#         wls_col_cycle = np.zeros((n_shuffles, bst.n_epochs))
#         radon_time_swap = np.zeros((n_shuffles, bst.n_epochs))
#         radon_col_cycle = np.zeros((n_shuffles, bst.n_epochs))
#         w_corr_time_swap = np.zeros((n_shuffles, bst.n_epochs))
#         w_corr_col_cycle = np.zeros((n_shuffles, bst.n_epochs))
        
#     for idx in range(bst.n_epochs):
#         posterior_array = posterior[:, bdries[idx]:bdries[idx+1]]
#         wls_scores[idx],_,_,_ = score_array(posterior_array)
#         _, _, _, radon_scores[idx] = detect_line_with_radon(posterior_array.T,dt,dp)
#         time = bst[idx].bin_centers
#         w_corr_scores[idx] = weighted_correlation(posterior_array.T, time, place_bin_centers)

#         for shflidx in range(n_shuffles):
#             posterior_ts = replay.time_swap_array(posterior_array)
#             posterior_cs = replay.column_cycle_array(posterior_array)
            
#             wls_time_swap[shflidx, idx],_,_,_  = score_array(posterior=posterior_ts)
#             wls_col_cycle[shflidx, idx],_,_,_  = score_array(posterior=posterior_cs)
            
#             _, _, _, radon_time_swap[shflidx, idx] = detect_line_with_radon(posterior_ts.T,dt,dp)
#             _, _, _, radon_col_cycle[shflidx, idx] = detect_line_with_radon(posterior_cs.T,dt,dp)
            
#             w_corr_time_swap[shflidx, idx] = weighted_correlation(posterior_ts.T, time, place_bin_centers)
#             w_corr_col_cycle[shflidx, idx] = weighted_correlation(posterior_cs.T, time, place_bin_centers)

#     return (
#         wls_scores,
#         radon_scores,
#         w_corr_scores,
#         wls_time_swap,
#         wls_col_cycle,
#         radon_time_swap,
#         radon_col_cycle,
#         w_corr_time_swap,
#         w_corr_col_cycle
#     )

def get_significant_events(scores, shuffled_scores, q=95):
    """Return the significant events based on percentiles.
    NOTE: The score is compared to the distribution of scores obtained
    using the randomized data and a Monte Carlo p-value can be computed
    according to: p = (r+1)/(n+1), where r is the number of
    randomizations resulting in a score higher than (ETIENNE EDIT: OR EQUAL TO?)
    the real score and n is the total number of randomizations performed.
    Parameters
    ----------
    scores : array of shape (n_events,)
    shuffled_scores : array of shape (n_shuffles, n_events)
    q : float in range of [0,100]
        Percentile to compute, which must be between 0 and 100 inclusive.
    Returns
    -------
    sig_event_idx : array of shape (n_sig_events,)
        Indices (from 0 to n_events-1) of significant events.
    pvalues :
    """

    n, _ = shuffled_scores.shape
    r = np.sum(abs(shuffled_scores) >= abs(scores), axis=0)
    pvalues = (r+1)/(n+1)

    # set nan scores to 1
    pvalues[np.isnan(scores)] = 1
    
    sig_event_idx = np.argwhere(scores > np.percentile(
        shuffled_scores,
        axis=0,
        q=q)).squeeze()

    return np.atleast_1d(sig_event_idx), np.atleast_1d(pvalues)

def shuff(posterior_array,time,place_bin_centers,dt,dp):
    
    posterior_ts = replay.time_swap_array(posterior_array)
    posterior_cs = replay.column_cycle_array(posterior_array)

#     radon_time_swap = detect_line_with_radon(posterior_ts,
#                                             dt,
#                                             dp,
#                                             incorporate_nearby_positions=False)
    
#     radon_col_cycle = detect_line_with_radon(posterior_cs,
#                                             dt,
#                                             dp,
#                                             incorporate_nearby_positions=False)

    w_corr_time_swap = weighted_correlation(posterior_ts, time, place_bin_centers)
    w_corr_col_cycle = weighted_correlation(posterior_cs, time, place_bin_centers)
    
    return w_corr_time_swap,w_corr_col_cycle
#     return radon_time_swap[-1],radon_col_cycle[-1],w_corr_time_swap,w_corr_col_cycle

def get_scores(bst, posterior, bdries, n_shuffles=1000,dt=0.02,dp=3,max_position=120,verbose=False):
    """
    runs score_array on observed data and then conducts a shuffle analysis using
    two types of procedures (time swap and column cycle).
    
    Will run through each epoch in your binned spike train
    """

    place_bin_edges = np.arange(0, max_position + dp, dp)
    place_bin_centers = place_bin_edges[:-1] + np.diff(place_bin_edges) / 2
        
    radon_scores = np.zeros(bst.n_epochs)
    w_corr_scores = np.zeros(bst.n_epochs)
    radon_pval_time_swap = np.zeros(bst.n_epochs)
    radon_pval_col_cycle = np.zeros(bst.n_epochs)
    w_corr_pval_time_swap = np.zeros(bst.n_epochs)
    w_corr_pval_col_cycle = np.zeros(bst.n_epochs) 
    
    # get n cores to know how many jobs to run  
    num_cores = multiprocessing.cpu_count()         
    
    for idx in range(bst.n_epochs):
        if verbose:
            print('event: ',str(idx))
            
        posterior_array = posterior[:, bdries[idx]:bdries[idx+1]]
#         _, _, _, radon_scores[idx] = detect_line_with_radon(posterior_array.T,
#                                                             dt,
#                                                             dp,
#                                                             incorporate_nearby_positions=False)
        time = bst[idx].bin_centers
        w_corr_scores[idx] = weighted_correlation(posterior_array.T, time, place_bin_centers)
            
        (
            w_corr_time_swap,
            w_corr_col_cycle
        ) = zip(*Parallel(n_jobs=num_cores)(delayed(shuff)(posterior_array.T,
                                                           time,
                                                           place_bin_centers,
                                                           dt,
                                                           dp) for i in range(n_shuffles)))
        
#         _,radon_pval_time_swap[idx] = get_significant_events(radon_scores[idx], np.expand_dims(radon_time_swap, axis=1))
#         _,radon_pval_col_cycle[idx] = get_significant_events(radon_scores[idx], np.expand_dims(radon_col_cycle, axis=1))
        _,w_corr_pval_time_swap[idx] = get_significant_events(w_corr_scores[idx], np.expand_dims(w_corr_time_swap, axis=1))
        _,w_corr_pval_col_cycle[idx] = get_significant_events(w_corr_scores[idx], np.expand_dims(w_corr_col_cycle, axis=1))
        
    return (
        w_corr_scores,
        w_corr_pval_time_swap,
        w_corr_pval_col_cycle
    )

def map_estimate(posterior, place_bin_centers):
    """ From Eric Denovellis """
    posterior[np.isnan(posterior)] = 0.0
    return place_bin_centers[posterior.argmax(axis=1)].squeeze()

def isotonic_regression(posterior, time, place_bin_centers):
    """ From Eric Denovellis """
    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    map_ = map_estimate(posterior, place_bin_centers)
    map_probabilities = np.max(posterior, axis=1)

    regression = IsotonicRegression(increasing='auto').fit(
        X=time,
        y=map_,
        sample_weight=map_probabilities,
    )

    prediction = regression.predict(time)

    return prediction

def get_features(bst_placecells,
                 posteriors,
                 bdries,
                 mode_pth,
                 pos,
                 ep_type,
                 figs=False,
                 max_position=120,
                 dt=0.02,
                 dp=3):
    """
    Using the posterior probability matrix, calculate several features on spatial trajectory
    and detects if the trajectory is foward or reverse depending on the rat's current position
    """
    
    place_bin_edges = np.arange(0, max_position + dp, dp)
    place_bin_centers = place_bin_edges[:-1] + np.diff(place_bin_edges) / 2
    
    traj_dist = []
    traj_speed = []
    traj_step = []
    replay_type = []
    dist_rat_start = []
    dist_rat_end = []
    position = []

    for idx in range(bst_placecells.n_epochs):
        posterior_array = posteriors[:, bdries[idx]:bdries[idx+1]]

        x = bst_placecells[idx].bin_centers

#         y = get_mode_pth_from_array(posterior_array,tuningcurve=tc)
        y = mode_pth[bdries[idx]:bdries[idx+1]]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        velocity, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
        y = x*velocity+intercept
#         slope, intercept, rsquared = linregress_array(posterior=posterior)

        position.append(y)

        # get spatial difference between bins
        dy = np.abs(np.diff(y))
        # get cumulative distance 
        traj_dist.append(np.nansum(dy))
        # calculate avg speed of trajectory (dist(cm) / time(sec))
        traj_speed.append(np.nansum(dy) / (np.nanmax(x) - np.nanmin(x)))
        # get mean step size 
        traj_step.append(np.nanmean(dy))
        
        # get slope to see the direction of replay
#         _,slope,_,_ = score_array(posterior_array)
        
        rat_event_pos = np.interp(x,pos.abscissa_vals,pos.data[0])
        rat_x_position = np.nanmean(rat_event_pos)
        
        # get dist of the start & end of trajectory to rat
        dist_rat_start.append(rat_x_position - y[0])
        dist_rat_end.append(rat_x_position - y[-1])

        if ep_type[idx] != "track":
            replay_type.append(np.nan)
        else:
            # what side of the track is the rat on ? 
            side = np.argmin(np.abs([0,120] - rat_x_position))
            if (side == 1) & (velocity < 0):
                replay_type.append('forward')
            elif (side == 1) & (velocity > 0):
                replay_type.append('reverse')
            elif (side == 0) & (velocity < 0):
                replay_type.append('reverse')
            elif (side == 0) & (velocity > 0):
                replay_type.append('forward')
            else:
                replay_type.append(np.nan)
        if figs:
            fig = plt.figure(figsize=(4,3))
            ax = plt.gca()
            npl.plot(x,rat_event_pos,"^",color='brown',linewidth=10,ax=ax)
            ax.plot(x,y,'k',linewidth=2)
            ax.scatter(x[0],y[0],color='g')
            ax.scatter(x[-1],y[-1],color='r')
            ax.set_title(replay_type[idx])

    return traj_dist,traj_speed,traj_step,replay_type,dist_rat_start,dist_rat_end,position

def run_all(session,data_path,spike_path,save_path,mua_df,df_cell_class,verbose=False):
    """
    Main function that conducts the replay analysis
    """
    if verbose:
        print('loading data')
    maze_size_cm,pos,st_all = get_base_data(data_path,spike_path,session)

    # to make everything more simple, lets restrict to just the linear track
    pos = pos[0]
    st_all = st_all[0]
    maze_size_cm = maze_size_cm[0]
    
    # compute and smooth speed
    speed1 = nel.utils.ddt_asa(pos, smooth=True, sigma=0.1, norm=True)

    # find epochs where the animal ran > 4cm/sec
    run_epochs = nel.utils.get_run_epochs(speed1, v1=4, v2=4)
    
    # set up results
    results = {}

    # loop through each area seperately
    areas = df_cell_class.area[df_cell_class.session == session] 
    for current_area in pd.unique(areas):
        if verbose:
            print('running through: ',current_area)  
            
        # subset units to current area
        st = st_all._unit_subset(np.where(areas==current_area)[0]+1)
        # reset unit ids like the other units never existed
        st.series_ids = np.arange(0,len(st.series_ids))+1
        
        # restrict spike trains to those epochs during which the animal was running
        st_run = st[run_epochs] 
        ds_run = 0.5 
        ds_50ms = 0.05
        # smooth and re-bin:
        #     sigma = 0.3 # 300 ms spike smoothing
        bst_run = st_run.bin(ds=ds_50ms).smooth(sigma=0.3 , inplace=True).rebin(w=ds_run/ds_50ms)

        sigma = 3 # smoothing std dev in cm
        tc = nel.TuningCurve1D(bst=bst_run,
                               extern=pos,
                               n_extern=40,
                               extmin=0,
                               extmax=maze_size_cm,
                               sigma=sigma,
                               min_duration=0)

        # locate pyr cells with >= 100 spikes, peak rate >= 1 Hz, peak/mean ratio >=1.5
        peak_firing_rates = tc.max(axis=1)
        mean_firing_rates = tc.mean(axis=1)
        ratio = peak_firing_rates/mean_firing_rates
        temp_df = df_cell_class[(df_cell_class.session == session) & (df_cell_class.area == current_area)]
        unit_ids_to_keep = (np.where(((temp_df.cell_type == "pyr")) &
                                     (temp_df.n_spikes >=100) &
                                     (tc.ratemap.max(axis=1) >= 1) &
                                     (ratio>=1.5))[0]+1).squeeze().tolist()

    
        if isinstance(unit_ids_to_keep, int):
            print('warning: only 1 unit...skipping')
            results[current_area] = {}
            continue
        elif len(unit_ids_to_keep) == 0:
            print('warning: no units...skipping')
            results[current_area] = {}
            continue
            
        sta_placecells = st._unit_subset(unit_ids_to_keep)
        tc = tc._unit_subset(unit_ids_to_keep)
        total_units = sta_placecells.n_active
        # tc.reorder_units(inplace=True)
        
        if verbose:
            print('decoding and scoring position')  
            
        # access decoding accuracy on behavioral time scale 
        decoding_r2, median_error, decoding_r2_shuff, _ = decode_and_shuff(bst_run.loc[:,unit_ids_to_keep],
                                                                         tc,
                                                                         pos,
                                                                         n_shuffles=1000)
        # check decoding quality against chance distribution
        _, decoding_r2_pval = get_significant_events(decoding_r2, decoding_r2_shuff)
        
        if decoding_r2_pval > 0.05:
            print('warning: poor decoding...skipping')
            results[current_area] = {}
            continue

        # create intervals for PBEs epochs
        # first restrict to current session and to track + pre/post intervals
#         temp_df = mua_df[((mua_df.session == session) &
#                           ((mua_df.ep_type == "pedestal_1") |
#                            (mua_df.ep_type == "track") |
#                            (mua_df.ep_type == "pedestal_2")))]
        temp_df = mua_df[mua_df.session == session]
    
        # restrict to events at least 80ms
        temp_df = temp_df[temp_df.ripple_duration >= 0.08]
        
        if temp_df.shape[0] == 0:
            print('warning: no PBE events...skipping')
            results[current_area] = {}
            continue
            
        # make epoch object
        PBEs = nel.EpochArray([np.array([temp_df.start_time,temp_df.end_time]).T])

        # bin data into 20ms 
        bst_placecells = sta_placecells[PBEs].bin(ds=0.02)

        # count units per event
        n_active = [bst.n_active for bst in bst_placecells]
        n_active = np.array(n_active) 
        # also count the proportion of bins in each event with 0 activity
        inactive_bin_prop = [sum(bst.n_active_per_bin == 0) / bst.lengths[0] for bst in bst_placecells]
        inactive_bin_prop = np.array(inactive_bin_prop) 
        # restrict bst to instances with >= 5 active units and < 50% inactive bins
        idx = (n_active >= 5) & (inactive_bin_prop < .5)
        bst_placecells = bst_placecells[np.where(idx)[0]]
        # restrict df to instances with >= 5 active units
        temp_df = temp_df[idx]
        n_active = n_active[idx]
        inactive_bin_prop = inactive_bin_prop[idx]

        # decode each event
        posteriors, bdries, mode_pth, mean_pth = nel.decoding.decode1D(bst_placecells,
                                                                       tc,
                                                                       xmin=0,
                                                                       xmax=maze_size_cm)
                
        # score each event using trajectory_score_bst (sums the posterior probability in a range (w) from the LS line)
        if verbose:
            print('scoring events')  

        scores, scores_time_swap, scores_col_cycle = replay.trajectory_score_bst(bst_placecells,
                                                                                 tc,
                                                                                 w=3,
                                                                                 n_shuffles=1500,
                                                                                 normalize=True)
        
        # find sig events using time and column shuffle distributions
        _,score_pval_time_swap = get_significant_events(scores, scores_time_swap)
        _,score_pval_col_cycle = get_significant_events(scores, scores_col_cycle)


        if verbose:
            print('extracting features')  
        (
            traj_dist,
            traj_speed,
            traj_step,
            replay_type,
            dist_rat_start,
            dist_rat_end,
            position
        ) = get_features(bst_placecells,posteriors,bdries,mode_pth,pos,list(temp_df.ep_type))
        
        slope, intercept, r2values = replay.linregress_bst(bst_placecells, tc)
    
        # package data into results dictionary
        results[current_area] = {}

        results[current_area]['sta_placecells'] = sta_placecells
        results[current_area]['bst_placecells'] = bst_placecells
        results[current_area]['tc'] = tc
        results[current_area]['posteriors'] = posteriors
        results[current_area]['bdries'] = bdries
        results[current_area]['mode_pth'] = mode_pth
        results[current_area]['position'] = position

        # add event by event metrics to df
        temp_df['n_active'] = n_active
        temp_df['inactive_bin_prop'] = inactive_bin_prop
        temp_df['trajectory_score'] = scores
        temp_df['r_squared'] = r2values
        temp_df['slope'] = slope
        temp_df['intercept'] = intercept
        temp_df['score_pval_time_swap'] = score_pval_time_swap
        temp_df['score_pval_col_cycle'] = score_pval_col_cycle
        temp_df['traj_dist'] = traj_dist
        temp_df['traj_speed'] = traj_speed
        temp_df['traj_step'] = traj_step
        temp_df['replay_type'] = replay_type
        temp_df['dist_rat_start'] = dist_rat_start
        temp_df['dist_rat_end'] = dist_rat_end
        results[current_area]['df'] = temp_df

        results[current_area]['session'] = session
        results[current_area]['decoding_r2'] = decoding_r2
        results[current_area]['decoding_r2_pval'] = decoding_r2_pval
        results[current_area]['decoding_median_error'] = median_error
        results[current_area]['total_units'] = total_units
        

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
    """
    function to loop through each session
    you can use a basic loop or run in parallel
    """
    warnings.filterwarnings('ignore')

    # find sessions to run
    sessions = pd.unique(mua_df.session)

    if parallel:
        num_cores = multiprocessing.cpu_count()         
        processed_list = Parallel(n_jobs=num_cores)(delayed(main_loop)(session,
                                                                       data_path,
                                                                       spike_path,
                                                                       save_path,
                                                                       mua_df,
                                                                       df_cell_class
                                                                       ) for session in sessions)
    else:    
        for session in sessions:
            print(session)
            main_loop(session,data_path,spike_path,save_path,mua_df,df_cell_class)