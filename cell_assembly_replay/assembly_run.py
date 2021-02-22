import itertools
import numpy as np
import os
from cell_assembly_replay import assembly,assembly_run
import pandas as pd
import nelpy as nel
import multiprocessing
from joblib import Parallel, delayed
import pickle

def load_add_spikes(spike_path,session,fs=32000):
    spikes = np.load(os.path.join(spike_path,session)+'.npy', allow_pickle=True)
    spikes_ = list(itertools.chain(*spikes))
    session_bounds = nel.EpochArray([min(spikes_), max(spikes_)])
    return nel.SpikeTrainArray(timestamps=spikes, support=session_bounds, fs=fs)
    
def run_all(session,spike_path,swr_df,cell_list):
    '''
    run_all: loads data and runs analysis 
    '''
    # load spikes & add to object
    st = load_add_spikes(spike_path,session)
    
    # bin spike data at 25ms (optimal co-activity timescale Harris et al. 2003)
    dt = 0.025
    binned_st = st.bin(ds=dt)
    
    # There may be multiple simultaneous brain regions recorded
    # Split and run each region seperately
    session_ = []
    session_ripple = []
    area = []
    area_ripple = []
    assembl_strength = []
    assembl_frac = []
    n_assembl = []  
    n_units = []  
    n_assembl_n_cell_frac = []  
    n_cells_per_assembl = []  
    patterns_ = []  
    significance_ = []  
    zactmat_ = []  
    assemblyAct_ = []
    
    areas = cell_list.area[cell_list.session == session] 
    for a in pd.unique(areas):
        
        # store brain region
        area.append(a)
        session_.append(session)   
        
        # detect assemblies using methods from Lopes-dos-Santos et al (2013)
        patterns, significance, zactmat = assembly.runPatterns(binned_st.data[areas==a,:])
        assemblyAct = assembly.computeAssemblyActivity(patterns, zactmat)

        patterns_.append(patterns)
        significance_.append(significance)
        zactmat_.append(zactmat)
        assemblyAct_.append(assemblyAct)
        
        # calc features per ripple
        if len(assemblyAct) == 0:
            # save area and session
            area_ripple.append(np.full([swr_df[swr_df.session == session].shape[0]], a))
            session_ripple.append(np.full([swr_df[swr_df.session == session].shape[0]], session))
            
            assembl_strength.append(np.full([swr_df[swr_df.session == session].shape[0]], np.nan))
            assembl_frac.append(np.full([swr_df[swr_df.session == session].shape[0]], np.nan))
            
            n_assembl.append(np.nan)
            n_units.append(np.nan)
            n_assembl_n_cell_frac.append(np.nan)
            n_cells_per_assembl.append(np.nan)
        else:
            for ripple in swr_df[swr_df.session == session].itertuples():
                # save area and session
                area_ripple.append(a)
                session_ripple.append(session)
                # pull out current assembly based on ripple width
                curr_assembl = assemblyAct[:,(binned_st.bin_centers >= ripple.start_time) & (binned_st.bin_centers <= ripple.end_time)]
                # Assembly strength during SPW-R periods
                assembl_strength.append(curr_assembl[curr_assembl > 5].mean())
                # fraction of active assemblies active during SPW-R 
                assembl_frac.append(sum(np.any(curr_assembl > 5,axis=1)) / curr_assembl.shape[0])

            n_assembl.append(patterns.shape[0])
            n_units.append(patterns.shape[1])
            n_assembl_n_cell_frac.append(patterns.shape[0]/patterns.shape[1])

            # number of cells that contribute significantly (>2 SD) to each assembly     
            n_cells_per_assembl_ = np.sum(patterns > (patterns.mean(axis=1) + patterns.std(axis=1)*2)[:, np.newaxis],axis=1)
            n_cells_per_assembl.append(n_cells_per_assembl_[n_cells_per_assembl_ > 0].mean())  

    # package data
    results = {}
    results['patterns'] = patterns_
    results['significance'] = significance_
    results['zactmat'] = zactmat_
    results['assemblyAct'] = assemblyAct_
    results['session'] = session_
    results['session_ripple'] = session_ripple
    results['area'] = area
    results['area_ripple'] = area_ripple
    results['assembl_strength'] = assembl_strength
    results['assembl_frac'] = assembl_frac
    results['n_assembl'] = n_assembl
    results['n_units'] = n_units
    results['n_assembl_n_cell_frac'] = n_assembl_n_cell_frac
    results['n_cells_per_assembl'] = n_cells_per_assembl
    
    return results

def main_loop(session,spike_path,save_path,swr_df,cell_list):
    '''
    main_loop: file management 
    '''
    
    base = os.path.basename(session)
    os.path.splitext(base)
    save_file = save_path + os.path.splitext(base)[0] + '.pkl'
    
    # check if saved file exists
    if os.path.exists(save_file):
        return
        
    # detect ripples and calc some features
    results = run_all(session,spike_path,swr_df,cell_list)   

    # save file
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
        
def assembly_run(spike_path,save_path,swr_df,cell_list,parallel=True):
    # find sessions to run
    sessions = pd.unique(swr_df.session)

    if parallel:
        num_cores = multiprocessing.cpu_count()         
        processed_list = Parallel(n_jobs=num_cores)(delayed(main_loop)(session,spike_path,save_path,swr_df,cell_list) for session in sessions)
    else:    
        for session in sessions:
            print(session)
            main_loop(session,spike_path,save_path,swr_df,cell_list)