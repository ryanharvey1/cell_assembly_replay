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
    
def run_all(session,spike_path,swr_df):
    '''
    run_all: loads data and runs analysis 
    '''
    # load spikes & add to object
    st = load_add_spikes(spike_path,session)
    
    # bin spike data at 25ms (optimal co-activity timescale Harris et al. 2003)
    dt = 0.025
    binned_st = st.bin(ds=dt)
    
    # detect assemblies using methods from Lopes-dos-Santos et al (2013)
    patterns, significance, zactmat = assembly.runPatterns(binned_st.data)
    assemblyAct = assembly.computeAssemblyActivity(patterns, zactmat)
    
    if len(assemblyAct) == 0:
        assembl_strength = np.full([swr_df[swr_df.session == session].shape[0]], np.nan)
        assembl_frac = np.full([swr_df[swr_df.session == session].shape[0]], np.nan)
        n_assembl = np.nan
        n_units = np.nan
        n_assembl_n_cell_frac = np.nan
        n_cells_per_assembl = np.nan
    else:
        assembl_strength = []
        assembl_frac = []
        for ripple in swr_df[swr_df.session == session].itertuples():
            curr_assembl = assemblyAct[:,(binned_st.bin_centers >= ripple.start_time) & (binned_st.bin_centers <= ripple.end_time)]
            # Assembly strength during SPW-R periods
            assembl_strength.append(curr_assembl[curr_assembl > 5].mean())
            # fraction of active assemblies active during SPW-R 
            assembl_frac.append(sum(np.any(curr_assembl > 5,axis=1)) / curr_assembl.shape[0])

        n_assembl = patterns.shape[0]
        n_units = patterns.shape[1]
        n_assembl_n_cell_frac = n_assembl/n_units
        
        # number of cells that contribute significantly (>2 SD) to each assembly     
        n_cells_per_assembl = np.sum(patterns > (patterns.mean(axis=1) + patterns.std(axis=1)*2)[:, np.newaxis],axis=1)
        n_cells_per_assembl = n_cells_per_assembl[n_cells_per_assembl > 0].mean()    

    # package data
    results = {}
    results['patterns'] = patterns
    results['significance'] = significance
    results['zactmat'] = zactmat
    results['assemblyAct'] = assemblyAct
    results['session'] = session
    results['assembl_strength'] = assembl_strength
    results['assembl_frac'] = assembl_frac
    results['n_assembl'] = n_assembl
    results['n_units'] = n_units
    results['n_assembl_n_cell_frac'] = n_assembl_n_cell_frac
    results['n_cells_per_assembl'] = n_cells_per_assembl
    
    return results

def main_loop(session,spike_path,save_path,swr_df):
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
    results = run_all(session,spike_path,swr_df)   

    # save file
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
        
def assembly_run(spike_path,save_path,swr_df,parallel=True):
    # find sessions to run
    sessions = pd.unique(swr_df.session)

    if parallel:
        num_cores = multiprocessing.cpu_count()         
        processed_list = Parallel(n_jobs=num_cores)(delayed(main_loop)(session,spike_path,save_path,swr_df) for session in sessions)
    else:    
        for session in sessions:
            print(session)
            main_loop(session,spike_path,save_path,swr_df)