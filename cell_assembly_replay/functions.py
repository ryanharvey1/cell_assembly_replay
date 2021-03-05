import numpy as np
import pandas as pd
import hdf5storage
import h5py
from scipy.signal import find_peaks
import os
import sys

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def loadXML(path):
    """
    path should be the folder session containing the XML file
    Function returns :
        1. the number of channels
        2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
            eeg file first if both are present or both are absent
        3. the mappings shanks to channels as a dict
    Args:
        path : string
    Returns:
        int, int, dict
    """
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, xmlfiles[0])

    from xml.dom import minidom
    xmldoc = minidom.parse(new_path)
    nChannels = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data

    shank_to_channel = {}
    groups = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
    return int(nChannels), int(fs), shank_to_channel

def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
    if type(channel) is not list:
        f = open(path, 'rb')
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        duration = n_samples/frequency
        interval = 1/frequency
        f.close()
        with open(path, 'rb') as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
            timestep = np.arange(0, len(data))/frequency
            # check if lfp time stamps exist
            lfp_ts_path = os.path.join(os.path.dirname(os.path.abspath(path)),'lfp_ts.npy')
            if os.path.exists(lfp_ts_path):
                timestep = np.load(lfp_ts_path).reshape(-1)

            return data, timestep # nts.Tsd(timestep, data, time_units = 's')
        
    elif type(channel) is list:
        f = open(path, 'rb')
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2

        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        duration = n_samples/frequency
        f.close()
        with open(path, 'rb') as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
            timestep = np.arange(0, len(data))/frequency
            # check if lfp time stamps exist
            lfp_ts_path = os.path.join(os.path.dirname(os.path.abspath(path)),'lfp_ts.npy')
            if os.path.exists(lfp_ts_path):
                timestep = np.load(lfp_ts_path).reshape(-1)
            return data,timestep # nts.TsdFrame(timestep, data, time_units = 's')
        
def get_session_path(session):
    f = h5py.File(session,'r')
    return f['session_path'][()].tobytes()[::2].decode()

def load_position(session):
    f = h5py.File(session,'r')
    # load frames [ts x y a s] 
    frames = np.transpose(np.array(f['frames']))
    return pd.DataFrame(frames,columns=['ts', 'x', 'y', 'hd', 'speed'])   

def get_epochs(path):
    f = h5py.File(path,'r')
    ex_ep = []
    for i in range(f['events'].shape[0]):
        ex_ep.append(f['events'][i])
    return ex_ep

def get_maze_size_cm(path):
    f = h5py.File(path,'r')
    maze_size_cm = []
    for i in range(f['maze_size_cm'].shape[0]):
        maze_size_cm.append(f['maze_size_cm'][i][0])
    return maze_size_cm  

def get_spikes(filename):
    data = hdf5storage.loadmat(filename,variable_names=['Spikes'])
    spike_times=data['Spikes']
    spike_times=np.squeeze(spike_times)
    for i in range(spike_times.shape[0]):
        spike_times[i]=np.squeeze(spike_times[i])
    return spike_times

def writeNeuroscopeEvents(path, ep, name):
    f = open(path, 'w')
    for i in range(len(ep)):
        f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
        #f.writelines(str(ep.as_units('ms').iloc[i]['peak']) + " "+name+" start "+ str(1)+"\n")
        f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
    f.close()
    return

def fastrms(x,window=5):
    window = np.ones(window)
    power = x**2
    rms = np.convolve(power,window,mode='same')
    return  np.sqrt(rms/sum(window))

def get_place_fields(ratemap,min_peak_rate=2,min_field_width=2,max_field_width=39,percent_threshold=0.2):
    
    std_rates = np.std(ratemap)
    
    locs,properties = find_peaks(fastrms(ratemap), height=min_peak_rate, width=min_field_width)
    pks = properties['peak_heights']

    exclude = []
    for j in range(len(locs)-1):
        if min(ratemap[locs[j]:locs[j+1]]) > ((pks[j] + pks[j+1]) / 2) * percent_threshold:
            if pks[j] > pks[j+1]:
                exclude.append(j+1)
            elif pks[j] < pks[j+1]:
                exclude.append(j)
       
    if any(ratemap[locs] < std_rates*.5):
        exclude.append(np.where(ratemap[locs] < std_rates*.5))
    if not exclude:
        pks = np.delete(pks, exclude)
        locs = np.delete(locs, exclude)
    
    fields = []
    for j in range(len(locs)):
        Map_Field = (ratemap > pks[j] * percent_threshold)*1
        start = locs[j]
        stop = locs[j]
        
        while (Map_Field[start] == 1)  & (start > 0):
            start -= 1
        while (Map_Field[stop] == 1)  & (stop < len(Map_Field)-1):
            stop += 1

        if ((stop - start) > min_field_width) & ((stop - start) < max_field_width):
            com = start
            while sum(ratemap[start:stop]) - sum(ratemap[start:com]) > sum(ratemap[start:com])/2:
                com += 1
            fields.append((start,stop,stop - start,pks[j],locs[j],com))
                        
    # add to data frames
    fields = pd.DataFrame(fields, columns=("start", "stop", "width", "peakFR", "peakLoc", "COM"))  
    return fields

def get_place_cell_idx(session):
    """
    find cells to include. At least 1 field from both directions
    """
    data = hdf5storage.loadmat(session,variable_names=['ratemap'])
    include = []
    field = 0
    for i in range(data['ratemap'].shape[0]):
        for d in range(2):
            fields = get_place_fields(data['ratemap'][i,d][0])
            if not fields.empty:
                field += 1
        if field > 0:
            include.append(1)
        else:
            include.append(0)
        field = 0
    return include  