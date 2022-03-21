import os
import numpy as np
from scipy.signal import find_peaks
from bokeh.plotting import figure, show
from tqdm import tqdm
import glob



# Build numpy arrays for a file
def process_file(filename):
    muon = np.load(filename)
    conv = 8*10**-9
    peak1_heights = np.zeros((len(muon), 4))
    peak2_heights = np.zeros((len(muon), 4))
    peak1_times = np.zeros((len(muon), 4))
    peak2_times = np.zeros((len(muon), 4))

    # for i in tqdm(range(len(muon))): # len(muon)
    for i in range(len(muon)): # len(muon)
        peak_heights, peak_times = get_peak_info(muon, i)
        peak1_heights[i] += peak_heights[:,0]
        peak2_heights[i] += peak_heights[:,1]
        peak1_times[i] += conv*peak_times[:,0]
        peak2_times[i] += conv*peak_times[:,1]

    return peak1_heights, peak2_heights, peak1_times, peak2_times

# Get statistics on readings
def get_statistics(peak1_heights, peak2_heights, peak1_times, peak2_times):
    A1_average_height = np.average(peak1_heights[:,0])
    num_nonzero = np.count_nonzero(peak2_heights,axis=0)
    p2_average_height = np.sum(peak2_heights,axis=0)/num_nonzero

    print('A p1 height:', A1_average_height)
    print('A p1 time:', np.average(np.average(peak1_times[:,0])))
    print('Count p2:', num_nonzero)
    print('Average height p2', p2_average_height)
    print('Double peak indices:', np.nonzero(peak2_heights))

    # Find indices with p2


# Get times and magnitudes of peaks (ABCD)
def get_peak_info(muon, num):
    peak2threshhold = 80
    event = muon[num]
    A_info = find_peaks(event[0,:], height=peak2threshhold)
    B_info = find_peaks(event[1,:], height=peak2threshhold)
    C_info = find_peaks(event[2,:], height=peak2threshhold)
    D_info = find_peaks(event[3,:], height=peak2threshhold)

    peak_heights = [A_info[1]['peak_heights'],B_info[1]['peak_heights'],
    C_info[1]['peak_heights'],D_info[1]['peak_heights']]
    peak_times = [A_info[0],B_info[0],C_info[0],D_info[0]]
    # print(peak_heights)
    # print(peak_times)

    clean_heights, clean_times = clean_info(peak_heights, peak_times)
    return clean_heights, clean_times


# Clean get_peak_info data into numpy
def clean_info(peak_heights, peak_times):
    clean_heights = np.zeros((4, 2))
    clean_times = np.zeros((4, 2))
    for i in range(4):
        if len(peak_heights[i]) == 1:
            clean_heights[i][0] += peak_heights[i][0]
            clean_times[i][0] += peak_times[i][0]
        elif len(peak_heights[i]) == 2:
            clean_heights[i] += np.array(peak_heights[i])
            clean_times[i] += np.array(peak_times[i])
    # print(clean_heights)
    # print(clean_times)
    return clean_heights, clean_times


# Graph the i-th event in a .np filename
def graph(filename, i):
    muon = np.load(filename)
    event = muon[i]
    channelA = event[0,:]
    channelB = event[1,:]
    channelC = event[2,:]
    channelD = event[3,:]

    t = np.linspace(0,2700*8, 2700)
    p = figure()
    p.line(t, channelA, legend_label='A', line_color='blue')
    p.line(t, channelB, legend_label='B', line_color='red')
    p.line(t, channelC, legend_label='C', line_color='yellow')
    p.line(t, channelD, legend_label='D', line_color='green')
    show(p)


# For testing
# filename = '2kV/muon_data_14bit_4.npy'
# filename = 'full_data/muon_data_14bit_10000_220315T1223.npy'
# muon = np.load(filename)

# get_peak_info(muon, 0)
def process_all(folder, outname):
    data_lengths, total_num_incidents = find_data_length(folder) 

    peak1_heights = np.zeros((total_num_incidents, 4))
    peak2_heights = np.zeros((total_num_incidents, 4))
    peak1_times = np.zeros((total_num_incidents, 4))
    peak2_times = np.zeros((total_num_incidents, 4))

    path = os.path.join(folder, '*.npy')
    filenames = glob.glob(path)
    filenames.sort()
    past = 0
    for i, filename in enumerate(tqdm(filenames)):
        p1_h, p2_h, p1_t, p2_t = process_file(filename)
        peak1_heights[past:past+data_lengths[i]] = p1_h
        peak2_heights[past:past+data_lengths[i]] = p2_h
        peak1_times[past:past+data_lengths[i]] = p1_t
        peak2_times[past:past+data_lengths[i]] = p2_t
        
        # This line wasn't here
        past += data_lengths[i]

    np.savez(outname, peak1_heights=peak1_heights, peak2_heights=peak2_heights,
    peak1_times=peak1_times, peak2_times=peak2_times)



def find_data_length(folder):
    path = os.path.join(folder, '*.npy')
    filenames = glob.glob(path)
    data_lengths = []
    filenames.sort()

    for filename in filenames:
        if "gamma" not in filename:
            num_incidents = int(filename.split('14bit_')[1].split('_')[0])
            data_lengths.append(num_incidents)
        else:
            return [10000], 10000

    return data_lengths, sum(data_lengths)

# "muon_data_14bit_gamma_threshold_80.npy")
process_all('gamma', 'gamma_data.npz')
# process_all('finalData2', 'full_muon_data.npz')
# print(np.load('full_muon_data.npz').files)#['peak1_heights'].shape)
# graph('full_muon_data.npz', 32)

# P2 32 (200),118 (100),99737 (620)




