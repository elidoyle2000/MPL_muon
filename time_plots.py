import numpy as np
from fetch_data import get_data
from postprocess import Channel
from bokeh.plotting import figure, show
import glob


def graph_events(sorted_file_indices, event_indices, conditionDesc, doCombineLeftAndRight=False):
    path = "finalData2/*.npy"
    filenames = glob.glob(path)
    filenames.sort()
    for file_index in sorted_file_indices:
        filename = filenames[int(file_index)]
        data = np.load(filename)
    # muon = np.load(filename)
    # peak1_heights, peak2_heights, peak1_times, peak2_times = process_file(filename)
    # condArray = getDoubles(peak1_heights, peak2_heights, peak1_times, peak2_times)
    # data = muon[condArray]
        if not doCombineLeftAndRight:
            color_order = ["red", "green", "blue", "purple"]
            label_order = ["Main left", "Main right", "Top", "Bottom"]
        else:
            color_order = ["magenta", "blue", "purple"]
            label_order = ["Main left + Main right", "Top", "Bottom"]

        for i in range(data.shape[0]):
            if i not in event_indices:# or i < 5500:
                continue
            t = np.arange(0, 2700*8, 8)
            event = data[i]
            tAfterPeak1 = t[(1700//8):]
            eventAfterPeak1 = event[:,(1700//8):]
            maxAfterPeak1 = np.max(eventAfterPeak1)
            tOfMaxAfterPeak1 = tAfterPeak1[np.argmax(np.max(eventAfterPeak1, 0))]
        
            # t = np.arange(0, 2700*8, 8)
            p = figure(title='%s %d in %s'%(conditionDesc, i, filename), background_fill_color="#fafafa")
            p.xaxis.axis_label = 'time (ns)'
            p.yaxis.axis_label = 'PMT voltage (mV)'
            for c in range(len(color_order)):
                t_max = tOfMaxAfterPeak1 + 100
                p.line(t[1550//8:t_max//8], event[c][1550//8:t_max//8], line_color=color_order[c], legend_label=label_order[c])
                # p.line(t, event[c], line_color=color_order[c], legend_label=label_order[c])
            show(p)
            kk = input("Continue?")
            # time.sleep(1)
            # return
                
        print(filename)

def viewOriginalDataFromCondition(path, fromFile=1, doCombineLeftAndRight=False, doPurify=False):
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLeftAndRight, doPurify=doPurify)
    
    # CHANGE STUFF HERE


    # cond:bool = (p1_h[:, Channel.leftAndRight] != 0) & (p2_h[:, Channel.leftAndRight] != 0)
    
    # cond:bool = ((p1_h[:, Channel.left] != 0) & (p2_h[:, Channel.left] != 0)) | ((p1_h[:, Channel.right] != 0) & (p2_h[:, Channel.right] != 0))
    # desc = "decay muons"

    # cond:bool = (p1_h[:, Channel.top] != 0) & (p1_h[:, Channel.bottom] != 0)
    # desc = "through muons"

    # # can't be purified
    # cond:bool = (p1_h[:, Channel.top] == 0) & (p1_h[:, Channel.bottom] == 0)
    # desc = "in sideways?"


    # cond:bool = ((p1_h[:, Channel.leftAndRight] != 0) & (p2_h[:, Channel.leftAndRight] != 0)) & (p2_h[:, Channel.leftAndRight] < 200)
    cond:bool = ((p2_h[:, Channel.left] + p2_h[:, Channel.right]) < 450) & ((p2_h[:, Channel.left] + p2_h[:, Channel.right]) > 0) & (p1_h[:, Channel.top] != 0)
    cond2:bool = (p2_h[:, Channel.left] > 30) & (p2_h[:, Channel.right] > 30)
    cond = cond & cond2
    desc = "combined decay muons"

    muon = np.load(path)
    sorted_file_index = muon['sorted_file_index']
    index_in_file = muon['index_in_file']
    a = np.nonzero(cond)

    file_indices = [sorted_file_index[thingy] for thingy in a[0]]
    indices = [index_in_file[thingy] for thingy in a[0] if sorted_file_index[thingy] == file_indices[0]]
    
    graph_events(file_indices, indices, conditionDesc=desc, doCombineLeftAndRight=doCombineLeftAndRight)

viewOriginalDataFromCondition("full_muon_data.npz", doCombineLeftAndRight=False)