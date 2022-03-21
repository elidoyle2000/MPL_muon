from typing import List
import numpy as np
import os

from bokeh.plotting import figure, show
from bokeh.layouts import row, gridplot
from scipy.optimize import curve_fit

# importing enum for enumerations
 
# creating enumerations using class
class channel():
    def __init__(self) -> None:
        self.left = 0
        self.right = 1
        self.top = 2
        self.bottom = 3
        self.leftAndRight = 0


Channel = channel()

def make_plot(title, hist, edges, xlabel, ylabel):
    p = figure(title=title, background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="#fafafa", alpha=0.5, 
           legend_label='Data')
    p.y_range.start = 0
    p.x_range.start = 0
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    p.grid.grid_line_color = "white"
    return p

def make_step_histogram(title:str, hists:List, channels:List[int], edgeses:List, xlabel:str, ylabel:str, LandRCombined:bool=False):
    if LandRCombined:
        color_order = ["magenta", "blue", "purple"]
        label_order = ["Main left + Main right", "Top", "Bottom"]
    else:
        color_order = ["red", "green", "blue", "purple"]
        label_order = ["Main left", "Main right", "Top", "Bottom"]
    p = figure(title=title, background_fill_color="#fafafa")
    p.y_range.start = 0
    p.x_range.start = -10
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    p.grid.grid_line_color = "#fafafa"

    for i in range(len(channels)):
        c = channels[i]
        hist = hists[i]
        edges = edgeses[i]
        hist = np.insert(hist, 0, [0])
        edges = np.insert(edges, 0, [0])

        p.step(edges[:-1],hist,mode="after", line_width=len(channels)-i, color=color_order[c], legend_label=label_order[c])

    return p

def get_data(path, combineLandR=False, doPurify=False):
    # Use npz file
    muon = np.load(path)
    p1_h = muon['peak1_heights']
    p2_h = muon['peak2_heights']
    p1_t = muon['peak1_times']
    p2_t = muon['peak2_times']
    if combineLandR:
        p1_h = combineLeftAndRight(p1_h)
        p2_h = combineLeftAndRight(p2_h)
        p1_t = combineLeftAndRight(p1_t)
        p2_t = combineLeftAndRight(p2_t)
    
    if doPurify:
        p1_h = purify(p1_h)
        p2_h = purify(p2_h)
        p1_t = purify(p1_t)
        p2_t = purify(p2_t)

    return p1_h, p2_h, p1_t, p2_t


# Lifetime model of muon
def lifetime(x, a, tau, bkg):
  return a * np.exp(-x/tau) + bkg


# Take only incident on top (A) and then a main (B, C)

# Incident on left and right
def graph5(path):
    p1_h, p2_h, p1_t, p2_t = get_data(path)
    # Both middle peaks
    cond = (p2_h[:,0] != 0) & (p2_h[:,1] != 0) 
    event_dt = p2_t[:,0][cond] - 200*(8*10**-9)
    event_dt = event_dt * 10**6
    
    hist, edges = np.histogram(event_dt, bins=20)
    p = make_plot('Second Peak Time Histogram (Channels A and B)', hist, edges, 
                'time (us)', 'count / bin')

    bin_center = ((edges[1:]+edges[:-1])/2)[hist!=0]
    hist = hist[hist!=0]

    constants, covariance = curve_fit(lifetime, bin_center, hist, sigma = np.sqrt(hist))

    x = np.linspace(edges[0], edges[-1], 50)
    fit_result = lifetime(x, *constants)
    p.line(x, fit_result, width = 2, legend_label='Best fit')
    show(p)

    print('number of decay events:', sum(hist))
    print(constants, np.sqrt(np.diag(covariance)))
    print('lifetime =', round(constants[1], 3), '+/-', round(np.sqrt(covariance[1][1]), 3), 'us')


def graph1(path, isGamma=False, doCombineLandR=False):
    
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLandR)


    if not doCombineLandR:
        L_heights = p1_h[:, Channel.left]
        R_heights = p1_h[:, Channel.right]
        histL, edges = np.histogram(L_heights, bins=40)
        histR, edges = np.histogram(R_heights, bins=40)

        if isGamma:
            p = make_step_histogram('Energy histogram from main scintillator with gamma source', [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLandR)
        else:
            p = make_step_histogram('Energy histogram from main scintillator', [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLandR)
    
    else:
        main_heights = p1_h[:, Channel.leftAndRight]
        histMain, edges = np.histogram(main_heights, bins=40)
        if isGamma:
            p = make_step_histogram('Energy histogram from main scintillator with gamma source', [histMain], [Channel.leftAndRight], [edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLandR)
        else:
            p = make_step_histogram('Energy histogram from main scintillator', [histMain], [Channel.leftAndRight], [edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLandR)

    # show(p)
    return p


def graph3(path):
    p1_h, p2_h, p1_t, p2_t = get_data(path)
    L_heights = p1_h[:,0]
    R_heights = p1_h[:,1]

    p = figure(title='Left vs. Right Scintillator', background_fill_color="#fafafa")
    p.xaxis.axis_label = 'Left Scintillator Voltage (mV)'
    p.yaxis.axis_label = 'Right Scintillator Voltage (mV)'
    p.scatter(L_heights, R_heights, size = 0.2)
    show(p)

def graph4(path):

    # (4) Show that events that have signal in PMT-Top and PMT-Bottom 
    #     have large energy in PMT1+2. These are vertical through going muons.

    p1_h, p2_h, p1_t, p2_t = get_data(path)

    cond:bool = (p1_h[:, Channel.top] != 0) & (p1_h[:, Channel.bottom] != 0)
    p1_h = p1_h[cond]
    histL, edges = np.histogram(p1_h[:, Channel.left], bins=20)
    histR, edges = np.histogram(p1_h[:, Channel.right], bins=20)

    
    p = make_step_histogram('Left and Right when Top and Bottom both have pulses', [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                'energy (AU)', 'count / bin')

    show(p)
    # return p

def graph7(path): 
    p1_h, p2_h, p1_t, p2_t = get_data(path)
    # isolate muon events that lead to decay measured by top
    cond:bool = (p2_h[:, Channel.top] != 0)
    top = p2_h[:,Channel.top][cond]

    bottom = p2_h[:, Channel.bottom][cond]
    numTopMeasuredDecay = top.shape[0]
    numTopAndBottomMeasuredDecay = np.sum(np.where(bottom != 0, 1, 0))


    x_labels = ["Top", "Top and Bottom"]
    counts = [numTopMeasuredDecay, numTopAndBottomMeasuredDecay]

    p = figure(title='Did bottom measure decay when top measured decay', background_fill_color="#fafafa", x_range=x_labels)
    p.vbar(x=x_labels, top=counts, width=0.9)
    # p.xaxis.axis_label = 'Top voltage (mV)'
    p.yaxis.axis_label = 'Num events decay measured'
    # p.scatter(top, bottom, size = 0.2)
    show(p)



def blankButNotBlankFires(path): 
    p1_h, p2_h, p1_t, p2_t = get_data(path)
    
    topFiresCondition:bool = (p1_h[:, Channel.top] != 0)
    topDoesntFireCondition:bool = (p1_h[:, Channel.top] == 0)
    leftFiresCondition:bool = (p1_h[:, Channel.left] != 0)
    leftDoesntFireCondition:bool = (p1_h[:, Channel.left] == 0)
    rightFiresCondition:bool = (p1_h[:, Channel.right] != 0)
    rightDoesntFireCondition:bool = (p1_h[:, Channel.right] == 0)
    bottomFiresCondition:bool = (p1_h[:, Channel.bottom] != 0)
    bottomDoesntFireCondition:bool = (p1_h[:, Channel.bottom] == 0)

    print("Left but not right: ", np.sum(np.logical_and(leftFiresCondition, rightDoesntFireCondition)))
    print("Right but not left: ", np.sum(np.logical_and(rightFiresCondition, leftDoesntFireCondition)))
    
    # should be zero
    print("Left or right but not top: ", np.sum(np.logical_and(topDoesntFireCondition, np.logical_or(leftFiresCondition, rightFiresCondition))))

    print("Top but not bottom: ", np.sum(np.logical_and(topFiresCondition, bottomDoesntFireCondition)))
    # cond:bool = (p2_h[:, Channel.top] != 0)
    # top = p2_h[:,Channel.top][cond]

    # bottom = p2_h[:, Channel.bottom][cond]
    # numTopMeasuredDecay = top.shape[0]
    # numTopAndBottomMeasuredDecay = np.sum(np.where(bottom != 0, 1, 0))


    # x_labels = ["Top", "Top and Bottom"]
    # counts = [numTopMeasuredDecay, numTopAndBottomMeasuredDecay]

    # p = figure(title='Did bottom measure decay when top measured decay', background_fill_color="#fafafa", x_range=x_labels)
    # p.vbar(x=x_labels, top=counts, width=0.9)
    # # p.xaxis.axis_label = 'Top voltage (mV)'
    # p.yaxis.axis_label = 'Num events decay measured'
    # # p.scatter(top, bottom, size = 0.2)
    # show(p)

folder = 'finalData2'
filename = 'full_muon_data.npz'
path = os.path.join(folder, filename)

"""
Ch0 (A) is left middle PMT (trigger)<br>
Ch1 (B) is right middle PMT<br>
Ch2 (C) Top Paddle<br>
Ch3 (D) Bottom Paddle<br>
"""

# graph3(filename) # GRAPH 3
# graph4(filename) # GRAPH 4


# graph7(filename)


# show(row(all, gamma))


def graph_double_event(filename):
    data = np.load(filename)
    color_order = ["red", "green", "blue", "purple"]
    label_order = ["Main left", "Main right", "Top", "Bottom"]
    for i in range(data.shape[0]):
        t = np.arange(0, 2700*8, 8)
        event = data[i]
        tAfterPeak1 = t[(1700//8):]
        eventAfterPeak1 = event[:,(1700//8):]
        maxAfterPeak1 = np.max(eventAfterPeak1)
        tOfMaxAfterPeak1 = tAfterPeak1[np.argmax(np.max(eventAfterPeak1, 0))]
        if maxAfterPeak1 > 100:
            # t = np.arange(0, 2700*8, 8)
            p = figure(title='Double peak event', background_fill_color="#fafafa")
            p.xaxis.axis_label = 'time (ns)'
            p.yaxis.axis_label = 'PMT voltage (mV)'
            for c in range(4):
                t_max = tOfMaxAfterPeak1 + 100
                p.line(t[1550//8:t_max//8], event[c][1550//8:t_max//8], line_color=color_order[c], legend_label=label_order[c])
            show(p)
            # return
            
    print(filename)

def combineLeftAndRight(arr):
    new_arr = np.zeros((arr.shape[0], arr.shape[1] - 1))
    new_arr[:,0] = arr[:,0] + arr[:,1]
    new_arr[:,1] = arr[:,2]
    new_arr[:,2] = arr[:,3]

    new_arr = new_arr[new_arr[:,1]==0, :]

    return new_arr


def purify(array):
    




    return

# blankButNotBlankFires(filename)
# graph7(filename)
# graph_double_event("finalData2/muon_data_14bit_10000_220315T1223.npy")

all = graph1(filename) # FINAL GRAPH 1  
gamma = graph1("gamma_data.npz", isGamma=True) # Gamma graph
allCombined = graph1(filename, doCombineLandR=True) # FINAL GRAPH 1  
gammaCombined = graph1("gamma_data.npz", isGamma=True, doCombineLandR=True) # Gamma graph

grid = gridplot(children = [[all, gamma], [allCombined, gammaCombined]], sizing_mode = 'stretch_both')
show(grid)