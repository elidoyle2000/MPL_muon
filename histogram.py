import glob
import time
from typing import List
import numpy as np
import os
import chromedriver_binary  # Adds chromedriver binary to path


from bokeh.plotting import figure, show
from bokeh.layouts import row, gridplot
from scipy.optimize import curve_fit
from bokeh.io import export_png

from fetch_data import get_data, Channel

min_right_border:int = 50
num_bins:int = 40
 
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

# what are you DOING step-histogram?! O_o
def make_step_histogram(title:str, hists:List, channels:List[int], edgeses:List, xlabel:str, ylabel:str, LandRCombined:bool=False, proportion=True):
    if LandRCombined:
        color_order = ["magenta", "blue", "purple"]
        label_order = ["Main left + Main right", "Top", "Bottom"]
    else:
        color_order = ["red", "green", "blue", "purple"]
        label_order = ["Main left", "Main right", "Top", "Bottom"]
    p = figure(title=title, background_fill_color="#fafafa")
    p.y_range.start = 0
    p.x_range.start = -10
    if not proportion:
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
    else:
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = "Fraction of total events measured"
    p.grid.grid_line_color = "#fafafa"

    for i in range(len(channels)):
        c = channels[i]
        if proportion:
            hist = hists[i] / np.sum(hists[i])
        else:
            hist = hists[i]
        edges = edgeses[i]
        hist = np.insert(hist, 0, [0])
        edges = np.insert(edges, 0, [0])

        p.step(edges[:-1],hist,mode="after", line_width=len(channels)-i, color=color_order[c], legend_label=label_order[c])

    return p


# Lifetime model of muon
def lifetime(x, a, tau, bkg):
  return a * np.exp(-x/tau) + bkg


def graph1(path, isGamma=False, doCombineLeftAndRight=False, doPurify=False, doShow=True):
    
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLeftAndRight, doPurify=doPurify)


    if not doCombineLeftAndRight:
        L_heights = p1_h[:, Channel.left]
        R_heights = p1_h[:, Channel.right]
        histL, edges = np.histogram(L_heights, bins=num_bins)
        histR, edges = np.histogram(R_heights, bins=num_bins)

        if isGamma:
            p = make_step_histogram('Energy histogram from main scintillator with gamma source\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLeftAndRight)
        else:
            p = make_step_histogram('Energy histogram from main scintillator\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLeftAndRight)
    
    else:
        main_heights = p1_h[:, Channel.leftAndRight]
        histMain, edges = np.histogram(main_heights, bins=num_bins)
        if isGamma:
            p = make_step_histogram('Energy histogram from main scintillator with gamma source\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), [histMain], [Channel.leftAndRight], [edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLeftAndRight)
        else:
            p = make_step_histogram('Energy histogram from main scintillator\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), [histMain], [Channel.leftAndRight], [edges], 
                        'voltage (mV)', 'count / bin', LandRCombined=doCombineLeftAndRight)

    if doShow:
        show(p)

    p.min_border_right = min_right_border
    return p


def graph3(path, doCombineLeftAndRight=False, doPurify=False, doShow=True):
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=False, doPurify=doPurify)
    L_heights = p1_h[:,Channel.left]
    R_heights = p1_h[:,Channel.right]

    p = figure(title='Left vs. Right Scintillator\n(doCombineLeftAndRight: %s, doPurify: %s)'%(False, doPurify), background_fill_color="#fafafa")
    p.xaxis.axis_label = 'Left Scintillator Voltage (mV)'
    p.yaxis.axis_label = 'Right Scintillator Voltage (mV)'
    p.scatter(L_heights, R_heights, size = 0.2)
    if doShow:
        show(p)
    p.min_border_right = min_right_border
    return p

def graph4(path, doCombineLeftAndRight=False, doPurify=False, doShow=True):
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLeftAndRight, doPurify=doPurify)

    # (4) Show that events that have signal in PMT-Top and PMT-Bottom 
    #     have large energy in PMT1+2. These are vertical through going muons.

    if not doCombineLeftAndRight:
        cond:bool = (p1_h[:, Channel.top] != 0) & (p1_h[:, Channel.bottom] != 0)
        p1_h = p1_h[cond]
        histL, edges = np.histogram(p1_h[:, Channel.left], bins=num_bins)
        histR, edges = np.histogram(p1_h[:, Channel.right], bins=num_bins)

        
        p = make_step_histogram('Through muon event\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                    'voltage (mV)', 'count / bin')

    else:
        cond:bool = (p1_h[:, Channel.topAfterCombined] != 0) & (p1_h[:, Channel.bottomAfterCombined] != 0)
        p1_h = p1_h[cond]
        histMain, edges = np.histogram(p1_h[:, Channel.leftAndRight], bins=num_bins)
        
        p = make_step_histogram('Through muon event\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), [histMain], [Channel.leftAndRight], [edges], 
                    'voltage (mV)', 'count / bin', LandRCombined=True)

    # p.title.align = 'center'
    if doShow:
        show(p)
    p.min_border_right = min_right_border
    return p
    # return p

# Incident on left and right
def graph5(path, doCombineLeftAndRight=False, doPurify=False, doShow=True):
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLeftAndRight, doPurify=doPurify)
    # Both middle peaks
    cond = (p2_h[:,Channel.left] != 0) & (p2_h[:,Channel.right] != 0) 
    event_dt = p2_t[:,0][cond] - 200*(8*10**-9)
    event_dt = event_dt * 10**6
    
    hist, edges = np.histogram(event_dt, bins=num_bins)
    p = make_plot('Second Peak Time Histogram (Channels A and B)', hist, edges, 
                'time (us)', 'count / bin')

    bin_center = ((edges[1:]+edges[:-1])/2)[hist!=0]
    hist = hist[hist!=0]

    constants, covariance = curve_fit(lifetime, bin_center, hist, sigma = np.sqrt(hist))

    x = np.linspace(edges[0], edges[-1], 50)
    fit_result = lifetime(x, *constants)
    p.line(x, fit_result, width = 2, legend_label='Best fit')
    
    
    print('number of decay events:', sum(hist))
    print(constants, np.sqrt(np.diag(covariance)))
    print('lifetime =', round(constants[1], 3), '+/-', round(np.sqrt(covariance[1][1]), 3), 'µs')
    if doShow:
        show(p)
    p.min_border_right = min_right_border
    return p

def graph6(path, doCombineLeftAndRight=True, doPurify=False, doShow=True):
    doCombineLeftAndRight=True
    # (6) Show the energy histogram (PMT1+2) of first pulse events (muon goes part way into scintillator) 
    #     and second pulse events (from electron decay)
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLeftAndRight, doPurify=doPurify)
    
    # first pulse: given that there is a second pulse, leftAndRight not 0
    cond:bool = (p1_h[:, Channel.leftAndRight] != 0) & (p2_h[:, Channel.leftAndRight] != 0)
    
    p1_h = p1_h[cond]
    p2_h = p2_h[cond]

    main_heights_p1 = p1_h[:, Channel.leftAndRight]
    main_heights_p2 = p2_h[:, Channel.leftAndRight]


    p = figure(title='Energy histogram of first pulse vs. second pulse in double pulse events\n(doCombineLeftAndRight: %s, doPurify: %s)'%(doCombineLeftAndRight, doPurify), background_fill_color="#fafafa")
    p.y_range.start = 0
    p.x_range.start = -10
    p.xaxis.axis_label = 'voltage (mV)'
    p.yaxis.axis_label = 'count / bin'
    p.grid.grid_line_color = "#fafafa"


    
    histMainP1, edges = np.histogram(main_heights_p1, bins=num_bins)
    histMainP1 = np.insert(histMainP1, 0, [0])
    edges = np.insert(edges, 0, [0])

    p.step(edges[:-1], histMainP1, mode="after", line_width=2, color="blue", legend_label="1st peak (Main left + Main right)")

    histMainP2, edges = np.histogram(main_heights_p2, bins=num_bins)
    histMainP2 = np.insert(histMainP2, 0, [0])
    edges = np.insert(edges, 0, [0])

    p.step(edges[:-1], histMainP2, mode="after", line_width=1, color="green", legend_label="2nd peak (Main left + Main right)")

    if doShow:
        show(p)
    p.min_border_right = min_right_border
    return p



def graph7(path, doCombineLeftAndRight=False, doPurify=False, doShow=True):
    p1_h, p2_h, p1_t, p2_t = get_data(path, combineLandR=doCombineLeftAndRight, doPurify=doPurify)
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
    if doShow:
        show(p)
    p.min_border_right = min_right_border
    return p
    


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


folder = 'finalData2'
filename = 'full_muon_data.npz'
path = os.path.join(folder, filename)


def all4plots(function, filename):
    normal =        function(filename, doCombineLeftAndRight=False, doPurify=False, doShow=False)
    onlyPurify =    function(filename, doCombineLeftAndRight=False, doPurify=True,  doShow=False)
    onlyCombine =   function(filename, doCombineLeftAndRight=True,  doPurify=False, doShow=False)
    both =          function(filename, doCombineLeftAndRight=True,  doPurify=True,  doShow=False)


    grid = gridplot(children = [[normal, onlyPurify], [onlyCombine, both]])#, sizing_mode = "fixed")#'stretch_both')
    export_png(grid, filename="graphs/" + function.__name__ + "_combine_purify_effect.png")
    # show(grid)
    
    


def generateGraphs1and2():
    doPurify:bool = False
    all =               graph1(filename,            isGamma=False,  doCombineLeftAndRight=False,    doPurify=doPurify, doShow=False) # FINAL GRAPH 1  
    gamma =             graph1("gamma_data.npz",    isGamma=True,   doCombineLeftAndRight=False,    doPurify=doPurify, doShow=False) # Gamma graph
    allCombined =       graph1(filename,            isGamma=False,  doCombineLeftAndRight=True,     doPurify=doPurify, doShow=False) # FINAL GRAPH 1  
    gammaCombined =     graph1("gamma_data.npz",    isGamma=True,   doCombineLeftAndRight=True,     doPurify=doPurify, doShow=False) # Gamma graph

    grid = gridplot(children = [[all, gamma], [allCombined, gammaCombined]])#, sizing_mode = 'stretch_both')
    show(grid)
    export_png(grid, filename="graphs/graph1and2_grid_purify_%s_proportions.png" % doPurify)

def generateGraph3():
    purifyFalse =    graph3(filename, doCombineLeftAndRight=False, doPurify=False, doShow=False)
    purifyTrue =     graph3(filename, doCombineLeftAndRight=False, doPurify=True,  doShow=False)
    r = row(children = [purifyFalse, purifyTrue])
    show(r)
    export_png(r, filename="graphs/graph3_row.png")

def generateGraph6():
    noPurify = graph6(filename, doPurify=False, doShow=False)
    purify =   graph6(filename, doPurify=True,  doShow=False)
    r = row(children = [noPurify, purify])
    show(r)
    export_png(r, filename="graphs/graph6_combine_purify_effect.png")

# all4plots(graph3, filename)

# generateGraphs1and2()
# generateGraph3()
generateGraph6()

