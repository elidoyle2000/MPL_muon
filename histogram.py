from typing import List
import numpy as np
import os

from bokeh.plotting import figure, show
from bokeh.layouts import row
from scipy.optimize import curve_fit

# importing enum for enumerations
 
# creating enumerations using class
class channel():
    def __init__(self) -> None:
        self.left = 0
        self.right = 1
        self.top = 2
        self.bottom = 3


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

def make_step_histogram(title:str, hists:List, channels:List[int], edgeses:List, xlabel:str, ylabel:str):
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

def get_data(path):
    # Use npz file
    muon = np.load(path)
    p1_h = muon['peak1_heights']
    p2_h = muon['peak2_heights']
    p1_t = muon['peak1_times']
    p2_t = muon['peak2_times']
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

def graph1(path, isGamma=False):
    p1_h, p2_h, p1_t, p2_t = get_data(path)
    L_heights = p1_h[:, Channel.left]
    R_heights = p1_h[:, Channel.right]
    
    histL, edges = np.histogram(L_heights, bins=40)
    histR, edges = np.histogram(R_heights, bins=40)

    if isGamma:
        p = make_step_histogram('Energy histogram from main scintillator with gamma source', [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                    'energy (AU)', 'count / bin')
    else:
        p = make_step_histogram('Energy histogram from main scintillator', [histL, histR], [Channel.left, Channel.right], [edges, edges], 
                    'energy (AU)', 'count / bin')

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
graph4(filename) # GRAPH 4



# all = graph1(filename) # FINAL GRAPH 1  
# gamma = graph1("gamma_data.npz", isGamma=True) # Gamma graph

# show(row(all, gamma))