import numpy as np
import matplotlib.pyplot as plt

def graph(n, k):
    path = 'muon_data_14bit_'+str(n)+'.npy'
    print('Graphing:', path)
    muon = np.load(path)
    t = np.linspace(0, 2700*8, 2700) # 2700 points between 0 and 2700*8
    for i in range(k):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,10))
        axs[1].set_xlim(1550, 1850)

        event0 = muon[i]
        
        axs[0].plot(t, event0[0,:], label='Ch A: Left')
        axs[0].plot(t, event0[1,:], label='Ch B: Right')
        axs[0].plot(t, event0[2,:], label='Ch C: Top')
        axs[0].plot(t, event0[3,:], label='Ch D: Bottom')

        axs[1].plot(t, event0[0,:], label='Ch A: Left')
        axs[1].plot(t, event0[1,:], label='Ch B: Right')
        axs[1].plot(t, event0[2,:], label='Ch C: Top')
        axs[1].plot(t, event0[3,:], label='Ch D: Bottom')
        
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()

    # avgs = np.zeros((4, len(t)))


    # avgs = np.average(muon, axis=0)
# graph("220314T1102", 1)

# for i in range(5):
#     graph(i, 2)

def make_subplots(n):
    # for i in range(5):
    #     muon = np.load('muon_data_14bit_'+str(i)+'.npy')
    #     print("Shape of muon_data_14bit_%d.npy: %s" % (i, muon.shape))

    #     try:
    #         muon = np.load('finalData/muon_data_14bit_'+str(i)+'.npy', allow_pickle=True)
    #     except:
    #         muon = np.load('finalData/muon_data_14bit_'+str(i)+'.npy')
    #     print("Shape of finalData/muon_data_14bit_%d.npy: %s" % (i, muon.shape))

    num_events = 100
    muon = np.load('finalData2/muon_data_14bit_'+str(n)+'.npy')
    t = np.linspace(0, 2700*8, 2700) # 2700 points between 0 and 2700*8

    for i in range(0, muon.shape[0], 100): # batch 
        print('Graphing subplots %d to %d for file %s' % (i, i+99, str(n)))
        fig, ax = plt.subplots(10, 10, sharex=True, sharey=True)
        for j in range(num_events):
            eventNum = i + j
            event = muon[eventNum]
            thisAx = np.reshape(ax, -1)[j]
            thisAx.set_xlim(1550,2000)#1850)
            # thisAx.axis('off')
            thisAx.set_xticks([])
            thisAx.set_yticks([])
            thisAx.plot(t, event[0,:])#, label='Ch A: Left')
            thisAx.plot(t, event[1,:])#, label='Ch B: Right')
            thisAx.plot(t, event[2,:])#, label='Ch C: Top')
            thisAx.plot(t, event[3,:])#, label='Ch D: Bottom')
        
        # plt.xlabel('Time (ns)')
        # plt.ylabel('Voltage (mV)')
        # plt.legend()
        plt.show()


make_subplots("220314T1600")
# make_subplots("threshold800")
# for i in range(5):
#     make_subplots(i)




def avg_graphs(n:int):
    print('Graphing avg over all events ', n)
    muon = np.load('muon_data_14bit_'+str(n)+'.npy')
    avgs = np.average(muon, axis=0)
    t = np.linspace(0,2700*8, 2700) # 2700 points between 0 and 2700*8
    # event0 = muon[2]
    
    plt.plot(t, avgs[0,:], label='Ch A: Left')
    plt.plot(t, avgs[1,:], label='Ch B: Right')
    plt.plot(t, avgs[2,:], label='Ch C: Top')
    plt.plot(t, avgs[3,:], label='Ch D: Bottom')
    
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.show()

    # avgs = np.zeros((4, len(t)))

    

"""
Load numpy(.npy) File 
- To load the output of Picoscope_acquire_muon.py, do following in Python: import numpy as np 
muon = np.load('Filename.npy') 
- Filename is your data file, e.g. `muon_data_14bit_123.npy` 
- Then, nth event can be called by muon[n]. It's a 2D numpy array looks like: [ [2700 Samples in Ch A], [2700 Samples in Ch B], [2700 Samples in Ch C], [2700 Samples in Ch D] ] 
- A simple example plotting 0th event: 
import numpy as np 
Import matplotlib.pyplot as plt 
muon = np.load('Filename.npy') #load data 
timeT = np.linspace(0, 2700*8, 2700) #sampling rate is 8 ns/samples, total number of samples in one event is 2700 
event0 = muon[0] #call 0th event 
# Plot 
plt.plot(timeT, event0[0, :], label='Ch A: left 
middle') 
plt.plot(timeT, event0[1, :], label='right middle') plt.plot(timeT, event0[2, :], label='top') 
plt.plot(timeT, event0[3, :], label='bottom') 
plt.xlabel('Time (ns)') 
plt.ylabel('Voltage (mV)') 
plt.legend() 
plt.show()
"""

"""
You need to write a piece of code that loads the time traces of 
the events and calculates Reduced Quantities (RQs) for each event 
such as 
1) Pulse heights in Channel A, Channel B …C …D
2) Pulse times in Channel A …D
Note that there may be one or two pulses in a given trace

Loads .npy files from the DAQ containing trace data[ ] 
Saves .npz files with RQs such as pulse1height[ ], pulse1time[ ], pulse2height[ ], pulse2time[ ]

The dimensions for a given variable could be
pulse1height[ event #, channel #]
"""