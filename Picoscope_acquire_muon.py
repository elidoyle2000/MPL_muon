# This code is based on:
# https://github.com/picotech/picosdk-python-wrappers/blob/master/ps5000aExamples/ps5000aBlockExample.py

# --- doc from original code ---
#
# Copyright (C) 2018 Pico Technology Ltd. See LICENSE file for terms.
#
# PS5000A BLOCK MODE EXAMPLE
# This example opens a 5000a driver device, sets up two channels and a trigger then collects a block of data.
# This data is then plotted as mV against time in ns.

import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
import time
import datetime

def acquire_data(fileIndex = datetime.datetime.now().strftime( '%y%m%dT%H%M' ), saveToDir="./"):
    # Parameters to be changed
    my_offset = 4 # V
    my_threshold = 200 #-20000000# - mV
    my_threshold = my_offset*1000 - my_threshold
    numEvents = 10000
    
    # Create chandle and status ready for use
    chandle = ctypes.c_int16()
    status = {}
    
    # Open 5000 series PicoScope
    # Resolution set to 14 Bit
    resolution =ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_14BIT"]
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    
    try:
        assert_pico_ok(status["openunit"])
    except: # PicoNotOkError:
    
        powerStatus = status["openunit"]
    
        if powerStatus == 286:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        elif powerStatus == 282:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        else:
            raise
    
        assert_pico_ok(status["changePowerSource"])
    
    # Set up channel and Bandwidth
    # ps5000aSetChannel(handle, channel, enabled, coupling type, range, analogueOffset)
    # ps5000aSetBandwidthFilter(handle, channel, Bandwidth Limiter)
    coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
    #chRange = ps.PS5000A_RANGE["PS5000A_10V"]
    chRange = ps.PS5000A_RANGE["PS5000A_5V"]
    limiter = 1 #PS5000A_BANDWIDTH_LIMITER['PS5000A_BW_20MHZ'] = 1
    for i in range(4):
        channel = i
        status["setCh"+str(i)] = ps.ps5000aSetChannel(chandle, channel, 1, coupling_type, chRange, my_offset)
        assert_pico_ok(status["setCh"+str(i)])
        status["setBandwidth"+str(i)] = ps.ps5000aSetBandwidthFilter(chandle, channel, limiter)
        assert_pico_ok(status["setBandwidth"+str(i)])
    '''
    # Check analogue offset
    a = ctypes.c_float()
    b = ctypes.c_float()
    ps.ps5000aGetAnalogueOffset(chandle, chRange, coupling_type, ctypes.byref(a), ctypes.byref(b))
    print(a.value, b.value)
    '''    
        
    
    # find maximum ADC count value
    # ps5000aMaximumValue(handle, pointer to value)
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])
    
    # Set up single trigger
    # ps5000aSetSimpleTrigger(handle, enable, channel source, threshold, THRESHOLD_DIRECTION, delay, autoTrigger_ms)
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    threshold = int(mV2adc(my_threshold,chRange, maxADC))
    direction = ps.PS5000A_THRESHOLD_DIRECTION['PS5000A_BELOW']
    #direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_FALLING_LOWER"] # this gives an error
    status["trigger"] = ps.ps5000aSetSimpleTrigger(chandle, 1, source, threshold, direction, 0, 2000)
    assert_pico_ok(status["trigger"])
    
    # Set number of pre and post trigger samples to be collected
    preTriggerSamples = 200
    postTriggerSamples = 2500
    maxSamples = preTriggerSamples + postTriggerSamples
    
    # Get timebase information
    #ps5000aGetTimebase2(handle, timebase, noSamples, timeIntervalNanoseconds, maxSamples, segmentIndex)
    timebase = 3 # 8 ns in 8, 12, 14 Bit modes
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["getTimebase2"] = ps.ps5000aGetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["getTimebase2"])

    # Create buffers ready for assigning pointers for data collection
    buffer0 = (ctypes.c_int16 * maxSamples)()
    buffer1 = (ctypes.c_int16 * maxSamples)()
    buffer2 = (ctypes.c_int16 * maxSamples)()
    buffer3 = (ctypes.c_int16 * maxSamples)()
    
    # Set data buffer location for data collection from channel A
    # ps5000aSetDataBuffer(handle, source, buffer, bufferLth, segmentIndex, ratio mode)
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    status["setDataBuffer0"] = ps.ps5000aSetDataBuffer(chandle, source, ctypes.byref(buffer0), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffer0"])
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    status["setDataBuffer1"] = ps.ps5000aSetDataBuffer(chandle, source, ctypes.byref(buffer1), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffer1"])
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_C"]
    status["setDataBuffer2"] = ps.ps5000aSetDataBuffer(chandle, source, ctypes.byref(buffer2), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffer2"])
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_D"]
    status["setDataBuffer3"] = ps.ps5000aSetDataBuffer(chandle, source, ctypes.byref(buffer3), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffer3"])
    
    # Empty array for data saving
    data = np.zeros((numEvents,4,maxSamples), dtype = np.int16)
    check = ctypes.c_int16(0)
    i=0
    startAt = time.time()
    
    # Run block capture
    # ps5000aRunBlock(handle, preTriggerSamples, PostTriggerSamples, timebase, time indisposed ms, segment index, lpReady, pParameter)
    status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
    assert_pico_ok(status["runBlock"])
    # Check for data collection to finish using ps5000aIsReady
    ready = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    # Retried data from scope to buffers assigned above
    # ps5000aGetValues(handle, start index, number of samples, downsample ratio, downsample ratio mode, overflow)
    overflow = ctypes.c_int16()
    cmaxSamples = ctypes.c_int32(maxSamples)
    status["getValues"] = ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
    assert_pico_ok(status["getValues"])
    # Stop the scope
    status["stop"] = ps.ps5000aStop(chandle)
    assert_pico_ok(status["stop"])
    # Repeat
    while i < numEvents-1:
        # print(i)
        status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
        assert_pico_ok(status["runBlock"])
            
        # convert ADC counts data to mV
        adc2mVCh0 = -np.array(adc2mV(buffer0, chRange, maxADC)) + my_offset*1000
        adc2mVCh1 = -np.array(adc2mV(buffer1, chRange, maxADC)) + my_offset*1000
        adc2mVCh2 = -np.array(adc2mV(buffer2, chRange, maxADC)) + my_offset*1000
        adc2mVCh3 = -np.array(adc2mV(buffer3, chRange, maxADC)) + my_offset*1000
        # Save data from previous block while the scope is collecting data
        thisEvent = np.array([adc2mVCh0, adc2mVCh1, adc2mVCh2, adc2mVCh3])
        data[i,:,:] = thisEvent
        # Show progress
        if i%100 == 0:
            print('Progress:', str(round(i/numEvents*100, 1)), '%   ', end = '\r')
        i = i + 1
        
        ready = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
        overflow = ctypes.c_int16()
        cmaxSamples = ctypes.c_int32(maxSamples)
        status["getValues"] = ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
        assert_pico_ok(status["getValues"])
        status["stop"] = ps.ps5000aStop(chandle)
        assert_pico_ok(status["stop"])
    
    adc2mVCh0 =  -np.array(adc2mV(buffer0, chRange, maxADC)) + my_offset*1000
    adc2mVCh1 =  -np.array(adc2mV(buffer1, chRange, maxADC)) + my_offset*1000
    adc2mVCh2 =  -np.array(adc2mV(buffer2, chRange, maxADC)) + my_offset*1000
    adc2mVCh3 =  -np.array(adc2mV(buffer3, chRange, maxADC)) + my_offset*1000
    thisEvent = np.array([adc2mVCh0, adc2mVCh1, adc2mVCh2, adc2mVCh3])
    data[numEvents-1,:,:] = thisEvent
    print("Progress: Done            ")
        
    endAt = time.time()
    '''
    # Create time data
    timeT = np.linspace(0, (cmaxSamples.value) * timeIntervalns.value, cmaxSamples.value)
    # plot the last data from channel A and B
    plt.plot(timeT, adc2mVCh0[:], label='left middle')
    plt.plot(timeT, adc2mVCh1[:], label='right middle')
    plt.plot(timeT, adc2mVCh2[:], label='top')
    plt.plot(timeT, adc2mVCh3[:], label='bottom')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.show()
    '''
    
    # Close unit Disconnect the scope
    status["close"]=ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status["close"])
    
    # Save data
    savefilename = saveToDir + 'muon_data_14bit_' + str(numEvents) + "_" + str(fileIndex) + '.npy'
    np.save(savefilename, data)
    
    # display status returns
    #print('progress: Done      ')
    print(status)
    print('time interval =', timeIntervalns.value)
    print('run time =', endAt - startAt)
    
if __name__ == '__main__':
    acquire_data()
