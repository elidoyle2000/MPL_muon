import datetime
import Picoscope_acquire_muon

num = 4*(100000//10000)
print('Starting', num, 'data acquisition')
for i in range(num):
    fileIndex = datetime.datetime.now().strftime( '%y%m%dT%H%M' )
    Picoscope_acquire_muon.acquire_data(fileIndex, saveToDir="./finalData2")

