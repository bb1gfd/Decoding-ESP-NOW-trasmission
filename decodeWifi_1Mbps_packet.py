import numpy as np
import matplotlib.pyplot as plt
from scipy import signal        

## hackRF info #######################################################

sampleRate=20000000     #Hz

## read hackRF IQ data ################################################
#fileName='beaconData.bin'
fileName='esp_now_transmitter.bin'
#fileName='esp_now_ack.bin'

fileobj1 = open(fileName, mode='rb')

data=np.fromfile(fileobj1, dtype = np.dtype('i1'), count=-1)
print('len(data)=',len(data))
fileobj1.close()

rr=data[::2]
qq=data[1::2]

timeSamples=np.arange(len(rr),dtype=float)
timeUS=1000000.0*timeSamples/sampleRate

# show hackRF IQ data
plt.ion()
plt.figure('IQ')
plt.plot(timeUS,rr)
plt.plot(timeUS,qq)
plt.xlabel('time in uS')
plt.title('hackRF sample values blue I and orange Q')
plt.grid()
plt.pause(0.01)

samples=rr.astype(float)+1.0j*qq.astype(float)

## Costas loop form pysdr.org #######################################

N = len(samples)
phase = 0
freq = 0
# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
alpha =     0.1
beta =      0.001
costas = np.zeros(N, dtype=complex)
maxAmpl=max(np.absolute(samples))
amplNorm=1.0/(maxAmpl*maxAmpl)
for i in range(N):
    costas[i] = samples[i] * np.exp(-1j*phase)
    error = np.real(costas[i]) * np.imag(costas[i])
    error=error*amplNorm
    freq += (beta * error)## freq is actually omega =d(phase)/d(t)
    phase += freq + (alpha * error)

# show Costas loop output
plt.figure('Costas loop output')
plt.plot(np.real(costas))
plt.plot(np.imag(costas))
plt.title('Costas loop output blue I and orange Q')
plt.grid()
plt.xlabel('hackRF samples')
plt.pause(0.01)

## upsample barker code and Costas output ############################
barker=np.array([1,-1, 1,1, - 1, 1, 1, 1, -1, - 1,- 1,],dtype=float)
costasUpsampled=signal.resample_poly(np.real(costas),len(barker),1)
samplesPer_uS=sampleRate/1000000
barkerUpsampled=signal.resample_poly(barker,samplesPer_uS,1)

## correlate upsampled
barkerCorr=np.correlate(costasUpsampled,barkerUpsampled,'same')

## find bit sequence from correlation peaks
bitArr=np.zeros(len(barkerCorr), dtype=int)
thresh=0.55*max(np.absolute(barkerCorr))## to be checked on plot
interpSamplesPer_uS=len(barker)*samplesPer_uS
k=0
peaksCount=0
jump=interpSamplesPer_uS-10
while k<len(barkerCorr):
    if abs(barkerCorr[int(k)])<thresh:
        k+=1;
        continue
    else:
        if peaksCount==0:
            # find first peak position
            xc0=np.argmax(np.absolute(barkerCorr[0:int(k+jump/2)]))
        if barkerCorr[int(k)]>thresh:
            bitArr[peaksCount]=1
        elif barkerCorr[int(k)]<-thresh:
            bitArr[peaksCount]=0
        peaksCount+=1
        k+=jump        
bitArr=bitArr[0:peaksCount]

## create horizontal scale with units of microseconds
xc=np.arange(len(barkerCorr))
xc=(xc-xc0)/interpSamplesPer_uS # place first peak at 0 time
## plot correlation to show peaks and thresholds
plt.figure('correlation of Costas output with Barker code')
plt.plot(xc,barkerCorr)
plt.plot(xc,thresh*np.ones(len(xc)),'r')
plt.plot(xc,-thresh*np.ones(len(xc)),'r')
plt.grid('True')
plt.title('Barker correlation peaks and red thresholds')
plt.xlabel('uS')

## decode differential encoding #########################

ndBits=np.zeros(len(bitArr), dtype=int)
ndBits[0]=bitArr[0]
for k in range(1,len(bitArr)):
    ndBits[k]=bitArr[k-1]^bitArr[k]

## descramble ###########################################

z=np.array([1,1,0,1,1,0,0],dtype=int)## shift register
finalBits=-np.ones(len(ndBits),dtype=int)
for k in range(len(ndBits)):
    xr=z[3]^z[6]
    finalBits[k]=ndBits[k]^xr
    z[1:]=z[0:-1]
    z[0]=ndBits[k]

plt.figure('descrambled Bits')
plt.plot(finalBits,'.-')
plt.grid('True')
plt.title('descrambled Bits')

## find Start Frame Delimiter
sfd=np.array([1,1,1,1,0,0,1,1,1,0,1,0,0,0,0,0],dtype=int)
sfd=np.flip(sfd)

corrSFD=np.correlate(2*finalBits[0:160]-1,2*sfd-1,'same')
sfdStart=int(np.argmax(corrSFD)-len(sfd)/2)
sfdRange=range(sfdStart,sfdStart+len(sfd))
plt.plot(sfdRange,sfd)
plt.title('descrambled Bits and orange SFD')
plt.xlabel('one bit per uS')

## convert bits to bytes
bitW=2.0**(np.arange(8))
bits=finalBits[sfdStart:]
numbOfBytes=int(len(bits)/8)
byteArr=np.zeros(numbOfBytes,dtype=int)
print('Length of byteArr=',len(byteArr))
for k in range(numbOfBytes):
    bitsOfByte=bits[k*8:k*8+8]
    byteArr[k]=int(sum(bitsOfByte*bitW))
print(byteArr)
durationOfPayload=byteArr[5]*256+byteArr[4]
print('Length of data Payload=',str(durationOfPayload),'uS')
print('Number of Payload bytes=',str(durationOfPayload/8))
payload=byteArr[8:]
destMac=np.vectorize(hex)(payload[4:10])
print('destMac=',destMac)
if payload[0]==0xd4:
    print('ACK frame')
else:
    sourcMac=np.vectorize(hex)(payload[10:16])
    print('sourcMac=',sourcMac)
if payload[0]==0x80:
    print('Beacon frame')
if payload[0]==0xd0:
    print('Action frame')
    userData=payload[39:-4]
    print('userData=',userData)
    ## show user data characters
    print('userData characters=',[chr(item) for item in userData])
