
import matplotlib.pyplot as plt
import numpy as np
import wave, struct


numCoefficients = 13 # choose the sive of mfcc array
minHz = 0
maxHz = 10000

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)
    

def melFilterBank(blockSize):
    numBands = int(numCoefficients)
    maxMel = int(freqToMel(maxHz))
    minMel = int(freqToMel(minHz))

    # Create a matrix for triangular filters, one row per filter
    filterMatrix = np.zeros((numBands, blockSize))

    melRange = np.array(xrange(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    # each array index represent the center of each triangular filter
    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (np.exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = np.floor(aux)  # Arredonda pra baixo
    centerIndex = np.array(aux, int)  # Get int values

    for i in xrange(numBands):
        start, centre, end = centerIndex[i:i + 3]
        k1 = np.float32(centre - start)
        k2 = np.float32(end - centre)
        up = (np.array(xrange(start, centre)) - start) / k1
        down = (end - np.array(xrange(centre, end))) / k2

        filterMatrix[i][start:centre] = up
        filterMatrix[i][centre:end] = down

    return filterMatrix.transpose()

















wav_file = wave.open("f2bjrop.wav", 'rb')
# from .wav file to binary data in hexadecimal
binary_data = wav_file.readframes(wav_file.getnframes())
wav_data = np.array(struct.unpack('{n}h'.format(n=wav_file.getnframes()*wav_file.getnchannels()), binary_data))




#Get bit depth
nb_bit = wav_file.getsampwidth()*8
max_nb_bit = float(2 ** (nb_bit - 1))

#Scale to  -1 .. 1
wav_data = wav_data / (max_nb_bit + 1)







#Sample frequency in Hz, Time in secs
Fs = float(wav_file.getframerate())
Ts = 1/Fs
N = 2048


start_idx = int(2/Ts)  #Get the start idx at 20th sec


signal = wav_data[ start_idx : start_idx + N ]
t = np.arange(0, N*Ts, Ts)


#Complex magnitude after DFT   [  a+bi  ]
ScomplexSpectrum = np.fft.fft(signal)

#Modulus of the Complex magnitudes return absolute real magnitudes
#idx is the according frequencies, Amp needed to scale down to fit real
powerSpectrum = np.abs(ScomplexSpectrum) ** 2

#0Hz to Fs Hz, N is the resolution: N=0,0Hz;   N=1,10Hz;  ...
Sfreq = np.linspace(0, Fs , N)

#filteredSpectrum = np.dot(powerSpectrum, melFilterBank(40))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 12))

axes[0].plot(t, signal)
axes[0].set_title('Input Wave')
axes[0].set_xlabel('time[sec]')
axes[0].set_ylabel('amplitude')
axes[0].grid(True)

axes[1].plot(Sfreq[:int(N/2)+1], powerSpectrum[:int(N/2)+1])
axes[1].set_title('Fast Fourier Transform')
axes[1].set_xlabel('freqency[Hz]')
axes[1].set_ylabel('amplitude')
axes[1].grid(True)



plt.tight_layout()
plt.show()



wav_file.close()
quit()
