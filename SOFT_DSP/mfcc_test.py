import numpy as np
import wave, struct
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


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

rate=Fs
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])


wav_file.close()
quit()


