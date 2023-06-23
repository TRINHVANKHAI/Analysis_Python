import matplotlib.pyplot as plt
import numpy as np

N = 2 ** 11
dt = 0.000025

freq1 = 10000  #10Hz
freq2 = 8000  #15Hz

ampl1 = 1.3   
ampl2 = 2.8

print ("Input signal");

print("Input 1: ")
print("Frequency : " + str(freq1) + " Hz")
print ("Amplitude: " + str(ampl1))

print("Input 2: ")
print("Frequency : " + str(freq2) + " Hz")
print ("Amplitude: " + str(ampl2))


t = np.arange(0, N*dt, dt)

print("Number of samples : " + str(N))
print("Sampling interval : " + str(dt) + " secs")
print("Sampling frequency: " + str(1/dt) + " Hz")

print("Total sampling time: " + str((N-1)*dt) + " secs")


signal = ampl1*np.sin(2*np.pi*freq1*t) + ampl2*np.sin(2*np.pi*freq2*t)

F = np.fft.fft(signal)

absf = np.abs(F)

absf_amp = absf / N * 2
absf_amp[0] = absf_amp[0] / 2

fq = np.linspace(0, 1.0/dt, N)

idx = np.argmax(signal)


idx = np.array(absf_amp[:int(N/2)+1])

idx = idx.argsort()[::-1]

F_ifft = np.fft.ifft(F)

F_ifft_real = F_ifft.real

F_ifft_real[:10]

idx = np.argmax(F_ifft_real)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))

axes[0, 0].plot(t, signal)
axes[0, 0].set_title('Input Wave')
axes[0, 0].set_xlabel('time[sec]')
axes[0, 0].set_ylabel('amplitude')
axes[0, 0].grid(True)

axes[0, 1].plot(fq[:int(N/2)+1], absf_amp[:int(N/2)+1])
axes[0, 1].set_title('Fast Fourier Transform')
axes[0, 1].set_xlabel('freqency[Hz]')
axes[0, 1].set_ylabel('amplitude')
axes[0, 1].grid(True)

axes[1, 0].plot(t, F_ifft_real, c="g")
axes[1, 0].set_title('Inverse Fast Fourier Transform')
axes[1, 0].set_xlabel('time[sec]')
axes[1, 0].set_ylabel('amplitude')
axes[1, 0].grid(True)

axes[1, 1].plot(t, signal, c="g")
axes[1, 1].plot(t, F_ifft_real, c="b")
axes[1, 1].set_title('Input Wave and IFFT')
axes[1, 1].set_xlabel('time[sec]')
axes[1, 1].set_ylabel('amplitude')
axes[1, 1].grid(True)

file_dir  = 'C:\\Users\\khai\\PYWORKSPACE\\SOFT_DSP\\'
file_name = 'ifft'
fig.savefig(file_dir + file_name + '0.0.jpg', bbox_unches="tight")


