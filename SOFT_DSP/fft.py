import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------
#SAMPLING
#-----------------------------------------------------------
Fs          = 1000                  # Sample frequency in Hz
Ts          = 1/Fs                  # Sample period in sec
Ls          = 1500                  # Number of samples
ts          = np.arange(Ls)*Ts      # Time space


#-----------------------------------------------------------
#SIGNAL
#-----------------------------------------------------------
SignalFreq      = 50    # Signal frequency 50Hz                                      
SignalAmp       = 0.7   # Signal amplitude
SignalOrg       = SignalAmp * np.sin(2*np.pi*SignalFreq*ts) + 1*np.sin(2*np.pi*120*ts)
Noise           = 0.0002*np.random.rand(Ls)
Signal          = SignalOrg+Noise
Yfft            = np.fft.fft(Signal)
P2              = abs(Yfft/Ls)
HalfP2          = int(Ls/2)
P1              = P2[1:HalfP2+1]
FilteredP1      = np.zeros(P1.shape[0])
for freq in range(len(P1)):
  if P1[freq] >0.3:
    FilteredP1[freq]  = P1[freq]
    print("Got frequency: %f\n" %(freq*(Fs/Ls)))
  
  
"""
------------------------------------------------------------
Plot result, extract only data in range to show, or full set
------------------------------------------------------------
Bt.shape = (k+1, 1)
"""
Xax     = np.arange(0,HalfP2)*(Fs/Ls)  #Correct frequency with (Fs/Ls)

Yax_S   = FilteredP1
Yax_t   = np.fft.ifft(FilteredP1);


fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.plot(Xax[0:200], Signal[0:200])
ax.plot(Xax[0:200], 2000*Yax_t[0:200])
ax.set_title('Predicted result yellow, expected blue')
ax.set_xlabel('Index')
ax.set_ylabel('y output')

plt.tight_layout()
plt.show()
