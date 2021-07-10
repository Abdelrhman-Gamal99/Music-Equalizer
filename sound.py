import numpy as np
from math import pi
import sounddevice as sd
from scipy.io.wavfile import write
# # Generate a sound of any signal
Fs = 16000 # sampling frequency
T=1/Fs
t= np.arange(3*Fs)*T
f1 = 10.0
f2=30.0
f3=50.0
f4=100.0
f5=400.0
f6=500.0
f7=1500.0
f8=2000.0
f9=7500.0
f10=8000.0
x1 =10*(np.sin(2*np.pi*f1*t))
x2 = 0.2*np.sin(2*pi*f2*t)
x3 = 0.05*np.sin(2*pi*f3*t)
x4 = 5*np.sin(2*pi*f4*t)
x5 = .01*np.sin(2*pi*f5*t)
x6 = 0.2*np.sin(2*pi*f6*t)
x7 = 0.5*np.sin(2*pi*f7*t)
x8 = 2*np.sin(2*pi*f8*t)
x9 = 3.5*np.sin(2*pi*f9*t)
x10= 15*np.sin(2*pi*f10*t)
x =  x1 + x2 + x3 +x4 + x5 + x6 + x7 + x8 + x9 + x10
waveform_integers = np.int16(x * 32767)
write('sound.wav', Fs,x)
sd.play(x,Fs)




