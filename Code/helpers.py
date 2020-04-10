#Used for getting 0-1 when there are negative samples. Source::https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
import numpy as np
from sklearn import preprocessing
from scipy.fftpack  import fft, rfft# for FFT
import math

#Unity based normalization, when you need interval 0-1 when there's negative numbers in data.
def unity_based_normalization(row):
  row = np.array(row, dtype=float)
  Min = min(row)
  Max = max(row)
  return (row-Min)/float(Max-Min)

#Normalize a row in the matrix
def norm(x):
  row = np.array(x, dtype=float)
  std = np.std(row)
  #Solves problem with all samples being the same in current row.
  if(std==0):
    std = 1
  return np.true_divide((row - np.mean(row)),std)

#For normalizing entire matrix row by row
def normalize(m):
  temp = np.empty(np.shape(m))
  for i in range(13):
    temp[i,:] = norm(np.array(m)[i,:])
  return temp

#Not used right now! Cause of memory.
def mean_m(m_list):
  if (m_list==[]): 
    return
  n = len(m_list)
  m = np.zeros(np.shape(m_list[0]))
  for i in range(n):
    m = m + m_list[i]
  return m/float(n)

#fourier stuff
def make_fourier(m):
  yf = []
  for ii in range(13):
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / 200.0
    #x = np.linspace(0.0, N*T, N)
    yf.append(rfft(m[ii,:]))
  return yf

# This one checks de mean frequencies for different EEG bands for each 10 second signal
def make_freq_list(data):
  egg_freq = []
  fs = 200              # Sampling rate (200 Hz for us)
  for ii in range(13):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data[ii,:]))
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data[ii]), 1.0/fs)
    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    egg_freq.append(eeg_band_fft)

  return egg_freq