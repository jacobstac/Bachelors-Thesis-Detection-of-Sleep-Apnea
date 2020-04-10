#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:56:39 2019

@author: antonivarsson
"""
#for reading .hea, .mat(V.4) and .arousal
import wfdb
#for reading .mat(V7.3)(HDF5)
import h5py
#for plotting
import matplotlib.pyplot as plt
#for matrix operations
import numpy as np

import scipy.signal as sci_sig
from scipy.fftpack  import fft, rfft# for FFT

from helpers import normalize, make_fourier,make_freq_list

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from scipy import signal

#Sets how big the chunks of data will be 200 cause data is sampled with 200Hz
seconds = 10
interval = seconds * 200


#calculates features for a interval/datapoint for a given signal(1-NS) param data should be a vector of 2000 measurements
def features(data,signalType=0):
    #wanted = [11,9,8]
    
    # if we want to downsamople. but we dont need too. hehe
    # change a to downsample
    a = 1
    want_to_downsample = False
    if(want_to_downsample):
        data = signal.resample(data, int(2000/a)) 

    # Fourier Transform
    fou = rfft(data)
    
    # Data pieces
    first_half = data[int(0/a):int(1000/a)]
    middle_half = data[int(500/a):int(1500/a)]
    last_half = data[int(1000/a):int(1999/a)]
    
    
    #### Independent Generic Features ###
    
    # Skewness features
    from scipy.stats import skew
    skew_first = skew(first_half)
    skew_mid = skew(middle_half)
    skew_last = skew(last_half)
    
    # kurtosis features
    from scipy.stats import kurtosis
    kurt_first = kurtosis(first_half)
    kurt_mid = kurtosis(middle_half)
    kurt_last = kurtosis(last_half)
    
    #### Independent Generic Features (statistical) ###
    
    # median
    median_first = np.median(first_half)
    median_mid = np.median(first_half)
    median_last = np.median(first_half)
    
    # correlate
    corr = np.correlate(first_half, last_half)[0]
    
    # det här ser helt galet ut
    return (
            skew_first, skew_mid, skew_last,
            median_first, median_mid, median_last,
            kurt_first, kurt_mid, kurt_last, 
            corr,
            np.var(data),
            np.std(data),
            np.mean(data), 
            np.mean(fou), 
            min(data), 
            max(data), 
            data[-1], 
            np.mean(data[0:int(1000/a)]), np.mean(data[int(500/a):int(1500/a)] ),  np.mean(data[0:int(1000/a)]), np.mean(data[int(1950/a):int(1999/a)]), 
            np.mean(fou[0:int(250/a)]), np.mean(fou[int(250/a):int(500/a)]), np.mean(fou[int(500/a):int(750/a)]), np.mean(fou[int(1750/a):int(1999/a)]),     
            )         

def covariance_distance(covariances, matrix):
  neglected_arousals = ['R', 'W', 'RANDOM', 'N1', 'N2', 'N3']
  t = 0.25
  differences = []
  for key in covariances.keys():
    if key not in neglected_arousals:
      ac = covariances[key]['Covariance']
      cc = np.cov(matrix)
      differences.append(np.linalg.norm(ac-cc))
  return differences

def fft_covariance_distance(covariances, matrix):
  neglected_arousals = ['R', 'W', 'RANDOM', 'N1', 'N2', 'N3']
  t = 0.25
  differences = []
  for key in covariances.keys():
    if key not in neglected_arousals:
      fc = np.cov(normalize(covariances[key]['FFT']))
      fm = make_fourier(matrix)
      fmc = np.cov(normalize(fm))
      differences.append(np.linalg.norm(fmc-fc))
  return differences


#extract the features from every type of signal
def extract_features(x,y,covariances,X=None,Y=None):

  #number of signals in loaded .mat file, usually 13.
  NS = np.shape(x)[0]
  #number of datapoints in loaded .mat file (amount of seconds recorded)*(200HZ), usually a really large number eg 5000000
  ND = np.shape(x)[1]
  #current start of the column being used. 0 at first, then 2000 then 4000...
  CS = 0
  #vector to put the extracted features in, row for each corresponding 10sec interval 
  if (X == None or Y == None):
    X = []
    Y = []
  #Might be a better way of doing this.
  while(CS+interval<=ND):  
    idx = list(range(CS,CS+interval))
    target_arousals = y[:,idx]
    add_point = False
    #if the region contains any 1:s it is and arousal region.
    if 1 in target_arousals:
      Y.append(1)
      add_point = True
    #if the region does not contain -1 it is and non arousal region, -1 is non classified regions and theese points are removed
    elif -1 not in target_arousals:
      Y.append(0)
      add_point = True
    if(add_point):
      local = normalize(x[:,idx])
      #maps the exctraction function to each signal, flattens list as multiple values are returned. 
      #OBS remove flatten if only one value is returned. that is datapoint = list(map(calculate_features,x[:,idx]))
      
      datapoint = list(sum([features(x,signalType=i) for i,x in enumerate(local)], ()))

      distances = covariance_distance(covariances, local)
      
      cov_list = fft_covariance_distance(covariances, x[:,idx])
      
      datapoint.append(max(distances))
      datapoint.append(min(distances))
      datapoint.append(np.mean(distances))
      
      datapoint.append(max(cov_list))
      datapoint.append(min(cov_list))
      datapoint.append(np.mean(cov_list))

      freqs = []
      for l in make_freq_list(local):
        for key in l.keys():
          freqs.append(l[key])
          
          

      X.append(datapoint + freqs)
      
    CS+=interval
  return X,Y