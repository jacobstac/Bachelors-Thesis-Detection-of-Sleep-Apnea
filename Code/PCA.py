#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:56:39 2019

@author: antonivarsson
"""
#for matrix operations
import numpy as np

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from helpers import make_fourier, normalize, make_freq_list

from sklearn.decomposition import PCA
pca = PCA(n_components = None)

#Sets how big the chunks of data will be 200 cause data is sampled with 200Hz
seconds = 10
interval = seconds * 200

#Returns a dictionary where key = a type of arousal and value is list of matrices of time before event,
#takes transposed data from .mat and none transposed from .arousal

def get_matrices(x,arousals,prev=None, add_random = True):
  arousals_pos = arousals.sample  
  types = arousals.aux_note
  
  if(add_random):
      num_of_random = 250
      np.random.seed(99) # to have a fixed random, same random for all patients
      rando_values = np.random.randint(low = 2000, high = arousals_pos[len(arousals_pos)-1], size = num_of_random)
      arousals_pos = np.append(arousals_pos, rando_values)
      #print("arousals_pos with RANDOM: " + str(arousals_pos))
      types = np.append(types, np.array(['RANDOM' for _ in range(num_of_random)]))
      #print("types + random: " + str(types))
    
    
  matrix_vectors = {} if prev == None else prev
  pos = 0

  #Iterate over each arousal point
  for arousal in arousals_pos:
    if(arousal-interval > 0):
      #get the corresponding indexes for points
      idx = list(range(arousal-interval,arousal))
      #get corresponding data-points
      #print("idx: ", idx)
      
      m_new = x[:,idx]
      m = normalize(m_new)    
     
      #remove ( and ) characters as the wierd formatting produces wrong keys
      A_type = types[pos].translate({ord('('):None}).translate({ord(')'):None})
      #Append matrix or create a list of matrix
      if A_type in matrix_vectors:
        matrix_vectors[A_type]['Matris'] = m + matrix_vectors[A_type]['Matris']
        matrix_vectors[A_type]['Cov'] = np.cov(m) + matrix_vectors[A_type]['Cov']
        matrix_vectors[A_type]['No_norm'] = m_new + matrix_vectors[A_type]['No_norm'] 
        matrix_vectors[A_type]['Instances'] = matrix_vectors[A_type]['Instances'] + 1
      else:
        matrix_vectors[A_type] = { 'Matris': m, 'Cov': np.cov(m), 'No_norm': m_new, 'Instances': 1}  

    #increment arousal
    pos = pos + 1
    
  return matrix_vectors

def PCA(arousals):
  for key in arousals.keys():
    avg_m = normalize(arousals[key]['Matris']/float(arousals[key]['Instances']))
    avg_cov = arousals[key]['Cov']/float(arousals[key]['Instances'])
    avg_m_no_norm = arousals[key]['No_norm']/float(arousals[key]['Instances'])
    avg = np.mean(avg_m,axis=1)
    arousals[key] = {
        'Average_m' : avg_m,
        'Average_covariance': avg_cov,
        'Covariance': np.cov(avg_m),
        'Frequences': make_freq_list(avg_m_no_norm),
        'FFT'       : make_fourier(avg_m_no_norm),
        'Mean'      : avg,
        'Instances' : arousals[key]['Instances'],
    }
  return arousals

