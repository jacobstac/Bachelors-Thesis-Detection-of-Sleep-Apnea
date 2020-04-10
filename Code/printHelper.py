# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack # for FFT
from helpers import normalize




def print_and_save_content(t, 
                           print_avg_cov = False, 
                           print_avg_mat = False,
                           print_avg_fft = False,
                           print_cov_fft = False):
    
  # plot all the computed data, Loopning through all de computed data 
  for key, value in t.items():
      
    #########################################
    # ----- AVERAGE COVARIANCE MATRIX ----- #
    #########################################
    if(print_avg_cov):
        string = str(key)+ " Covariance. Trials: " + str(t[key]['Instances'])
        b = t[key]['Covariance']
        plt.figure(figsize=(9,7))
        plt.pcolor(b, cmap='viridis')
        plt.colorbar()
        plt.title(string)
        plt.savefig(str(string) +".png")
        plt.show()

    #######################################
    # ----- PLOTTING AVERAGE MATRIX ----- #
    #######################################
    if(print_avg_mat):
        aa = t[key]['Average_m']
        plt.figure(figsize=(12,20))
        string = str(key)+ " Average_m. Trials: " + str(t[key]['Instances'])
        for ii in range(13):
            ax = aa[ii,:]
            ab = ax - np.mean(ax)
            ab = ab/np.std(ax)
            plt.subplot(13,1,ii+1)
            plt.subplots_adjust(wspace = 0.4 )
            plt.plot(ab)
            plt.xlim(1800,2000)
        plt.savefig(str(string) + ".png")
        plt.show()
    
    ####################################   
    # ----- PLOTTING AVERAGE FFT ----- #
    ####################################   
    m = t[key]['FFT']
    if(print_avg_fft):
        yf = []
        f, axarr = plt.subplots(13)
        f.set_figheight(45)
        f.set_figwidth(9)

        for ii in range(13):
            # Number of samplepoints
            N = 600
            # sample spacing
            T = 1.0 / 200.0
            #x = np.linspace(0.0, N*T, N)
            yf.append(m[ii])
            xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
            string = str(key)+ " FFT diagram: trials: " + str(t[key]['Instances']) + ", signal nr: " + str(ii) 
            axarr[ii].set_title(string)
            axarr[ii].plot(xf, 2.0/N * np.abs(yf[ii][:N//2]))
            plt.title(string)
        plt.savefig(str(key)+ " FFT diagram: trials: " + str(t[key]['Instances']) +".png")
        plt.show()
        
        
    ######################################################   
    # ----- PLOTTING AVERAGE FFT COVARIANCE MATRIX ----- #
    ######################################################  
    if(print_cov_fft):
        fft_array = np.zeros((13,2000))
        for i in range(13):
            fft_array[i] = m[i]
        cov_fft_array = np.cov(normalize(fft_array))
        string = str(key)+ " FFT Covariance. Trials: " + str(t[key]['Instances'])
        plt.figure(figsize=(9,7))
        plt.pcolor(cov_fft_array, cmap='viridis')
        plt.colorbar()
        plt.title(string)
        plt.savefig(str(string) +".png")
        plt.show()

        ###################################
        # ----- PLOTTING WELSCH FFT ----- #
        ###################################
    
        '''sf = 200
        win = 2 * sf
        data = t[key]['FFT2'][1,:]
        freqs, psd = signal.welch(data, sf, nperseg=win)
        
        # Plot the power spectrum
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's periodogram for ", t[key]['Instances'])
        plt.xlim([0, freqs.max()])
        sns.despine()
        '''
        
        ###################################
        # ----- EEG FREQUENCES ----- #
        ###################################    
    #f, axarr = plt.subplots(13)
    #f.set_figheight(45)
    #f.set_figwidth(9) 


    for ii in range(13):
    
        import pandas as pd
        df = pd.DataFrame(columns=['band', 'val'])
        df['band'] = t[key]['Frequences'][ii].keys()
        string = str(key) + " Frequences, signal: " + str(ii)
        df['val'] = [t[key]['Frequences'][ii][band] for band in t[key]['Frequences'][ii]]
        ax = df.plot.bar(x='band', y='val', legend=False, title = string)
        ax.set_xlabel("EEG band")
        ax.set_ylabel("Mean band Amplitude")
        #ax.savefig(string + ".png")