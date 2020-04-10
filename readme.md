### Thesis project Anton Ivarsson & Jacob Stachowicz 2019 Computer Science KTH Royal Institute of Technology
Working with polysomnography data provided by physioNet data repository, to find abnormalities in sleep using machine learning. 

##### **Abstract:**

Sleep arousal is a phenomenon that affects the sleep of a large amount of people. 
The process of predicting and classifying arousal events is done manually with the aid of certified technologists, although some research has been done on automation using Artificial Neural Networks (ANN). This study explored how a Support Vector Machine performed(SVM) compared to an ANN on this task. Polysomnography (PSG) is a sort of sleep study which produces the data that is used in classifying sleep disorders. The PSG-data used in this thesis consists of 13 wave forms sampled at or resampled at 200Hz. There were samples from 994 patients totalling approximately 6.98 1010 data points, processing this amount of data is time consuming and presents a challenge. 2000 points of each signal was used in the construction of the data set used for the models. Extracted features included: Median, Max, Min, Skewness, Kurtosis, Power of EEG-band frequencies and more. Recursive feature elimination was used in order to select the best amount of extracted features. The extracted data set was used to train two ”out of the box” classifiers and due to memory issues the testing had to be split in four batches. When taking the mean of the four tests, the SVM scored ROC AUC of 0,575 and the ANN 0.569 respectively. As the difference in the two results was very modest it was not possible to conclude that either model was better suited for the task at hand. It could however be concluded that SVM can perform as well as ANN on PSG-data. More work has to be done on feature extraction, feature selection and the tuning of the models for PSG-data to conclude anything else. Future thesis work could include research questions as ”Which features performs best for a SVM in the prediction of Sleep arousals on PSG-data” or ”What feature selection technique performs best for a SVM in the prediction of Sleep arousals on PSG-data”, etc.

[Link to the full paper](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1354207&dswid=-1498)



### The Project in short:

#### INTRODUCTION

Poor sleep quality can cause health issues. Diagnosing sleep illnesses is done by classifying sleep abnormalities. A clinical term for sleep abnormality  is *Sleep Arousal.*

#### Problem formulation

***“*** Can a ***support vector machine*** perform better than 
   an ***artificial neural network*** on Polysomnography data 
   for ***classifying Sleep Arousals ? ”***

<img src="images/sheep.png" width="50%">



### installing needed packages for the usage of extraction

* Install anaconda if not installed already
* conda install -c conda-forge hdf5storage
* If you can't find conda add the path variable by running: export PATH=~/anaconda3/bin:$PATH