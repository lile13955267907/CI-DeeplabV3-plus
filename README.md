Basic information of the code
1. Title: Deep Learning-Based Method for the Identification of Typical Crops Using Dual Polarimetric SAR and High-Resolution Optical Images
2. Key words: Crop classification, GF-2, polarimetric synthetic aper-ture radar, CI-DeepLabV3+
3. Author: Xiaoshuang Ma, Le Li， Yingchun Wang
4. Contents of the main document: 1)  The Slove2.py is the master model, you can use the code to train all model; 2) The SegDataFolder.py is to processing data. Which include channel data, mean and std. If you want to train new data, you should pay attention on the channel data and the value of mean and std; 3) The getSetting.py is to change the hyper-parameter such as optimizer, scheduler, criterion, backbone et.al.
