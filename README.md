# TSception
This is the PyTorch implementation of the TSception in our paper:

*Yi Ding, Neethu Robinson, Qiuhao Zeng, Dou Chen, Aung Aung Phyo Wai, Tih-Shih Lee, Cuntai Guan, "TSception: A Deep Learning Framework for Emotion Detection Using EEG", in IJCNN 2020, WCCI'20* available [Arxiv](https://arxiv.org/abs/2004.02965), [IEEE Xplore](https://ieeexplore.ieee.org/document/9206750)

It is an end-to-end deep learning framework to do classification from raw EEG signals. 
# Requirement
```
python == 3.6 and above
torch == 1.2.0 and above
numpy == 1.16.4
h5py == 2.9.0
pathlib 
```
# Run the code
please save the data into a folder and set the path of the data in 'PrepareData.py'.
> python PrepareData.py

After running the above script, a file named 'data_split.hdf' will be generated at the same location of the script. Please set the location of data_split.hdf in 'Train.py' before running it.

> python Train.py

# Acknowledgment
This code is double-checked by Quihao Zeng and Ravikiran Mane.
# EEG data
Different from images, the EEG data can be treated as 2D time series, whose dimensions are channels (EEG electrodes) and time respectively, (Fig.1) The channels here are the EEG electrodes instead of RGB dimensions in image or the input/output channels for convolutional layers. Because the electrodes are located on different areas on the surface of the human's head, the channel dimension contains spatial information of EEG; The time dimension is full of temporal information instead. In order to train a classifier, the EEG signal will be split into shorter time segments by a sliding window with a certain overlap along the time dimension. Each segment will be one input sample for the classifier.
<p align="center">
<img src="https://user-images.githubusercontent.com/58539144/74715094-ca284500-5266-11ea-9919-9e742e72e37d.png" width=600 align=center>
</p>
<p align="center">
 Fig.1 EEG data. The hight is channel dimesion and the width is the time dimension.
</p>

# Data to use
There are 2 subjects' data available for researchers to run the code. Please find the data in the folder named 'data' in this repo. The data is cleared by a band-pass filter(0.3-45) and [ICA (MNE)](https://mne.tools/stable/auto_tutorials/preprocessing/plot_40_artifact_correction_ica.html). The file is in '.hdf' format. To load the data, please use:
> dataset = h5py.File('NAME.hdf','r')

After loading, the keys are 'data' for data and 'label' for the label. The dimension of the data is (trials x channels x data). The dimension of the label is (trials x data). To use the data and label, please use:

> data = dataset['data']

> label = dataset['label']

The visilizations of the 2 subjects' data are shown in Fig.3:

<p align="center">
<img src="https://user-images.githubusercontent.com/58539144/86339561-51aaa980-bc86-11ea-9cf0-c44ffadd1c3e.png" width=800 align=center>
</p>
<p align="center">
 Fig.3 Visilizations of the 2 subjects' data. Amplitudes of the data are in uV.
</p>

# Structure of TSception
TSception can be divided into 3 main parts: temporal learner, spatial learner and classifier(Fig.2). The input is fed into the temporal learner first followed by spatial learner. Finally, the feature vector will be passed through 2 fully connected layer to map it to the corresponding label. The dimension of input EEG segment is (channels x 1 x timepoint_per_segment), in our case, it is (4 x 1 x 1024), since it has 4 channels, and 1024 data points per channel. There are 9 kernels for each type of temporal kernels in temporal learner, and 6 kernels for each type of spatial kernels in spatial learner. The multi-scale temporal convolutional kernels will operate convolution on the input data parallelly. For each convolution operation, Relu() and average pooling is applied to the feature. The output of each level temporal kernel are concatenated along feature dimension, after which batch normalization is applied. In the spatial learner, the global kernel and hemisphere kernel are used to extract spatial information. Specially, the output of the two spatial kernels will be concatenated along channel dimension after Relu, and average pooling. The flattened feature map will be fed into a fully connected layer. After the dropout layer and softmax activation function, the classification result will be generated. For more details, please see the comments in the code and our paper.  
<p align="center">
<img src="https://user-images.githubusercontent.com/58539144/74716976-80415e00-526a-11ea-9433-02ab2b753f6b.PNG" width=800 align=center>
</p>

<p align="center">
 Fig.2 TSception structure
</p>

# Cite
Please cite our paper if you use our code in your own work:
```
@INPROCEEDINGS{9206750,
  author={Y. {Ding} and N. {Robinson} and Q. {Zeng} and D. {Chen} and A. A. {Phyo Wai} and T. -S. {Lee} and C. {Guan}},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={TSception:A Deep Learning Framework for Emotion Detection Using EEG}, 
  year={2020},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IJCNN48605.2020.9206750}}
```
