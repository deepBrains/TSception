import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
This is the models of TSception and its variant

To use the models, please manage the data into
the dimension of(mini-batch, 1, EEG-channels,data point)
before feed the data into forward()

For more details about the models, please refer to our paper:

Yi Ding, Neethu Robinson, Qiuhao Zeng, Dou Chen, Aung Aung Phyo Wai, Tih-Shih Lee, Cuntai Guan,
"TSception: A Deep Learning Framework for Emotion Detection Useing EEG"(IJCNN 2020)

'''

    
################################################## TSception ######################################################
class TSception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hiden, dropout_rate):
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[0]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[1]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[2]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        
        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0]),1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0]*0.5),1), stride=(int(input_size[0]*0.5),1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        size = self.get_size(input_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())
        
    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(x)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        z = self.Sception1(out)        
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final,z),dim = 2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_size(self,input_size):
        # here we use and array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1,1,input_size[0],input_size[1]))

        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(data)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        z = self.Sception1(out)        
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final,z),dim = 2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()
######################################### Temporal ########################################
class Tception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, hiden, dropout_rate):
        # input_size: channel x datapoint
        super(Tception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[0]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[1]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[2]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        
        self.BN_t = nn.BatchNorm2d(num_T)

        size = self.get_size(input_size,sampling_rate,num_T)
        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())
        
    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(x)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    def get_size(self,input_size,sampling_rate,num_T):
        data = torch.ones((1,1,input_size[0],input_size[1]))        
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(data)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        out = out.view(out.size()[0], -1)
        return out.size()
    
############################################ Spacial ########################################  
class Sception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_S, hiden, dropout_rate):
        # input_size: channel x datapoint
        super(Sception, self).__init__()
        
        self.Sception1 = nn.Sequential(
            nn.Conv2d(1, num_S, kernel_size=(int(input_size[0]),1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(1, num_S, kernel_size=(int(input_size[0]*0.5),1), stride=(int(input_size[0]*0.5),1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))

        self.BN_s = nn.BatchNorm2d(num_S)
        
        size = self.get_size(input_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())
        
    def forward(self, x):
        y = self.Sception1(x)
        out = y
        y = self.Sception2(x)
        out = torch.cat((out,y),dim = 2)
        out = self.BN_s(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    def get_size(self, input_size):
        data = torch.ones((1,1,input_size[0],input_size[1]))
        y = self.Sception1(data)
        out = y
        y = self.Sception2(data)
        out = torch.cat((out,y),dim = 2)
        out = self.BN_s(out)
        out = out.view(out.size()[0], -1)
        return out.size()

if __name__ == "__main__":
    model = TSception(2,(4,1024),256,9,6,128,0.2)
    #model = Sception(2,(4,1024),256,6,128,0.2)
    #model = Tception(2,(4,1024),256,9,128,0.2)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
