#This is a script to do pre-processing on the EEG data
import numpy as np
import math
import h5py
import os
from pathlib import Path
class Processer:
    def __init__(self):
        self.data = None
        self.label = None
        self.data_processed = None
        self.label_processed = None   
    def load_data(self, path, subject):
        path = Path(path)
        data_list = []
        label_list = []
        for i in range(subject):
            file_code = 'sub_'+ str(i)+'.hdf'
            file = path / file_code
            data_dictionary = h5py.File(file, 'r')
            data = data_dictionary['data']
            label = data_dictionary['label']
            data_list.append(data)
            label_list.append(label)
            print('The shape of data is:'+ str(data_list[-1].shape))
            print('The shape of label is:' + str(label_list[-1].shape))
        self.data = np.stack(data_list, axis = 0)
        self.label = np.stack(label_list, axis = 0)
        # data: subject x trial x channels x datapoint
        # label: subject x trial x datapoint
        print('***************Data loaded successfully!***************')
       
    def format_data(self):
        # data: subject x trial x channels x datapoint
        # label: subject x trial x datapoint
        data = self.data
        label = self.label

        # change the label representation 1.0 -> 0.0; 2.0 -> 1.0
        label[label == 1.0] = 0.0
        label[label == 2.0] = 1.0
        
        #Expand the frequency dimention
        self.data_processed = np.expand_dims(data, axis = 2)       

        self.label_processed = label
        
        print("The data shape is:" + str(self.data_processed.shape))
        
    def split_data(self, segment_length = 1, overlap = 0, sampling_rate = 256, save = True):
        #data: subject x trial x 1 x channels x datapoint
        #label: subject x trial x datapoint
        #Parameters
        data = self.data_processed
        label = self.label_processed
        #Split the data given
        data_shape = data.shape
        label_shape = label.shape
        data_step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []
        label_split = []
        
        number_segment = int((label_shape[2]-data_segment)//(data_step)) + 1
        for i in range(number_segment):
            data_split.append(data[:,:,:,:,(i * data_step):(i * data_step + data_segment)])
            label_split.append(label[:,:,(i * data_step)])
        data_split_array = np.stack(data_split, axis = 2)
        label_split_array = np.stack(label_split, axis = 2)
        print("The data and label are splited: Data shape:" + str(data_split_array.shape) +" Label:" + str(label_split_array.shape))
        self.data_processed = data_split_array
        self.label_processed = label_split_array
        

        #TODO: Save the processed data here
        if save == True:
            if self.data_processed.all() != None:
              
              save_path = Path(os.getcwd())
              filename_data = save_path / Path('data_split.hdf')
              save_data = h5py.File(filename_data, 'w')
              save_data['data'] = self.data_processed
              save_data['label'] = self.label_processed
              save_data.close()
              print("Data and Label saved successfully! at: " + str(filename_data))
            else :
              print("data_splited is None")
        

        
if __name__ == "__main__":
    Pro = Processer() 
    # e.g. path = '/Users/mac/TSception/data'
    Pro.load_data(path='Your path of the data file',subject=2) 
    Pro.format_data()      
    Pro.split_data(segment_length = 4, overlap = 0.975, sampling_rate = 256, save = True)

        
              

        
            
        
           
                
            
        
        
