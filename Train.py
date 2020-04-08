import torch
import time
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn

from pathlib import Path
from EEGDataset import *
from torch.utils.data import DataLoader
from Models import *

# Acknowledgement:
# Thanks to this tutorial:
# [https://colab.research.google.com/github/dvgodoy/PyTorch101_ODSC_London2019/blob/master/PyTorch101_Colab.ipynb]
class TrainModel():
    def __init__(self):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = None # should be (eeg_channel, time data point)
        self.model = 'TSception'
        self.cross_validation = 'Session' # Subject
        self.sampling_rate = 256
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 1e-3
        self.num_epochs = 200
        self.num_class = 2
        self.batch_size = 128
        self.patient = 4
        
        # Parameters: Model
        self.dropout = 0.3
        self.hiden_node = 128
        self.T = 9
        self.S = 6
        self.Lambda = 1e-6
        
    def load_data(self, path):
        '''
        This is the function to load the data
        Data format : .hdf
        Input : path
                the path of your data
                type = string
        Data dimension : (subject x trials x segments x 1 x channel x data) type = numpy.array
        Label dimension : (subject x trials x segments) type = numpy.array
        Note : For different data formats, please change the loading
               functions, (e.g. use h5py.File to load NAME.hdf)
               
        '''
        path = Path(path)
        dataset = h5py.File(path, 'r')
        self.data = np.array(dataset['data'])
        self.label = np.array(dataset['label'])

        # The input_shape should be (channel x data)
        self.input_shape = self.data[0,0,0,0].shape
        
        print('Data loaded!\n Data shape:[{}], Label shape:[{}]'
              .format(self.data.shape,self.label.shape))
        
    def set_parameter(self, cv, model, number_class, sampling_rate,
                      random_seed, learning_rate, epoch, batch_size,
                      dropout, hiden_node, patient,
                      num_T, num_S, Lambda):
        '''
        This is the function to set the parameters of training process and model
        All the settings will be saved into a NAME.txt file 
        Input : cv --
                   The cross-validation type
                   Type = string
                   Default : Leave_one_session_out
                   Note : for different cross validation type, please add the
                          corresponding cross validation function. (e.g. self.Leave_one_session_out())
                          
                model --
                   The model you want choose
                   Type = string
                   Default : TSception
                   
                number_class --
                   The number of classes
                   Type = int
                   Default : 2

                sampling_rate --
                   The sampling rate of the EEG data
                   Type = int
                   Default : 256

                random_seed --
                   The random seed
                   Type : int
                   Default : 42

                learning_rate --
                   Learning rate
                   Type : flaot
                   Default : 0.001

                epoch --
                   Type : int
                   Default : 200

                batch_size --
                   The size of mini-batch
                   Type : int
                   Default : 128
                
                dropout --
                   dropout rate of the fully connected layers
                   Type : float
                   Default : 0.3

                hiden_node --
                   The number of hiden node in the fully connected layer
                   Type : int
                   Default : 128

                patient --
                   How many epoches the training process should wait for
                   It is used for the early-stopping
                   Type : int
                   Default : 4
                   
                num_T --
                   The number of T kernels
                   Type : int
                   Default : 9
                   
                num_S --
                   The number of S kernels
                   Type : int
                   Default : 6

                Lambda --
                   The L1 regulation coefficient in loss function
                   Type : float
                   Default : 1e-6
                
        '''
        self.model = model
        self.sampling_rate = sampling_rate
        # Parameters: Training process
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = epoch
        self.num_class = number_class
        self.batch_size = batch_size
        self.patient = patient
        self.Lambda = Lambda
        
        # Parameters: Model
        self.dropout = dropout
        self.hiden_node = hiden_node
        self.T = num_T
        self.S = num_S

        
        #Save to log file for checking
        if cv == "Leave_one_subject_out":
            file = open("result_subject.txt",'a')
        elif cv == "Leave_one_session_out":
            file = open("result_session.txt",'a')
        elif cv == "K_fold":
            file = open("result_k_fold.txt",'a')
        file.write("\n"+ str(datetime.datetime.now())+
              "\nTrain:Parameter setting for " + str(self.model) +
              "\n1)number_class:" + str(self.num_class) + "\n2)random_seed:" + str(self.random_seed)+
              "\n3)learning_rate:" + str(self.learning_rate) + "\n4)num_epochs:" + str(self.num_epochs) +
              "\n5)batch_size:" + str(self.batch_size)+
              "\n6)dropout:" + str(self.dropout) + "\n7)sampling_rate:" + str(self.sampling_rate) +
              "\n8)hiden_node:" + str(self.hiden_node) + "\n9)input_shape:" + str(self.input_shape) +
              "\n10)patient:" + str(self.patient) + "\n11)T:" + str(self.T) + 
              "\n12)S:" + str(self.S) + "\n13)Lambda:" + str(self.Lambda) + '\n')
        
        file.close()

    def Leave_one_session_out(self):
        '''
        This is the function to achieve 'Leave one session out' cross-validation
        To know more details about 'Leave one session out', please refer to our paper

        Note : all the acc and std will be logged into the result_session.txt
               
               The txt file is located at the same location as the python script
        
        '''
        save_path = Path(os.getcwd())
        if not os.path.exists(save_path / Path('Result_model/Leave_one_session_out/history')):
            os.makedirs(save_path / Path('Result_model/Leave_one_session_out/history'))
        #Data dimension: subject x trials x segments x 1 x channel x data
        #Label dimension: subject x trials x segments
        #Session: trials[0:2]-session 1; trials[2:4]-session 2; trials[4:end]-session 3
        data = self.data
        label = self.label
        shape_data = data.shape
        shape_label = label.shape
        subject = shape_data[0]
        trial = shape_data[1]
        session = int(shape_data[1]/2)
        channel = shape_data[4]
        frequency = shape_data[3]
        print("Train:Leave_one_session_out \n1)shape of data:" + str(shape_data) + " \n2)shape of label:" + str(shape_label)+
              " \n3)trials:" + str(trial) + " \n4)session:" + str(session) +
              " \n5)datapoint:" + str(frequency) + " \n6)channel:" + str(channel))
        #Train and evaluate the model subject by subject
        ACC = []
        ACC_mean = []
        for i in range(subject):
            index = np.arange(trial)
            ACC_subject = []
            ACC_session = []
            for j in range(session):
                # Split the data into training set and test set
                # One session(contains 2 trials) is test set
                # The rest are training set
                index_train = np.delete(index,[2*j,2*j+1])
                index_test = index[2*j:2*(j+1)]

                data_train = data[i,index_train,:,:,:,:]
                label_train = label[i,index_train,:]

                data_test = data[i,index_test,:,:,:,:]
                label_test = label[i,index_test,:]

                # Split the training set into training set and validation set
                data_train,label_train, data_val, label_val = self.split(data_train, label_train)
                
                # Prepare the data format for training the model
                data_train = torch.from_numpy(data_train).float()
                label_train = torch.from_numpy(label_train).long()

                data_val = torch.from_numpy(data_val).float()
                label_val = torch.from_numpy(label_val).long()
                

                data_test = torch.from_numpy(np.concatenate(data_test, axis = 0)).float()
                label_test = torch.from_numpy(np.concatenate(label_test, axis = 0)).long()
                
                # Check the dimension of the training, validation and test set
                print('Training:', data_train.size(), label_train.size())
                print('Validation:', data_val.size(), label_val.size())
                print('Test:', data_test.size(), label_test.size())

                # Get the accuracy of the model
                ACC_session = self.train(data_train,label_train,
                                         data_test,label_test,
                                         data_val, label_val,
                                         subject = i, session = j,
                                         cv_type = "leave_one_session_out")
                
                ACC_subject.append(ACC_session)
                '''
                # Log the results per session
                
                file = open("result_session.txt",'a')
                file.write('Subject:'+str(i) +' Session:'+ str(j) + ' ACC:' + str(ACC_session) + '\n')
                file.close()
                '''
            ACC_subject = np.array(ACC_subject)
            mAcc = np.mean(ACC_subject)
            std = np.std(ACC_subject)
            
            print("Subject:" + str(i) + "\nmACC: %.2f" % mAcc)
            print("std: %.2f" % std)

            # Log the results per subject
            file = open("result_session.txt",'a')
            file.write('Subject:'+str(i) +' MeanACC:'+ str(mAcc) + ' Std:' + str(std) + '\n')
            file.close()
        
            ACC.append(ACC_subject)
            ACC_mean.append(mAcc)

        self.result = ACC
        # Log the final Acc and std of all the subjects
        file = open("result_session.txt",'a')
        file.write("\n"+ str(datetime.datetime.now()) +'\nMeanACC:'+ str(np.mean(ACC_mean)) + ' Std:' + str(np.std(ACC_mean)) + '\n')
        file.close()
        print("Mean ACC:" + str(np.mean(ACC_mean)) + ' Std:' + str(np.std(ACC_mean)))

        # Save the result
        save_path = Path(os.getcwd())
        filename_data = save_path / Path('Result_model/Result.hdf')
        save_data = h5py.File(filename_data, 'w')
        save_data['result'] = self.result
        save_data.close()
        
    def split(self, data, label):
        '''
        This is the function to split the training set into training set and validation set
        Input : data --
                The training data
                Dimension : trials x segments x 1 x channel x data
                Type : np.array
                
                label --
                The label of training data
                Dimension : trials x segments
                Type : np.array
                
        Output : train --
                 The split training data
                 Dimension : trials x segments x 1 x channel x data
                 Type : np.array

                 train_label --
                 The corresponding label of split training data
                 Dimension : trials x segments
                 Type : np.array

                 val --
                 The split validation data
                 Dimension : trials x segments x 1 x channel x data
                 Type : np.array

                 val_label --
                 The corresponding label of split validation data
                 Dimension : trials x segments
                 Type : np.array
        '''
        #Data dimension: trials x segments x 1 x channel x data
        #Label dimension: trials x segments
        np.random.seed(0)
        data = np.concatenate(data, axis = 0)
        label = np.concatenate(label, axis = 0)
        #data : segments x 1 x channel x data
        #label : segments
        index = np.arange(data.shape[0])
        index_randm = index
        np.random.shuffle(index_randm)
        label = label[index_randm]
        data = data[index_randm]

        # get validation set
        val = data[int(data.shape[0]*0.8):]
        val_label = label[int(data.shape[0]*0.8):]

        train = data[0:int(data.shape[0]*0.8)]
        train_label = label[0:int(data.shape[0]*0.8)]

        return train, train_label, val, val_label
    
    def make_train_step(self, model, loss_fn, optimizer):
        def train_step(x,y):
            model.train()
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item() / len(pred)
            # L1 regularization
            loss_r = self.regulization(model,self.Lambda)
            # yhat is in one-hot representation;
            loss = loss_fn(yhat, y) + loss_r
            #loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item(), acc
        return train_step

    def regulization(self, model, Lambda):
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err
    
    def train(self, train_data, train_label, test_data, test_label, val_data,
              val_label, subject, session, cv_type):
        print('Avaliable device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        # Train and validation loss
        losses = []
        accs = []
        
        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []
        
        test_losses = []
        test_acc = []
        Acc_test = []
        
        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        # build the model
        if self.model == 'Sception':
            model = Sception(num_classes = self.num_class, input_size = self.input_shape,
                             sampling_rate = self.sampling_rate, num_S = self.S,
                             hiden = self.hiden_node, dropout_rate = self.dropout)
        elif self.model == 'Tception':
            model = Tception(num_classes = self.num_class, input_size = self.input_shape,
                             sampling_rate = self.sampling_rate, num_T = self.T,
                             hiden = self.hiden_node, dropout_rate = self.dropout)
        elif self.model == 'TSception':
            model = TSception(num_classes = self.num_class, input_size = self.input_shape,
                              sampling_rate = self.sampling_rate, num_T = self.T, num_S = self.S,
                              hiden = self.hiden_node, dropout_rate = self.dropout)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)


        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data 
        dataset_train = EEGDataset(train_data, train_label)
        dataset_test = EEGDataset(test_data, test_label)
        dataset_val = EEGDataset(val_data, val_label)
        
        # Dataloader for training process
        train_loader = DataLoader(dataset = dataset_train, batch_size = self.batch_size, shuffle = True,pin_memory = False)
        
        val_loader = DataLoader(dataset = dataset_val, batch_size = self.batch_size, pin_memory = False)
        
        test_loader = DataLoader(dataset = dataset_test, batch_size = self.batch_size, pin_memory = False)
        
        total_step = len(train_loader)

        
        ######## Training process ########
        Acc = []
        acc_max = 0
        patient = 0

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch,y_batch) in enumerate(train_loader):
                
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                

                loss, acc = train_step(x_batch,y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                
            losses.append(sum(loss_epoch)/len(loss_epoch))
            accs.append(sum(acc_epoch)/len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print ('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}' 
                   .format(epoch+1, num_epochs,losses[-1] , accs[-1]))
            

            ######## Validation process ########
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]
                    correct = (pred == y_val).sum()
                    acc = correct.item() / len(pred)
                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)
                    
                Acc_val.append(sum(val_acc)/len(val_acc))
                Loss_val.append(sum(val_losses)/len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc: {:.4f}'
                  .format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_acc = []

            ######## early stop ########
            Acc_es = Acc_val[-1]
             
            if Acc_es > acc_max:
                acc_max = Acc_es
                patient = 0
                print('----Model saved!----')
                torch.save(model,'max_model.pt')
            else :
                patient += 1
            if patient > self.patient:
                print('----Early stopping----')
                break
            
             
        ######## test process ########
        model = torch.load('max_model.pt')
        with torch.no_grad(): 
            for x_test, y_test in test_loader:
            
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                model.eval()

                yhat = model(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test).sum()
                acc = correct.item() / len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)
                
            print('Test Loss:{:.4f}, Acc: {:.4f}'
                  .format(sum(test_losses)/len(test_losses), sum(test_acc)/len(test_acc)))
            Acc_test = (sum(test_acc)/len(test_acc))
            test_losses = []
            test_acc = []
        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        if cv_type == "leave_one_session_out":
            filename_callback = save_path / Path('Result_model/Leave_one_session_out/history/'
                                                 + 'history_subject_' + str(subject) + '_session_'
                                                 + str(session)+ '_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()
        return Acc_test
    
if __name__ == "__main__":
    train = TrainModel()
    train.load_data('D:\Code\CNN\Data_processed\data_split.hdf')
    # Please set the parameters here. You can also use for loops to select the parameters automatically,
    # if you have enough computation resources.
    train.set_parameter( cv = 'Leave_one_session_out',
                         model = 'TSception',
                         number_class = 2,
                         sampling_rate = 256,
                         random_seed = 42,
                         learning_rate = 0.001,
                         epoch = 200,
                         batch_size = 128,
                         dropout = 0.3,
                         hiden_node = 128,
                         patient = 4,
                         num_T = 9,
                         num_S = 6,
                         Lambda = 0.000001)
    train.Leave_one_session_out()


