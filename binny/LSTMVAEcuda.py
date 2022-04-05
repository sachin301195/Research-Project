# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:26:53 2022

@author: patbin
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
import torch
import math
import time

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 6, 4
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('Datasets/BASE/NOC/Unaligned_set_1_BASE_NOC_0.csv',delimiter=";")
unwantedFeatures = ['Feed substrate concentration','Hot/cold switch']
numberOfFeatures = df.shape[1]
class DataPreprocessing():
    
    def __init__(self, dataframe):
        self.dataframe=dataframe
        self.n_features=dataframe.shape[1]
        self.seqLen = dataframe.shape[0]
        
    def getMiniDataList(self, plot=False):
        l = []
        '''Now make 2000 to 400 by averaging every 5 points'''
        for i in range(0,self.dataframe.shape[0]-5,5):
            l.append(self.dataframe.iloc[i:i+5].mean())
        if (plot==True):
            plt.plot(self.dataframe, label="Original Data ")
            plt.plot(l)
        self.seqLen=len(l)
        return l
    
    def createTensorFromList(self, dataList):
        tensor=torch.tensor(np.array(dataList),dtype=torch.float)
        tensor=tensor.reshape(len(dataList),self.n_features)
        return tensor
    
    # Not needed for transformer Implementation
    def scaleTensor(self, data):
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        scaledData = torch.tensor(scaler.transform(data), dtype=torch.float)
        return scaledData, scaler

    def getBatchTensor(self, data, batch_size):
        total_records, features=data.shape[0],data.shape[1]
        extra_element=total_records%batch_size
        if extra_element!=0: data=data[:-extra_element]
        data=data.reshape(self.seqLen//batch_size,batch_size,features)
        return data

batch_size = 50

#if __name__ == "__main__":
def getData(path, batchSize, unwantedFeatures):
    df = pd.read_csv(path, delimiter=";")
    df = df.drop(unwantedFeatures, axis = 1)
    numberOfFeatures = df.shape[1]
    dataPreProc = DataPreprocessing(df)    
    miniDataList = dataPreProc.getMiniDataList()
    dataTensor = dataPreProc.createTensorFromList(miniDataList)
    dataTensor, scaler = dataPreProc.scaleTensor(dataTensor)
    batchDataTensor = dataPreProc.getBatchTensor(dataTensor, batchSize)
    
    return batchDataTensor, dataTensor, numberOfFeatures, scaler, df.columns

path = 'Datasets/BASE/NOC/Unaligned_set_1_BASE_NOC_{}.csv'
bs = 5
batchDataTensor, dataTensor, numberOfFeatures, scaler, columns = getData(path.format(2), bs, unwantedFeatures)

class TransformerVAE(nn.Module):
    
    def __init__(self,  n_features, feedforward_dim_enc, feedforward_dim_dec, n_layers=1):
        super(TransformerVAE, self).__init__()

        self.n_layers = n_layers
        self.feature_size=n_features
        self.src_mask = None
        
        self.feedforward_dim_enc = feedforward_dim_enc
        self.feedforward_dim_dec = feedforward_dim_dec
        
        
        #Encoder
        self.encoder1 = nn.LSTM(self.feature_size, self.feedforward_dim_enc)
        self.encoder2 = nn.LSTM(self.feedforward_dim_enc, self.feature_size*2)
        
        #self.encoder3 = nn.LSTM(self.feature_size, self.feature_size * 2, num_layers = 3)

        #decoder
        #self.decoder_layer1 = nn.Linear(self.feature_size, self.feedforward_dim_dec)
        #self.decoder_layer2 = nn.Linear(self.feedforward_dim_dec, self.feature_size*2)
        self.decoder_layer3 = nn.Linear(self.feature_size, self.feature_size)

    def get_sample(self, mu, logvar):
        if self.training: #inbuilt object which according to me is inherited from nn.Module, comes when model.train() used
            stan_dev=logvar.mul(0.5).exp_() 
            eps = stan_dev.data.new(stan_dev.size()).normal_()#acually epsilon is from normal dis table, sample is the whole Mew+X*sigma
            return eps.mul(stan_dev).add_(mu)#Mew+X*sigma
        else:
            return mu
        
    def get_three_layered_encoder_output(self, sequences):
        bottleneck, _ = self.encoder3(sequences)
        mu1, logvar1 = bottleneck[:,:,:self.feature_size], bottleneck[:,:,self.feature_size:]
        
        return mu1, logvar1

    def get_encoder_output(self, sequences):
        sequences, _ = self.encoder1(sequences)
        bottleneck, _ = self.encoder2(sequences)
        mu1, logvar1 = bottleneck[:,:,:self.feature_size], bottleneck[:,:,self.feature_size:]
        
        return mu1, logvar1
    

    def forward(self, sequences):
        
        mu1, logvar1 = self.get_encoder_output(sequences)
#        mu1, logvar1 = self.get_three_layered_encoder_output(sequences)        
        sample = self.get_sample(mu1, logvar1)
        
        #output = self.decoder_layer1(sample)
        #output = self.decoder_layer2(output) 
        output = self.decoder_layer3(sample)
        
        mu2, logvar2 = output[:,:,:self.feature_size], output[:,:,self.feature_size:]
        
        return mu2, logvar2, mu1, logvar1
#%%
def loss_function(mu2, logvar2, mu1, logvar1, x):
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    torch.pi=torch.tensor(torch.pi)
    KLD=-0.5*torch.mean(logvar1+1-(torch.exp(logvar1)+mu1*mu1))
    c=torch.log(2*torch.pi)
    b=logvar2
    
    a= ((x - mu2)*(x - mu2)/torch.exp(logvar2))
    NLL=0.5*(a+b+c)
    NLL=torch.mean(NLL)
    MSE=torch.mean((x-mu2)**2)
    
    return NLL+KLD

def trainDataFrame(train, optimizer):
    train_loss=0
    for i in range(0, len(train)):# making upper limit lower because it sees 10,10,1 into the future and it will not exist at the end
        #print(train[i].shape)
        mu2, logvar2, mu1, logvar1 = model(train[i].unsqueeze(0)).to(device)
        loss = loss_function(mu2, logvar2, mu1, logvar1, train[i])
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss/train.shape[0]

def testDataFrame(test):
    test_loss=0
    for i in range(0, len(test)):# making upper limit lower because it sees 10,10,1 into the future and it will not exist at the end
        mu2, logvar2, mu1, logvar1 = model(test[i].unsqueeze(0))
        loss = loss_function(mu2, logvar2, mu1, logvar1, test[i])
        test_loss+=loss.item()
    return test_loss/test.shape[0]

def train_model(
    model, num_epochs, optimizer, path, features, l1, h1, l2, h2, batchSize, unwantedFeatures
):    
    epochTimes=[]
    
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    
    for t in range(num_epochs):
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #with torch.profiler.profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        start=time.time()
        sum_train_loss=0
        sum_test_loss=0
        for i in range(l1,h1):
            data_path=path.format(i)
            #print(data_path)
            #train, _, _ = getData(data_path, batchSize)
            train=pd.read_csv(data_path,delimiter=';')
            train = train.drop(unwantedFeatures, axis = 1)            
            dataPreProc = DataPreprocessing(train)
            dataTensor = dataPreProc.createTensorFromList(train)
            if (i==0):
                dataTensor, scaler = dataPreProc.scaleTensor(dataTensor)
            else:
                dataTensor, _ = dataPreProc.scaleTensor(dataTensor)            
            train = dataPreProc.getBatchTensor(dataTensor, batchSize)
            sum_train_loss+=trainDataFrame(train, optimizer)
#           Heeerreeee
            #train=train.unsqueeze(0)
            #model.reset_hidden_state()
                #if test_data is not None:
        with torch.no_grad():
            model.eval()
            for j in range(l2,h2):
                data_path=path.format(j)
                #test = getData(data_path, batchSize)
                test=pd.read_csv(data_path,delimiter=';')
                test = test.drop(unwantedFeatures, axis = 1)
                dataPreProc = DataPreprocessing(test)
                dataTensor = dataPreProc.createTensorFromList(test)
                dataTensor, scaler = dataPreProc.scaleTensor(dataTensor)                
                test=dataPreProc.getBatchTensor(dataTensor,batch_size)                
                n_batches=test.shape[0]
                sum_test_loss+=testDataFrame(test)
        sum_train_loss=sum_train_loss/(h1-l1)
        sum_test_loss=sum_test_loss/(h2-l2)
        print("For epoch no.{} Train loss is {} and Test loss is {}".format(t,sum_train_loss, \
                                                                            sum_test_loss/(train.shape[0])))
        end=time.time()
        test_hist[t] = sum_test_loss
        train_hist[t] = sum_train_loss
        epochTimes.append(end-start)
    
    #rint(prof.key_averages().table(sort_by="self_cpu_memory_usage",)) 
    print('Training time (s) for {} epoch : {} '.format(t, end-start))
    
    return model.eval(), train_hist, epochTimes, test_hist, scaler
#%%
#Creating instance and training
feedforwardDimEnc = 20
feedforwardDimDec = 20
model = TransformerVAE(n_features = numberOfFeatures, feedforward_dim_enc = feedforwardDimEnc,
                             feedforward_dim_dec = feedforwardDimDec, n_layers=1)
unwantedFeatures = ['Feed substrate concentration','Hot/cold switch']
epochs = 50
learn_rate = 1e-3
l1,h1 = 0, 100
l2,h2 = 100, 120
batch_size = 50
path = 'Datasets/BASE/NOC/Unaligned_set_1_BASE_NOC_{}.csv'
optimizer = torch.optim.Adam(
     model.parameters(), lr=learn_rate
)
modelEval, trainLoss, epochTime, testLoss, scaler = train_model(
  model, epochs, optimizer, path, numberOfFeatures, l1, h1, l2, h2, batch_size, unwantedFeatures
)
#%%
'''Plot losses and time'''
plt.figure(figsize=(14,12),dpi=80)
plt.subplot(2,2,1)
plt.plot(trainLoss, label = "Train Loss")
plt.legend()
plt.plot(testLoss, label = "Test Loss")
plt.legend()
plt.subplot(2,2,2)
MeanEpochTime = sum(epochTime)/len(epochTime)
plt.plot(epochTime, label = "Time in seconds Mean with average time :  {} s".format(round(MeanEpochTime, 4)))
plt.legend()
plt.savefig("LSTMd2VAELossTrainingTime.png")
#%% 
'''Signal reproduction'''
path = 'Datasets/BASE/NOC/Unaligned_set_1_BASE_NOC_169.csv'
batchSize = 5
testData, _, _, _, features = getData(path, batchSize, unwantedFeatures)
print(testData.shape)
with torch.no_grad():
    Mu = torch.zeros(1, batchSize, numberOfFeatures)
    Sigma = torch.zeros(1, batchSize, numberOfFeatures)
    #prediction = torch.zeros(1, batchSize, numberOfFeatures)
    for i in range(0, len(testData)):
        mu2, logvar2, mu1, logvar1 = model(testData[i].unsqueeze(0))
        #loss = loss_function(mu2, logvar2, mu1, logvar1, testData[i])
        #analysisLossTest += loss.item()
        Mu = torch.cat((Mu, mu2),axis = 0)                
        Sigma = torch.cat((Sigma, torch.sqrt(torch.exp(logvar2))), axis = 0)
        #Sigma = torch.cat((Sigma, torch.exp(logvar2)), axis = 0)
Mu = torch.flatten(Mu[1:], end_dim = -2)
Sigma = torch.flatten(Sigma[1:], end_dim = -2)
trueValue = torch.flatten(testData, end_dim = -2)
Mu = scaler.inverse_transform(Mu)#.flatten()
Sigma = scaler.inverse_transform(Sigma)
trueValue = scaler.inverse_transform(trueValue)
alpha = 1
plt.figure(figsize=(14,40),dpi=80)
for featurePlot in range(0, numberOfFeatures):
    plt.subplot(numberOfFeatures,1,featurePlot+1)
    plt.plot(trueValue[:, featurePlot], '#1f77b4', label='Data {}'.format(features[featurePlot]))
    plt.plot(Mu[:,featurePlot], 'red', label='Model mean {}'.format(features[featurePlot]))
    plt.fill_between(range(0, trueValue.shape[0]), (Mu[:, featurePlot] + alpha * Sigma[:, featurePlot]), (Mu[:, featurePlot] - alpha * Sigma[:,featurePlot]), facecolor='red', alpha=0.4, label='Standard deviation')
    #'1σ-Umgebung')
    plt.legend()
    plt.savefig("LSTMd2VAESignal_Reproduction.png")  
    plt.tight_layout()
#%%
import logging

dt_string = time.strftime("%d%m%Y-%H%M%S")
logger = logging.getLogger(dt_string)
LogHandler = logging.FileHandler("C:\\Workspace\\Git_clone_2\\src\\reporting_tool\\logs\\ "+dt_string +".log"
                                , mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
LogHandler.setFormatter(formatter)
logger.addHandler(LogHandler)
logger.setLevel(logging.DEBUG)
#%%
'''Saving all fault distributions'''
def getLossDistribution(model, dp, l, h, flag, unwantedFeatures, batchSize):
    with torch.no_grad():
        model=model.eval()
        model = model.float()
        #losses = []
        numberOfRecords = h - l
        losses = np.zeros(shape = (numberOfRecords, 2)) # loss and anomalous or normal label
        totLoss = 0
        #dp = 'Datasets/BASE/NOC/Unaligned_set_1_BASE_NOC_{}.csv'
        for i, file_no in enumerate(range(l, h)):
            print("iiiiiiiiii",i)
            data_path = dp.format(file_no)
            print(data_path)
            data = pd.read_csv(data_path, delimiter=';')
            data = data.drop(unwantedFeatures, axis =1)
            dataPreProc = DataPreprocessing(data)
            data = dataPreProc.getMiniDataList()            
            dataTensor = dataPreProc.createTensorFromList(data)    
            dataTensor, _ = dataPreProc.scaleTensor(dataTensor)
            test = dataPreProc.getBatchTensor(dataTensor, batchSize)
            print(test.shape)
            for j in range(0, len(test)):
                optimizer.zero_grad()
                mu2, logvar2, mu1, logvar1 = model(test[j].unsqueeze(0))
                #print("shapee",train[j].shape, output.shape)
                loss = loss_function(mu2, logvar2, mu1, logvar1, test[j]).item()    ##same as training
                #losses.append(loss)    
                #print(train.shape[1])
                totLoss += loss
            if (flag == "NORMAL"):
                 losses[i] = [totLoss, 0]
            elif (flag == "FAULTY"):
                 losses[i] = [totLoss, 1]                
            #losses.append(totLoss)
            #losses.append(0)
            print("Loss is: ", totLoss, "for df ", file_no)
            totLoss=0
    return losses
batchSize = 5
path = 'Datasets/BASE/NOC/Unaligned_set_1_BASE_NOC_{}.csv'
flag = 'NORMAL'
normalTest = getLossDistribution(model, path, 100, 379, flag, unwantedFeatures, batchSize)
normalDistr = getLossDistribution(model, path, 379, 400, flag, unwantedFeatures, batchSize)
flag = "FAULTY"
anoDistrDict = {}
for fault_no in range (1,16):
    anoPath = 'Datasets/BASE/fault_{}/Unaligned_set_1_BASE_fault_{}_{}.csv'.format(fault_no, fault_no, "{}")
    l, h = 0, 21
    anoDistr = getLossDistribution(model, anoPath, l, h, flag, unwantedFeatures, batchSize)#model, dp, l, h, flag, batchSize
    anoDistrDict.update({fault_no : anoDistr})
    sns.distplot(normalTest[:,0], bins=50, kde=True, label = "Normal")
    sns.distplot(anoDistr[:,0], bins=50, kde=True, label = "Fault {}".format(fault_no))
    plt.xlabel("NLL + KL-Divergence")
    plt.legend()
    plt.savefig("plt_f/LSTMd2VAEFault{}".format(fault_no)
    plt.figure()
#%%
'''Setting threshold up and low based on the following'''
sns.distplot(normalTest[:,0], label = "Normal Data", bins=50, kde=True)
plt.legend()
plt.xlabel("MSE loss")
#%%
'''Setting threshold up and low based on the following'''
# THRESHOLD_DOWN = 700
# THRESHOLD_UP = 3000
# #%%
# '''Code for logging'''
# import logging

# dt_string = time.strftime("%d%m%Y-%H%M%S")
# logger = logging.getLogger(dt_string)
# LogHandler = logging.FileHandler("C:\\Workspace\\Git_clone_2\\src\\reporting_tool\\logs\\ "+dt_string +".log"
#                                 , mode='w') ---change
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# LogHandler.setFormatter(formatter)
# logger.addHandler(LogHandler)
# logger.setLevel(logging.DEBUG)

# #%%
# '''Faultwise score for anomalies'''
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import fbeta_score #(...beta = 1)
# def PerformanceMetrics(Distribution, j):
#     true_predict = np.zeros(shape = (len(Distribution), 2))
#     for i in range(0, len(Distribution)):
#         if(Distribution[i,0] < THRESHOLD_DOWN or THRESHOLD_UP < Distribution[i,0]): prediction = 1
#         else: prediction = 0
#         #print(Distribution[i,1])
#         true_predict[i] = [Distribution[i,1], prediction]
#     #print(true_predict)
#     print("\nFor fault ", j)
#     logger.info("\nFor fault " + j)
#     print("accuracy",accuracy_score(true_predict[:,0], true_predict[:,1]))
#     logger.info("accuracy" +accuracy_score(true_predict[:,0], true_predict[:,1]))
#     print("precision",precision_score(true_predict[:,0], true_predict[:,1]))
#     logger.info("precision " + precision_score(true_predict[:,0], true_predict[:,1]))    
#     print("recall",recall_score(true_predict[:,0], true_predict[:,1]))
#     logger.info("recall "+recall_score(true_predict[:,0], true_predict[:,1]))        
#     f1 = fbeta_score(true_predict[:,0], true_predict[:,1], beta=1)
#     print("F1",f1)
#     logger.info("F1" + f1)        
    
# for i in range(1,16):
#     Distribution = anoDistrDict.get(i)
#     PerformanceMetrics(Distribution, i)
# #%%
# '''Total score of all records combined'''
# Temp = np.zeros(shape =(21,2))
# for i in range(1,16):
#     Temp = np.concatenate((Temp, anoDistrDict.get(i)))
# TotalDistribution = np.concatenate((Temp[21:], normalDistr))
# PerformanceMetrics(TotalDistribution, " Actually total distribution:")
