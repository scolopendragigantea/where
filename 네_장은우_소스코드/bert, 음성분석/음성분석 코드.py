import librosa
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Conv2D, Activation, MaxPool2D, Flatten, Conv3D,MaxPool3D , GlobalAveragePooling2D
from tqdm import tqdm
import librosa.display
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from tensorflow.keras.models import load_model


audio_file = "1.wav"
audio, sr = librosa.load(audio_file, sr=16000)
print('sr:', sr, ', audio shape:', audio.shape)
print('length:', audio.shape[0]/float(sr), 'secs')
mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)

pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
padded_mfcc = pad2d(mfcc, 40)
librosa.display.specshow(padded_mfcc, sr=16000, x_axis='time')

#for filename in os.listdir(DATA_DIR + "train2/"):
from sklearn.preprocessing import MinMaxScaler
import random
trainset = []
filename = audio_file
num = "0987654321"

try:
  k =[1,2,3,4]
  #k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,'a','b','c','d','e','f','g','h','i','j','k']
  for i in k:
    filename = str(i) +".wav"
    audio, sr = librosa.load(filename, sr=16000)
    
    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    padded_mfcc = pad2d(mfcc, 40)
    #print(padded_mfcc)
    # 추임새 별로 dataset에 추가
    if str(i) in num:
      trainset.append((padded_mfcc, 0))
    else:
      trainset.append((padded_mfcc, 1))
except Exception as e:
 # print(filename, e)
  raise
random.shuffle(trainset)
# 학습 데이터를 무작위로 섞는다.


testset = []
filename = audio_file
num = "0987654321"
try:
  k =[1,2,3,4]
  #k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,'a','b','c','d','e','f','g','h','i','j','k']
  for i in k:
      
    audio, sr = librosa.load(str(i)+".wav", sr=16000)

    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    padded_mfcc = pad2d(mfcc, 40)

    # 추임새 별로 test dataset에 추가
    if str(i) in num:
      print(1)
      testset.append((padded_mfcc, 0))
    else:
      print(2)
      testset.append((padded_mfcc, 1))
except Exception as e:
  #print(i)
  raise

# 테스트 데이터를 무작위로 섞는다.
random.shuffle(testset)

train_mfccs = [a for (a,b) in trainset]
train_y = [b for (a,b) in trainset]

test_mfccs = [a for (a,b) in testset]
test_y = [b for (a,b) in testset]

train_mfccs = np.array(train_mfccs)
train_y = to_categorical(np.array(train_y))

test_mfccs = np.array(test_mfccs)
test_y = to_categorical(np.array(test_y))

print('train_mfccs:', train_mfccs.shape)
print('train_y:', train_y.shape)

print('test_mfccs:', test_mfccs.shape)
print('test_y:', test_y.shape)

train_X_ex = np.expand_dims(train_mfccs, -1)
test_X_ex = np.expand_dims(test_mfccs, -1)
print('train X shape:', train_X_ex.shape)
print('test X shape:', test_X_ex.shape)



ip = Input(shape=train_X_ex[0].shape,)
print(ip)
m = Conv2D(32, kernel_size=(4,4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4,4))(m)

m = Conv2D(32*2, kernel_size=(4,4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4,4))(m)

m = Conv2D(32*2, kernel_size=(4,4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4,4))(m)

m = Flatten()(m)

m = Dense(64, activation='relu')(m)
m = Dense(64, activation='relu')(m)
m = Dense(64, activation='relu')(m)
op = Dense(2, activation='softmax')(m)

model = Model(ip, op)
model.summary()
audio, sr = librosa.load('c.wav')

mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
padded_mfcc = pad2d(mfcc, 40)
padded_mfcc= np.expand_dims(padded_mfcc, 0)

model.predict(padded_mfcc)
