import numpy as np 
import random

 
def train_test_split(data,test_size):
    data_copy=data.copy()
    np.random.shuffle(data_copy)
    split_idx = int(data_copy.shape[0]*(1-test_size))
    train_data=data_copy[:split_idx]
    test_data=data_copy[split_idx:]
    X_train=train_data[:,:-1]
    X_test=test_data[:,:-1]
    y_train=train_data[:,-1]
    y_test=test_data[:,-1]
    return X_train,y_train,X_test,y_test