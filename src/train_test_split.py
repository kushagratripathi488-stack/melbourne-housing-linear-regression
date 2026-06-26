import copy 
import random

 
def train_test_split(data,test_size):
    data_copy=copy.deepcopy(data)
    random.shuffle(data_copy)
    split_idx = int(len(data_copy)*(1-test_size))
    train_data=data_copy[:split_idx]
    test_data=data_copy[split_idx:]
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    for row in train_data: 
        X_train.append(row[:-1])
        y_train.append(row[-1])
    for row in test_data: 
        X_test.append(row[:-1])
        y_test.append(row[-1])
    return X_train,y_train,X_test,y_test