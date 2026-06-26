import time 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error
from src.data_loader import load_csv 
from src.preprocessing_numpy import fill_missing_with_mean
from src.preprocessing_numpy import winsorization
from src.preprocessing_numpy import mean_normalization
from src.train_test_split_numpy import train_test_split
from src.numpy_regression import cost_func

def run_sklearn():
#Loading the data
    columns,rows=load_csv("D:/CSOC 2026/INTELLIGENCE/Level1 project/melbourne-housing-linear-regression(Pure python)/melb_data.csv")
    rows=np.array(rows)
    #Data preprocessing 
    data_column=np.array(['Rooms','Bathroom','Car','YearBuilt','Distance','Lattitude','Longtitude',
        'Price'])
    feature_column=np.array(['Rooms','Bathroom','Car','YearBuilt','Distance','Lattitude','Longtitude',
       ])

    data_indices=[columns.index(features) for features in data_column]
    data=[]
    for row in rows : 
        data_row=[]
        for i in data_indices: 
            if row[i]=="" or row[i]=="NaN" : 
                 data_row.append(np.nan)
            else:
                data_row.append(float(row[i]))
        data.append(data_row)
    data=np.array(data,dtype=float)

    for i,column in enumerate(data_column): 
        if column !='Price' :
            data=fill_missing_with_mean(data,i)
            data=winsorization(data,i)
            data=mean_normalization(data,i)


    X_train,y_train,X_test,y_test=train_test_split(data,0.2)

    #Linear progression model using the gradient descent 
    model=LinearRegression()
    start=time.time()
    model.fit(X_train,y_train)
    end=time.time()

    #Evaluation Criteria 
    start_pre=time.time()
    y_pred=model.predict(X_test)
    end_pred=time.time()
    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=root_mean_squared_error(y_test,y_pred)
    print("r2_score : ",r2)
    print("mean absolute value :",mae)
    print("Root mean square error :",rmse)
    print("Total time taken for sklearn model fitting : ",end-start," seconds")
    print("Prediction Time : ",end_pred-start_pre)

    return {
        "training_time": end-start,
        "prediction_time":end_pred-start_pre,
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }

