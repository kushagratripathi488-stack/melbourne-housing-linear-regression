import time 
import numpy as np 
import matplotlib.pyplot as plt
from src.data_loader import load_csv 
from src.preprocessing_numpy import fill_missing_with_mean
from src.preprocessing_numpy import winsorization
from src.preprocessing_numpy import mean_normalization
from src.train_test_split_numpy import train_test_split
from src.numpy_regression import cost_func
from src.numpy_regression import gradient_descent
from src.evaluaton import r2_score
from src.evaluaton import root_mean_squared_error
from src.evaluaton import mean_absolute_error
#Loading the data
def run_numpy():
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


    #Train Test Split
    X_train,y_train,X_test,y_test=train_test_split(data,0.2)

    #Linear progression model using the gradient descent 
    weights=np.zeros(len(feature_column))
    bias=0.0
    learning_rate = 0.1
    epochs = 1001
    start=time.time()
    weights,bias,costs,times=gradient_descent(X_train,y_train,weights,bias,learning_rate,epochs)
    end=time.time()
    final_train_cost=cost_func(X_train,y_train,weights,bias)
    final_test_cost=cost_func(X_test,y_test,weights,bias)
    #Evaluation Criteria 
    y_pred=np.dot(X_test,weights)+bias
    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=root_mean_squared_error(y_test,y_pred)
    print("r2_score : ",r2)
    print("mean absolute value :",mae)
    print("Root mean square error :",rmse)
    print("Total time taken for convergence : ",end-start," seconds")
    print("Final training data cost of the model : ",final_train_cost)
    print("Final testing data cost of the model : ",final_test_cost)

    #plt.figure(figsize=(8, 5))

    #plt.plot(times,costs, label="NumPy",marker='o')  
    #plt.xlabel("Time(Seconds)")
    #plt.ylabel("Cost")
    #plt.title("Cost vs Time",fontdict={'fontsize': 16, 'fontweight': 'semibold'})
    #plt.grid(True)
    #plt.legend()
    #plt.show()

    return {
        "costs": costs,
        "times": times,
        "final_cost": final_test_cost,
        "training_time": end-start,
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }

