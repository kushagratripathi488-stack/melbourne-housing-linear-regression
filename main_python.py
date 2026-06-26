import time 
import matplotlib.pyplot as plt
from src.data_loader import load_csv 
from src.preprocessing import fill_missing_with_mean
from src.preprocessing import winsorization
from src.preprocessing import mean_normalization
from src.train_test_split import train_test_split
from src.pure_python_regression import predic_func
from src.pure_python_regression import cost_func
from src.pure_python_regression import gradient_descent
from src.evaluaton import r2_score
from src.evaluaton import root_mean_squared_error
from src.evaluaton import mean_absolute_error
#Loading the data
def run_pure_python():
    columns,rows=load_csv("D:/CSOC 2026/INTELLIGENCE/Level1 project/melbourne-housing-linear-regression(Pure python)/melb_data.csv")

#Data preprocessing 
    data_column=['Rooms','Bathroom','Car','YearBuilt','Distance','Lattitude','Longtitude',
         'Price']
    feature_column=['Rooms','Bathroom','Car','YearBuilt','Distance','Lattitude','Longtitude',
        ]

    for column in data_column: 
        if column !='Price' :
            rows=fill_missing_with_mean(rows,columns.index(column))
            rows=winsorization(rows,columns.index(column))
            rows=mean_normalization(rows,columns.index(column))

    #Feature Scaling
    data_indices=[columns.index(features) for features in data_column]
    data=[]
    for row in rows : 
        data_row=[]
        for i in data_indices: 
            data_row.append(float(row[i]))
        data.append(data_row)


    #Train Test Split
    X_train,y_train,X_test,y_test=train_test_split(data,0.2)

    #Linear progression model using the gradient descent 
    weights=[0.0]*len(feature_column)
    bias=0.0
    learning_rate = 0.1
    epochs = 1001
    start=time.time()
    weight,bias,costs,times=gradient_descent(X_train,y_train,weights,bias,learning_rate,epochs)
    end=time.time()
    final_cost=cost_func(X_test,y_test,weight,bias)
#Evaluation Criteria 
    y_pred=[predic_func(features,weight,bias) for features in X_test]
    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=root_mean_squared_error(y_test,y_pred)
    print("r2_score : ",r2)
    print("mean absolute value :",mae)
    print("Root mean square error :",rmse)
    print("Total time taken for convergence : ",end-start," seconds")
    print("Final cost of the model : ",final_cost)

    # plt.figure(figsize=(8, 5))

    # plt.plot(times,costs, label="Pure python",marker='o')  
    # plt.xlabel("Time(Seconds)")
    # plt.ylabel("Cost")
    # plt.grid(True)
    # plt.title("Cost vs Time",fontdict={'fontsize': 16, 'fontweight': 'semibold'})
    # plt.show()
    # plt.legend()

    return {
        "costs": costs,
        "times": times,
        "final_cost": final_cost,
        "training_time": end-start,
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }
