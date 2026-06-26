import time 
import numpy as np

def cost_func(feature_data, target_data, weights, bias):
    m = len(feature_data)
    errors = np.dot(feature_data,weights)+bias-target_data
    total=np.dot(errors,errors)
    return total / (2 * m)

def gradient_descent(features_data,target_data,weights,bias,learning_rate,epochs): 
    m=len(features_data)
    prev_cost=cost_func(features_data,target_data,weights,bias)
    costs=[prev_cost]
    start=time.time()
    times=[0]
    for epoch in range(epochs):
        errors = np.dot(features_data,weights)+bias-target_data
        db=np.mean(errors)
        dw=np.dot(features_data.T,errors)/m
        
        bias-=learning_rate*db
        weights-=learning_rate*dw
        
        if epoch % 20 == 0:
            cost = cost_func(features_data,target_data,weights,bias)
            costs.append(cost)
            if epoch % 100 ==0:
                print(f"Epoch {epoch}: Cost = {cost}")
            temp=time.time()
            times.append(temp-start)
    return weights,bias,costs,times