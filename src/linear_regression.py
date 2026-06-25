import time 
def predic_func(features,weights,bias) : 
    return bias+sum(f*w for f,w in zip(features,weights))

def cost_func(feature_data, target_data, weights, bias):
    m = len(feature_data)

    total = sum(
        (predic_func(f, weights, bias) - y) ** 2
        for f, y in zip(feature_data, target_data)
    )
    return total / (2 * m)

def gradient_descent(features_data,target_data,weights,bias,learning_rate,epochs): 
    m=len(features_data)
    prev_cost=cost_func(features_data,target_data,weights,bias)
    costs=[prev_cost]
    start=time.time()
    times=[0]
    for epoch in range(epochs):
        errors = [
    predic_func(f, weights, bias) - y
    for f, y in zip(features_data, target_data)
]
        db=sum(error for error in errors)/m
        dw=[]
        for i in range(len(weights)): 
            derivative=sum((error*f[i] for error,f in zip(errors,features_data)))/m
            dw.append(derivative)
        bias-=learning_rate*db
        for j in range(len(weights)):
            weights[j]-=learning_rate*dw[j]
        
        if epoch % 20 == 0:
            cost = cost_func(features_data,target_data,weights,bias)
            costs.append(cost)
            if epoch % 100 ==0:
                print(f"Epoch {epoch}: Cost = {cost}")
            temp=time.time()
            times.append(temp-start)
            prev_cost=cost
    return weights,bias,costs,times