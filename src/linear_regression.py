def predic_func(features,weights,bias) : 
    return bias+sum(f*w for f,w in zip(features,weights))

def cost_func(feature_data,target_data,weights,bias):
    count=len(feature_data)*2
    initial=sum(((predic_func(f,weights,bias)-y)**2 for f,y in zip(feature_data,target_data)))
    cost=initial/count
    return cost

def gradient_descent(features_data,target_data,weights,bias,learning_rate,epochs): 
    m=len(features_data)
    for epoch in range(epochs): 
        db=sum(((predic_func(f,weights,bias)-y) for f,y in zip(features_data,target_data)))/m
        dw=[]
        for i in range(len(weights)): 
            derivative=sum(((predic_func(f,weights,bias)-y)*f[i] for f,y in zip(features_data,target_data)))/m
            dw.append(derivative)
        bias-=learning_rate*db
        for j in range(len(weights)):
            weights[j]-=learning_rate*dw[j]
        if epoch % 100 == 0:
            cost = cost_func(features_data,target_data,weights,bias)
            print(f"Epoch {epoch}: Cost = {cost}")
    return weights,bias