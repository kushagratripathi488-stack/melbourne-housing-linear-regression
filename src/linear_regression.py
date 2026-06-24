def predic_func(features,weights,bias) : 
    return bias+sum(f*w for f,w in zip(features,weights))

def cost_func(feature_data,target_data,weights,bias):
    count=len(feature_data)*2
    initial=sum(((predic_func(f,weights,bias)-y)**2 for f,y in zip(feature_data,target_data)))
    cost=initial/count
    return cost
