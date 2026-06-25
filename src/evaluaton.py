import math 
def r2_score(y_true,y_pred):
    y_mean=sum(y_true)/len(y_true)
    ss_total=sum((y-y_mean)**2 for y in y_true)
    ss_residual=sum((y-y_cap)**2 for y,y_cap in zip(y_true,y_pred))
    r2=1-ss_residual/ss_total
    return r2

def mean_absolute_error(y_true,y_pred): 
    total=sum(abs(y-y_cap) for y,y_cap in zip(y_true,y_pred))
    n=len(y_pred)
    mae=total/n
    return mae

def root_mean_squared_error(y_true,y_pred) : 
    squared_sum=sum((y-y_cap)**2 for y,y_cap in zip(y_true,y_pred))
    n=len(y_pred)
    mse=(squared_sum/n)
    return math.sqrt(mse)

