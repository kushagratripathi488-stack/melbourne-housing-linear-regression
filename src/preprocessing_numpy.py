import numpy as np 
def fill_missing_with_mean(rows,col_idx) :
    column=rows[:,col_idx]
    mean=np.nanmean(column)
    column[np.isnan(column)]=mean
    return rows

def winsorization(data,col_idx) : 
    column=data[:,col_idx]
    q1,q3=np.percentile(data,[25,75])

    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    return np.clip(data,lower,upper)

def mean_normalization(data,col_idx):
    column=data[:,col_idx]
    mean=np.mean(column)
    mx=np.max(column)
    mn=np.min(column)
    data[:,col_idx]=(column-mean)/mx-mn
    return data


    