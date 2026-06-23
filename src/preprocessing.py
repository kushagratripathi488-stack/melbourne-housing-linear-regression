def fill_missing_with_mean(data,col_idx) :
    total=0.0
    count=0
    for row in data : 
        value=row[col_idx]
        if value!="NaN" : 
            total+=float(value)
            count+=1
    mean=total/count
    for row in data : 
        if row[col_idx]=="NaN" : 
            row[col_idx]=mean
    return mean

def winsorization(data,col_idx) : 
    values=sorted(float(row[col_idx] for row in data))
    n=len(values)
    q1=values[n//4]
    q3=values[3*n//4]

    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    for row in data : 
        value=row[col_idx]
        if(value < lower) : 
            row[col_idx]=lower
        if(value>upper): 
            row[col_idx]=upper
    
    return data 

def remove_row_wrt_column(data,col_idx):
    cleaned_data=[]
    for row in data :
        if row[col_idx] !="NaN":
            cleaned_data.append(row)
    return cleaned_data

def mean_normalization(data,col_idx):
    total=0.0
    count=0
    max_value=max(float(row[col_idx]) for row in data)
    min_value=min(float(row[col_idx]) for row in data)
    for row in data: 
        total+=float(row[col_idx])
        count+=1
    mean=total/count
    diff=max_value-min_value
    for row in data: 
        value=float(row[col_idx])
        row[col_idx]=(value-mean)/diff
    return data


    