import csv

def load_csv(path):
    with open (path,"r") as file:
        reader=csv.reader(file)

        columns=next(reader)
        
        rows=[]
        for row in reader : 
            rows.append(row)

        return columns,rows
    
