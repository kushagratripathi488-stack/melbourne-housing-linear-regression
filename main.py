import csv 
from src.data_loader import load_csv 

columns,rows=load_csv("D:/CSOC 2026/INTELLIGENCE/Level1 project/melbourne-housing-linear-regression(Pure python)/melb_data.csv")

print("Columns : ",columns)
print("Total rows : ",len(rows))
print("First row :",rows[0])

