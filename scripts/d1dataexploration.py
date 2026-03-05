import pandas as pd
import numpy as np
print('The Libraries are imported successfully')
print('Loading the Dataset.....')
try:
  df = pd.read_csv('Telco-Customer-Churn.csv')
  print("Loaded from saved file")
except:
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    print("Loaded from URL")

print(df.head(3))
print(f'{df.shape[0]}:Number of Customers')
print(f'{df.shape[1]}:Number of Features')

for i,col in enumerate(df.columns):
   print(f'{i}-{col}')


print(f'{df.dtypes} Number of datatypes')

type_count=df.dtypes.value_counts()
for dtype,count in type_count.items():
   print(f'{dtype}:{count}')


cat_cols=df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
   print(f'--{col}')

cat_nums=df.select_dtypes(include=['float64','int64']).columns.to_list()
for col in cat_nums:
   print(f'**{col}')


print(f'Maxmium Customer Stayed Months{df['tenure'].max()}')
print(f'Minimum number of Months Stayed by Customer {df['tenure'].min()}')
print(f'Average Number of Months Stayed by customer{df["tenure"].mean():2f}')
