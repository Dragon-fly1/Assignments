import pandas as pd

def imputation(data,type=0):    #type = 0,1,2 for mean,median and mode respectively
    
    imputation_data = dict()    #col:(mean,median,mean of modes/mode)
    
    for col in data.select_dtypes(include='number'):
        if data[col].isnull().any():
            imputation_data[col]=((data[col]).mean(),(data[col]).median(),(data[col]).mode().mean())
    
    for row in range(data.shape[0]):
        for col in data.select_dtypes(include='number'):
            if pd.isnull(data.at[row,col]):
                data.loc[row,col] = imputation_data[col][type]


DATA_FILE = "train.csv"
data = pd.read_csv(DATA_FILE)

imputation(data,1)

data.to_csv("imputated_data.csv",index=False)