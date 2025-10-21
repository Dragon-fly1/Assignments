import pandas as pd

DATA_FILE = "data.csv"
data = pd.read_csv(DATA_FILE)

target_cols = []    #data to be encoded

for col in target_cols:
    categories = set()
    for row in range(data.shape[0]):
        if pd.isnull(data.at[row,col]):
            continue
        elif data.at[row,col] not in categories:
            categories.add(data.at[row,col])
            data[data.at[row,col]]=0
        data.at[row,data.at[row,col]]=1
    data.drop(col,axis=1,inplace=True)

data.to_csv("one_hot_encoding.csv",index=False)