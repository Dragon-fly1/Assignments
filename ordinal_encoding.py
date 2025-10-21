import pandas as pd

DATA_FILE = " "    #path to data file
data = pd.read_csv(DATA_FILE)

target_cols = []    #data columns to be encoded

for col in target_cols:
    count = 0
    encountered_categories = dict()
    for row in range(data.shape[0]):
        if pd.isnull(data.at[row,col]):
            data.at[row,col]= -1
        elif data.at[row,col] in encountered_categories:
            data.at[row,col]= encountered_categories[data.at[row,col]]
        else:
            encountered_categories[data.at[row,col]] = count
            data.at[row,col]= encountered_categories[data.at[row,col]]
            count+=1

data.to_csv("ordinal_encoding.csv",index=False)
