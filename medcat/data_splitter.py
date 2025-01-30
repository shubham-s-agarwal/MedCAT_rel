n_samples =  {"Reason-Drug" :  9982,
"Duration-Drug": 1282,
"ADE-Drug" : 2200,
"Dosage-Drug" : 8436,
"Strength-Drug" : 13396,
"Route-Drug": 11072,
"Frequency-Drug" : 12612,
"Form-Drug" :  13302}

counter = {"Reason-Drug" :  0,
"Duration-Drug": 0,
"ADE-Drug" : 0,
"Dosage-Drug" : 0,
"Strength-Drug" : 0,
"Route-Drug": 0,
"Frequency-Drug" : 0,
"Form-Drug" :  0}

import pandas as pd
csv_path = "/Users/k2370999/Downloads/rels_SA.csv"
df = pd.read_csv(csv_path, index_col=False,
                             encoding='utf-8')

df = df.sample(frac=1)

train_data = []
test_data = []
for i in range(len(df)):
    if counter[df.loc[i, "label"]] < n_samples[df.loc[i, "label"]] * 0.8:
        train_data.append(df.iloc[i])
        counter[df.loc[i, "label"]]+=1
    else:
        test_data.append(df.iloc[i])

train_df = pd.DataFrame(train_data,columns=df.columns)
test_df = pd.DataFrame(test_data,columns=df.columns)

print(df.label.value_counts())
print(train_df.label.value_counts())
print(test_df.label.value_counts())

train_df.to_csv("/Users/k2370999/Downloads/Relation Extraction/train_fixed_final.csv", index=False)
test_df.to_csv("/Users/k2370999/Downloads/Relation Extraction/test_fixed_final.csv", index=False)


