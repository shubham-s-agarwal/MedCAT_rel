import pickle
with open ('relcat_traindata_labels_expo', 'rb') as fp:
    itemlist = pickle.load(fp)

print(itemlist[0])

# import pandas as pd
#
# csv_path = "/Users/k2370999/Downloads/RELCAT_TRAIN_CSV_v2.csv"
# df = pd.read_csv(csv_path, index_col=False,
#                              encoding='utf-8')
#
# train_data_expo = df.text.tolist()
# train_data_labels = df.label.tolist()
# train_data_ent1 = []
# train_data_ent2 = []
# print(train_data_expo[0])
#
# for i in train_data_expo:
#     i = i.split(" ")
#     for j in range(len(i)):
#         if i[j]=='[s1]':
#
#             # print(i)
#             this_ent = []
#             k = j + 1
#             while True:
#                 if '[e1]' in i[k]:
#                     break
#                 else:
#                     this_ent.append(i[k])
#                     k=k+1
#             train_data_ent1.append(this_ent)
#
#         if i[j]=='[s2]':
#             # print(i)
#             this_ent = []
#             k = j + 1
#             while True:
#                 if '[e2]' in i[k]:
#                     break
#                 else:
#                     this_ent.append(i[k])
#                     k=k+1
#             train_data_ent2.append(this_ent)
# # print(train_data_ent2)
#
# import pickle
# with open('relcat_traindata_ent1_expo', 'wb') as fp:
#     pickle.dump(train_data_ent1, fp)
#     print("File written!")
#
# with open('relcat_traindata_ent2_expo', 'wb') as fp:
#     pickle.dump(train_data_ent2, fp)
#     print("File written!")
#
# # import pickle
# #
# # with open('relcat_traindata_expo', 'wb') as fp:
# #     pickle.dump(train_data_expo, fp)
# #     print("File written!")
# #
# # with open('relcat_traindata_labels_expo', 'wb') as fp:
# #     pickle.dump(train_data_labels, fp)
# #     print("File written!")
