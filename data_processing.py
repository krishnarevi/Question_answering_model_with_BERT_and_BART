# -*- coding: utf-8 -*-
"""Data_processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/krishnarevi/END2.0/blob/main/Data_processing.ipynb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:19:13 2021

@author:  KNI9KOR

"""


import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#%%
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        elif (fullPath[-4:]=='xlsx'):
            allFiles.append(fullPath)

    return allFiles
path=r'D:\personal\TSAI\Capstone\END2QnADatasets-main'

#%%
folder = "corrected_format"
os.chdir(path)
print("current dir is: %s" % (path))

if os.path.isdir(folder):
    print("Report directory exists")
else:
    print("Report directory Doesn't exists, creating one")
    os.mkdir(folder)


#%%
"Stackoverflow team"
df_sof = pd.read_excel(path+r'\\Combined_annotated_QandA_stackoverflow_v3.xlsx')
print(df_sof.isna().sum())
print(df_sof.info())
df_sof.to_json(path+'\\corrected_format\\stackoverflow_qna.json', orient='records', lines=True)
# df_sof.to_json(path+'\\corrected_format\\stackoverflow_qna_2.json', orient='records')
#format json dictionary {x:"str",y:"str",z:"str"} ,{x:"str",y:"str",z:"str"}
#%%
df_1= pd.read_json(path+r'\\corrected_format\stackoverflow_qna.json',lines=True)


#%%
"Pytorch discuss team 1"
# df_pyd_1 = pd.read_json(path+r'\\Json_Pytoch_Discuss.json',lines=True,orient='data')
df_pyd_1 =pd.read_json(path+r'\\Json_Pytoch_Discuss.json',orient='split')
df_pyd_1.to_excel(path+r'\\pytorch_discuss_1.xlsx',index= False)
# remove manually unwanted columns
df_2 = pd.read_excel(path+r'\\corrected_format\\excel\\pytorch_discuss_1.xlsx')
print(df_2 .isna().sum())
print(df_2 .info())
# re_df_pyd_1.to_json(path+'\\corrected_format\\Pytoch_Discuss_1.json', orient='records', lines=True)
#%%
"Pytorch discuss team 2"
df_pyd_2 = pd.read_json(path+r'\\pytorch_discuss_qa.json',orient='records')
print(df_pyd_2.isna().sum())
print(df_pyd_2.info())
df_pyd_2 =df_pyd_2[['x','z','y']]
df_pyd_2.to_excel(path+r'\\corrected_format\\excel\\pytorch_discuss_qa.xlsx',index=False)
df_3 = pd.read_excel(path+r'\\corrected_format\\excel\\pytorch_discuss_qa.xlsx')
print(df_3.isna().sum())
print(df_3.info())

# re_df_pyd_2.to_json(path+'\\corrected_format\\Pytoch_Discuss_2.json', orient='records', lines=True)
#%%
"Pytorch github team 1"
df_pygit_1 = pd.read_json(path+r'\\data_pytorch_github.json',orient='records')
print(df_pygit_1.info())
print(df_pygit_1.isna().sum())
df_pygit_1=df_pygit_1.dropna()#remove missing values
df_pygit_1.to_excel(path+r'\\corrected_format\\excel\\data_pytorch_github.xlsx',index =False)

df_4 = pd.read_excel(path+r'\\corrected_format\\excel\\data_pytorch_github.xlsx')
print(df_4.isna().sum())
print(df_4.info())

# df_pygit_1.to_json(path+'\\corrected_format\\Pytoch_Github_1.json', orient='records', lines=True)
#%%
"Pytorch github team 2"
df_pygit_2 = pd.read_json(path+r'\\data.json',orient='records')

print(df_pygit_2.info())
print(df_pygit_2.isna().sum())
df_pygit_2 = df_pygit_2.dropna().reset_index(drop =True)
df_pygit_2 = df_pygit_2[['x','z','y']]

print(df_pygit_2.info())
print(df_pygit_2.isna().sum())
df_pygit_2.to_excel(path+r'\\corrected_format\\excel\\data.xlsx',index = False)


df_5 = pd.read_excel(path+r'\\corrected_format\\excel\\data.xlsx')
print(df_5.isna().sum())
print(df_5.info())

# df_pygit_2.to_json(path+'\\corrected_format\\Pytoch_Github_2.json', orient='records', lines=True)
#%%
"Youtube video "

df_1 = pd.read_json(path+r'\\capstone-1-100.json',orient='split')
df_2 = pd.read_json(path+r'\\capstone-100-2.json',orient='split')
df_3 = pd.read_json(path+r'\\capstone-300.json',orient='split')
pdList = [df_1, df_2, df_3]  # List of your dataframes
yt_df = pd.concat(pdList)
yt_df = yt_df[['x','z','y']]
print(yt_df.info())
print(yt_df.isna().sum())
yt_df.to_excel(path + r'\\corrected_format\\excel\\youtube_data.xlsx',index =False)


df_6 = pd.read_excel(path+r'\\corrected_format\\excel\\youtube_data.xlsx')
print(df_6.isna().sum())
print(df_6.info())

# new_df.to_json(path+'\\corrected_format\\Pytoch_youtube.json', orient='records', lines=True)
#%%
"Pytorch documentation "
df_pydoc = pd.read_json(path+r'\\pytorchdocumentatioQA.json',orient='records')
print(df_pydoc.info())
print(df_pydoc.isna().sum())
df_pydoc.rename(columns = {'X':'x','Z':'z','Y':'y'}, inplace = True)
df_pydoc.dropna()
df_pydoc = df_pydoc[['x','z','y']]
print(df_pydoc.info())
df_pydoc.to_excel(path + r'\\corrected_format\\excel\\Pytorch_doc_data.xlsx',index =False)


df_7 = pd.read_excel(path+r'\\corrected_format\\excel\\Pytorch_doc_data_useful2k.xlsx')
print(df_7.isna().sum())
print(df_7.info())
#%%
list_df = [df_1,df_2,df_3,df_4,df_5,df_6,df_7]
all_data = pd.concat(list_df)
print(all_data.info())
print(all_data.isna().sum())

# print("Duplicate Rows :")
# duplicate =all_data[all_data.duplicated(keep = False)]
# duplicate=duplicate.sort_values(['x']).reset_index(drop=True) # no hard constraint

"remove duplicate rows "
data = all_data.drop_duplicates(ignore_index =True)
#%%
"data cleaning "
from copy import deepcopy
import re
df =deepcopy(data)

def dfreplace(df, *args, **kwargs):
    s = pd.Series(df.values.flatten())
    for i,j in zip(*args, **kwargs):
        s = s.str.replace(i, j)
    return pd.DataFrame(s.values.reshape(df.shape), df.index, df.columns)

df= dfreplace(df, ["&quot;",":&quot" ],
                  ['\"','\"'])
cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
cleanr_1 =  re.compile('" rel="nofollow noreferrer">')
df['y'] = [ re.sub(cleanr, '', x) for x in df.y]
df['y'] = [ re.sub(cleanr_1, '', x) for x in df.y]
df['y'] = df['y'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
df['y'] = [x.split('\\r\\n') for x in df.y]
df['y'] = ["\n".join(l for l in s if l).replace(r"```", '') for s in df.y]
print(df.isna().sum())
print(df.info())

#%%
df['z'] = [ re.sub(cleanr, '', x) for x in df.z]
df['z'] = [ re.sub(cleanr_1, '', x) for x in df.z]

df['z'] = [x.split('\\r\\n') for x in df.z]
df['z'] = ["\n".join(l for l in s if l).replace(r"```", '') for s in df.z]

print(df.isna().sum())
print(df.info())

final_df = df.drop_duplicates(
  subset = ['x', 'y'],
  keep = 'last').reset_index(drop = True)
final_df.to_excel(r'D:\personal\TSAI\Capstone\END2QnADatasets-main\corrected_format\excel\processed_data.xlsx',index =False)
#%%
from copy import deepcopy
final_df=pd.read_excel(r'D:\personal\TSAI\Capstone\END2QnADatasets-main\corrected_format\processed_data.xlsx')
dk = deepcopy(final_df)
dk = dk.drop_duplicates(
  subset = ['x'],
  keep = 'first').reset_index(drop = True)

dk=dk.dropna().reset_index(drop=True)
# "Aggregate data based on unique question"
# agg_data= dk.groupby(['x']).agg({'y': 'sum','z' : 'sum'})
# agg_data = agg_data.reset_index()
# agg_data.to_excel(r'D:\personal\TSAI\Capstone\END2QnADatasets-main\corrected_format\excel\agg_data.xlsx',index =False)

master= dk.to_dict(orient='records')
dk.to_excel(path+'\\corrected_format\\master.xlsx',index=False)
import json
with open(path+'\\corrected_format\\master.json', 'w') as fout:
    json.dump(master, fout)
#%%
"create json file for all context "
context_df = final_df[['z']].drop_duplicates(ignore_index =True)

context = context_df.to_dict(orient='records')
with open(path+'\\corrected_format\\context_master.json', 'w') as fout:
    json.dump(context, fout)


#%%
'Split data for modeling¶'

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(dk, test_size=0.2, random_state=123)

print('Train values shape:', X_train.shape)
print('Test values shape:', X_test.shape)


X_train['id'] = range(1, len(X_train) + 1)
X_test['id'] = range(1, len(X_test) + 1)
X_train.to_excel(path+'\\corrected_format\\train_data.xlsx',index=False)
X_test.to_excel(path+'\\corrected_format\\test_data.xlsx',index=False)
train = X_train.to_dict(orient='records')
test = X_test.to_dict(orient='records')
import json
with open(path+'\\corrected_format\\train_data.json', 'w') as fout:
    json.dump(train, fout)
with open(path+'\\corrected_format\\test_data.json', 'w') as fout:
    json.dump(test, fout)
# #%%
# with open(path+'\\corrected_format\\train_data.json') as f:
#         g = json.load(f)
# #%%