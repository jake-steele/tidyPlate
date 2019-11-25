# Goals for project:
# 1.0 Input xlsx: two separate 8x12 tables (one layout, one raw data)
# 1.1 Row and Col headers, same sheet
# 1.2 Table of standard concentrations and units
# 2.0 Link layout label for each well to raw data reading
# 3.0 Average raw data of wells labeled 'BLANK' (More than one BLANK?)
# 3.1 Use BLANK avg to generate -BLANK values for each well
# 4.0 Generate and optimize Linear/5PL Std curve
# 5.0 Insert Std Curve graph as image in xlsx
# https://stackoverflow.com/questions/30716911/writing-pandas-matplotlib-image-directly-into-xlsx-file
# 6.0 Calc ind. and avg. values for each sample
# 6.1 Insert new graph containing calculated values and statistics

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


excel_file = "C:/Users/Jake/Documents/Programming/tidyPlate/sampleData.xlsx"
layout_df = pd.read_excel(excel_file, sheet_name='Input', index_col=0, nrows=8)
layout_df.index.names = [None]
layout_df


# In[3]:


raw_df = pd.read_excel(excel_file, sheet_name='Input', index_col=0, skiprows=11, nrows=8)
raw_df.index.names = [None]
raw_df


# In[4]:


combined_df = pd.DataFrame({
    'row': pd.Series(['A','B','C','D','E','F','G','H']).repeat(12),
    'col': pd.np.tile(np.array(np.arange(1,13)),8)},
    columns=['row','col'])
combined_df


# In[5]:


combined_df['well'] = combined_df['row'] + combined_df['col'].map(str)
combined_df.reset_index(drop=True,inplace=True)
combined_df


# In[6]:


labels_ser = [''] * 96
raw_ser = [0] * 96
for index, row in combined_df.iterrows():
    labels_ser[index] = layout_df[row['col']][row['row']]
    raw_ser[index] = raw_df[row['col']][row['row']]
combined_df['label'] = labels_ser
combined_df['raw'] = raw_ser
combined_df


# In[7]:


STD_BLANK = combined_df.loc[combined_df['label'] == 'STD BLANK']['raw'].mean()
STD_BLANK


# In[8]:


SAMP_BLANK = combined_df.loc[combined_df['label'] == 'BLANK']['raw'].mean()
SAMP_BLANK


# In[9]:


minBlk_ser = [0] * 96
for index, row in combined_df.iterrows():
    if 'STD' in row['label']:
        if STD_BLANK != 0:
            minBlk_ser[index] = row['raw'] - STD_BLANK
        else:
            minBlk_ser[index] = row['raw'] - SAMP_BLANK
    else:
        minBlk_ser[index] = row['raw'] - SAMP_BLANK
combined_df['minusBlank'] = minBlk_ser
combined_df


# In[10]:


user_std_options = pd.read_excel(excel_file, sheet_name='Input', index_col=0, skiprows=22, nrows=4, usecols=[0,1])
user_std_options.index.names = [None]
user_std_options


# In[11]:


std_df = pd.DataFrame()
if user_std_options['userPref']['Serial Diluted (y/n)?'] == 'y':
    std_df = pd.read_excel(excel_file, sheet_name='Input', index_col=1, skiprows=28,
                           nrows=user_std_options['userPref']['# of Concentrations:'],
                          usecols=[0,1,2])
else:
    std_df = pd.read_excel(excel_file, sheet_name='Input', index_col=1, skiprows=46,
                           nrows=user_std_options['userPref']['# of Concentrations:'],
                          usecols=[0,1,2])
std_df


# In[12]:


std_rep_dict = { }
for name, row in std_df.iterrows():
    if name not in std_rep_dict.keys():
        std_rep_dict[name] = []
std_rep_dict


# In[13]:


for index, row in combined_df.iterrows():
    if row['label'] in std_rep_dict.keys():
        std_rep_dict[row['label']].append(row['minusBlank'])
std_rep_dict


# In[14]:


std_rep_df = pd.DataFrame.from_dict(std_rep_dict, orient='index')
std_rep_df


# In[15]:


STD_REPS = 0
for key in std_rep_dict.keys():
    if len(std_rep_dict[key]) > STD_REPS:
        STD_REPS = len(std_rep_dict[key])
STD_REPS


# In[16]:


col_rep_names = []
for c in range(STD_REPS):
    col_rep_names.append('Rep' + str(c + 1))
std_rep_df.columns = col_rep_names
std_rep_df


# In[17]:


std_df = pd.concat([std_df, std_rep_df], axis=1)
std_df


# In[18]:


std_df_rep_cols = [col for col in std_df.columns if 'Rep' in col]
std_df_rep_cols


# In[19]:


std_df['avg'] = std_df.loc[:,std_df_rep_cols].mean(axis=1)
std_df


# In[20]:


std_df['sd'] = std_df.loc[:,std_df_rep_cols].std(axis=1)
std_df


# In[21]:


std_df['cv'] = std_df.loc[:,'sd'] / std_df.loc[:,'avg']
std_df


# In[ ]:




