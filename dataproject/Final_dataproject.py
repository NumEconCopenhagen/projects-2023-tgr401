#Things needed
#Installations
# %pip install git+https://github.com/alemartinello/dstapi

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

# for import of DST
import requests
from IPython.display import display
from io import StringIO
# for fetching data
import pandas_datareader
from dstapi import DstApi


#Fetching
nan1 = DstApi('NAN1')
nan1.tablesummary(language='en')

params = nan1._define_base_params(language='en')

params = {
    'table': 'nan1',
 'format': 'BULK',
 'lang': 'en',
 'variables': [{'code': 'TRANSAKT', 'values': ['B1GQK','P7K','TFSPR','P6D','P31S1MD','P3S13D','P5GD','TFUPR']},
  {'code': 'PRISENHED', 'values': ['LAN_M']},
  {'code': 'Tid', 'values': ['*']}]}


#Rename variable values
nan1_api = nan1.get_data(params=params)
rename = {'B.1*g Gross domestic product':'Y',
          'P.3 Government consumption expenditure':'G',
          'P.31 Private consumption':'C',
          'P.5g Gross capital formation':'I',
          'P.6 Exports of goods and services':'X',
          'P.7 Imports of goods and services':'M',
          'Final demand':'Demand'
}
nan1_api['TRANSAKT']=nan1_api['TRANSAKT'].replace(rename)
nan1_api.sort_values(by=['TID','TRANSAKT'], inplace=True)

#Save the data in a csv file
nan1_api.to_csv('nan1_api.csv', index=False)

dkdata = pd.read_csv('nan1_api.csv')
del dkdata['PRISENHED']


#Ting at gemme og slette
# Group the data by "TRANSAKT" and "TID", and sum the values
grouped = nan1_api.groupby(['TRANSAKT', 'TID'])['INDHOLD'].sum()

# Unstack the "TID" column to create a dataframe with years as columns
df = grouped.unstack(level=-1)


# Print the resulting dataframe


#mere
nan1_api.set_index('TID',inplace=True)
nan1_api.droplevel(0,axis=1).rename_axis(None,axis=1).reset_index()
pvt_df=pd.pivot_table(nan1_api,index='TID',columns='TRANSAKT').reset_index


#Load the dataset again from the csv file
dkdata=pd.read_csv('nan1_api.csv')
#Create a table 
dkbal = dkdata.groupby(['TID', 'TRANSAKT'])['INDHOLD'].mean().unstack(fill_value=0)

dkbal =dkbal.reset_index()

print(dkdata.loc[dkdata['TRANSAKT']=='C'].mean())
print(dkdata.loc[(dkdata['TRANSAKT'] == 'I') & (dkdata['TID'].between(2008,2010))])
display(dkbal.loc[[1990,2000]])

empl_roskilde = dkdata.loc[dkdata['TRANSAKT'] == 'X', :]

# Plot the content of the data frame
empl_roskilde.plot(x='TID',y='INDHOLD',legend=False);

dkbal['NX'].describe()
print(dkdata.loc[(dkdata['TRANSAKT'] == 'I') & (dkdata['TID'].between(2008,2010))])
display(dkbal.loc[[1990,2000]])

c_y_ratio = dkdata.loc[(dkdata['TRANSAKT'] == 'C') & (dkdata['TID'] == 2020), 'INDHOLD'].values[0] / dkdata.loc[(dkdata['TRANSAKT'] == 'Y') & (dkdata['TID'] == 2020), 'INDHOLD'].values[0]
print('C/Y ratio for 2020:', c_y_ratio)

nan1._define_base_params(language='en')