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