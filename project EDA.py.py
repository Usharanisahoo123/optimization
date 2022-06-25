# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:26:08 2022

@author: samir kumar das
"""

##pip install psycopg2
#pip install streamlit
import numpy as np
import pandas as pd
import psycopg2 as psy
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
connection = psy.connect(user = "postgres",
                         password = "usha#1234",
                         host = "localhost",
                         port = "5432",
                         database = "Project")

cur = connection.cursor()
cur.execute('SELECT * FROM public."DATA"')

#cursor.execute('SELECT * FROM public."DATA"')
data = cur.fetchall()
data = pd.DataFrame(data)

### rename the coulmns name 
data.rename(columns={0:"UID",1:"NAME",2:"ZONE",3:"Brand",4:"MC",5:"Fdate",6:"NSU",7:"NSV",8:"GST Value",9:"NSV-GST",10:"Sales at Cost",11:"SALES AT COST",12:"MARGIN%         ",13:"Gross Sales",14:"Gross RGM(P-L)",15:"Gross Margin %(Q/P*100)",16:"MRP",17:"SP",18:"DIS",19:"DIS%"},inplace=True)
data.columns
del data['UID'] ### delete unnecessery column ##
data.shape ## to know file size

# check for count of NA'sin each column
data.isna().sum()
data.dropna(subset=['MRP','SP','DIS','DIS%'], inplace=True) ### DROP null value 
data.info
data.dtypes
#@@@@@ CHANGING THE DATA TYPES
#data["UID"] = str(data['UID'])
data['Fdate'] = pd.to_datetime(data['Fdate'])
data["NSU"] = pd.to_numeric(data["NSU"])
data["NSV"] = pd.to_numeric(data["NSV"])
data["GST Value"] = pd.to_numeric(data["GST Value"])
data["NSV-GST"] = pd.to_numeric(data["NSV-GST"])
data["Sales at Cost"] = pd.to_numeric(data["Sales at Cost"])
data["SALES AT COST"] = pd.to_numeric(data["SALES AT COST"])
data["MARGIN%         "] = pd.to_numeric(data["MARGIN%         "])
data["Gross Sales"] = pd.to_numeric(data["Gross Sales"])
data["Gross RGM(P-L)"] = pd.to_numeric(data["Gross RGM(P-L)"])
data["Gross Margin %(Q/P*100)"] = pd.to_numeric(data["Gross Margin %(Q/P*100)"])
data["MRP"] = pd.to_numeric(data["MRP"])
data["SP"] = pd.to_numeric(data["SP"])
data["DIS"] = pd.to_numeric(data["DIS"])
data["DIS%"] = pd.to_numeric(data["DIS%"])
data.apply(lambda x: sum(x.isnull()),axis=0)

from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data["MRP"] = pd.DataFrame(median_imputer.fit_transform(data[["MRP"]]))
data["SP"] = pd.DataFrame(median_imputer.fit_transform(data[["SP"]]))
data["DIS"] = pd.DataFrame(median_imputer.fit_transform(data[["DIS"]]))
data["DIS%"]=pd.DataFrame(median_imputer.fit_transform(data[["DIS%"]]))
data.isna().sum()

#Replace the outlier with the lower_limit and upper_limit
lower=[]
upper=[]
list=['NSU', 'NSV','GST Value', 'NSV-GST', 'Sales at Cost', 'SALES AT COST', 'MARGIN%         ','Gross Sales', 'Gross RGM(P-L)', 'Gross Margin %(Q/P*100)', 'MRP', 'SP','DIS', 'DIS%']
for i in list:
    IQR = data[i].quantile(0.75) - data[i].quantile(0.25)
    lower_limit = data[i].quantile(0.25) - (IQR * 1.5)
    lower.append(lower_limit)
    upper_limit = data[i].quantile(0.75) + (IQR * 1.5)
    upper.append(upper_limit)

# Now let's replace the outliers by the lower limit and upper limit
def outlier(list):
    z=0
    j=0
    for x in list:
   
        data[x] = pd.DataFrame(np.where(data[x] > upper[z], upper[z], np.where(data[x] < lower[j], lower[j], data[x])))
        z=z+1
        j=j+1
    return data
data1=outlier(list)
superstore = data1