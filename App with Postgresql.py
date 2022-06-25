# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:07:43 2022

@author: samir kumar das
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import psycopg2 as psy
connection = psy.connect(user = "postgres",
                         password = "usha#1234",
                         host = "localhost",
                         port = "5432",
                         database = "Project")
cur=connection.cursor()
curs = connection.cursor()
curs.execute("ROLLBACK")
connection.commit()
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

data = data.loc[data['MARGIN%         '] > 0,:]

st.title('Price Optimization')

Unique_Products =pickle.load(open('Unique_Products.pkl','rb'))
Zone = pickle.load(open('Zone.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

Selected_Product_Name = st.selectbox(
    'Select Product Name',
     (Unique_Products.values))

Selected_Zone = st.selectbox(
    'Select Zone',
     (Zone.values))

data = data.loc[data['NAME'] == Selected_Product_Name,:]
data_new = data.loc[data['ZONE'] == Selected_Zone,:]
values_at_max_profit = 0
def find_optimal_price(data_new):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    # demand curve
    # sns.lmplot(x = "price", y = "quantity",data = data_new,fit_reg = True, size = 4)
    # fit OLS model
    model = ols("quantity ~ price", data=data_new).fit()
    # print model summary
    print(model.summary())
    prams = model.params

    # plugging regression coefficients
    # quantity = prams.Intercept + prams.price * price # eq (5)
    # the profit function in eq (3) becomes
    # profit = (prams.Intercept + prams.price * price) * price - cost # eq (6)

    # a range of diffferent prices to find the optimum one
    start_price = data_new.price.min()
    end_price = data_new.price.max()
    Price = np.arange(start_price, end_price, 0.05)
    Price = list(Price)

    # assuming a fixed cost
    k1 = data_new['NSV'].div(data_new['quantity'])
    cost = k1.min()
    Revenue = []
    for i in Price:
        quantity_demanded = prams.Intercept + prams.price * i

        # profit function
        Revenue.append((i - cost) * quantity_demanded)
    # create data frame of price and revenue
    profit = pd.DataFrame({"Price": Price, "Revenue": Revenue})

    # plot revenue against price
    #plt.plot(profit["Price"], profit["Revenue"])

    # price at which revenue is maximum

    ind = np.where(profit['Revenue'] == profit['Revenue'].max())[0][0]
    values_at_max_profit = profit.iloc[[ind]]
    return values_at_max_profit


#optimal_price = {}
#optimal_price[Selected_Product_Name] = find_optimal_price(data_new)
#optimal_price[Selected_Product_Name]

if st.button('Predict Optimized Price'):
    values_at_max_profit = find_optimal_price(data_new)
    st.write('Optimized Price of the Product', values_at_max_profit )

