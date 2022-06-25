# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:43:09 2022

@author: samir kumar das
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
#%matplotlib inline

superstore = pd.read_csv(r"C:\Users\samir kumar das\Downloads\priceOptimize.csv")
superstore.shape

#superstore.rename(columns={'Sales at Cost' : 'SaC'}, inplace=True)
superstore.rename(columns={'NSU' : 'quantity','SP':"price"}, inplace=True)


superstore = superstore.dropna()
superstore.columns

superstore = superstore.loc[superstore['MARGIN%         '] > 0,:]
top_10_items  = superstore['NAME'].value_counts().head(10)
print(top_10_items)

name = input("Enter the product name:")
zone = input("Enter the Zone:")

data = superstore.loc[superstore['NAME'] == name,:]
data_new = data.loc[data['ZONE'] == zone,:]

# revenue
revenue = quantity * price # eq (1)

revenue = NSU * SP
# profit
profit = revenue - cost # eq (2)



# revised profit function
profit = quantity * price - cost # eq (3)

profit = NSV * SP - cost
def find_optimal_price(data_new):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols  
    # demand curve
    #sns.lmplot(x = "price", y = "quantity",data = data_new,fit_reg = True, size = 4)
    # fit OLS model
    model = ols("quantity ~ price", data = data_new).fit()
    # print model summary
    print(model.summary())
    prams = model.params
    prams.Intercept
    prams.price
 # plugging regression coefficients
    # quantity = prams.Intercept + prams.price * price # eq (5)
    # the profit function in eq (3) becomes
    # profit = (prams.Intercept + prams.price * price) * price - cost # eq (6)


   # a range of diffferent prices to find the optimum one
    start_price = data_new.price.min() 
    end_price   = data_new.price.max()
    Price  = np.arange(start_price, end_price,0.05)
    Price = list(Price)

   # assuming a fixed cost
    k1   = data_new['NSV'].div(data_new['quantity'])
    cost = k1.min()
    Revenue = []
    for i in Price:
       quantity_demanded = prams.Intercept + prams.price * i
   
      # profit function
       Revenue.append((i-cost) * quantity_demanded)
   # create data frame of price and revenue
    profit = pd.DataFrame({"Price": Price, "Revenue": Revenue})
    
   #plot revenue against price
    plt.plot(profit["Price"], profit["Revenue"])


   # price at which revenue is maximum


    ind = np.where(profit['Revenue'] == profit['Revenue'].max())[0][0]
    values_at_max_profit = profit.iloc[[ind]]
    return values_at_max_profit


optimal_price = {}
optimal_price[name] = find_optimal_price(data_new)
optimal_price[name] 


######### Check For Different Items ##########

top_10_items  = superstore['NAME'].value_counts().head(10)
print(top_10_items)

name = input("Enter the product name:")
zone = input("Enter the Zone:")

data = superstore.loc[superstore['NAME'] == name,:] 
data_new = data.loc[data['ZONE'] == zone,:]
print(data)

optimal_price[name] = find_optimal_price(data_new)
optimal_price[name]

#print(data_new['price'].max())