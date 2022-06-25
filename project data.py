# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:47:13 2022

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




          #### MODEL BUILDING ####

superstore.rename(columns={'NSU' : 'quantity','SP':"price"}, inplace=True)


superstore = superstore.dropna()

superstore.isnull().sum()

superstore.columns

superstore = superstore.loc[superstore['MARGIN%         '] > 0,:]          

top_10_items  = superstore['NAME'].value_counts().head(10)
print(top_10_items)

name = input("Enter the product name:")
 
zone = input("Enter the Zone:")

data = superstore.loc[superstore['NAME'] == name,:]
data_new = data.loc[data['ZONE'] == zone,:]

# revenue
#revenue = superstore.quantity * superstore.price # eq (1)

#revenue = superstore.NSU * superstore.SP
# profit
#profit = revenue - cost # eq (2)



# revised profit function
#profit = quantity * price - cost # eq (3)

#profit = NSV * SP - cost
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


########################
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
%matplotlib inline

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






















import pickle

Unique_Products = superstore['NAME'].unique()

Unique_Products = pd.Series(Unique_Products)

pickle.dump(Unique_Products,open('Unique_Products.pkl','wb'))

Zone = superstore['ZONE'].unique()

Zone = pd.Series(Zone)

pickle.dump(Zone,open('Zone.pkl','wb'))

Model = superstore['ZONE'].unique()

Model = pd.Series(Model)

pickle.dump(Zone,open('Model.pkl','wb'))







Unique_Product=

############### 


########### DEPLOMENT ####
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

cur = connection.cursor()
cur.execute('SELECT * FROM public."DATA"')
connection.commit()
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

data = data.drop_duplicates()

st.title('Price Optimization')

Unique_Products =pickle.load(open('Unique_Products.pkl','rb'))
Zone = pickle.load(open('Zone.pkl','rb'))



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
