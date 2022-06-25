# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:42:54 2022

@author: samir kumar das
"""

import numpy as np
import pandas as pd
price = pd.read_csv(r"C:\Users\samir kumar das\Downloads\priceOptimize.csv")
price.shape ## to know file size

price.dtypes ## to know file data types
### to know if there is duplicate value or not ###
duplicate = price.duplicated()
duplicate
sum(duplicate)
##  so there is no duplicate value ##

del price['UID']## there is no need this  UID columns
price.info

price.columns

######### FOR NSU COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

price.NSU.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# NSU
plt.bar(height = price.NSU, x = np.arange(1, 37438, 1))
plt.hist(price.NSU) #histogram
plt.boxplot(price.NSU) #boxplot , there is outlier
### we have do winsorisation on NSU column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['NSU'])

price['NSU'] = winsor.fit_transform(price[['NSU']])

######### FOR NSV COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

price.NSV.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# NSU
plt.bar(height = price.NSV, x = np.arange(1, 37438, 1))
plt.hist(price.NSV) #histogram
plt.boxplot(price.NSV) #boxplot , there is outlier
### we have do winsorisation on NSV column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['NSV'])

price['NSV'] = winsor.fit_transform(price[['NSV']])

######### FOR GST Value COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

price['GST Value'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# NSU
plt.bar(height = price.C, x = np.arange(1, 37438, 1))
plt.hist(price['GST Value']) #histogram
plt.boxplot(price['GST Value']) #boxplot , there is outlier
### we have do winsorisation on NSV column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['GST Value'])

price['GST Value'] = winsor.fit_transform(price[['GST Value']])

price['Gross RGM(P-L)'].describe()





price.columns


lower=[]
upper=[]
list=['A', 'B', 'C', 'B-GST', 'E','F', 'G         ', 'H', 'Gross RGM(P-L)', 'Gross Margin %(Q/P*100)','K', 'L', 'M', 'M%']
for i in list:
    IQR = price[i].quantile(0.75) - price[i].quantile(0.25)
    lower_limit = price[i].quantile(0.25) - (IQR * 1.5)
    lower.append(lower_limit)
    upper_limit = price[i].quantile(0.75) + (IQR * 1.5)
    upper.append(upper_limit)
def outlier(list):
    j=0
    z=0
    for x in list:
        price[x] = pd.DataFrame(np.where(price[x] > upper[z],upper[z], np.where(price[x] < lower[j], lower[j],price[x])))
        z=z+1
        j=j+1
    return price
df2=outlier(list)
    
import seaborn as sns
sns.boxplot(data=df2)

#### COUNT UNIQUE VALUE ####
price.NAME.unique()
price['NAME'].value_counts().to_dict()
price['ZONE'].value_counts().to_dict()
price['Brand'].value_counts().to_dict()
price['MC'].value_counts().to_dict()

#### creating frequency encoder
lis = ['NAME','Brand','MC','ZONE']

encoder_dictionary = {}
for var in lis:
    encoder_dictionary[var]=(price[var].value_counts()/len(price)).to_dict()

   
for var in lis:
    price[var] = price[var].map(encoder_dictionary[var])
price.head()    

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)






















