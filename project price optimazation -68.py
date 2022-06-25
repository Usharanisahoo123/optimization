# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:31:27 2022

@author: samir kumar das
"""

import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\samir kumar das\Downloads\priceOptimize.csv")
data.dtypes ### data types all columns ##
data.columns ### columns name ##

del data['UID'] ### delete unnecessery column ##
data.shape ## to know file size

# check for count of NA'sin each column
data.isna().sum()
data.dropna(subset=['MRP','SP','DIS','DIS%'], inplace=True) ### DROP null value 
# Create an imputer object that fills 'Nan' values
# for Mean, Meadian, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer
# Mean Imputer For MRP
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data["MRP"] = pd.DataFrame(mean_imputer.fit_transform(data[["MRP"]]))
data["MRP"].isna().sum()

# Median Imputer FOR SP
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data["SP"] = pd.DataFrame(median_imputer.fit_transform(data[["SP"]]))
data["SP"].isna().sum()  

# Mean Imputer For DIS
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data["DIS"] = pd.DataFrame(mean_imputer.fit_transform(data[["DIS"]]))
data["DIS"].isna().sum()

# Median Imputer FOR DIS%
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data["DIS%"] = pd.DataFrame(median_imputer.fit_transform(data[["DIS%"]]))
data["DIS%"].isna().sum()

data.info
######### FOR NSU COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data.NSU.describe() ### 

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# NSU
plt.bar(height = data.NSU, x = np.arange(1, 37438, 1))
plt.hist(data.NSU) #histogram
plt.boxplot(data.NSU) #boxplot , there is outlier
### we have do winsorisation on NSU column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['NSU'])

data['NSU'] = winsor.fit_transform(data[['NSU']])

######### FOR NSV COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data.NSV.describe()
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# NSV
plt.bar(height = data.NSV, x = np.arange(1, 37438, 1))
plt.hist(data.NSV) #histogram
plt.boxplot(data.NSV) #boxplot , there is outlier
### we have do winsorisation on NSV column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['NSV'])

data['NSV'] = winsor.fit_transform(data[['NSV']])
######### FOR GST Value COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['GST Value'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# GST-Value
plt.bar(height = data['GST Value'], x = np.arange(1, 37438, 1))
plt.hist(data['GST Value']) #histogram
plt.boxplot(data['GST Value']) #boxplot , there is outlier
### we have do winsorisation on NSV column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['GST Value'])

data['GST Value'] = winsor.fit_transform(data[['GST Value']])

######### FOR NSV-GST COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['NSV-GST'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# NSV-GST
plt.bar(height = data['NSV-GST'], x = np.arange(1, 37438, 1))
plt.hist(data['NSV-GST']) #histogram
plt.boxplot(data['NSV-GST']) #boxplot , there is outlier
### we have do winsorisation on NSV column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['NSV-GST'])

data['NSV-GST'] = winsor.fit_transform(data[['NSV-GST']])

######### FOR Sales at Cost COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['Sales at Cost'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Sales at Cost
plt.bar(height = data['Sales at Cost'], x = np.arange(1, 37438, 1))
plt.hist(data['Sales at Cost']) #histogram
plt.boxplot(data['Sales at Cost']) #boxplot , there is outlier
### we have do winsorisation on Sales at Cost column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Sales at Cost'])

data['Sales at Cost'] = winsor.fit_transform(data[['Sales at Cost']])

######### FOR SALES AT COST COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['SALES AT COST'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# SALES AT COST
plt.bar(height = data['SALES AT COST'], x = np.arange(1, 37438, 1))
plt.hist(data['SALES AT COST']) #histogram
plt.boxplot(data['SALES AT COST']) #boxplot , there is outlier
### we have do winsorisation on SALES AT COST column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['SALES AT COST'])

data['SALES AT COST'] = winsor.fit_transform(data[['SALES AT COST']])

######### FOR MARGIN%COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['MARGIN%         '].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# MARGIN%         
plt.bar(height = data['MARGIN%         '], x = np.arange(1, 37438, 1))
plt.hist(data['MARGIN%         ']) #histogram
plt.boxplot(data['MARGIN%         ']) #boxplot , there is outlier
### we have do winsorisation on SALES AT COST column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['MARGIN%         '])

data['MARGIN%         '] = winsor.fit_transform(data[['MARGIN%         ']])



######### FOR Gross Sales COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['Gross Sales'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Gross Sales        
plt.bar(height = data['Gross Sales'], x = np.arange(1, 37438, 1))
plt.hist(data['Gross Sales']) #histogram
plt.boxplot(data['Gross Sales']) #boxplot , there is outlier
### we have do winsorisation on Gross Sales column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Gross Sales'])

data['Gross Sales'] = winsor.fit_transform(data[['Gross Sales']])

######### FOR Gross RGM(P-L) COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['Gross RGM(P-L)'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Gross RGM(P-L)      
plt.bar(height = data['Gross RGM(P-L)'], x = np.arange(1, 37438, 1))
plt.hist(data['Gross RGM(P-L)']) #histogram
plt.boxplot(data['Gross RGM(P-L)']) #boxplot , there is outlier
### we have do winsorisation on Gross RGM(P-L) column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Gross RGM(P-L)'])

data['Gross RGM(P-L)'] = winsor.fit_transform(data[['Gross RGM(P-L)']])

######### Gross Margin %(Q/P*100) COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['Gross Margin %(Q/P*100)'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Gross Margin %(Q/P*100)     
plt.bar(height = data['Gross Margin %(Q/P*100)'], x = np.arange(1, 37438, 1))
plt.hist(data['Gross Margin %(Q/P*100)']) #histogram
plt.boxplot(data['Gross Margin %(Q/P*100)']) #boxplot , there is outlier
### we have do winsorisation on Gross Margin %(Q/P*100) column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Gross Margin %(Q/P*100)'])

data['Gross Margin %(Q/P*100)'] = winsor.fit_transform(data[['Gross Margin %(Q/P*100)']])


######### MRP COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['MRP'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# MRP   
plt.bar(height = data['MRP'], x = np.arange(1, 37438, 1))
plt.hist(data['MRP']) #histogram
import seaborn as sns
sns.boxplot(data.MRP)#boxplot , there is outlier
### we have do winsorisation on Gross Margin %(Q/P*100) column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['MRP'])

data['MRP'] = winsor.fit_transform(data[['MRP']])

######## SP COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['SP'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# SP  
plt.bar(height = data['SP'], x = np.arange(1, 37438, 1))
plt.hist(data['SP']) #histogram
import seaborn as sns
sns.boxplot(data.SP)#boxplot , there is outlier
### we have do winsorisation on Gross Margin %(Q/P*100) column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['SP'])

data['SP'] = winsor.fit_transform(data[['SP']])

######## DIS COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['DIS'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# DIS  
plt.bar(height = data['DIS'], x = np.arange(1, 37438, 1))
plt.hist(data['DIS']) #histogram
import seaborn as sns
sns.boxplot(data.DIS)#boxplot , there is outlier
### we have do winsorisation on Gross Margin %(Q/P*100) column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['DIS'])

data['DIS'] = winsor.fit_transform(data[['DIS']])

######## DIS% COLUMN #####
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

data['DIS%'].describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# DIS  
plt.bar(height = data['DIS%'], x = np.arange(1, 37438, 1))
plt.hist(data['DIS%']) #histogram
plt.boxplot(data['DIS%'])#boxplot , there is outlier
### we have do winsorisation on Gross Margin %(Q/P*100) column
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['DIS%'])

data['DIS%'] = winsor.fit_transform(data[['DIS%']])


