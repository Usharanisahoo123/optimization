import psycopg2 as psy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy import stats
from feature_engine.encoding import CountFrequencyEncoder as cf
from sklearn.preprocessing import StandardScaler as sc
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier

connection = psy.connect(user = "postgres",
                         password = "M@nita99",
                         host = "localhost",
                         port = "5432",
                         database = "Project")

cur = connection.cursor()
cur.execute('SELECT * FROM public."DATA"')

#cursor.execute('SELECT * FROM public."DATA"')
data = cur.fetchall()
data = pd.DataFrame(data)
#@@@@@ RENAMING COLUMNS
data = data.rename(columns={0:"UID",1:"NAME",2:"ZONE",3:"BRAND",4:"MC",5:"F_DATE",6:"NSU"})
data = data.rename(columns={7:"NSV",8:"GST_VALUE",9:"NSV-GST",10:"SALES_AT_COST",11:"SALESATCOST",12:"MARGIN%",13:"GROSS_SALE"})
data = data.rename(columns={14:"GROSS_RGM(P-L)",15:"GROSS_MARGIN",16:"MRP",17:"SP",18:"DISCOUNT",19:"DIS%"})
data.info()
data_0 = data
#@@@@@ CHANGING THE DATA TYPES
#data["UID"] = str(data['UID'])
data['F_DATE'] = pd.to_datetime(data['F_DATE'])
data["NSU"] = pd.to_numeric(data["NSU"])
data["NSV"] = pd.to_numeric(data["NSV"])
data["GST_VALUE"] = pd.to_numeric(data["GST_VALUE"])
data["NSV-GST"] = pd.to_numeric(data["NSV-GST"])
data["SALES_AT_COST"] = pd.to_numeric(data["SALES_AT_COST"])
data["SALESATCOST"] = pd.to_numeric(data["SALESATCOST"])
data["MARGIN%"] = pd.to_numeric(data["MARGIN%"])
data["GROSS_SALE"] = pd.to_numeric(data["GROSS_SALE"])
data["GROSS_RGM(P-L)"] = pd.to_numeric(data["GROSS_RGM(P-L)"])
data["GROSS_MARGIN"] = pd.to_numeric(data["GROSS_MARGIN"])
data["MRP"] = pd.to_numeric(data["MRP"])
data["SP"] = pd.to_numeric(data["SP"])
data["DISCOUNT"] = pd.to_numeric(data["DISCOUNT"])
data["DIS%"] = pd.to_numeric(data["DIS%"])

#@@@@@@@ EXPORT DATA TO EXCEL FILE
#data.to_excel("C:\\Users\\Biki\\Desktop\\ppa\\data.xlsx")

#@@@@@@@ FEATURE ENGINEERING

data['WEIGHT'] = data['NAME'].astype(str).replace('\.0', '', regex=True)


data["WEIGHT"] = data["WEIGHT"].str.replace('1k','1000')
data["WEIGHT"] = data["WEIGHT"].str.replace('1kg','1000')
data["WEIGHT"] = data["WEIGHT"].str.replace('1 kg','1000')
data["WEIGHT"] = data["WEIGHT"].str.replace('1Kg','1000')
data["WEIGHT"] = data["WEIGHT"].str.replace('1K','1000') 
data["WEIGHT"] = data["WEIGHT"].str.replace('1l','1000')
data["WEIGHT"] = data["WEIGHT"].str.replace('1L','1000')
data["WEIGHT"] = data["WEIGHT"].str.replace('2K','2000')
data["WEIGHT"] = data["WEIGHT"].str.replace('5Kg','5000')
data["WEIGHT"] = data["WEIGHT"].str.replace('5K','5000')
data["WEIGHT"] = data["WEIGHT"].str.replace('5L','5000')
data["WEIGHT"] = data["WEIGHT"].str.replace('15K','15000')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 1Kg', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 8g', '8')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 18g', '18')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 1kg', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 1K', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 1L', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 5kg', '5000')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 1g', '1')
data['WEIGHT'] = data['WEIGHT'].str.replace('PP 1kgg', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('0.5g', '0.5', regex = True)
data['WEIGHT'] = data['WEIGHT'].str.replace('1kg', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('5Kg', '5000')
data['WEIGHT'] = data['WEIGHT'].str.replace('20Kg', '20000')
data['WEIGHT'] = data['WEIGHT'].str.replace('1kgg', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('1Kg', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('10Kg', '10000')
data['WEIGHT'] = data['WEIGHT'].str.replace('1K', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('5g', '5')
data['WEIGHT'] = data['WEIGHT'].str.replace('2 Kg', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('7g', '7')
data['WEIGHT'] = data['WEIGHT'].str.replace('8g', '8')
data['WEIGHT'] = data['WEIGHT'].str.replace('10kg', '10000')
data['WEIGHT'] = data['WEIGHT'].str.replace('2kg', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('170g', '170')
data['WEIGHT'] = data['WEIGHT'].str.replace('600g', '600')
data['WEIGHT'] = data['WEIGHT'].str.replace('5L', '5000')
data['WEIGHT'] = data['WEIGHT'].str.replace('2Kg', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('2K', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('60g', '60')
data['WEIGHT'] = data['WEIGHT'].str.replace('500ml', '500')
data['WEIGHT'] = data['WEIGHT'].str.replace('380g', '380')
data['WEIGHT'] = data['WEIGHT'].str.replace('300g', '300')
data['WEIGHT'] = data['WEIGHT'].str.replace('42g', '42')
data['WEIGHT'] = data['WEIGHT'].str.replace('180g', '180')
data['WEIGHT'] = data['WEIGHT'].str.replace('40g', '40')
data['WEIGHT'] = data['WEIGHT'].str.replace('1g', '1')
data['WEIGHT'] = data['WEIGHT'].str.replace('70g', '70')
data['WEIGHT'] = data['WEIGHT'].str.replace('15K', '15000')
data['WEIGHT'] = data['WEIGHT'].str.replace('5K', '5000')
data['WEIGHT'] = data['WEIGHT'].str.replace('1k', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('90g', '90')
data['WEIGHT'] = data['WEIGHT'].str.replace('30Kg', '30000')
data['WEIGHT'] = data['WEIGHT'].str.replace('20g', '20')
data['WEIGHT'] = data['WEIGHT'].str.replace('6g', '6')
data['WEIGHT'] = data['WEIGHT'].str.replace('2 kg', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('20kg', '20000')
data['WEIGHT'] = data['WEIGHT'].str.replace('2L', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('3K', '3000')
data['WEIGHT'] = data['WEIGHT'].str.replace('5 kg', '5000')
data['WEIGHT'] = data['WEIGHT'].str.replace('4g', '4')
data['WEIGHT'] = data['WEIGHT'].str.replace('2k', '2000')
data['WEIGHT'] = data['WEIGHT'].str.replace('450Gg GP', '450')
data['WEIGHT'] = data['WEIGHT'].str.replace('5 Kg', '5000')
data['WEIGHT'] = data['WEIGHT'].str.replace('10KG', '10000')
data['WEIGHT'] = data['WEIGHT'].str.replace('1L', '1000')
data['WEIGHT'] = data['WEIGHT'].str.replace('5kg', '5000')

data['WEIGHT'] = data['WEIGHT'].str.replace(r'[^\d.]+', '')

data["WEIGHT"] = pd.to_numeric(data["WEIGHT"])
data['WEIGHT'] = data['WEIGHT'].fillna(1000)


#@@@@@@@@@@ PRE PROCESSING
#@@@@@@ DATA CLEANING
data.isna().sum()
data = data.dropna()
data = data.drop(columns=["SALESATCOST","F_DATE"])

data.duplicated().sum()
data = data.drop_duplicates()

#@@@@@@ EXPLARATORY DATA ANALYSIS
#@@@@@@@@ PLOTS
sns.distplot(data['MRP'])
data['BRAND'].value_counts().plot(kind='bar')

sns.barplot(x=data['BRAND'],y=data['MRP'])
plt.xticks(rotation='vertical')
plt.show()

data['MC'].value_counts().plot(kind='bar')
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(data['GROSS_SALE'])
sns.distplot(data['DISCOUNT'])

sns.heatmap(data.corr())

sns.barplot(x=data['ZONE'],y=data['MRP'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(data['WEIGHT'])

sns.scatterplot(x=data['WEIGHT'],y=data['MRP'])

sns.barplot(x=data['WEIGHT'],y=data['MRP'])
plt.xticks(rotation='vertical')
plt.show()


#@@@@@@@ NORMALISATION
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
a = list(data.columns)
b = a[5:15] + a[16:18]

for i in b:
    data[str(i)] = norm_func(data[str(i)])



#@@@@@@@ MODEL BULDING
#@@@@@@@@@@ TRAIN TEST SPLIT
x = data.drop(columns=['SP','NAME','UID'])
y = np.log(data['SP'])

x.isna().sum()
y = y.fillna(0)
x = x.fillna(0)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)



#@@@@@@@@@@ RANDOM FOREST

step1 = ColumnTransformer(transformers=[('col_tnf', cf(encoding_method='frequency'),[0,1,2])],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=1000,
                              random_state=30,
                              max_samples=0.95,
                              max_features=1,
                              max_depth=20)

pipe = Pipeline([('step1',step1),('step2',step2)])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


#@@@@@@@@@ DECISION TREE
step1 = ColumnTransformer(transformers=[('col_tnf', cf(encoding_method='frequency'),[0,1,2])],
                          remainder='passthrough')


step2 = DecisionTreeRegressor(max_depth=40,
                              max_features='auto',
                              max_leaf_nodes=50,
                              splitter='best',
                              criterion='friedman_mse')

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

#@@@@@@@@ DECISION TREE CLASSIFIER
step1 = ColumnTransformer(transformers=[('col_tnf', cf(encoding_method='frequency'),[0,1,2])],
                          remainder='passthrough')

step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(sparse=False,drop='first'),
                                         [0,1,2])],remainder='passthrough')


step2 = DecisionTreeClassifier(max_depth=40,
                              max_features='auto',
                              max_leaf_nodes=50,
                              splitter='best',
                              criterion='friedman_mse')

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

#@@@@@@@@@@@ EXPORTING MODEL
import pickle
brand = set(data_0['BRAND'])
pickle.dump(data_0,open('C:\\Users\\Biki\\Desktop\\flaskProject\\data_0.pkl','wb'))
pickle.dump(pipe,open('C:\\Users\\Biki\\Desktop\\flaskProject\\pipe.pkl','wb'))
pickle.dump(brand,open('C:\\Users\\Biki\\Desktop\\flaskProject\\brand.pkl','wb'))




















         
        
        
        
        
        
