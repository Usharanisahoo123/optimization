import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import pylab

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Internship Project\priceOptimize.csv")
data.columns


data1= data[(data["MC"] == "Sunflower Oil") & (data["NAME"]=='FRESH & PURE SUNFLOWER OIL PP 1L')]
data1.head()

data1.columns

sns.boxplot(data1.NSU)
sns.boxplot(data1.SP)

from feature_engine.outliers import Winsorizer

winsor = Winsorizer( capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables =['SP'])

data1['SP'] = winsor.fit_transform(data1[['SP']])


sns.pairplot(data1[['SP','NSU',]], plot_kws={'alpha':0.8})


sunflower_model = ols("NSU ~ SP ", data= data1).fit()

print(sunflower_model.summary())

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(sunflower_model, fig=fig)

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(sunflower_model, "SP", fig=fig)

elasticities = {}

def create_model_and_find_elasticity(data):
    model = ols("NSU ~ SP", data).fit()
    price_elasticity = model.params[1]
    print("Price elasticity of the product: " + str(price_elasticity))
    print(model.summary())
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_partregress_grid(model, fig=fig)
    return price_elasticity, model

price_elasticity, model_sunflower = create_model_and_find_elasticity(data1)
elasticities['Sunflower oil 1L'] = price_elasticity

data1.SP.min()
data1.SP.max()


start_price = 73
end_price   = 95
cost_price_oil = 65

test = pd.DataFrame(columns = ["SP", "NSU"])

test['SP'] = np.arange(start_price, end_price,0.05)

test['NSU'] = model_sunflower.predict(test['SP']) ####

GST = 0.0476*test['SP']

test['PROFIT'] = (test["SP"] - cost_price_oil - GST) * test["NSU"]

plt.plot(test['SP'],test['NSU'])
plt.plot(test['SP'],test['PROFIT'])
plt.show()


ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
test.loc[[ind]]


##################################################################


