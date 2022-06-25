import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from scipy import stats
import pylab

data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Internship Project\kabuli chana.csv")
data.columns
data.drop(['Unnamed: 2', 'Unnamed: 3'], axis=1, inplace = True)

sns.boxplot(data.SP)
sns.boxplot(data.NSU)
stats.probplot(data['SP'], dist = 'norm',plot = pylab)
stats.probplot(data['NSU'], dist = 'norm', plot = pylab)
plt.scatter(x = data['SP'], y = data['NSU'], color = 'green') 
sns.pairplot(data)
sns.lmplot(x = "SP", y = "NSU", data = data, fit_reg = True, size = 4)

np.corrcoef(data.SP, data.NSU) 

cov_output = np.cov(data.SP, data.NSU)[0, 1]
cov_output



# Simple Linear Regression

# Import library
import statsmodels.formula.api as smf

model = smf.ols('NSU ~ SP', data = data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data))
pred1

# Regression Line
plt.scatter(data.SP, data.NSU)
plt.plot(data.SP, data.NSU, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data.NSU - pred1
res1 
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(SP); y = NSU

plt.scatter(x = np.log(data['SP']), y = data['NSU'], color = 'brown')
np.corrcoef(np.log(data.SP), data.NSU) #correlation

model2 = smf.ols('NSU ~ np.log(SP)', data = data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data))
pred2

# Regression Line
plt.scatter(np.log(data.SP), data.NSU)
plt.plot(np.log(data.SP), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data.NSU - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = SP; y = log(NSU)

plt.scatter(x = data['SP'], y = np.log(data['NSU']), color = 'orange')
np.corrcoef(data.SP, np.log(data.NSU)) #correlation

model3 = smf.ols('np.log(NSU) ~ SP', data = data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(data.SP, np.log(data.NSU))
plt.plot(data.SP, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data.NSU - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation
# x = SP; x^2 = SP*SP; y = log(SP)

model4 = smf.ols('np.log(NSU) ~ SP + I(SP*SP)', data = data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 1:].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(data.SP, np.log(data.NSU))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data.NSU - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


model5 = smf.ols('np.log(NSU) ~ SP + I(SP*SP) + I(SP*SP*SP) ', data = data).fit()
model5.summary

pred5 = model5.predict(pd.DataFrame(data))
pred5_at = np.exp(pred5)
pred5_at

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X = data.iloc[:, 1:].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values

plt.scatter(data.SP, np.log(data.NSU))
plt.plot(X, pred5, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

res5 = data.NSU - pred5_at
res_sqr5 = res5 * res5
mse5 = np.mean(res_sqr5)
rmse5 = np.sqrt(mse5)
rmse5


# revenue
gross_sale = NSU * SP # eq (1)
# profit
profit = gross_sale - cost_to_compay - GST  # eq (2)

# revised profit function
profit = NSU * SP - cost_to_compay - GST   # eq (3)


####### 
def norm_fun(i):
    x = (i-i.min()/(i.max()-i.min()))
    return x

data_norm = norm_fun(data.iloc[:,1:])


data1 = pd.concat([data.NSU, data_norm], axis = 1)

stats.probplot(data1['SP'], dist = 'norm',plot = pylab)
stats.probplot(data1['NSU'], dist = 'norm', plot = pylab)
plt.scatter(x = data1['SP'], y = data1['NSU'], color = 'green') 

# nothing change in the result
#######


# demand curve
sns.lmplot(x = "SP", y = "NSU", data = data, fit_reg = True, size = 4)
# fit OLS model
model = ols("NSU ~ SP", data = data ).fit()
# print model summary
print(model.summary())

# plugging regression coefficients
quantity =  5436.7196   - 50.1994  * SP # eq (5)
# the profit function in eq (3) becomes
profit = ( 5436.7196   - 50.1994  * SP) * SP - cost_to_company - GST # eq (6)

# a range of diffferent prices to find the optimum one we will take range for giving the discount between +/- 20%
SP = [67.7796, 69.6903, 74.9315 ,78.4898, 79.0944 ,81.4244, 82.1309 ,84.0000, 84.9490, 89.8896, 90.7242, 92.4321, 93.2629, 96.0478, 96.1425, 96.2146, 97.8196 ]

# assuming a fixed cost
Profit = []
Quantity =[]

for i in SP:
   cost = 0.88* i
   GST = 0.05* i
   
     #    y         =      c         +   m      x
   quantity_demanded =   5436.7196   - 50.1994 * i
    # profit function
   Profit.append((i - cost - GST) * quantity_demanded)  # Its nothing but final Profit 
   Quantity.append(quantity_demanded)

# create data frame of price and revenue

optimize = pd.DataFrame({"Price": SP, "Quantity": Quantity , "Profit": Revenue})
#plot revenue against price

plt.plot(optimize["Price"],optimize["Profit"])

# price at which revenue is maximum
optimize[optimize['Profit'] == optimize['Profit'].max()]
