###################### problem 1 #################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

startup = pd.read_csv('C:/Users/usach/Desktop/lasso-ridge regression/50_Startups (1).csv')
startup.info()
startup.describe() 
#rename the columns
startup.rename(columns = {'R&D Spend':'rd_spend', 'Marketing Spend' : 'm_spend'} , inplace = True)  
# covariance for data set 
covariance = startup.cov()
covariance
# Correlation matrix 
co = startup.corr()
co
# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(startup.iloc[:,[0,1,2]])
df_norm.describe()
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=startup.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())

# Create dummy variables on categorcal columns
enc_df = pd.get_dummies(startup.iloc[:,[3]])
enc_df.columns
enc_df.rename(columns={"State_New York":'State_New_York'},inplace= True)
model_df = pd.concat([enc_df, df_norm, startup.iloc[:,4]], axis =1)
# Rearrange the order of the variables
model_df = model_df.iloc[:, [6, 0,1, 2, 3,4,5]]

###LASSO MODEL###
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.7, normalize = True)
#model building
lasso.fit(model_df.iloc[:, 1:], model_df.Profit)
# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(model_df.columns[1:]))
#state columns has lowest coefficent 
lasso.alpha
pred_lasso = lasso.predict(model_df.iloc[:, 1:])
# Adjusted r-square#
lasso.score(model_df.iloc[:, 1:], model_df.Profit)
#RMSE
np.sqrt(np.mean((pred_lasso - model_df.Profit)**2))
#####################
#lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters = {'alpha' : [1e-15,1e-10,0,40,80,160,320,1000,1900,1960,1970,2000,2001,4000]}
lasso_reg = GridSearchCV(lasso, parameters , scoring = 'r2' ,cv = 5)
lasso_reg.fit(model_df.iloc[:,1:],model_df.Profit)
lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(model_df.iloc[:,1:])
#Adjusted R- square
lasso_reg.score(model_df.iloc[:,1:],model_df.Profit)
#RMSE
np.sqrt(np.mean((lasso_pred-model_df.Profit)**2))
### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)
rm.fit(model_df.iloc[:, 1:], model_df.Profit)
#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_
plt.bar(height = pd.Series(rm.coef_), x = pd.Series(model_df.columns[1:]))
rm.alpha
pred_rm = rm.predict(model_df.iloc[:, 1:])
# adjusted r-square#
rm.score(model_df.iloc[:, 1:],model_df.Profit)
#RMSE
np.sqrt(np.mean((pred_rm - model_df.Profit)**2))
#####################
#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha' : [4000,8000,16000,32000,64000,150000,300000,600000,1200000]}
ridge_reg = GridSearchCV(ridge, parameters , scoring = 'r2' ,cv = 5)
ridge_reg.fit(model_df.iloc[:,1:], model_df.Profit)
ridge_reg.best_params_
ridge_reg.best_score_
ridge_pred = ridge_reg.predict(model_df.iloc[:,1:])
#Adjusted R- square
ridge_reg.score(model_df.iloc[:,1:], model_df.Profit)
#RMSE
np.sqrt(np.mean((ridge_pred- model_df.Profit)**2))
############################### problem 2 ##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
comp=pd.read_csv("C:/Users/usach/Desktop/lasso-ridge regression/Computer_Data (1).csv")

# Rearrange the order of the variables
comp.columns
# Correlation matrix 
a = comp.corr()
a
# EDA
a1 = comp.describe()
a1
import seaborn as sns
# Sctter plot and histogram between variables
sns.pairplot(comp) # sp-hp, wt-vol multicolinearity issue
# Preparing the model on train data 
model_train = smf.ols('price ~ speed+hd+ram+screen+ads+trend', data = comp).fit()
model_train.summary()
# Prediction
pred = model_train.predict(comp)
# Error
resid  = pred - comp.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
comp=comp.drop(['premium'],axis=1)
comp=comp.drop(['cd'],axis=1)
comp=comp.drop(['multi'],axis=1)
# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13, normalize = True)
lasso.fit(comp.iloc[:, 2:], comp.price)
# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(comp.columns[2:]))
lasso.alpha
pred_lasso = lasso.predict(comp.iloc[:, 2:])
# Adjusted r-square
lasso.score(comp.iloc[:, 2:], comp.price)
# RMSE
np.sqrt(np.mean((pred_lasso - comp.price)**2))
### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)
rm.fit(comp.iloc[:, 2:], comp.price)
# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_
plt.bar(height = pd.Series(rm.coef_), x = pd.Series(comp.columns[2:]))
rm.alpha
pred_rm = rm.predict(comp.iloc[:, 2:])
# Adjusted r-square
rm.score(comp.iloc[:, 2:], comp.price)
# RMSE
np.sqrt(np.mean((pred_rm - comp.price)**2))
######################### problem 3 ###########################
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

ToyotaCorolla = pd.read_csv(r"C:/Users/usach/Desktop/lasso-ridge regression/ToyotaCorolla (1).csv",encoding='unicode_escape')
ToyotaCorolla.head()

ToyotaCorolla.drop(['Model','Mfg_Year','Id','Color'],axis=1,inplace=True)

ToyotaCorolla['Fuel_Type'].value_counts()
#Setting Dummy variables
ToyotaCorolla = pd.get_dummies(data=ToyotaCorolla,columns=['Fuel_Type'],drop_first=True)
ToyotaCorolla.head()

#RMSE of perfect multiple linear model with all the aspects
# RMSE: 1231.6935032422946
#Buidling model with Lasso and Ridge

#Lasso model
lasso = Lasso(alpha=0.8,normalize=True)
lasso.fit(ToyotaCorolla.drop(['Price'],axis=1),ToyotaCorolla['Price'])
pred = lasso.predict(ToyotaCorolla.drop(['Price'],axis=1))
resid = ToyotaCorolla['Price'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse

plt.figure(figsize=(15,5))
plt.bar(height=lasso.coef_,x=ToyotaCorolla.drop(['Price'],axis=1).columns);plt.axhline(y=0,color='r')

#Ridge model
ridge = Ridge(alpha=0.8,normalize=True)
ridge.fit(ToyotaCorolla.drop(['Price'],axis=1),ToyotaCorolla['Price'])
pred = ridge.predict(ToyotaCorolla.drop(['Price'],axis=1))
resid = ToyotaCorolla['Price'] - pred
rmse = np.sqrt(np.mean(np.square(resid)))
rmse
plt.figure(figsize=(15,5))
plt.bar(height=ridge.coef_,x=ToyotaCorolla.drop(['Price'],axis=1).columns);plt.axhline(y=0,color='r')
########################### problem 4 ##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
life = pd.read_csv("C:/Users/usach/Desktop/lasso-ridge regression/Life_expectencey_LR.csv")
life.info()
life.describe()          
#droping index colunms 
life.drop(["Country"], axis = 1, inplace = True)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
life['Status'] = LE.fit_transform(life['Status'])
#droping na value row
life.dropna(axis=0,inplace=True)
# covariance for data set 
covariance = life.cov()
covariance
# Correlation matrix 
Correlation = life.corr()
Correlation

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(life.iloc[:,[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]])
df.describe()
#final dataframe
model_df = pd.concat([life.iloc[:,[2,1]],df ], axis =1)
####lasso model #########
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.07, normalize = True)
lasso.fit(model_df.iloc[:, 1:], model_df.Life_expectancy)
# coefficient values for all independent variables
lasso.coef_
lasso.intercept_
plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(model_df.columns[1:]))
#state columns has lowest coefficent 
lasso.alpha
pred_lasso = lasso.predict(model_df.iloc[:, 1:])
# Adjusted r-square#
lasso.score(model_df.iloc[:, 1:], model_df.Life_expectancy)
#RMSE
np.sqrt(np.mean((pred_lasso -model_df.Life_expectancy)**2))
#####################
#lasso Regression 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters = {'alpha' : [1e-16,1e-15,1e-14,1e-13,1e-12,1e-10,0,20,30,40,50,60,80]}
lasso_reg = GridSearchCV(lasso, parameters , scoring = 'r2' ,cv = 5)
lasso_reg.fit(model_df.iloc[:,1:],model_df.Life_expectancy)
lasso_reg.best_params_
lasso_reg.best_score_
lasso_pred = lasso_reg.predict(model_df.iloc[:,1:])
#Adjusted R- square
lasso_reg.score(model_df.iloc[:,1:],model_df.Life_expectancy)
#RMse
np.sqrt(np.mean((lasso_pred-model_df.Life_expectancy)**2))
### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.077, normalize = True)
rm.fit(model_df.iloc[:, 1:], model_df.Life_expectancy)
#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_
plt.bar(height = pd.Series(rm.coef_), x = pd.Series(model_df.columns[1:]))
rm.alpha
pred_rm = rm.predict(model_df.iloc[:, 1:])
# adjusted r-square#
rm.score(model_df.iloc[:, 1:],model_df.Life_expectancy)
#RMSE
np.sqrt(np.mean((pred_rm - model_df.Life_expectancy)**2))
#####################
#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha' : [0,40,80,160,320,334,640]}
ridge_reg = GridSearchCV(ridge, parameters , scoring = 'r2' ,cv = 5)
ridge_reg.fit(model_df.iloc[:,1:], model_df.Life_expectancy)
ridge_reg.best_params_
ridge_reg.best_score_
ridge_pred = ridge_reg.predict(model_df.iloc[:,1:])
#Adjusted R- square
ridge_reg.score(model_df.iloc[:,1:], model_df.Life_expectancy)
#RMSE
np.sqrt(np.mean((ridge_pred- model_df.Life_expectancy)**2))
