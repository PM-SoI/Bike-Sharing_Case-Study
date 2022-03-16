#!/usr/bin/env python
# coding: utf-8

# ## Bike Rental Case Study
# 
# ### Problem Statement:
# 
# A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.
# 
# A US bike-sharing provider BoomBikes has recently suffered considerable dips in revenues due to the ongoing Corona pandemic. The company is looking for possible solution for increasing the revenues after the lockdown period.To predict the user demand for shared bikes among the people after the Covid-19 and make huge profits, they have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes. 
# 
# The company wants to know:
# •	Which variables are significant in predicting the demand for shared bikes.
# •	How well those variables describe the bike demands?
# •   To create a linear model that predicts the user demand.

# ## Step 1: Reading and Understanding the Data
# 
# Let us first import NumPy and Pandas and read the housing dataset

# In[60]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[62]:


bikes = pd.read_csv("day.csv")


# In[63]:


bikes.head()


# In[64]:


bikes.shape


# In[65]:


bikes.info()


# In[66]:


#There are no null values in the dataset
#data types of all variables seems to be fine


# In[67]:


bikes.describe()


# ## Step 2: Visualising the Data
# 
# **Understanding the data**.
# - Univariate analysis was carried out for identifying outliers in the continious variables but nothing significant worth requiring treatment was observed.
# - 'casual' variable had outliers, since it is a response variable so no treatment offered as it will not be used aspredictor variable in model.
# - Find out the multicollinearity in predictor variables
# - Also identify if some predictors directly have a strong association with the outcome variable
# 
# #### Visualising Numeric Variables
# 
# Let's make a pairplot of all the numeric variables

# In[68]:


sns.pairplot(bikes)
plt.show()


# In[69]:


# It is evident from above that:
# 'cnt' i.e. total user demand is actually sum of 'registered' and 'casual' user demands
# 'cnt', 'registered' and 'casual' relationship is evident in pair plots.
#  Scatter plot between 'cnt'/'registered'/'casual' vs 'temp', 'atemp' is exhibiting linear relationship. 


# #### Visualising Categorical Variables
# 
# Categorical variables to be vsualized using boxplots.

# In[70]:


plt.figure(figsize=(40, 24))
plt.subplot(3,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bikes)
plt.subplot(3,3,2)
sns.boxplot(x = 'yr', y = 'cnt', data = bikes)
plt.subplot(3,3,3)
sns.boxplot(x = 'mnth', y = 'cnt', data = bikes)
plt.subplot(3,3,4)
sns.boxplot(x = 'holiday', y = 'cnt', data = bikes)
plt.subplot(3,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = bikes)
plt.subplot(3,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bikes)
plt.subplot(3,3,7)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bikes)
plt.show()


# It is evident from above plots that:<br>
# 1. Seasonal user demand is increasing from spring-summer-fall-winter with lowest in spring and highest in fall.<br>
# 2. Aevrage user demand in year 2019 is much higher than year 2018 in all seasons and through out the year.<br>
# 3. Month wise user demand is increasing steadily from January to May-June-July-Aug (Fall season) and then decreased till december.<br>
# 4. User demand dips on weather holidays whereas average demands is almost same during week days & working days (weekend+holiday).<br>
# 5. User demand is max in Clear weather days and min in Light Snow/Rain days, almost nil on Heavy Rain/Ice Pallets days.<br>

#  Let's visualise multiple categorical features parallely to understand the user demand patterns. 

# In[71]:


plt.figure(figsize = (10, 5))
sns.boxplot(x = 'weathersit', y = 'cnt', hue = 'yr', data = bikes)
plt.show()


# In[72]:


plt.figure(figsize = (10, 5))
sns.boxplot(x = 'weekday', y = 'cnt', hue = 'workingday', data = bikes)
plt.show()


# In[73]:


plt.figure(figsize = (10, 5))
sns.boxplot(x = 'mnth', y = 'cnt', hue = 'season', data = bikes)
plt.show()


# In[74]:


plt.figure(figsize = (10, 5))
sns.boxplot(x = 'season', y = 'cnt', hue = 'weathersit', data = bikes)
plt.show()


# ## Step 3: Data Preparation

# - First, we will drop 'instant' variable being record index followed with 'dteday' as month & weekday are available.
# - Variables 'yr', 'holiday' and 'working day' are in 0s and 1s so no binary encoding required.
# - 04 Categorical variables ('season','mnth',weekday',weathersit') with multilevels are also in numerical values.
# - Hence, no dummy variables needs to be generated here.

# In[75]:


bikes.drop(['instant','dteday'], axis =1 , inplace =True)
bikes.head()


# ## Step 4: Splitting the Data into Training and Testing Sets
# 
# We will go for 70:30 train & test split.

# In[76]:


# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
bikes_train, bikes_test = train_test_split(bikes, train_size = 0.7, test_size = 0.3, random_state = 100)


# ### Rescaling the Features 
# Some features have small integer values, so to ensure the comparable scales and proper coefficients we will use Min-Max scaling.

# In[77]:


scaler = MinMaxScaler()


# In[78]:


# Apply scaler() to seven variables/columns as under except the 'binary' and 'dummy' variables
num_vars = ['temp','atemp','hum','windspeed','casual','registered','cnt']
bikes_train[num_vars] = scaler.fit_transform(bikes_train[num_vars])


# In[79]:


bikes_train.head()


# In[80]:


bikes_train.describe()


# In[81]:


# Let's check the correlation coefficients to see which variables are highly correlated
plt.figure(figsize = (16, 10))
sns.heatmap(bikes_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# 1. It is evident from above that 'temp' or'atemp' (highly correlated,one is to be dropped) is important predictor variables correlated to users demand. <br>
# 2. 'casual' and 'registered' variables are adding and contributing to the 'cnt' so their correlation is obvious.<br>
# 3. Let's make two prediction models one each for 'registered' and 'casual', final count can be found by adding the both.

# In[82]:


#dropping the 'temp' and keeping the 'atemp' as it makes better sense to keep temperature actually being felt by user.
bikes_train.drop(['temp'], axis =1 , inplace =True)
bikes_train.head()


# In[83]:


plt.figure(figsize=[6,6])
plt.scatter(bikes_train.atemp, bikes_train.casual)
plt.show()


# So, we pick `atemp` as the first variable and we'll try to fit a regression line to that.

# ### Dividing into X and Y sets for the model building

# In[84]:


y_train = bikes_train.pop('casual')
bikes_train.pop('cnt')
bikes_train.pop('registered')
X_train = bikes_train
bikes_train.head()


# ## Step 5: Building a Linear Model
# 
# Fit a regression line through the training data using `statsmodels` by adding a constant using `sm.add_constant(X)`to ensure that `statsmodels` does not fit a regression line passing through the origin, as the default option.

# In[85]:


# Add a constant
X_train_lm = sm.add_constant(X_train[['atemp']])
# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()


# In[86]:


# Check the parameters obtained
lr.params


# In[87]:


# Let's visualise the data with a scatter plot and the fitted regression line
plt.scatter(X_train_lm.iloc[:, 1], y_train)
plt.plot(X_train_lm.iloc[:, 1], -0.021445+ 0.538229*X_train_lm.iloc[:, 1], 'r')
plt.show()


# In[88]:


# Print a summary of the linear regression model obtained
print(lr.summary())


# ### Adding another variable
# 
# The R-squared value obtained is `0.308`. Since we have so many variables, we can clearly do better than this. So let's go ahead and add `yr`.

# In[90]:


# Assign all the feature variables to X
X_train_lm = X_train[['atemp', 'yr']]


# In[91]:


# Build a linear model
X_train_lm = sm.add_constant(X_train_lm)
lr = sm.OLS(y_train, X_train_lm).fit()
print(lr.summary())


# We have clearly improved the model as the value of adjusted R-squared as its value has gone up to `0.308` from `0.348`.
# Let's go ahead and add another variable, `season`.

# In[92]:


# Assign all the feature variables to X
X_train_lm = X_train[['atemp', 'yr','season']]


# In[93]:


# Build a linear model
X_train_lm = sm.add_constant(X_train_lm)
lr = sm.OLS(y_train, X_train_lm).fit()
print(lr.summary())


# We have improved the adjusted R-squared from '0.348' to '0.349'. Now let's go ahead and add all the feature variables.

# ### Adding all the variables to the model

# In[94]:


# Check all the columns of the training data
bikes_train.columns


# In[95]:


#Build a linear model
X_train_lm = sm.add_constant(X_train[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
       'weathersit', 'atemp', 'hum', 'windspeed']])
lr_1 = sm.OLS(y_train, X_train_lm).fit()
print(lr_1.summary())


# Looking at the p-values, variables with high p-values aren't really significant (in the presence of other variables).Maybe we could drop some? We could simply drop the variable with the highest, non-significant p value. <br>
# ### A better way would be to supplement this with the VIF information. 

# ### Checking VIF
# 
# Variance Inflation Factor or VIF, gives a basic quantitative idea about how much the feature variables are correlated with each other. It is an extremely important parameter to test our linear model. The formula for calculating `VIF` is:
# 
# ### $ VIF_i = \frac{1}{1 - {R_i}^2} $

# In[97]:


# Check for the VIF values of the feature variables. 
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We generally want a VIF that is less than 5. So there are clearly some variables we need to drop.

# ### Dropping the variable and updating the model
# 
# On dropping the variable 'hum' with high p-value 0.054 and high VIF value of 24.67.

# In[98]:


# Dropping highly correlated variables and insignificant variables
X = X_train.drop('hum', 1,)
# Build a third fitted model
X_train_lm = sm.add_constant(X)
lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model
print(lr_2.summary())


# In[99]:


# Calculate the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Dropping the variable and updating the model
# 
# On dropping the variable 'season' with high p-value 0.025 and high VIF value of 23.60.

# In[100]:


# Dropping highly correlated variables and insignificant variables
X = X.drop('season', 1)
# Build a second fitted model
X_train_lm = sm.add_constant(X)
lr_3 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model
print(lr_3.summary())


# In[101]:


# Calculate the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Dropping the variable and updating the model

# ### Dropping the Variable and Updating the Model
#  Now dropping the variable `mnth` with high p-value 0.456.

# In[102]:


X = X.drop(['mnth'], axis = 1)
# Build a fourth fitted model
X_train_lm = sm.add_constant(X)
lr_4 = sm.OLS(y_train, X_train_lm).fit()
print(lr_4.summary())


# In[103]:


# Calculate the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Now as you can see, the VIFs and p-values both are within an acceptable range. So we go ahead and make our predictions using this model only.

# ## Step 7: Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[105]:


y_train_pred = lr_4.predict(X_train_lm)


# In[106]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# ## Step 8: Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final, i.e. fourth model.

# #### Applying the scaling on the test sets

# In[107]:


bikes_test.columns


# In[108]:


num_vars = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
bikes_test[num_vars] = scaler.transform(bikes_test[num_vars])


# In[109]:


bikes_test.describe()


# #### Dividing into X_test and y_test

# In[110]:


bikes_test.head()


# In[111]:


y_test = bikes_test.pop('casual')                   
X_test = bikes_test


# In[112]:


# Adding constant variable to test dataframe
X_test_m4 = sm.add_constant(X_test)
X_test_m4.head()


# In[113]:


# Creating X_test_m4 dataframe by dropping variables from X_test_m4
X_test_m4 = X_test_m4.drop(["temp","mnth","hum","season","registered","cnt"], axis = 1)
X_test_m4.head()


# In[114]:


# Making predictions using the fourth model
y_test_pred = lr_4.predict(X_test_m4)


# ## Step 9: Model Evaluation
# 
# Let's now plot the graph for actual versus predicted values.

# In[115]:


# Plotting y_test and y_pred to understand the spread
fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_test_pred', fontsize = 18)      


# In[116]:


r2_score (y_true = y_test , y_pred= y_test_pred)


# In[117]:


print(lr_4.summary())


# ### Since r2_score of the LR model and r2-score of the output test result is within 5 % , it means model strength is very good and model accuracy is 65.1 %.****

# We can see that the equation of our best fitted line is:
# 
# $ registered = 0.2081+ 0.5147  \times  atemp + 0.0884  \times  yr - 0.2490 \times workingday 0.1295 \times holiday + 0.0072 \times weekday - 0.0550 \times weathersit - 0.1090 \times windspeed $

# The above model is the proposed model for predicting the "Casual" users, we would predict the "Registered" users in similar fashion by preparing a separate model.
