
# coding: utf-8

# In[28]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

bike_rentals = pd.read_csv("bike_rental_hour.csv")

print(bike_rentals.head())


# In[29]:

plt.hist(bike_rentals['cnt'])
plt.show()


# In[30]:

bike_rentals.corr()


# # Calculating Features

# - We are going to introduce some order in hour data by separating the hours in terms of morning, afternoon, evening and night

# In[31]:

def assign_label(hour):
    if (hour > 6 ) & (hour <= 12):
        return 1
    if (hour > 12 ) & (hour <= 18):
        return 2
    if (hour > 18 ) & (hour <= 24):
        return 3
    if (hour >= 0 ) & (hour <= 6):
        return 4

bike_rentals['time_label'] = bike_rentals['hr'].apply(lambda x: assign_label(x))


# In[32]:

print(bike_rentals.head())


# # Splitting The Data Into Train And Test Sets

# - In this case we are going to use the Mean Squared error, which make sense for our continous data.

# In[33]:

how_many_rows = np.int(len(bike_rentals) * .8)

train = bike_rentals.sample(frac=.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]


# In[34]:

predictors = list(bike_rentals.columns)

predictors.remove('cnt')
predictors.remove('casual')
predictors.remove('dteday')
predictors.remove('registered')

print(predictors)


# In[35]:

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train[predictors], train['cnt'])

predict = model.predict(test[predictors])

mse = np.mean((test['cnt'] - predict) ** 2)

print('error: ', mse)


# - Thought about the error:
#     - It is very high, so it should be due to the reason that some data is too high and then there comes this high mean squared.

# # Applying Decision Trees

# - In order to choose which model fits better our data we are going to checck how decision trees work with this dataframe and after that we will see which one though us a bigger error so we can compare.
# 

# In[36]:

from sklearn.tree import DecisionTreeRegressor

model_tree = DecisionTreeRegressor()
model_tree.fit(train[predictors], train['cnt'])
predictions_tree = model_tree.predict(test[predictors])

mse_tree = np.mean((test['cnt'] - predictions_tree) ** 2)

print('mse_tree: ', mse_tree)


# - As we can see, using decision tree we are going to get more accuracy in our predictions than with linearregression, because as we proved, the mean squared error is smaller in the dec tree than in Linear regression.

# # Applying Random Forest

# In[37]:

from sklearn.ensemble import RandomForestRegressor

model_forest = RandomForestRegressor()
model_forest.fit(train[predictors], train['cnt'])
predict_forest = model_forest.predict(test[predictors])

mse_forest = np.mean((test['cnt'] - predict_forest) ** 2)

print('mse_forest: ', mse_forest)


# In[ ]:



