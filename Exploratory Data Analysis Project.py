#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis 

# #### Project Name - Exploratory Data Analysis on  Used_cars price
# 
# Aim - Analyze the used car's price and identify the factors influencing the car price using EDA.

# #### dataset = https://www.kaggle.com/datasets/sukhmanibedi/cars4u

# In[1]:


#Import necessary library to do EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load dataset

UsedCar_df = pd.read_csv('used_cars.csv')


# ### Basic information

# In[3]:


#Check total number of observations and features.

UsedCar_df.shape


# ### Duplicate observation

# In[4]:


UsedCar_df.duplicated().sum()


# #### Observation:
#     
# We are getting 0. this means their is not a single duplicate observation in our dataset and that is good.

# In[5]:


#See top 5 and bottom 5 records

UsedCar_df.head()


# In[6]:


UsedCar_df.tail()


# #### Observation:
# * We can see that the in Mileage feature their is difference in unit.
# * We have to convert km/kg to kmpl for all the observations
# 
# * Or remove units from Mileage ,Engine Power and New_price.

# In[7]:


#Replace km/kg to kmpl

UsedCar_df["Mileage"] = UsedCar_df['Mileage'].str.replace('km/kg' , 'kmpl')


# In[8]:


#Replacing units from features
import warnings
warnings.filterwarnings("ignore")

UsedCar_df["Mileage"] = UsedCar_df['Mileage'].str.replace('[^\d.]' ,'')
UsedCar_df['Engine'] = UsedCar_df['Engine'].str.replace('[^\d.]' ,'')
UsedCar_df['Power'] = UsedCar_df['Power'].str.replace('[^\d.]' , '')
UsedCar_df['New_Price'] = UsedCar_df['New_Price'].str.replace('[^\d.]' , '')

UsedCar_df.head()


# In[ ]:





# #### Observation:
# So we have successfully handled the units of Mileage,Engine,Power and New_Price.

# In[9]:


#We have to know some basic information about our dataset for better understanding.

UsedCar_df.info()


# #### Observation:
# We have to convert datatypes of Mileage , Engine , Power and New_Price to int/float 
# because we can not perform mean,median on string datatype.
# 

# ### DataType Conversion:

# In[10]:


UsedCar_df[['Mileage' , 'Engine' , 'New_Price']] = UsedCar_df[['Mileage' , 'Engine' , 'New_Price']].astype('float')


# #### * DataType of Power is not converting through astype() so changed it through apply(pd.to_numeric)

# In[11]:


UsedCar_df['Power'] = UsedCar_df['Power'].apply(pd.to_numeric)


# In[12]:


UsedCar_df.info()


# ### Missing values

# In[13]:


UsedCar_df.isnull().sum()


# ####  We have missing values in Mileage , Engine , Power , Seats and New_price.
# * Because for this features datatype is object so we have to convert it to integer.
# * Beacause we have nan we can not convert datatype.
# * So first we have to raplace nan with 0 then convert datatype to int then fill 0 with mean ,median ,mode.

# In[14]:


#Calculating total percentage of missing values 

MissingValues_Pct = UsedCar_df.isnull().sum()/len(UsedCar_df.index)*100


# In[15]:


print(MissingValues_Pct)


# #### Observation:
# 
# 1. We can see that the some features like Mileage , Engine , Power , Seats , New_Price has missing values.
# 2. New_Price feature has 86% of missing data followed by Price 7%.

# #### We can drop New_Price column because it has more than 85% of missing values

# In[16]:


UsedCar_df = UsedCar_df.drop(['New_Price'] , axis = 1)


# In[17]:


UsedCar_df.head()


# ### Handling Missing Values:

# In[18]:


UsedCar_df.isnull().sum()


# In[19]:


Mileage_mean = UsedCar_df['Mileage'].mean()
Mileage_median = UsedCar_df['Mileage'].median()
Mileage_mode = UsedCar_df['Mileage'].mode()

print(Mileage_mean)
print(Mileage_median)
print(Mileage_mode)


# #### Observation:
# So we have 0 in Mileage looks like data entry error

# In[20]:


#UsedCar_df['Mileage'].replace(0 , np.nan , inplace = True)


# In[21]:


UsedCar_df.loc[UsedCar_df['Mileage'] == 0 , 'Mileage']


# #### Observation:
# We have zero 81 times in Mileage feature so we have to replace 0's with nan. 

# In[22]:


UsedCar_df.loc[UsedCar_df['Mileage'] == 0 , 'Mileage'] = np.nan


# In[23]:


#Number of nan

UsedCar_df['Mileage'].isnull().sum()


# Beacause mean and median are alomost same so we can replace nan with anyone of the central tendency.

# In[24]:


#filling nun values with mean

UsedCar_df['Mileage'].fillna(Mileage_mean , inplace = True)


# In[25]:


#Check null values

UsedCar_df['Mileage'].isnull().sum()


# In[26]:


UsedCar_df['Seats'].unique()


# In[27]:


UsedCar_df.loc[UsedCar_df['Seats'] == 0 , 'Seats']


# #### Observation:
# * their is some data entry error because we have 0 in seats feature.
# * Car seats can not be 0 right.
# * so we have to replace 0 with nan then with central tendancy.

# In[28]:


#UsedCar_df['Seats'].fillna(value = np.nan , inplace = True)


# In[29]:


UsedCar_df.loc[UsedCar_df['Seats'] == 0 , 'Seats'] = np.nan


# In[30]:


UsedCar_df['Seats'].isnull().sum()


# In[31]:


UsedCar_df['Seats'].unique()


# In[32]:


Seats_mean  = UsedCar_df['Seats'].mean()


# In[33]:


#filling nun values with mean

UsedCar_df['Seats'].fillna(Seats_mean , inplace = True)


# In[34]:


UsedCar_df['Seats'].isnull().sum()


# In[35]:


UsedCar_df.loc[UsedCar_df['Engine'] == 0 , 'Engine']


# No 0 value

# In[36]:


Engine_median = UsedCar_df['Engine'].median()


# In[37]:


UsedCar_df['Engine'].fillna(Engine_median ,inplace = True)


# In[38]:


UsedCar_df['Engine'].isnull().sum()


# In[39]:


UsedCar_df.loc[UsedCar_df['Power'] == 0 , 'Power']


# No 0 value

# In[40]:


UsedCar_df['Power'].isnull().sum()


# In[41]:


Power_median = UsedCar_df['Power'].median()


# In[42]:


UsedCar_df['Power'].fillna(Power_median ,inplace = True)


# In[43]:


UsedCar_df['Power'].isnull().sum()


# In[44]:


UsedCar_df.isnull().sum()


# In[45]:


UsedCar_df.loc[UsedCar_df['Price'] == 0 , 'Price']


# No 0 value that's good

# In[46]:


Price_mean = UsedCar_df['Price'].mean()


# In[47]:


UsedCar_df['Price'].fillna(Price_mean , inplace = True)


# In[48]:


UsedCar_df['Price'].isnull().sum()


# In[49]:


UsedCar_df['Price'].isnull().sum()


# In[50]:


UsedCar_df.isnull().sum()


# #### So we have successfully handled missing values for numerical features.

# Note-
# UsedCar_df['Mileage'] = 0
# 
# Do not do this it will change your entire feature to zero.

# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


UsedCar_df.nunique()


# #### Observation:
# 
# We can get unique number of values for each feature.

# In[52]:


UsedCar_df['Location'].unique()


# In[53]:


UsedCar_df['Fuel_Type'].unique()


# In[54]:


UsedCar_df['Owner_Type'].unique()


# #### Observation:
#     
# So their is no duplicate values for categorical features.

# ### Data reduction
# 
# Some columns or variables can be dropped if they do not add value to our analysis.
# 
# In our dataset, the column 'S.No' have only ID values, assuming they don’t have any predictive power to predict the dependent variable i.e. price of the car.

# In[55]:


UsedCar_df = UsedCar_df.drop(['S.No.'] , axis = 1)


# In[56]:


UsedCar_df.info()


# In[ ]:





# ### Feature Engineering

# #### Feature creation
# 
# We have 'Year' that is basically manufacturing price of the car so we can create new feature 'Car_age'
# because age of the car is a contributing factor to the car's price.

# In[57]:


from datetime import date

UsedCar_df['Car_age'] = date.today().year - UsedCar_df['Year']


# In[58]:


UsedCar_df.head()


# #### Feature split
# 
# Since car names will not be great predictors of the price in our current data. But we can process this column to extract important information using brand and Model names.
# 
# We can create new features like 'Brand' and 'Model' hy substracting from 'Name' feature. 
# 
# #Feature split on 'Name'

# In[59]:


UsedCar_df['Brand'] = UsedCar_df.Name.str.split().str.get(0)


# In[60]:


UsedCar_df['Model'] = UsedCar_df.Name.str.split().str.get(1) + UsedCar_df.Name.str.split().str.get(2)


# In[61]:


UsedCar_df[['Name' , 'Brand' , 'Model']]


# In[62]:


UsedCar_df.head(2)


# In[63]:


UsedCar_df.info()


# In[64]:


UsedCar_df.isnull().sum()


# #### Observation:
#     
# 1. After feature split all the columns looks good.
# 2. For all the columns datatype is correct.
# 3. Missing values have taken care for numerical features but we can see that the we have one null value for 
#    'Model' feature

# print('Total number of unique brand:' , UsedCar_df['Brand'].nunique())
# 
# print(UsedCar_df['Brand'].unique())
# 
# #Total number of unique brand: 33
# #['Maruti' 'Hyundai' 'Honda' 'Audi' 'Nissan' 'Toyota' 'Volkswagen' 'Tata'
#  'Land' 'Mitsubishi' 'Renault' 'Mercedes-Benz' 'BMW' 'Mahindra'
#  'Ford' 'Porsche' 'Datsun' 'Jaguar' 'Volvo' 'Chevrolet' 'Skoda'
#  'Mini' 'Fiat' 'Jeep' 'Smart' 'Ambassador' 'Isuzu' 'ISUZU' 'Force'
#  'Bentley' 'Lamborghini' 'Hindustan' 'OpelCorsa']

# #### Observation:
#     
# 1. Data entry errors, and some variables may need data type conversion.
# 2. The brand name ‘Isuzu’ ‘ISUZU’ and ‘Mini’ and ‘Land’ looks incorrect. This needs to be corrected

# In[65]:


UsedCar_df['Brand'].replace({'ISUZU':'Isuzu' , 'Mini':'Mini Cooper' , 'Land':'Land Rover'} , inplace = True)


# In[66]:


print('Total number of unique brand:' , UsedCar_df['Brand'].nunique())

print(UsedCar_df['Brand'].unique())


# In[ ]:





# ### Statistic summary
# 
# Statistics summary gives a high-level idea to identify whether the data has any outliers, data entry error, distribution of data such as the data is normally distributed or left/right skewed

# In[67]:


UsedCar_df.describe().T


# #### Observation:
#     
# 1. We can see that the year's range from 1996 to 2019 which shows that the used cars contain both new models and
#    new models cars.
# 2. Average KM driven is ~58K but we can see that the huge difference between min and max valuse where max value is 650000 KM
#    which shows the evidence of outlier.
# 3. Average seat in a car is 5. seat is an important feature in price contribution.
# 4. Price is very high for used car which is 160K whereas average price for used cars is 9.5K.here may be an outlier or data 
#    entry issue.
#    

# In[68]:


UsedCar_df.describe(include = 'all').T


# In[ ]:





# ### Treating Outliers

# In[178]:


figure ,axes = plt.subplots(3,3 , figsize =(18,18))

sns.boxplot(ax = axes[0,0] ,x = 'Price' , data = UsedCar_df)
sns.boxplot(ax = axes[0,1] , x = 'Kilometers_Driven' , data = UsedCar_df)
sns.boxplot(ax = axes[0,2] , x = 'Seats' , data = UsedCar_df)
sns.boxplot(ax = axes[1,0] , x = 'Mileage' , data = UsedCar_df)
sns.boxplot(ax = axes[1,1] , x = 'Engine' , data = UsedCar_df)
sns.boxplot(ax = axes[1,2] , x = 'Power' , data = UsedCar_df)
sns.boxplot(ax = axes[2,0] , x = 'Car_age' , data = UsedCar_df)


# ### first method:

# In[135]:


Q1 = np.percentile(UsedCar_df['Price'] , 25)
Q3 = np.percentile(UsedCar_df['Price'] , 75)
print('Q1:',Q1)
print('Q3:',Q3)

IQR =Q3-Q1
print('IQR:',IQR)


# In[136]:


upper_bound = Q3+1.5*IQR
lower_bound = Q1-1.5*IQR

print('upper_bound:',upper_bound)
print('lower_bound:',lower_bound)


#UsedCar_df[UsedCar_df['Price']>upper_bound]

# ### Using function:

# In[162]:


def find_outliers_IQR(UsedCar_df):
    Q1 = UsedCar_df.quantile(0.25)
    Q3 = UsedCar_df.quantile(0.75)
    
    IQR = Q3-Q1
    
    upper_bound = Q3+1.5*IQR
    lower_bound = Q1-1.5*IQR
    
    outliers = UsedCar_df[((UsedCar_df>upper_bound) | (UsedCar_df<lower_bound))]
    
    return outliers 


# In[165]:


outliers = find_outliers_IQR(UsedCar_df['Price'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# In[166]:


outliers = find_outliers_IQR(UsedCar_df['Kilometers_Driven'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# In[179]:


outliers = find_outliers_IQR(UsedCar_df['Seats'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# In[180]:


outliers = find_outliers_IQR(UsedCar_df['Mileage'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# In[181]:


outliers = find_outliers_IQR(UsedCar_df['Engine'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# In[182]:


outliers = find_outliers_IQR(UsedCar_df['Power'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# In[183]:


outliers = find_outliers_IQR(UsedCar_df['Car_age'])

print('number of outliers: ', str(len(outliers)))
print('max outlier value: ', str(outliers.max()))
print('min outlier value: ', str(outliers.min()))


# #### Observation:
#     
# *  
# *
# *
# *
# *
# *
# *

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Lets separate Numerical and categorical features 

# In[69]:


#Categorical features

Cat_features = UsedCar_df.select_dtypes(include = ['object']).columns

print('Number of categorical features:')
print(Cat_features.nunique())

print('Categorical features:')
print(Cat_features)


# In[70]:


#Numerical features

Num_features = UsedCar_df.select_dtypes(include = np.number).columns

print('Number of numerical features:')
print(Num_features.nunique())

print('Numerical features:')
print(Num_features)


# #### Observation:
#     
# 1. Categorical features = 7
# 2. Numerical features = 8

# In[192]:


Usedcar_df_num_features = UsedCar_df.select_dtypes(include=np.number)

Usedcar_df_num_features


# In[191]:


Usedcar_df_cat_features = UsedCar_df.select_dtypes(include=['object'])

Usedcar_df_cat_features


# ### Univariate Analysis
# 
# Analyzing/visualizing the dataset by taking one variable at a time:
# 
# 1. Univariate analysis can be done for both Categorical and Numerical variables.
# 2. Categorical variables can be visualized using a Count plot, Bar Chart, Pie Plot, etc.
# 3. Numerical Variables can be visualized using Histogram, Box Plot, Density Plot, etc.
# 
# * In our dataset, we have done a Univariate analysis using Histogram and  Box Plot for continuous Variables.
# * A histogram and box plot is used to show the pattern of the variables, as some variables have skewness and outliers.

# In[71]:


Num_features


# In[72]:


for col in Num_features:
    print(col)
    print('skew:' , round(UsedCar_df[col].skew() ,2))

    plt.figure()
    sns.distplot(UsedCar_df[col])
    plt.show()


# #### Observation:
#     
# All the numerical features have skewness so we just don't reduce skewness from all the columns 
# first check correlation of features with target feature which is Price.

# In[73]:


#Check Correlation between features

sns.heatmap(UsedCar_df.corr(),annot = True)
plt.show()


# #### Observation:
#     
# * Price has good correlation with year,Mileage,Engine,Power and car_age variabel
#   but bad correlation with Kilometers_Driven and seats.
# * so we have to transform or reduce skewness from Kilometers_Driven and seats 

# #### Handling skewness

# #### Handling right skewness
# 
# * log transformation
# * Root Transformation - 
# square root transformation
#  and Cube root transformatoin
# * reciprocals transformation

# In[74]:


#log transformation for Kilometers_Driven

log_kilometers_Driven = np.log(UsedCar_df['Kilometers_Driven'])


# In[75]:


log_kilometers_Driven.skew()


# In[76]:


sns.distplot(log_kilometers_Driven , hist = True)


# In[77]:


#Root transformation

sqrt_kilometers_driven = np.sqrt(UsedCar_df['Kilometers_Driven'])


# In[78]:


sqrt_kilometers_driven.skew()


# In[79]:


sns.distplot(sqrt_kilometers_driven , hist = True)


# In[80]:


cube_root_kilometers = np.cbrt(UsedCar_df['Kilometers_Driven'])


# In[81]:


cube_root_kilometers.skew()


# In[82]:


sns.distplot(cube_root_kilometers , hist = True)


# #### Observation:  Kilometers_driven
#    
# After applying multiple skewness technique we can see that the cube root transformation gives us more moderately skewed data.

# In[83]:


#log transformation for Seats

log_seats = np.log(UsedCar_df['Seats'])


# In[84]:


log_seats.skew()


# In[85]:


sns.distplot(log_seats ,hist = True)


# #### Observation: Seats
# So we have successfully handle the skewness for seats from 1.96 to 0.86

# #### Note
# 
#   It's giving nan because log transformation doesn't work whenever their is nan.
#   so for that we have to convert nan to 0.01 or 0.001.

# In[ ]:





# #### categorical variables are being visualized using a count plot. Categorical variables provide the pattern of factors influencing car price

# In[86]:


#Categorical features

Cat_features


# In[87]:


figure ,axes = plt.subplots(3,3 , figsize =(18,18))
sns.countplot(ax = axes[0,0],x = 'Fuel_Type', data = UsedCar_df , order = UsedCar_df['Fuel_Type'].value_counts().index , color = 'blue')

sns.countplot(ax= axes[0,1],x = 'Location', data = UsedCar_df , order = UsedCar_df['Location'].head(6).value_counts().index , color = 'blue')

sns.countplot(ax = axes[0,2],x = 'Transmission', data = UsedCar_df , order = UsedCar_df['Transmission'].value_counts().index , color = 'blue')

sns.countplot(ax= axes[1,0],x = 'Owner_Type', data = UsedCar_df , order = UsedCar_df['Owner_Type'].value_counts().index , color = 'blue')

sns.countplot(ax= axes[1,1],x = 'Brand', data = UsedCar_df , order = UsedCar_df['Brand'].head(5).value_counts().index , color = 'blue')

sns.countplot(ax= axes[1,2],x = 'Model', data = UsedCar_df , order = UsedCar_df['Model'].head(5).value_counts().index , color = 'blue')

sns.countplot(ax= axes[2,0],x = 'Mileage', data = UsedCar_df , order = UsedCar_df['Mileage'].head(5).value_counts().index , color = 'blue')

sns.countplot(ax= axes[2,1],x = 'Engine', data = UsedCar_df , order = UsedCar_df['Engine'].head(5).value_counts().index , color = 'blue')

sns.countplot(ax= axes[2,2],x = 'Power', data = UsedCar_df , order = UsedCar_df['Power'].head(5).value_counts().index , color = 'blue')

plt.show()


# #### Observation:
#     
# * We can see that the customers are buying large number of diesel cars followed by petrol cars.
# * Location - Mumbai follwed by hyderabad then coimbatore
# * In terms of transmission large number of buyers prefer mannual cars over automatic cars.
# * People prefer buying first hand car over second hand.
# * Maruti is the most demand car follwed by hyundai.
# * Customers are buying WagonR most of the time.
# * In terms of Engine people buying - 1248cc.

# ### Bivariate Analysis
# 
# Bivariate Analysis helps to understand how variables are related to each other and the relationship between dependent and independent variables present in the dataset.

# In[88]:


#A bar plot can be used to show the relationship between Categorical variables and continuous variables


fig, axarr = plt.subplots(2,3 ,figsize=(18,16))

UsedCar_df.groupby('Location')['Price'].mean().sort_values(ascending=False).plot.bar(ax= axarr[0,0],fontsize=12)

UsedCar_df.groupby('Fuel_Type')['Price'].mean().sort_values(ascending=False).plot.bar(ax= axarr[0,1] ,fontsize=12)

UsedCar_df.groupby('Transmission')['Price'].mean().sort_values(ascending=False).plot.bar(ax= axarr[0,2] ,fontsize=12)

UsedCar_df.groupby('Brand')['Price'].mean().head(5).sort_values(ascending=False).plot.bar(ax= axarr[1,0] ,fontsize=12)

UsedCar_df.groupby('Model')['Price'].mean().head(5).sort_values(ascending=False).plot.bar(ax= axarr[1,1] ,fontsize=12)

UsedCar_df.groupby('Owner_Type')['Price'].mean().sort_values(ascending=False).plot.bar(ax= axarr[1,2] ,fontsize=12)

plt.show()


# #### Observation:
#     
# * Prices are very high in cities like coimbatore followed by banglore then kochi.
# * For electric cars prices are more.
# * Automatic cars are more expensive than mannual cars.
# * We can see that the most expensive brand is Bentley.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




