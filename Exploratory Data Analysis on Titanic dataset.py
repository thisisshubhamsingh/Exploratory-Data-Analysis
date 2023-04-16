#!/usr/bin/env python
# coding: utf-8

# ## Preparing data for Machine Learning model

# #### Project Name - Exploratory Data Analysis on  titanic dataset
# 
# Aim - Analyze the titanic_train dataset and identify how many passenger's survived or deceased.

# #### dataset = https://github.com/krishnaik06/EDA1/blob/master/titanic_train.csv

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_excel('Titanic_train.xlsx')


# In[3]:


train.head()


# In[4]:


train.shape


# ### Missing Values

# In[5]:


#Let's see total number of missing values for each column and percentage of that.

total = train.isnull().sum().sort_values(ascending=False)
percentage = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_values = pd.concat([total , percentage] , axis = 1 , keys = ['Total' , 'Percent'])

missing_values


# #### We can also represent missing values in terms of visualization.

# In[6]:


#Graphical representation of Missing Values 

plt.figure(figsize=(6,4))
sns.heatmap(train.isnull() , yticklabels=False , cbar = False ,cmap = 'viridis')
plt.show()


# ### Observation:
# We can see that approx 20% of data is missing in Age column and around 77% in Cabin column, so for Age variable we can use
# imputation and we can drop Cabin variable because we can't use imputation because it has 77% of missing values.
# 
# And we can also see that the Embarked variable has also 2 missing values for that we can drop nan.

# ### Data Cleaning

# #### let's drop unnecessary columns 
# We can see that Passengerid , Name and Ticket doesn't provide any value.

# In[7]:


train.drop(['PassengerId' , 'Name' , 'Ticket'] , axis = 1 , inplace= True)


# #### Perform imputation on Age
# 
# Let's check average age of passengers w.r. to Pclass then we can impute it with avg age respectively.

# In[8]:


train.groupby(['Pclass'])['Age'].mean()


# In[9]:


plt.figure(figsize=(6,4))
sns.boxplot(x = 'Pclass' , y = 'Age' , data = train)
plt.show()


# ### Observation:
# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. 
# We'll use these average age values to impute based on Pclass for Age.
# 
# 

# In[10]:


#We can impute average age by creating a function-

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age
                  


# #### Let's apply that function

# In[11]:


train['Age'] = train[['Age' , 'Pclass']].apply(impute_age , axis = 1)


# #### Now let's check that heatmap again!

# In[12]:


plt.figure(figsize=(6,4))
sns.heatmap(train.isnull() , yticklabels=False , cbar = False , cmap = 'viridis')
plt.show()


# In[13]:


#Drop cabin

train.drop('Cabin' , axis = 1 , inplace = True)


# In[14]:


#Dropping the rowns having NaN/NaT values under certain label

train.dropna(subset = ['Embarked'], inplace=True)


# In[15]:


train.reset_index(inplace = True)


# #### Now let's check that heatmap again!

# In[16]:


plt.figure(figsize=(6,4))
sns.heatmap(train.isnull() , yticklabels=False , cbar = False , cmap = 'viridis')
plt.show()


# #### Observation:
# * So we have successfully handled all missing values.
# * Now we can see that the we aren't seeing any missing values in heatmap.

# #### Let's see our cleaned data

# In[17]:


train.head()


# #### Let's get some more information about data 

# In[18]:


plt.figure(figsize=(5,3))
sns.set_style('whitegrid')
sns.countplot(data = train , x = 'Survived' , width = 0.4)
plt.show()


# #### Observation:
# * We can see that the more than 500 pessenger didn't survive.
# * Around 330 pessengers have survived.
#     

# In[19]:


plt.figure(figsize=(5,3))
sns.set_style('whitegrid')
sns.countplot(data = train , x = 'Survived' , hue = 'Sex' , width = 0.4)
plt.show()


# #### Observation:
# 
# * So More than 450 men didn't survive as compared to female.
# * we can see that the more than 200 female pessenger survived as compared to mens.
# 

# In[20]:


plt.figure(figsize=(5,3))
sns.set_style('whitegrid')
sns.countplot(data = train , x = 'Survived' , hue = 'Pclass' , width = 0.4)
plt.show()


# #### Observation:
# * In terms of pessenger class who didn't survive were from third class.
# * those pessengers who were in first class they had survived/

# In[21]:


plt.figure(figsize=(5,3))
sns.distplot(train['Age'] ,kde = False, bins = 40)


# #### Observation:
# 
# We can see that the most of the pessengers are between 20 to 30 years of age group.

# In[22]:


plt.figure(figsize=(5,3))
sns.set_style('whitegrid')
sns.countplot(data = train ,x = 'SibSp' , width = 0.6)
plt.show()


# #### Observation:
# * Most of the pessengers are travelling alone followed by couples. 

# In[23]:


train['Fare'].hist(bins = 40 , figsize = (5,3))


# #### Observation:
#     
# So the most of fare < 100.  
# 

# In[ ]:





# In[24]:


train.head()


# ## Converting Categorical Features
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[25]:


train.info()


# In[26]:


sex = pd.get_dummies(train['Sex'] , drop_first = True)
sex.head()


# In[30]:


embarked = pd.get_dummies(train['Embarked'] , drop_first = True)
embarked.head()


# In[28]:


train.head()


# In[31]:


train.drop(['Sex' , 'Embarked'] ,axis = 1, inplace=True)


# In[32]:


train.head()


# In[33]:


train = pd.concat([train , sex , embarked] , axis = 1)


# In[34]:


train.head()


# #### Great! Now our data is ready for machine learning model.
