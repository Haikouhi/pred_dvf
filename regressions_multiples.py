#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install folium')


# In[2]:

# basic imports
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import csv
import pandas as pd
from pandas.tools import plotting
import folium
import warnings # supprssion des warnings pandas
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# In[3]:

# scikit-learn imports
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[4]:

# running the csv files
f8 = "/home/utilisateur/SIMPLON_2019/projet_04_06_2019/valeursfoncieres_2018.csv"
d8 = pd.read_csv(f8,header=0, parse_dates=True)


# In[5]:


f7 = "/home/utilisateur/SIMPLON_2019/projet_04_06_2019/valeursfoncieres_2017.csv"
d7 = pd.read_csv(f7,header=0, parse_dates=True)


# In[6]:


d8.head()


# In[7]:

# adding both files together
d= pd.concat([d7,d8])


# In[8]:

# shows the first five rows
d.head()


# In[9]:

# extracting ZIP codes
l0 = d[d['code_postal']==69000]


# In[10]:


l1=d[d['code_postal']==69001]


# # Extraction de données concernant les arrondissements de Lyon 18 et 17

# In[11]:


l2=d[d['code_postal']==69002]
l3=d[d['code_postal']==69003]
l4=d[d['code_postal']==69004]
l5=d[d['code_postal']==69005]
l6=d[d['code_postal']==69006]
l7=d[d['code_postal']==69007]
l8=d[d['code_postal']==69008]
l9=d[d['code_postal']==69009]
l10=d[d['code_postal']==69010]


# In[12]:

# regrouping all info on boroughs in one dataframe
df_17_18 = pd.concat([l1, l2, l3, l4, l5, l6, l7, l9], ignore_index=True)
df_17_18.head()


# # Extraction des variables pertinentes

# In[13]:

# choosing which variables will be used, they are then added to a dataframe
ly_1718= df_17_18[["date_mutation","valeur_fonciere","adresse_code_voie","code_postal","type_local","surface_reelle_bati","nombre_pieces_principales","latitude","longitude"]]


# In[14]:

# removing empty data
ly_1718_clean= ly_1718.dropna()# clear missing values by columns


# In[15]:

# calculating real estate value per square meter 
ly_1718_clean['surface_au_mettre_carre'] = ly_1718_clean['valeur_fonciere']/ly_1718_clean['surface_reelle_bati']


# In[16]:

# shows statistical data of the dataframe
print(ly_1718_clean['valeur_fonciere'].describe(include='all'))


# In[17]:

# extracting only the appartments becuas that is teh data we need
l_appart=ly_1718_clean[ly_1718_clean['type_local']=='Appartement']


# In[18]:

# shows statistical data of specific estate types (smmal, big spaces)
print(l_appart['type_local'].describe(include='all'))


# In[19]:

# shows statistical data of real estate value
print(l_appart['valeur_fonciere'].describe(include='all'))


# In[20]:

# shows statistical data of floor surface
print(l_appart['surface_reelle_bati'].describe(include='all'))


# In[ ]:





# In[21]:

# representation of real estate value and floor surface 
l_appart.plot(x='surface_reelle_bati', y='valeur_fonciere', style='o')  
plt.title('surface_reelle_bati vs valeur_fonciere')  
plt.xlabel('surface_reelle_bati')  
plt.ylabel('valeur_fonciere')  
plt.show()


# In[22]:

# shows distribution of the real estate value
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(l_appart['valeur_fonciere'])


# In[23]:

# removing value (price)
l_appart_neat = l_appart[l_appart["valeur_fonciere"]<2500000]


# In[24]:

# removing value (floor surface)
l_ap_bon = l_appart_neat[l_appart_neat["surface_reelle_bati"]<260]


# In[25]:

# representation of real estate value and floor surface 
l_ap_bon.plot(x='surface_reelle_bati', y='valeur_fonciere', style='o')  
plt.title('surface_reelle_bati vs valeur_fonciere')  
plt.xlabel('surface_reelle_bati')  
plt.ylabel('valeur_fonciere')  
plt.show()


# In[26]:

# removing value (price)
l_appart_neat = l_appart[l_appart["valeur_fonciere"]<1500000]


# In[27]:

# removinf value (floor surface)
l_ap_bon = l_appart_neat[l_appart_neat["surface_reelle_bati"]<250]


# In[28]:

# representation of real estate value and floor surface 
l_ap_bon.plot(x='surface_reelle_bati', y='valeur_fonciere', style='o')  
plt.title('surface_reelle_bati vs valeur_fonciere')  
plt.xlabel('surface_reelle_bati')  
plt.ylabel('valeur_fonciere')  
plt.show()


# In[29]:

# statistics of floor surface after the changes previously applied
print(l_ap_bon['surface_reelle_bati'].describe(include='all'))


# In[30]:

# statistics of floor surface after the changes previously applied
print(l_ap_bon['surface_au_mettre_carre'].describe(include='all'))


# In[31]:

# shows number of values
l_ap_bon.shape



# In[33]:

# extraction
x = l_ap_bon.iloc[:, [5]] # surface réelle bati
y = l_ap_bon.iloc[:, [6]] # nombre de pièces
z = l_ap_bon.iloc[:, [1]] # valeur fonctière


# In[34]:

# creating df with only two variables 
dx= pd.DataFrame(l_ap_bon, columns=['surface_reelle_bati','nombre_pieces_principales'])


# In[35]:


dx.shape


# In[36]:

# normalizing variables 
x1= l_ap_bon['surface_reelle_bati'].values.reshape(-1,1)
y1= l_ap_bon['valeur_fonciere'].values.reshape(-1,1)


# In[37]:

# multi-linear regression
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=0) # 20% pour le test


# In[38]:

# linear regression (training the model)
regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm


# In[39]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[40]:

# target variable
y_pred = regressor.predict(x_test)


# In[41]:

# compares the start data to final result (old values to the prediction)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[42]:

# representation 
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[43]:

# dots and line for visuel purposes 
plt.scatter(x_test, y_test,  color='gray') # representation en gris
plt.plot(x_test, y_pred, color='red', linewidth=2) # la droite de regression
plt.show()


# In[44]:

# recalculating regression coeff
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[45]:

# calculting coeff regression
regressor = LinearRegression()  
regressor.fit(x, y) #training the algorithm


# In[46]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# # Etude en 3 D

# # Normalisation des données et feature Scaling

# In[47]:

# same as previous but in 3D
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_scaled = scale.fit_transform(dx[['surface_reelle_bati', 'nombre_pieces_principales']].as_matrix())


# In[48]:


import statsmodels.api as sm


# In[49]:

# shows statistical data in 3D
est = sm.OLS(z,dx[['surface_reelle_bati', 'nombre_pieces_principales']]).fit()
 
print(est.summary())
