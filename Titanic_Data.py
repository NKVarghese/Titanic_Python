#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
##import pandas_profiling
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[39]:


Titanic_data=pd.read_csv("C:\\Users\\betsy\\OneDrive\\Desktop\\DA\\Datasets\\train.csv")
Titanic_test=pd.read_csv("C:\\Users\\betsy\\OneDrive\\Desktop\\DA\\Datasets\\test.csv")

combine=[Titanic_data,Titanic_test]


# In[6]:


get_ipython().system('pip install pandas-profiling')


# In[ ]:





# In[ ]:





# In[40]:


get_ipython().system('pip install pandas-profiling')
import seaborn as sn
df=sn.load_dataset('iris')
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ProfileReport(df)


# In[41]:


Titanic_data.shape


# In[42]:


from pandas_profiling import ProfileReport
ProfileReport(Titanic_data)


# In[7]:


Titanic_data.head()


# In[8]:


Titanic_data.info()


# In[9]:


Titanic_data.shape


# In[43]:


Titanic_data.describe()


# In[44]:


Titanic_data.isnull().sum()


# In[6]:


Titanic_data.drop('Cabin',axis=1,inplace=True)


# In[7]:


Titanic_data.isnull().sum()


# In[45]:


Titanic_data.head()


# In[9]:


New_Age=Titanic_data.Age.median()
New_Age


# In[11]:


Titanic_data.Age.fillna(New_Age,inplace=True)


# In[46]:


Titanic_data.head(20)


# In[47]:


Titanic_data.isnull().sum()


# In[48]:


Titanic_data.Embarked=Titanic_data.Embarked.fillna(Titanic_data['Embarked'].mode()[0])
Titanic_data.isnull().sum()


# In[49]:


Titanic_data.drop('Ticket',axis=1,inplace=True)
Titanic_data.isnull().sum()


# In[50]:


Titanic_data.AgeBand=0
Titanic_data.loc[Titanic_data.Age<=1, 'AgeBand']='Infant'
Titanic_data.loc[(Titanic_data.Age>1)&(Titanic_data.Age<=12),'AgeBand']='Child'
Titanic_data.loc[(Titanic_data.Age>12), 'AgeBand']='Adult'

Titanic_data.head()


# In[51]:


Titanic_data.FareBand=0
Titanic_data.loc[(Titanic_data.Fare>=1)& (Titanic_data.Fare<=15), 'FareBand']=1
Titanic_data.loc[(Titanic_data.Fare>15)&(Titanic_data.Fare<=30),'FareBand']=2
Titanic_data.loc[(Titanic_data.Fare>30)& (Titanic_data.Fare<=50), 'FareBand']=3
Titanic_data.loc[(Titanic_data.Fare>50),'FareBand']=4

Titanic_data.head()


# In[52]:


Titanic_data.drop('PassengerId',axis=1,inplace=True)
Titanic_data.head()


# In[54]:


Titanic_data['Title']=Titanic_data.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(Titanic_data['Title'],Titanic_data['Sex'])


# In[55]:


Titanic_data['Title']=Titanic_data['Title'].replace(["Capt","Col","Countess","Don","Dr","Jonkheer","Major","Rev","Sir"],"Rare")
Titanic_data['Title']=Titanic_data['Title'].replace(["Mlle","Ms"],"Miss")
Titanic_data['Title']=Titanic_data['Title'].replace(["Mme","Lady"],"Mrs")

Titanic_data[['Title','Survived']].groupby(['Title'],as_index=False).mean()


# In[56]:


TitleMap={'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Rare':5}

Titanic_data['Title']=Titanic_data['Title'].map(TitleMap)
Titanic_data['Title']-Titanic_data['Title'].fillna(0)

Titanic_data.head()


# In[57]:



Titanic_data.Title=Titanic_data.Title.map({'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Rare':5})
Titanic_data.Title=Titanic_data.Title.fillna(0)

Titanic_data.head()


# In[84]:


MapTitle={'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Rare':5}
Titanic_data.Title=Titanic_data.Title.map(MapTitle)
Titanic_data.Title=Titanic_data.Title.fillna(0)

Titanic_data.head()


# In[58]:


Titanic_data.Sex=Titanic_data.Sex.map({'male':0,'female':1})
Titanic_data.head()


# In[37]:


Titanic_data.drop('Name',axis=1,inplace=True)

Titanic_data.head()


# In[59]:


Titanic_data.Embarked=Titanic_data.Embarked.map({'S':0,'C':1,'Q':2})
Titanic_data.head()


# In[103]:


Titanic_data['Age*Pclass']=Titanic_data.Age * Titanic_data.Pclass

Titanic_data.loc[:,['Age*Pclass','Age','Pclass']].head()


# In[110]:


import pandas_profiling
profile=pandas_profiling.ProfileReport(Titanic_data)
profile.to_file(output_file="Titanic Post Process Report.html")


# In[111]:


# Data Visulaisations


# In[112]:


#Count of Survival and Victims


# In[113]:


Titanic_data.groupby(['Survived'])['Survived'].count()


# In[13]:


plt=Titanic_data.Survived.value_counts().plot(kind='bar')
plt.set_xlabel('Died or Survived')
plt.set_ylabel('Count')


# In[117]:


#More survival rate based on gender


# In[14]:


Titanic_data.groupby(['Survived','Sex'])['Sex'].count()


# In[20]:


Titanic_data['Sex'].Survived.groupby(Titanic_data.Survived.count().plot(kind='pie', figsize=(6,6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(['Died','Survived'])


# In[28]:


Titanic_data.groupby(['Sex']).size().plot(kind='pie', figsize=(6,6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(['Died','Survived'])


# In[24]:


data = Titanic_data.groupby(['Sex','Survived']).size()
explode = np.zeros(len(data))
explode[data.index.get_loc(('male',1))] = 0.05
data.plot(kind='pie', figsize=(6,6),explode=explode,autopct='%1.1f%%')
plt.axis('equal')
plt.legend(['Died','Survived'])


# In[25]:


data = Titanic_data.groupby(['Sex','Survived']).size()
explode = np.zeros(len(data))
explode[data.index.get_loc(('male',1))] = 0.05
data.plot(kind='pie', figsize=(6,6),explode=explode,autopct='%1.1f%%', labels=data.index)
plt.axis('equal')
plt.legend(title='Survived')


# In[62]:


Titanic_data[Titanic_data['AgeBand']=='Adult'].Survived.groupby(Titanic_data.Survived).size().plot(kind='pie',figsize=(6,6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(['Died','Survived'])


# In[63]:


Titanic_data[Titanic_data['AgeBand']=='Child'].Survived.groupby(Titanic_data.Survived).size().plot(kind='pie',figsize=(6,6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(['Died','Survived'])


# In[64]:


Titanic_data[Titanic_data['AgeBand']=='Infant'].Survived.groupby(Titanic_data.Survived).size().plot(kind='pie',figsize=(6,6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(['Died','Survived'])


# In[71]:


sns.barplot(x='Pclass',y='Survived',data=Titanic_data)


# In[77]:


sns.barplot(x='Pclass',y='Survived', hue='Sex',data=Titanic_data)


# In[78]:


sns.barplot(x='Pclass',y='Survived', hue='AgeBand',data=Titanic_data)


# In[79]:


sns.barplot(x='Pclass',y='AgeBand', hue='Survived',data=Titanic_data)


# In[80]:


sns.barplot(x='Pclass',y='Survived', hue='Embarked',data=Titanic_data)


# In[85]:


sns.countplot(x='Embarked',data=Titanic_data)


# In[90]:


plt=Titanic_data[['Embarked','Survived']].groupby('Embarked').mean().Survived.plot(kind='bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survived')


# In[91]:


pd.crosstab([Titanic_data.Sex,Titanic_data.Survived,Titanic_data.Pclass],[Titanic_data.Embarked], margins=True)


# In[94]:


sns.violinplot(x='Pclass',y='Embarked', hue='Sex',data=Titanic_data)


# In[95]:


sns.violinplot(x='Pclass',y='Embarked', hue='Survived',data=Titanic_data)


# In[98]:


sns.catplot(x='Pclass',y='Embarked', hue='Sex',col='Survived', kind= 'bar',data=Titanic_data)


# In[101]:


Titanic_data['Fare'].max()


# In[102]:


Titanic_data['Fare'].min()


# In[103]:


Titanic_data['Fare'].mean()


# In[104]:


Titanic_data['Fare'].mode()


# In[107]:


Titanic_data[['FareBand','Survived']].groupby(['FareBand']).mean()


# In[108]:


#Average fare by P Class and Embark location


# In[109]:


sns.boxplot(x='Pclass',y='Fare', hue='Embarked',data=Titanic_data)


# In[110]:


#Features that impacted survival rate


# In[112]:


sns.heatmap(Titanic_data.corr(),annot=True)


# In[ ]:




