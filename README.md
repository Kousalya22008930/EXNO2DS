# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic.csv")
dt
```
![312083120-e06452cb-fff5-46d9-91e5-1e13daed721d](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/76b7df12-8bcb-43de-b72b-48c769c5d085)
```
dt.info()
```
![312083583-8bde848f-e44f-44a9-ac0f-e259f11c65a7](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/b40918a7-88f6-495d-a405-1b4218c445e7)
```
dt.set_index('PassengerId',inplace=True)
dt.describe()
```
![312084768-813b0217-8139-4f0f-bb4c-7cef93046ca5](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/63e1c0a3-88da-4d82-9a5c-446dddae91d2)
```
dt.shape
```
![312084953-fe9f5eef-962b-4984-9f62-0928129a36d6](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/45d17fe3-7d54-4d66-ab55-c6e822755df1)
```
dt.nunique()
```
![312085145-428f6980-9a18-48f0-9c0a-cc1f8090bee1](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/798e13a3-7413-471c-851a-2cc16e2b8e8a)
```
dt["Survived"].value_counts()
```
![312085389-6b13cbf2-e40f-477b-854c-6b035f60eb08](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/0bc72985-b8d0-478c-bf4b-894a5dbf06f2)
```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![312085594-c6951eef-9034-4f9f-9f1a-8724a018dbea](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/f9781b7a-1c27-45ab-9dfc-43d320bdb25b)
```
sns.countplot(data=dt,x="Survived")
```
![312085841-07df1378-7f18-4d9d-b891-2f839ccb05b0](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/a9e6ea13-d420-4975-a0eb-964217033ce2)
```
dt.Pclass.unique()
```
![312086072-c421cbe0-9718-48cf-812a-266a17c27ff9](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/01d6694b-c9fe-4cfa-926f-280a229770eb)
```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![312086422-76377e69-143a-4fdf-937a-d3d0260ad75b](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/fbfd6a76-0190-49ef-96b6-c07874404d4b)
```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![312086721-5269a77c-0399-4250-aec3-228d0ab6d31c](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/91cdf801-594e-41e9-9a45-d432378b7d6f)
```
sns.catplot(x='Survived',hue="Gender",data=dt,kind="count")
```
![312086960-cf409002-66a2-49c1-bab9-daed33d5d5f9](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/65a0cf6a-7406-40f4-b061-79cdc0e26799)
```
dt.boxplot(column="Age",by="Survived")
```
![312087277-7de0dd77-83ba-47c1-aac4-292b6f552717](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/439dc51a-a01f-4252-976d-1b49186e0873)
```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![312087598-f44a4475-948b-4070-87e8-5b5c1cf03ea6](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/6038949d-b63b-4e15-bf71-043b8aa7440b)
```
sns.jointplot(x=dt["Age"],y=dt["Fare"],data=dt)
```
![312088001-73f0aaf9-cd78-411c-b8af-5e857c0db614](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/fe270d79-155c-442b-b344-20d86b3813ee)
```
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=dt)
```
![312088316-8b2d554b-a089-4e15-8b76-29f26fe6f40f](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/e8390e88-def8-4b75-971f-9f5a77bddc0b)
```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![312088646-570d378c-9f74-4ec8-9a20-df375e105912](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/3c0813da-f33e-40d8-9487-b529a0a66de3)
```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![312088959-4b2293e7-e5d7-41fe-b0b1-64c59e5cfd49](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/7c6feb4b-e17a-4636-9e94-86bebba30ca2)
```
sns.pairplot(dt)
```
![312089950-4de84806-e138-419d-b9d9-776f376baff5](https://github.com/Kousalya22008930/EXNO2DS/assets/119389108/0e41c294-d1fe-441e-b191-16c15a519560)
# RESULT
Thus,Data Analyzing of the given dataset was successful.
