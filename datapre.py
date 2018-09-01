import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r'C:\Users\USER\Desktop\all\train.csv')
dataset2=pd.read_csv(r'C:\Users\USER\Desktop\all\test.csv')

#removing not needed
RemoveList=['Id','MSSubClass','LotFrontage','Condition2','OverallCond','YearBuilt','MiscFeature','MoSold','SaleType',
            'PoolQC','GarageCond','GarageQual','GarageCars','GarageYrBlt','GarageType','FireplaceQu','KitchenAbvGr','Heating',
            'BsmtExposure','BsmtFinType2','BsmtFinType1','ExterCond','Exterior2nd']

for i in RemoveList:
    dataset = dataset.drop(i, 1)
    dataset2 = dataset2.drop(i, 1)

#filling the numerical columns / Can use Imputer as well
dataset = dataset.fillna(method='ffill')
dataset2 = dataset2.fillna(method='ffill')
#dropping the missing value rows for categorical variable
#(can check by dataset.isna().any())
dataset=dataset.dropna()
dataset2=dataset2.dropna()

#encoding the categorical variables and dropping the prediction column
D1=pd.get_dummies(dataset,drop_first=True)
D2=pd.get_dummies(dataset2,drop_first=True)
D1=D1.drop('SalePrice',1)

#getting the value of X,y
train_X=D1.iloc[:,:].values
train_y=dataset['SalePrice'].values
test_X=D2.iloc[:,:].values

#splitting the dataset
#from sklearn.cross_validation import train_test_split
#train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.25,random_state=0)

#changing the shape to get vector and not matrixOfFeatures
train_y=train_y.reshape(-1,1)
test_y=test_y.reshape(-1,1)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X= sc_X.transform(test_X)
sc_y = StandardScaler()
train_y = sc_y.fit_transform(train_y)
test_y= sc_y.transform(test_y)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
#fitting the dataset
#making a function for getting Mean Absolute Error as well
def getmae(max_leaf_node,train_X,train_y,test_X,test_y):
    regressor=DecisionTreeRegressor(max_leaf_node=max_leaf_node,random_state=0)
    regressor.fit(train_X,train_y)
    y_pred=regressor.predict(test_X)
    mae=mean_absolute_error(sc_X.inverse_transform(test_y),sc_y.inverse_transform(y_pred))
    return mae

#compare mae with different values of max_leaf_node
for max_leaf_node in [500,1000,2000,5000]:
    mae=getmae(max_leaf_node,train_X,train_y,test_X,test_y)
    print("Max Leaf Node : %d \t\t Mean Absolute Error : %d"%(max_leaf_node,mae)

#getting unscaled value
sc_y.inverse_transform(y_pred)
sc_X.inverse_transform(test_y)
