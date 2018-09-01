import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('train.csv')
dataset['label'] = 'train'
test_dataset=pd.read_csv('test.csv')
test_dataset['label'] = 'score'
y=dataset.iloc[:,3].values
dataset = dataset.drop('SalePrice', 1)
#test_dataset =test_dataset.drop('Id',1)

#concatinating the two sets
concat_df = pd.concat([dataset , test_dataset])
#X1=concat_df.iloc[:,:].values

#removing not needed
RemoveList=['Id','LotFrontage','Condition2','OverallCond','YearBuilt','MiscFeature','MoSold','SaleType',
           'PoolQC','GarageCond','GarageQual','GarageCars','GarageYrBlt','GarageType','FireplaceQu',
           'KitchenAbvGr','Heating','BsmtExposure','BsmtFinType2','BsmtFinType1','ExterCond','Exterior2nd']

for i in RemoveList:
    concat_df = concat_df.drop(i, 1)

gg=list((concat_df.select_dtypes(['object'])).columns.values)    

concat_df=concat_df.fillna(concat_df.mean())
#train_df=train_df.fillna(method='ffill')
#concat_df=concat_df.dropna()

nominal=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
         'Neighborhood','Condition1','BldgType','HouseStyle',
         'RoofStyle','RoofMatl','Exterior1st','MasVnrType','Foundation',
         'Electrical','Functional','GarageFinish','PavedDrive','Fence',
         'SaleCondition','ExterQual','BsmtQual','BsmtCond',
         'HeatingQC','CentralAir','KitchenQual']

features_df = pd.get_dummies(concat_df,columns=nominal,drop_first=True)
# Split your data
train_df = features_df[features_df['label'] == 'train']
score_df = features_df[features_df['label'] == 'score']

# Drop your labels
train_df = train_df.drop('label', axis=1)
score_df = score_df.drop('label', axis=1)

#train_df = train_df.drop('Id', 1)
#score_df = score_df.drop('Id',1)
ss=train_df.isnull().any()

#matrix of features
X=train_df.iloc[:,:].values

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)



from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X,y)

y_pred=regressor.predict(X2)

mae=mean_absolute_error(test_y,y_pred)
return mae

#train_df.columns[train_df.isnull().any()].tolist()
# train_df.mean()
#np.isnan(mat.any()) #and gets False
#np.isfinite(mat.all())






