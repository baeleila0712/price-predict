import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error




train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')


#將train, test合併成一個df
all_data = pd.concat([train,test])


#將所有資料先做labelencode

obj_column = all_data.dtypes[all_data.dtypes == 'object'].index
mapingdf = pd.DataFrame()

house_baseline = all_data.copy()

for column in obj_column:
    labelencoder = LabelEncoder()
    house_baseline[column] = labelencoder.fit_transform(all_data[column])
    mapingdf[column] = house_baseline[column]
    mapingdf['_'+column] =  labelencoder.inverse_transform(house_baseline[column])




print('Total missing values', sum(house_baseline.isna().sum()))
print('Total missing values', sum(house_baseline.isnull().sum()))


#空值佔比

total = house_baseline.isnull().sum().sort_values(ascending=False)
percent = (house_baseline.isnull().sum()/house_baseline.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)



#將空值進行處理

house_baseline["LotFrontage"] = house_baseline["LotFrontage"].fillna(house_baseline['LotFrontage'].mean())
house_baseline["GarageYrBlt"] = house_baseline["GarageYrBlt"].fillna(house_baseline['GarageYrBlt'].median())
house_baseline["MasVnrArea"] = house_baseline["MasVnrArea"].fillna(house_baseline['MasVnrArea'].mean())
house_baseline["BsmtFullBath"] = house_baseline["BsmtFullBath"].fillna(house_baseline['BsmtFullBath'].mean())
house_baseline["BsmtHalfBath"] = house_baseline["BsmtHalfBath"].fillna(house_baseline['BsmtHalfBath'].mean())
house_baseline["TotalBsmtSF"] = house_baseline["TotalBsmtSF"].fillna(house_baseline['TotalBsmtSF'].mean())
house_baseline["GarageArea"] = house_baseline["GarageArea"].fillna(house_baseline['GarageArea'].mean())
house_baseline["BsmtUnfSF"] = house_baseline["BsmtUnfSF"].fillna(house_baseline['BsmtUnfSF'].mean())
house_baseline["BsmtFinSF2"] = house_baseline["BsmtFinSF2"].fillna(house_baseline['BsmtFinSF2'].mean())
house_baseline["BsmtFinSF1"] = house_baseline["BsmtFinSF1"].fillna(house_baseline['BsmtFinSF1'].mean())
house_baseline["GarageCars"] = house_baseline["GarageCars"].fillna(house_baseline['GarageCars'].mean())


#確認一下是否還有空值

total = house_baseline.isnull().sum().sort_values(ascending=False)
percent = (house_baseline.isnull().sum()/house_baseline.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


#將資料切分成train, test


df_train = house_baseline.iloc[:1460, :]
df_test = house_baseline.iloc[1460:, :]


#確認一下是否符合submission要求

print('train shape', df_train.shape)
print('test shape', df_test.shape)



# print('No. of records in train dataset: ', len(df_train.index))
# print('No. of columns in train dataset: ', len(df_train.columns))
# print('No. of records in test dataset: ', len(df_test.index))
# print('No. of columns in test dataset: ', len(df_test.columns))



#再確認一下是否沒有空值

print ('Total missing values in train set', sum(df_train.isna().sum()))
print ('Total missing values in test set', sum(df_test.isna().sum()))


print ('Total missing values in train set', sum(df_train.isnull().sum()))
print ('Total missing values in test set', sum(df_test.isnull().sum()))





##🌟對於特徵作處理 丟一些outlier

fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], c = "lightcoral")
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()



df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df_train['GrLivArea'], df_train['SalePrice'],  c = "lightcoral")
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


df_train = df_train.drop(df_train[(df_train['LotArea']>150000) & (df_train['SalePrice']<300000)].index)



df_train['LotArea']=np.log(df_train['LotArea'])
sns.displot(x = 'LotArea', data = df_train, kde = True)
skewness=str(df_train["LotArea"].skew())
kurtosis=str(df_train["LotArea"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.title("After applying transform technique")
plt.show()





sns.displot(x = 'GrLivArea', data = df_train, kde = True)
skewness=str(df_train["GrLivArea"].skew())
kurtosis=str(df_train["GrLivArea"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.title("Before applying transform technique")
plt.show()




df_train['GrLivArea']=np.log(df_train['GrLivArea'])
sns.displot(x = 'GrLivArea', data = df_train, kde = True)
skewness=str(df_train["GrLivArea"].skew())
kurtosis=str(df_train["GrLivArea"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.title("After applying transform technique")
plt.show()


# 簡單處理後 先train看看效果

x = df_train.drop(['SalePrice'], axis = 1)
y = df_train['SalePrice']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 31)


len(x_train), len(x_test), len(y_train), len(y_test)



from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)




#model evaluation function
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

def model_evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae


# ## Lasso

from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=0.1, random_state = 32).fit(x_train, y_train)
cv_r2 = cross_val_score(lasso_reg, x_train, y_train, cv = 10)
y_preds = lasso_reg.predict(x_test)
cv_r2 = np.mean(cv_r2)
print("Cross val score: " + str(cv_r2))
r2, mae = model_evaluate(y_test, y_preds)
print("R^2 score: " + str(r2))
print("Mean Absolute Error: " + str(mae))


# ## Ridge

ridge_reg = linear_model.Ridge(alpha=.5).fit(x_train, y_train)
cv_r2 = cross_val_score(ridge_reg, x_train, y_train, cv = 10)
y_preds = ridge_reg.predict(x_test)
cv_r2 = np.mean(cv_r2)
print("Cross val score: " + str(cv_r2))
r2, mae = model_evaluate(y_test, y_preds)
print("R^2 score: " + str(r2))
print("Mean Absolute Error: " + str(mae))


# ## RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=1000).fit(x_train, y_train)
cv_r2 = cross_val_score(rf_reg, x_train, y_train, cv = 10)
y_preds = rf_reg.predict(x_test)
cv_r2 = np.mean(cv_r2)
print("Cross val score: " + str(cv_r2))
r2, mae = model_evaluate(y_test, y_preds)
print("R^2 score: " + str(r2))
print("Mean Absolute Error: " + str(mae))


# ## GradientBoostingRegressor


from sklearn.ensemble import GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=1, random_state=31).fit(x_train, y_train)
cv_r2 = cross_val_score(gbr_reg, x_train, y_train, cv = 10)
y_preds = gbr_reg.predict(x_test)
cv_r2 = np.mean(cv_r2)
print("Cross val score: " + str(cv_r2))
r2, mae = model_evaluate(y_test, y_preds)
print("R^2 score: " + str(r2))
print("Mean Absolute Error: " + str(mae))

# ## xgboost

import xgboost as XGB
xgb = XGB.XGBRegressor(learning_rate=0.01, n_estimators=1000, objective='reg:squarederror', random_state = 31).fit(x_train, y_train)
cv_r2 = cross_val_score(xgb, x_train, y_train, cv = 10)
y_preds = xgb.predict(x_test)
cv_r2 = np.mean(cv_r2)
print("Cross val score: " + str(cv_r2))
r2, mae = model_evaluate(y_test, y_preds)
print("R^2 score: " + str(r2))
print("Mean Absolute Error: " + str(mae))


# 再來處理test data好做predict

df_test['GrLivArea']=np.log(df_test['GrLivArea'])
df_test['LotArea']=np.log(df_test['LotArea'])


df_test = df_test.drop(['SalePrice'], axis=1)


df_test = scale.transform(df_test)



import xgboost as XGB
predictions = ridge_reg.predict(df_test)


# 儲存結果
with open('house_predict1.csv', 'w') as f:
    f.write('id,SalePrice\n')
    for i in range(len(predictions)):
        f.write(str(i + 1461) + ',' + str(float(predictions[i])) + '\n')