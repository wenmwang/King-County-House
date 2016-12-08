import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def process_features(df):
    tmp = df.copy()
    tmp['date'] = pd.to_datetime(df['date'])
    tmp['month'] = tmp.date.dt.month
    tmp['year'] = tmp.date.dt.year
    tmp['most_recent_renov'] = np.maximum(tmp['yr_built'], tmp['yr_renovated'])
    tmp.drop(['yr_built', 'yr_renovated'], axis=1, inplace=True)
    return tmp.iloc[:,3:]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
price_train = train['price']
price_test = test['price']
sqftPrice_train = train['price'] / train['sqft_living']
sqftPrice_test = test['price'] / test['sqft_living']
train2 = process_features(train)
test2 = process_features(test)
train2_knn = train2.copy()
test2_knn = test2.copy()
train2_knn['grade'] = np.exp(train2_knn['grade'])
test2_knn['grade'] = np.exp(test2_knn['grade'])
c = train2.columns
scaler = StandardScaler()
train_s = pd.DataFrame(scaler.fit_transform(train2), columns=c)
test_s = pd.DataFrame(scaler.transform(test2), columns=c)
scaler_knn = StandardScaler()
train_s_knn = pd.DataFrame(scaler_knn.fit_transform(train2_knn), columns=c)
test_s_knn = pd.DataFrame(scaler_knn.transform(test2_knn), columns=c)

print 'KNN:'
m1 = KNeighborsRegressor(n_neighbors=12)
knn_train = train_s_knn[['lat', 'long', 'sqft_living', 'waterfront', 'grade']]
knn_test = test_s_knn[['lat', 'long', 'sqft_living', 'waterfront', 'grade']]
m1.fit(knn_train, sqftPrice_train)
sqftPrice_fit = m1.predict(knn_train)
price_knn_train = sqftPrice_fit * train['sqft_living']
sqftPrice_pred = m1.predict(knn_test)
price_knn_pred = sqftPrice_pred * test['sqft_living']
print '\tTest RMSE of Sales Price: %d' % np.sqrt(mean_squared_error(price_test, price_knn_pred))
print '\tAverage Training Error: %.4f' % np.mean(np.true_divide(abs(price_train - price_knn_train), price_train))
print '\tAverage Test Error: %.4f\n' % np.mean(np.true_divide(abs(price_test - price_knn_pred), price_test))

print 'Extra Tree:'
m2 = ExtraTreesRegressor(n_estimators=500, min_samples_split=5)
m2.fit(train2, price_train)
price_et_train = m2.predict(train2)
price_et_pred = m2.predict(test2)
print '\tTest RMSE of Sales Price: %d' % np.sqrt(mean_squared_error(price_test, price_et_pred))
print '\tAverage Training Error: %.4f' % np.mean(np.true_divide(abs(price_train - price_et_train), price_train))
print '\tAverage Test Error: %.4f\n' % np.mean(np.true_divide(abs(price_test - price_et_pred), price_test))

print 'Gradient Boosting:'
m3 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5)
m3.fit(train2, price_train)
price_sgb_train = m3.predict(train2)
price_sgb_pred = m3.predict(test2)
print '\tTest RMSE of Sales Price: %d' % np.sqrt(mean_squared_error(price_test, price_sgb_pred))
print '\tAverage Training Error: %.4f' % np.mean(np.true_divide(abs(price_train - price_sgb_train), price_train))
print '\tAverage Test Error: %.4f\n' % np.mean(np.true_divide(abs(price_test - price_sgb_pred), price_test))

print 'Neural Network:' ## iterate with the same hyper parameters until training error is under threshold.
train_error = 1
while train_error > 0.13:
    m4 = MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu', alpha=0.2, max_iter=500, learning_rate_init=0.02)
    m4.fit(train_s, price_train)
    price_nn_train = m4.predict(train_s)
    price_nn_pred = m4.predict(test_s)
    train_error = np.mean(np.true_divide(abs(price_nn_train - price_train), price_train))
print '\tTest RMSE of Sales Price: %d' % np.sqrt(mean_squared_error(price_test, price_nn_pred))
print '\tAverage Training Error: %.3f' % train_error
print '\tAverage Test Error: %.3f\n' % np.mean(np.true_divide(abs(price_test - price_nn_pred), price_test))

print 'Ensemble:'
combine_train = np.column_stack((price_sgb_train, price_et_train,  price_nn_train, price_knn_train))
combine_test = np.column_stack((price_sgb_pred, price_et_pred, price_nn_pred, price_knn_pred))
scaler_ens = StandardScaler()
combine_train = scaler_ens.fit_transform(combine_train)
combine_test = scaler_ens.transform(combine_test)
combiner = Ridge(alpha=1200)
combiner.fit(combine_train, price_train)
price_fit = combiner.predict(combine_train)
price_pred = combiner.predict(combine_test)
wgt = combiner.coef_
wgt = wgt / sum(wgt)
print '\tTest RMSE of Sales Price: %d' % np.sqrt(mean_squared_error(price_test, price_pred))
print '\tAverage Training Error: %.4f' % np.mean(np.true_divide(abs(price_fit - price_train), price_train))
print '\tAverage Test Error: %.4f' % np.mean(np.true_divide(abs(price_test - price_pred), price_test))
print '\tFinal model consists of %d%% SGB, %d%% extra tree, %d%% neural net, and %d%% knn' % (int(round(wgt[0]*100)), int(round(wgt[1]*100)), int(round(wgt[2]*100)), int(round(wgt[3]*100)))