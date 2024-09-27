

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta

sns.set()

#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)


orig_cols = df.columns.values.tolist()

#we are going to enter the trade at the next open, not wait till the next close
#save the open for white reality check
openp = df['<OPEN>'].copy()
close = df['<CLOSE>'].copy()


high = df['<HIGH>'].shift(1)
low = df['<LOW>'].shift(1)
close = df['<CLOSE>'].shift(1)


n=10
df['RSI']=ta.RSI(np.array(close), timeperiod=n)
df['SMA'] = close.rolling(window=n).mean()
df['Corr']= close.rolling(window=n).corr(df['SMA'])
df['SAR']=ta.SAR(np.array(high),np.array(low), 0.2, 0.2)
df['ADX']=ta.ADX(np.array(high),np.array(low), np.array(df['<OPEN>']), timeperiod =n)
df['OO']= df['<OPEN>']-df['<OPEN>'].shift(1)
df['OC']= df['<OPEN>']-close
df.fillna(0, inplace=True)

##build day time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(["hour","day"], axis=1, inplace=True)

#build target
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade at the open
#df['retFut1'] = df['<CLOSE>'].pct_change(1).shift(-1).fillna(0) #if you wait until the close to enter the trade
#df.dropna(inplace=True) #make sure no Nans in df
#df = np.log(df+1)

#Preserve for calculations of system return
retFut1 = df['retFut1'].copy()

#build lags
for n in list(range(0,15)):
    name = 'lag_ret' + str(n)
    df[name] =  df['<OPEN>'].pct_change(1).shift(n).fillna(0)

#select the features (by dropping)
orig_cols = ['<HIGH>', '<LOW>', '<CLOSE>', '<SPREAD>',  '<VOL>'] #keep the open
df.drop(orig_cols, axis=1, inplace=True)

#distribute the df data into X inputs and y target
X = df.drop(['retFut1'], axis=1)
y = df[['retFut1']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

retFut1_train = retFut1[0:10000]
retFut1_test = retFut1[10000:12000]


##########################################################################################################################
#set up the grid search and fit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer 
from sklearn.inspection import permutation_importance
import detrendPrice 
import WhiteRealityCheckFor1 
from sklearn.preprocessing import FunctionTransformer


def information_coefficient(y_true, y_pred):
    rho, pval = spearmanr(y_true,y_pred) #spearman's rank correlation
    print (rho)
    return rho

def sharpe(y_true, y_pred):
    positions = np.where(y_pred> 0,1,-1 )
    dailyRet = pd.Series(positions).shift(1).fillna(0).values * y_true
    dailyRet = np.nan_to_num(dailyRet)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
    return ratio

"""
INSTRUCTIONS
Define a select_scaler that does not scale dummies
"""

def select_scaler(x_train):
        scaler = StandardScaler()
        df = pd.DataFrame(x_train)
        df_filtered = df.select_dtypes(exclude=['uint8'])
        df_scaled = pd.DataFrame(scaler.fit_transform(df_filtered))
        df_dummies = df.select_dtypes(include=['uint8'])
        df_scaled = df_scaled.join(df_dummies)
        return df_scaled.values

#myscorer = "neg_mean_squared_error"
myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)


#we turn off scaling because we should not scale dummies (and returns are already mostly scaled)
scaler = StandardScaler(with_mean=False, with_std=False)

custom_scaler = FunctionTransformer(select_scaler)

svr = NuSVR()

pipe = Pipeline([("scaler", scaler), ("select_scaler", custom_scaler), ("svr", svr)])


c_rs = np.linspace(0.001, 5, num=5, endpoint=True) #1 default
#gamma_rs = ["scale","auto"]
gamma_rs = ["scale"]
kernel_rs = ['rbf']
nu_rs = [1]


#set of parameters for random search
param_grid = {'svr__kernel': kernel_rs,
               'svr__C': c_rs,
               'svr__gamma': gamma_rs,
               'svr__nu': nu_rs}

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("svrreggression_results.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
positions = np.where(grid_search.predict(x_train.values)> 0,1,-1 ) #################


dailyRet = pd.Series(positions).fillna(0).values * retFut1_train
dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated SVRRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')


cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

# Test set
# Make "predictions" on test set (out-of-sample)

#positions2 = np.where(best_model.predict(x_test.values)> 0,1,-1 )
positions2 = np.where(grid_search.predict(x_test.values)> 0,1,-1 ) #################


dailyRet2 = pd.Series(positions2).fillna(0).values * retFut1_test
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated SVRRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("Cumulative"))

rho, pval = spearmanr(y_test,grid_search.predict(x_test.values)) #spearman's rank correlation: very small but significant

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  Rho={:0.6} PVal={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, rho, pval))


#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test.values)
residuals = np.subtract(true_y, pred_y)

from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout();
#plt.show()
plt.savefig(r'Results\%s.png' %("Residuals"))

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])


#white reality check
detrended_open = detrendPrice.detrendPrice(openp[10000:12000])
detrended_retFut1 = detrended_open.pct_change(periods=1).shift(-1).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()

"""
RESULTS
myscorer = make_scorer(information_coefficient, greater_is_better=True)
Out-of-sample: CAGR=0.0191385 Sharpe ratio=0.600173 maxDD=-0.0506941 maxDDD=498 Calmar ratio=0.377529  Rho=0.0607563 PVal=0.00656941
p_value:
0.27780000000000005

Out-of-sample: CAGR=0.0199874 Sharpe ratio=0.625854 maxDD=-0.0702316 maxDDD=501 Calmar ratio=0.284592  Rho=0.0589514 PVal=0.00836354
p_value:
0.4232

"""

#support vector machines do not expose any importances or any coefficients, so
#the only way to look at the feature importances is to use feature pemrutation.
#However, feature permutation is a method that works well only for linear models, so
#these permutation importances are to be taken with a grain of salt.
#For details see: https://towardsdatascience.com/stop-permuting-features-c1412e31b63f
#Unfortunately, the permutation_importance function cannot use our custom scorer,
#so we need to use 'neg_mean_squared_error' (not ‘r2’ which is only for linear models)

#THIS TAKES A LONG TIME TO RUN
#result = permutation_importance(best_model[2], x_train.values, y_train.values.ravel(), scoring='neg_mean_squared_error', n_repeats=5)
#importance = pd.DataFrame(zip(x_train.columns,result.importances_mean))
#importance.columns = ['feature_name','Per_score']
#importance_plot = sns.barplot(x=importance['feature_name'], y=importance['Per_score'], data=importance,orient='v',dodge=False,order=importance.sort_values('Per_score',ascending=False).feature_name)
#for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
#    item.set_rotation(90)
#plt.show()
#plt.savefig(r'Results\%s.png' %("Feature Importance"))

