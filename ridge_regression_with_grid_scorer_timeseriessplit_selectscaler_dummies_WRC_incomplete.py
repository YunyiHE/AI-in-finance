"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
"RESULTS" comments explain what results to expect from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.

You will be filling in code in two types of models:
1. a regression model and
2. a classification model.

Most of the time, because of similarities,
you can cut and paste from one model to the other.
But in a few instances, you cannot do this, so
you need to pay attention.
Also, in some cases,
you will find a "hint" for a solution 
in one of the two scripts (regression or classification)
that you can use as inspiration for the other.

This double task gives you the opportunity to look at the results
in both regression and classification approaches.

At the bottom, you will find some questions that we pose.
You do not need to write and turn in the answer to these questions,
but we strongly recommend you find out the answers to them.
"""

"""
We are going to show you the simplest way to customize a scaler to apply to some features and not others
For this objective, we will use the Function Transformer utility.
However, be aware that scikit-learn has a Column Transformer utility which 
when used together with the Function Transformer utility provides maximum flexibility.
The idea of this homework is to introduce you to the concept of:
using models to build features that other models use, 
all of it within the same pipeline.
This is called chaining models.
It is an advanced concept, but you can start reading here:
Function Transformer Reading (optional):
#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
"""

"""
IMPORTANT
This program effects two additional changes you have not seen before:
1. we enter positions at the open instead of waiting for the close and
2. we calculate the system returns by multiplying positions at t by target returns at t (more intuitive)
However, doing things this way does require a little extra housekeeping
"""


import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns

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

#save the close and open for white reality check
openp = df['<OPEN>'].copy() #for the case we want to enter trades at the open
close = df['<CLOSE>'].copy() #for the case we want to enter trades at the close


##build categorical features (cannot be scaled)
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
dumcols = df_dummies_hour.columns.values.tolist() + df_dummies_day.columns.values.tolist()

df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(["hour","day"], axis=1, inplace=True)

##build momentum features
for n in list(range(1,15)):
    df_period = df.pct_change(periods=n).fillna(0)
    name = 'mom_ret' + str(n)
    df[name] = df_period['<OPEN>']
del df_period


#build target
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0) #if you enter the trade at the open
#df['retFut1'] = df['<CLOSE>'].pct_change(1).shift(-1).fillna(0) #if you wait until the close to enter the trade
#df.dropna(inplace=True) #make sure no Nans in df
#df = np.log(df+1)


#preserve for system return calculations
retFut1 = df['retFut1'].copy()


#select the features (by dropping)
orig_cols = ['<HIGH>', '<LOW>', '<CLOSE>', '<SPREAD>',  '<VOL>'] #keep the open (given)
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
import detrendPrice 
import WhiteRealityCheckFor1 
from sklearn.preprocessing import FunctionTransformer

def information_coefficient(y_true, y_pred):
    rho, pval = spearmanr(y_true,y_pred) #spearman's rank correlation
    print (rho)
    return rho

def sharpe(y_true, y_pred):
    positions = np.where(y_pred> 0,1,-1 )
    dailyRet = pd.Series(positions).fillna(0).values * y_true
    dailyRet = np.nan_to_num(dailyRet)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
    return ratio


"""
INSTRUCTIONS:
instantiate a StandardScaler and save it in scaler
put the x_train data into a dataframe, save it in df
use the select_dtypes method of the dataframe to exclude the categorical features (having uint8 datatype)
save the non-categorical features into df_filtered
use scaler's fit_transform method to scale the df_filtered
put the scaled features into a dataframe called df_scaled
use the select_dtypes of the df dataframe to include the categorical features (having uint8 datatype)
save the categorical features in df_dummies
join the df_scaled to the df_dummies using the join method of df_scaled
save the combined features into df_scaled
return df_scaled.values

Note:
How do you test the select_scaler to see if you have coded it correctly?
The fact is that when you use Scikit-Learn's FunctionTransformer
the output is an object that 
is able to execute the fit_transform method of a regular Scikit-Learn scaler.

For example:
When you use myscaler = FunctionTransformer(select_scaler)
myscaler will automatically be able to use the fit_transform method:
result = myscaler.fit_transform(dataarray)
and the variable result can be inspected to see if select_scaler is coded correctly

"""

def select_scaler(x_train):
        scaler = #####
        df = #####
        df_filtered = #####
        df_scaled = #####
        df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan).fillna(0) #optional
        df_dummies = #####
        df_scaled = #####
        return #####

#myscorer = None
myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
print(encoded)

#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)

#when using rolling_scaler, use TimesSeriesSplit
#split = 5 
split = TimeSeriesSplit(n_splits=5, max_train_size=2000)
#split = TimeSeriesSplit(n_splits=5)

#we turn off scaling because we are using dummies (returns are already mostly scaled)
scaler = StandardScaler(with_mean=False, with_std=False)

select_scaler = FunctionTransformer(select_scaler)

ridge = Ridge(max_iter=1000) 

pipe = Pipeline([("scaler", scaler), ("select_scaler", select_scaler), ("ridge", ridge)])
a_rs = np.logspace(-7, 0, num=10, endpoint = True)

param_grid =  [{'ridge__alpha': a_rs}]

#when using rolling_scaler, use TimesSeriesSplit
grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)
#grid_search = GridSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_ridgereg.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
positions = np.where(grid_search.predict(x_train.values)> 0,1,-1 ) #################

#dailyRet = fAux.backshift(1, positions) * x[:train_set,0] # x[:train_set,0] = ret1
dailyRet = pd.Series(positions).fillna(0).values * retFut1_train 
dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated RidgeRegression on currency: train set')
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
plt.title('Cross-validated RidgeRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.show()

#metrics
rho, pval = spearmanr(y_test,grid_search.predict(x_test.values)) #spearman's rank correlation: very small but significant

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}   Rho={:0.6} PVal={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, rho, pval))
    
#residuals
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
plt.show()

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])


#white reality check (entering at the open)
detrended_open = detrendPrice.detrendPrice(openp[10000:12000])
detrended_retFut1 = detrended_open.pct_change(periods=1).shift(-1).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()

"""
RESULTS
using myscorer = make_scorer(information_coefficient, greater_is_better=True)
Out-of-sample: CAGR=0.0248019 Sharpe ratio=0.771224 maxDD=-0.0534287 maxDDD=343 Calmar ratio=0.464206   Rho=0.0960752 PVal=1.67827e-05
p_value:
0.04700000000000004
"""

#coefficients
importance = pd.DataFrame(zip(best_model[2].coef_, x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
plt.show()

"""
QUESTION
Why is it better to enter the trade at the open rather than wait to enter the trade at the close?
"""
