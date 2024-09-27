"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
"RESULTS" comments explain what results to expect from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
"""

"""
Below we use a FunctionTransformer and ColumnTransformer together 
to program the pipeline for a random forest that
uses wavelets to smooth some numerical features.
So you need to review the homework answer where we showed how to use ColumnTransformer and
you need to install wavelets:
conda install -c conda-forge pywavelets

"""
#conda install -c conda-forge pywavelets
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

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


#build date time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(["hour","day"], axis=1, inplace=True)

 
#build momentum features
for n in list(range(1,5)):
    df_period = df.pct_change(periods=n).fillna(0)
    name = 'ret' + str(n)
    df[name] = df_period['<OPEN>']
del df_period


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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer 
import detrendPrice 
import WhiteRealityCheckFor1 
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer



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
We have programmed a wavelet smoother function that uses one of the Daubechies wavelets.
As per the instructions in:
https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html
Our wavelet_soother has a hard_coded parameter wavelet="db6" but
there are many such wavelets:
db (Daubechies) family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
You could use a parameter grid to optimize the wavelet parameter choice 
by including a suitable parameter grid for it.
However, including a parameter grid for this homework is optional.
You only need to insert the wavelet_soother as it is (with a hard_coded parmeter wavelet="db6")
into the pipeline, using a FunctionTransformer and a ColumnTransformer following the example:
logistic_regression_with_grid_scorer_timeseriessplit_rollingscaler_dummies_coltransfomer_WRC.py
provided in the last homework answer.
NOTE: 
To make ColumnTransformer work properly,
when fitting the model, send in an x_train dataframe.
RandomForest should not turn the x_train dataframe into an array, so 
there is no need to make any corrections regarding this issue.

"""

def wavelet_smoother(x_train, scale=None):
    
        wavelet = "db6"
        df_wavelets = x_train.copy()
        
        for i in x_train.columns:
            signal = x_train[i]
            coefficients = pywt.wavedec(signal, wavelet, mode='per')
            coefficients[1:] = [pywt.threshold(i, value=scale*signal.max(), mode='soft') for i in coefficients[1:]]
            reconstructed_signal = pywt.waverec(coefficients, wavelet, mode='per')
            df_wavelets[i] = reconstructed_signal
        
        df_wavelets = df_wavelets.fillna(0)
        return df_wavelets
    
myscorer = None #is mse
myscorer = make_scorer(information_coefficient, greater_is_better=True)
#myscorer = make_scorer(sharpe, greater_is_better=True)

#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)


"""
INSTRUCTIONS

Use the FunctionTransformer to construct a wavelet_smoother object,
put the outcome in wavelet_smoother
There is no need to scale the inputs of a tree so 
comment out the standard scaler
Instantiate a PCA model,
save it in pca.
"""
#random forest doesn't really require scalng
#scaler = StandardScaler(with_mean=False, with_std=False)
wavelet_smoother = FunctionTransformer(wavelet_smoother)
pca = PCA()

"""
INSTRUCTIONS

The function dataframe.columns.difference() 
gives you complement of the values that you provide as argument. 
It can be used to create a new dataframe from an existing dataframe with exclusion of some columns
Use dataframe.columns.difference() to construct a dataframe without '<TICKVOL>','<OPEN>'
put the outcome in dfs.
Construct an index (a list of column names) to include the remaining numerical columns (float64) in dfs i.e. the lags
put the outcome in numerical_ix
"""

# determine the numerical features I want to smooth 
dfs = x_train[x_train.columns.difference(['<TICKVOL>','<OPEN>'])]
numerical_ix = dfs.select_dtypes(include=['float64']).columns

"""
INSTRUCTIONS

Define the transformer sub-pipeline with two steps: 
step1: 'wav' with wavelet_smoother and numerical_ix 
step2: 'pca' with pca and numerical_ix
save the transformer sub-pipeline in t
Define a ColumnTransformer using the parameters: transformers=t and remainder='passthrogh'
put the column transformer in col_transform
Note that with this sub-pipeline, you avoid applying wavelets and pca on categorical features
"""

# define the data preparation for the columns
t = [('wav', wavelet_smoother, numerical_ix), ('pca', pca, numerical_ix)]
col_transform = ColumnTransformer(transformers=t, remainder='passthrough')


"""
INSTRUCTIONS

As usual, 
define a model and save it in rforest
construct the pipeline from two steps:
step1: 'prep' with the sub-pipeline col_transform
step2: 'rforest' with rforest
"""

rforest = RandomForestRegressor()


#do not use pca for dummies
pipe = Pipeline([('prep', col_transform), ('rforest', rforest)])

#max_depth_rs = [200, 300, 500]
n_estimators_rs = [200, 300, 500]
ncomponents_rs =   list(range(10,x_test.shape[1]))
scales_rs = [{'scale': .5},{'scale': .1}]

"""
INSTRUCTIONS

set the param_grid paying attention to the double underscores
scales_rs pertains to wavelets
set up the grid_search
you may use n_jobs=-1 to speed up processing 
but it is not always compatible with the user's operating system
if you are selecting the Daubechies wavelet among the 38 possible ones
you will need n_jobs=-1 but
for this homework you can skip it since we hardcoded just one wavelet.
"""

#set of parameters for random search
#param_grid =  [{'prep__wav__kw_args': scales_rs, 'prep__pca__n_components':ncomponents_rs, 'rforest__max_depth': max_depth_rs}]
param_grid =  [{'prep__wav__kw_args': scales_rs, 'prep__pca__n_components':ncomponents_rs, 'rforest__n_estimators': n_estimators_rs}]

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True, n_jobs = -1)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True, n_jobs = -1)

#grid_search.fit(x_train.values, y_train.values.ravel())
grid_search.fit(x_train, y_train.values.ravel())


best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("randomforestregression_results.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
positions = np.where(grid_search.predict(x_train)> 0,1,-1 ) #################

#dailyRet = fAux.backshift(1, positions) * x[:train_set,0] # x[:train_set,0] = ret1
dailyRet = pd.Series(positions).fillna(0).values * retFut1_train
dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated RandomForestRegression on currency: train set')
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
positions2 = np.where(grid_search.predict(x_test)> 0,1,-1 ) #################

dailyRet2 = pd.Series(positions2).fillna(0).values * retFut1_test
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated RandomForestRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("Cumulative"))

rho, pval = spearmanr(y_test,grid_search.predict(x_test)) #spearman's rank correlation: very small but significant

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  Rho={:0.6} PVal={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, rho, pval))

"""
RESULTS
myscorer = make_scorer(information_coefficient, greater_is_better=True)
Out-of-sample: CAGR=0.168216 Sharpe ratio=5.03764 maxDD=-0.0264985 maxDDD=69 Calmar ratio=6.34814  Rho=0.394247 PVal=2.3077e-75
p_value:
0.0

"""

#plot the residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test)
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



#plot the importances
importance = pd.DataFrame(zip(best_model[1].feature_importances_, x_train.columns.values.tolist()))
importance.columns = ['importance','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['importance'], data=importance,orient='v',dodge=False,order=importance.sort_values('importance',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Importances"))

"""
QUESTION
Not Optional:
What happens to the profits if you
 "roll the prediction 1 period forward".
have the model predict the return from the next period's (3 hours ahead) open 
to the following period's (6 hours ahead) open

Try it and save it under a different name and turn it in.
"""