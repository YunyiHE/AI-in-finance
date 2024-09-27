import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import functions as ff

sns.set()

def single_autocorr(series, lag):
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0

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
#df["hour"] = df.index.hour.values
#df["day"] = df.index.dayofweek.values
#df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
#df_dummies_day = pd.get_dummies(df["day"], prefix='day')
#df =df.join(df_dummies_hour)
#df=df.join(df_dummies_day)
#df.drop(["hour","day"], axis=1, inplace=True)

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

#transform the target
df['retFut1'] = np.where((df['retFut1'] > 0), 1, 0)


#select the features (by dropping)
orig_cols = ['<HIGH>', '<LOW>', '<CLOSE>', '<SPREAD>', '<VOL>' ] #do not drop the open price
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
import phik
from phik.report import plot_correlation_matrix
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import FunctionTransformer
from scipy.special import ndtr
import detrendPrice 
import WhiteRealityCheckFor1 
import math

#global variables
fin_arr = np.zeros(x_train.shape[1])
pval_arr = np.zeros(x_train.shape[1])
counter = 0


def phi_k(y_true, y_pred):
    dfc = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    try:
        phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
        phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
        phi_k_p_val = 1 - ndtr(phi_k_sig) 
    except:
        phi_k_corr = 0
        phi_k_p_val = 0
    print(phi_k_corr)
    print(phi_k_p_val)
    return phi_k_corr

def fin_select(X, y):
    #Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
    #Model: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
    return fin_arr, pval_arr

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def select_prepare(X, target, retFut1, model):
    global fin_arr, pval_arr, counter
    if counter < 2:
        #select_prepare is called multiple times by the batch generator, you may want to run this loop only once if you have a ton of data
        for i in range(X.shape[1]):
            print("in select prepare")
            target = target[-X.shape[0]:] #this is needed because the last batch is a leftover with smaller size
            retFut1 = retFut1.iloc[-X.shape[0]:] #this is needed because the last batch is a leftover with smaller size
            model.fit(X[:, i].reshape(-1,1), target)
            preds = model.predict(X[:, i].reshape(-1,1))
            positions = np.where(preds> 0,1,-1 )
            dailyRet = pd.Series(positions).fillna(0).values * retFut1
            dailyRet = dailyRet.fillna(0)
            cumret = np.cumprod(dailyRet + 1) - 1
            cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
            sharpe_ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
            fin_arr[i] = cagr #use cagr or sharpe_ratio here
            pval_arr[i] = 1-sigmoid(cagr) #use cagr or sharpe_ratio here
    counter = counter + 1
    return X


#myscorer = None
myscorer = make_scorer(phi_k, greater_is_better=True) #very slow

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
print(encoded)

#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)

#we turn off scaling if dummies (returns are already mostly scaled)
scaler = StandardScaler(with_mean=True, with_std=True) #we kept the open

logistic = LogisticRegression(max_iter=1000, solver='liblinear') 

select_prep = FunctionTransformer(select_prepare, kw_args={'target': y_train.values.ravel(), 'retFut1': retFut1_train, 'model':logistic}) 

selector = SelectPercentile(score_func=fin_select, percentile=50)
#selector = SelectPercentile(score_func=f_classif, percentile=50)

pipe = Pipeline([("scaler", scaler), ("select_prep", select_prep), ("select", selector), ("logistic", logistic)])

c_rs = np.logspace(3, 0, num=20, endpoint = True)

p_rs= ["l1", "l2"]

param_grid =  [{'logistic__C': c_rs, 'logistic__penalty': p_rs}]

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True, n_jobs = -1)
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=myscorer, return_train_score=True, n_jobs = -1)

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logisticreg.csv")


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
plt.title('Cross-validated LogisticRegression on currency: train set')
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
plt.title('Cross-validated LogisticRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("Cumulative"))

#metrics
accuracy_score = accuracy_score(y_test.values.ravel(), grid_search.predict(x_test.values))

#If this figure does not plot correctly select the lines and press F9 again
arr1 = y_test.values.ravel()
arr2 = grid_search.predict(x_test.values)
dfc = pd.DataFrame({'y_true': arr1, 'y_pred': arr2})
phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
significance_overview = dfc.significance_matrix(interval_cols=[])
phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
phi_k_p_val = 1 - ndtr(phi_k_sig) 
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Significance of the coefficients", 
                        usetex=False, fontsize_factor=1.5, figsize=(7, 5))
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Significance"))

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  phi_k_corr={:0.6} phi_k_p_val={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, phi_k_corr, phi_k_p_val, accuracy_score))


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


#Detrending Prices and Returns and white reality check
detrended_open = detrendPrice.detrendPrice(openp[10000:12000])
detrended_retFut1 = detrended_open.pct_change(periods=1).shift(-1).fillna(0)
detrended_syst_rets = detrended_retFut1 * pd.Series(positions2).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()



#number of features kept:
print("the number of features kept was: ", np.size(best_model[3].coef_))

#plot the coefficients
importance = pd.DataFrame(zip(best_model[3].coef_.ravel().tolist(), x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients"))