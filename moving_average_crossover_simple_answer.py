
import WhiteRealityCheckFor1 #you can ignore this for now
import detrendPrice #you can ignore this fornow
import pandas as pd
import numpy as np
from datetime import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

start_date = '2000-01-01' 
end_date = '2018-12-31' 
#end_date = datetime.now() 

symbol = '^GSPC' 
msg = "" 
address = symbol + '.csv'

try:
    dfP = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    dfP.to_csv(address, header = True, index=True, encoding='utf-8') 
except Exception:
    msg = "yahoo problem"
    dfP = pd.DataFrame()

dfP = pd.read_csv(address, parse_dates=['Date'])
dfP = dfP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)

#dfP['Close'].plot(grid=True,figsize=(8,5))
dfP['42d'] = np.round(dfP['Close'].rolling(window=42).mean(),2)
dfP['252d'] = np.round(dfP['Close'].rolling(window=252).mean(),2)
#print(dfP.tail)

#dfP[['Close','42d','252d']].plot(grid=True,figsize=(8,5))
dfP['42-252'] = dfP['42d'] - dfP['252d']

dfP['pct_rets'] = (dfP['Close']/dfP['Close'].shift(1))-1

X = 0


dfP['Stance'] = np.where((dfP['42-252'] > X), 1, 0)
dfP['Stance'] = np.where(dfP['42-252'] < X, -1, dfP['Stance'])
#print(dfP['Stance'].value_counts())

#dfP['Stance'].plot(lw=1.5,ylim=[-1.1,1.1])
dfP['syst_rets'] = dfP['pct_rets'] * dfP['Stance'].shift(1) #using percent returns
dfP['syst_cum_rets'] = (dfP['syst_rets']+1).cumprod()-1+1 #using percent returns to get cumulative
dfP['mkt_cum_rets'] = ((dfP['pct_rets']+1).cumprod())-1+1 #using percent returns to get cumulative

dfP[['mkt_cum_rets','syst_cum_rets']].plot(grid=True,figsize=(8,5)) #plotting returns percent cumul

#both are ok
start = 1
start =  dfP['syst_cum_rets'].iloc[2]

start_val = start
end_val = dfP['syst_cum_rets'].iat[-1]

start_date = dfP.iloc[0].name
end_date = dfP.iloc[-1].name
days = (end_date - start_date).days 

periods = 360 #360 accounting days

TotalAnnReturn = (end_val-start_val)/start_val/(days/periods)

years = days/periods
CAGR = ((((end_val/start_val)**(1/years)))-1)


try:
    sharpe =  (dfP['syst_rets'].mean()/dfP['syst_rets'].std()) * np.sqrt(periods)
except ZeroDivisionError:
    sharpe = 0.0

print ("TotalAnnReturn in percent = %f" %(TotalAnnReturn*100))
print ("CAGR = %f" %(CAGR*100))
print ("Sharpe Ratio = %f" %(round(sharpe,2)))

"""
#white reality check
#Detrend prices before calculating detrended returns
dfP['DetClose'] = detrendPrice.detrendPrice(dfP.Close).values #you can ignore this for now
#these are the detrended returns to be fed to White's Reality Check
dfP['Det_pct_rets']= (dfP['DetClose']- dfP['DetClose'].shift(1)) / dfP['DetClose'].shift(1) #you can ignore this for now
dfP['Det_syst_rets']= dfP['Det_pct_rets']*dfP['Stance'].shift(1) #you can ignore this for now
WhiteRealityCheckFor1.bootstrap(dfP.Det_syst_rets #you can ignore this for now

"""
dfP.to_csv(r'Results\dfP_simple_MACO.csv')
#dfP[['Close','42d','252d']].plot(grid=True,figsize=(8,5))



