# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:13:46 2020

@author: tarun
"""


#!python
import logging
from kiteconnect import KiteConnect
#import json
import datetime as dt
import pandas as pd
import pickle
import copy
from stocktrends import Renko
import numpy as np
import statsmodels.api as sm

def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['high']-df['low'])
    df['H-PC']=abs(df['high']-df['close'].shift(1))
    df['L-PC']=abs(df['low']-df['close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df2 = Renko(df)
    atr_df = ATR(DF,120)
    #print(atr_df["ATR"])
    df2.brick_size = max(0.5,round(atr_df["ATR"].iloc[-1],0))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df


def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252*78)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252*78)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

logging.basicConfig(level=logging.DEBUG)

kite = KiteConnect(api_key="b1nuuiufnmsbhwxx")

# Redirect the user to the login url obtained
# from kite.login_url(), and receive the request_token
# from the registered redirect url after the login flow.
# Once you have the request_token, obtain the access_token
# as follows.

#data = kite.generate_session("j5baS4PWcm9djWUGnooW0CcDqhBrBsan", api_secret="3azcz6fawz1bzet7bqe8624t7c9mkchp")
kite.set_access_token("IpDeZM5KJyxYFR8sktcrbCPFrraJy1Bk")

tickers = ["ICICIBANK","SBIN","WIPRO","TATAMOTORS","IOC", "INFRATEL","ITC","ZEEL","ONGC","NTPC"]
tickers_to_insid = dict()
tickers_to_insid["ICICIBANK"] = "136236548"
tickers_to_insid["SBIN"] = "128028676"
tickers_to_insid["WIPRO"] = "129967364"
tickers_to_insid["TATAMOTORS"] = "128145924"
tickers_to_insid["IOC"] = "135927044"
tickers_to_insid["INFRATEL"] = "136912900"
tickers_to_insid["ITC"] = "128224004"
tickers_to_insid["ZEEL"] = "129417476"
tickers_to_insid["ONGC"] = "128079876"
tickers_to_insid["NTPC"] = "136334084"


ohlc_intraday = {}

start = dt.datetime.today()-dt.timedelta(100)
end = dt.datetime.today()

for ticker in tickers:
    ohlc_intraday[ticker] = kite.historical_data(tickers_to_insid[ticker], start, end, "5minute")


# ohlc_intraday = pickle.load(open( "ohlc_intraday.pickle", "rb" ))

# for ticker in tickers:
#     ohlc_intraday[ticker] = pd.DataFrame(ohlc_intraday[ticker])

# ohlc_renko = {}
# df = copy.deepcopy(ohlc_intraday)
# tickers_signal = {}
# tickers_ret = {}
# for ticker in tickers:
#     print("merging for ",ticker)
#     renko = renko_DF(df[ticker])
#     renko.columns = ["date","open","high","low","close","uptrend","bar_num"]
#     renko["date"] = pd.to_datetime(renko["date"], format='%Y-%m-%d %H:%M:%S%z')
#     df[ticker]["Date"] = df[ticker].index
#     ohlc_renko[ticker] = df[ticker].merge(renko.loc[:,["date","bar_num"]],how="outer",on="date")
#     ohlc_renko[ticker]["bar_num"].fillna(method='ffill',inplace=True)
#     ohlc_renko[ticker]["macd"]= MACD(ohlc_renko[ticker],12,26,9)[0]
#     ohlc_renko[ticker]["macd_sig"]= MACD(ohlc_renko[ticker],12,26,9)[1]
#     ohlc_renko[ticker]["macd_slope"] = slope(ohlc_renko[ticker]["macd"],5)
#     ohlc_renko[ticker]["macd_sig_slope"] = slope(ohlc_renko[ticker]["macd_sig"],5)
#     tickers_signal[ticker] = ""
#     tickers_ret[ticker] = []
    
# #Identifying signals and calculating daily return
# for ticker in tickers:
#     print("calculating daily returns for ",ticker)
#     for i in range(len(ohlc_intraday[ticker])):
#         if tickers_signal[ticker] == "":
#             tickers_ret[ticker].append(0)
#             if i > 0:
#                 if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
#                     tickers_signal[ticker] = "Buy"
#                 elif ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
#                     tickers_signal[ticker] = "Sell"
        
#         elif tickers_signal[ticker] == "Buy":
#             tickers_ret[ticker].append((ohlc_renko[ticker]["close"][i]/ohlc_renko[ticker]["close"][i-1])-1)
#             if i > 0:
#                 if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
#                     tickers_signal[ticker] = "Sell"
#                 elif ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
#                     tickers_signal[ticker] = ""
                
#         elif tickers_signal[ticker] == "Sell":
#             tickers_ret[ticker].append((ohlc_renko[ticker]["close"][i-1]/ohlc_renko[ticker]["close"][i])-1)
#             if i > 0:
#                 if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
#                     tickers_signal[ticker] = "Buy"
#                 elif ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
#                     tickers_signal[ticker] = ""
#     ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])

# #calculating overall strategy's KPIs
# strategy_df = pd.DataFrame()
# for ticker in tickers:
#     strategy_df[ticker] = ohlc_renko[ticker]["ret"]
# strategy_df["ret"] = strategy_df.mean(axis=1)
# CAGR(strategy_df)
# sharpe(strategy_df,0.025)
# max_dd(strategy_df)  

# #visualizing strategy returns
# (1+strategy_df["ret"]).cumprod().plot()

# #calculating individual stock's KPIs
# cagr = {}
# sharpe_ratios = {}
# max_drawdown = {}
# for ticker in tickers:
#     print("calculating KPIs for ",ticker)      
#     cagr[ticker] =  CAGR(ohlc_renko[ticker])
#     sharpe_ratios[ticker] =  sharpe(ohlc_renko[ticker],0.025)
#     max_drawdown[ticker] =  max_dd(ohlc_renko[ticker])

# KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=["Return","Sharpe Ratio","Max Drawdown"])      
# KPI_df.T
