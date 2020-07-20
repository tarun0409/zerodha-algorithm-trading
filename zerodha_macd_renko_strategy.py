# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:22:49 2020

@author: tarun
"""


import logging
from kiteconnect import KiteConnect
#import json
import datetime as dt
import pandas as pd
import copy
from stocktrends import Renko
import numpy as np
import statsmodels.api as sm
import time
import json
import pickle

logging.basicConfig(level=logging.DEBUG)

kite = KiteConnect(api_key="b1nuuiufnmsbhwxx")

kite.set_access_token("fKDhMi8e4VztzWj1EEHx2hhngwb5ewJn")

starttime = time.time()

# timeout = time.time() + 60*15

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

def renko_merge(DF):
    "function to merging renko df with original ohlc df"
    df = copy.deepcopy(DF)
    renko = renko_DF(df)
    renko.columns = ["date","open","high","low","close","uptrend","bar_num"]
    renko["date"] = pd.to_datetime(renko["date"], format='%Y-%m-%d %H:%M:%S%z')
    merged_df = df.merge(renko.loc[:,["date","bar_num"]],how="outer",on="date")
    merged_df["bar_num"].fillna(method='ffill',inplace=True)
    merged_df["macd"]= MACD(merged_df,12,26,9)[0]
    merged_df["macd_sig"]= MACD(merged_df,12,26,9)[1]
    merged_df["macd_slope"] = slope(merged_df["macd"],5)
    merged_df["macd_sig_slope"] = slope(merged_df["macd_sig"],5)
    return merged_df

def trade_signal(MERGED_DF,l_s):
    "function to generate signal"
    signal = ""
    df = copy.deepcopy(MERGED_DF)
    if l_s == "":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Buy"
        elif df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Sell"
            
    elif l_s == "long":
        if df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Sell"
        elif df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
            
    elif l_s == "short":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Buy"
        elif df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
    return signal

open_positions = {}
stocks = ["ICICIBANK","SBIN","WIPRO","TATAMOTORS","IOC", "INFRATEL","ITC","ZEEL","ONGC","NTPC"]
# tickers_to_insid = dict()
# tickers_to_insid["ICICIBANK"] = "136236548"
# tickers_to_insid["SBIN"] = "128028676"
# tickers_to_insid["WIPRO"] = "129967364"
# tickers_to_insid["TATAMOTORS"] = "128145924"
# tickers_to_insid["IOC"] = "135927044"
# tickers_to_insid["INFRATEL"] = "136912900"
# tickers_to_insid["ITC"] = "128224004"
# tickers_to_insid["ZEEL"] = "129417476"
# tickers_to_insid["ONGC"] = "128079876"
# tickers_to_insid["NTPC"] = "136334084"

profit = 0


def main(start_time, end_time):
    try:
        for stock in stocks:
            long_short = ""
            if len(open_positions)>0 and stock in open_positions:
                long_short = open_positions[stock]["position"]
            instrument_name = "BSE:"+stock
            curr_ohlc = kite.ohlc([instrument_name])
            instrument_token = curr_ohlc[instrument_name]["instrument_token"]
            hd = kite.historical_data(instrument_token, start_time, end_time, "5minute")
            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
            filename = stock+timestr+".pickle"
            with open(filename,"wb") as fp:
                pickle.dump(hd,fp)
            ohlc = pd.DataFrame(hd)
            signal = trade_signal(renko_merge(ohlc),long_short)
            if signal == "Buy":
                if stock in open_positions:
                    print(stock,"already has an open position")
                else:
                    #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_BUY, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                    open_positions[stock] = dict()
                    open_positions[stock]["position"] = "long"
                    curr_price = ohlc['close'].tolist()[-1]
                    open_positions[stock]["cost_price"] = curr_price
                    print("")
                    print("****************************************")
                    print("New long position initiated for ", stock)
                    print("Bought 100 shares of ",stock, "at",curr_price,"per stock")
                    print("****************************************")
                    print("")
                    
            elif signal == "Sell":
                if stock in open_positions:
                    print(stock,"already has an open position")
                else:
                    #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_SELL, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                    open_positions[stock] = dict()
                    open_positions[stock]["position"] = "short"
                    curr_price = ohlc['close'].tolist()[-1]
                    open_positions[stock]["sell_price"] = curr_price
                    print("")
                    print("****************************************")
                    print("New short position initiated for ", stock)
                    print("Sold 100 shares of ",stock, "at",curr_price,"per stock")
                    print("****************************************")
                    print("")
            elif signal == "Close":
                if stock not in open_positions:
                    print(stock,"does not have an open position")
                else:
                    if open_positions[stock]["position"] == "long":
                        #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_SELL, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                        curr_price = ohlc['close'].tolist()[-1]
                        open_positions[stock]["sell_price"] = curr_price
                        curr_profit = open_positions[stock]["sell_price"]-open_positions[stock]["cost_price"]
                        profit += curr_profit
                        print("")
                        print("****************************************")
                        print("Sold 100 shares of ",stock, "at",curr_price,"per stock")
                        print("Total Profit on this position:",curr_profit)
                        print("Overall Profit:",profit)
                        print("****************************************")
                        print("All positions closed for ", stock)
                        print("")
                    else:
                        #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_BUY, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                        curr_price = ohlc['close'].tolist()[-1]
                        open_positions[stock]["cost_price"] = curr_price
                        curr_profit = open_positions[stock]["sell_price"]-open_positions[stock]["cost_price"]
                        profit += curr_profit
                        print("")
                        print("****************************************")
                        print("Bought 100 shares of ",stock, "at",curr_price,"per stock")
                        print("Total Profit on this position:",curr_profit)
                        print("Overall Profit:",profit)
                        print("****************************************")
                        print("All positions closed for ", stock)
                        print("")
                    del open_positions[stock]
            elif signal == "Close_Buy":
                if stock not in open_positions:
                    print(stock, "does not have an open position")
                elif open_positions[stock]["position"] == "long":
                    print(stock,"already in long position!")
                else:
                    #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_BUY, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                    curr_price = ohlc['close'].tolist()[-1]
                    open_positions[stock]["cost_price"] = curr_price
                    curr_profit = open_positions[stock]["sell_price"]-open_positions[stock]["cost_price"]
                    profit += curr_profit
                    print("")
                    print("****************************************")
                    print("Bought 100 shares of ",stock, "at",curr_price,"per stock")
                    print("Total Profit on this position:",curr_profit)
                    print("Overall Profit:",profit)
                    print("Existing Short position closed for ", stock)
                    print("****************************************")
                    print("")
                    #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_BUY, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                    curr_price = ohlc['close'].tolist()[-1]
                    open_positions[stock]["cost_price"] = curr_price
                    print("")
                    print("****************************************")
                    print("New long position initiated for ", stock)
                    print("Bought 100 shares of ",stock, "at",curr_price,"per stock")
                    print("****************************************")
                    print("")
                    open_positions[stock]["position"] = "long"
            elif signal == "Close_Sell":
                if stock not in open_positions:
                    print(stock, "does not have an open position")
                elif open_positions[stock]["position"] == "short":
                    print(stock,"already in short position!")
                else:
                    #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_SELL, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                    curr_price = ohlc['close'].tolist()[-1]
                    open_positions[stock]["sell_price"] = curr_price
                    curr_profit = open_positions[stock]["sell_price"]-open_positions[stock]["cost_price"]
                    profit += curr_profit
                    print("")
                    print("****************************************")
                    print("Sold 100 shares of ",stock, "at",curr_price,"per stock")
                    print("Total Profit on this position:",curr_profit)
                    print("Overall Profit:",profit)
                    print("Existing long position closed for ", stock)
                    print("****************************************")
                    print("")
                    #kite.place_order(kite.VARIETY_REGULAR, "BSE", stock, kite.TRANSACTION_TYPE_SELL, 100, kite.PRODUCT_MIS, kite.ORDER_TYPE_MARKET)
                    curr_price = ohlc['close'].tolist()[-1]
                    open_positions[stock]["sell_price"] = curr_price
                    print("")
                    print("****************************************")
                    print("New short position initiated for ", stock)
                    print("Sold 100 shares of ",stock, "at",curr_price,"per stock")
                    print("****************************************")
                    print("")
                    open_positions[stock]["position"] = "short"
            orders = kite.orders()
            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
            filename = "orders"+timestr+".pickle"
            with open(filename,"wb") as fp:
                pickle.dump(orders,fp)
        print("Current open positions : ",open_positions)
    except Exception as e:
        print(e)
        print("Some error occurred. Skipping this iteration. Going to next iteration")
            

end_hour = 20
end_minute = 00

while dt.datetime.now() < dt.datetime.now().replace(hour=end_hour,minute=end_minute):    
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        start = dt.datetime.now()-dt.timedelta(6.48)
        end = dt.datetime.now()-dt.timedelta(3.34)
        main(start,end)
        print("start_day:",start,"end_day",end, "overall profit till now:",profit)
        time.sleep(300 - ((time.time() - starttime) % 300)) # 5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
