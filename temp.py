
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

logging.basicConfig(level=logging.DEBUG)

kite = KiteConnect(api_key="b1nuuiufnmsbhwxx")

kite.set_access_token("fKDhMi8e4VztzWj1EEHx2hhngwb5ewJn")

ohlc = kite.ohlc(["BSE:WIPRO","BSE:ITC"])

print(ohlc["BSE:ITC"]["last_price"])

import pickle