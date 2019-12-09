import pandas as pd
import numpy as np
import datetime
import time

import cbpro  # https://github.com/danpaquin/coinbasepro-python

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import datetime

import cbpro  # https://github.com/danpaquin/coinbasepro-python

from matplotlib import pyplot as plt

# examples
# a = innitPriceHistory()
# b, didUpdate = updateValues(a)
# print(didUpdate)
# z = featureCalc(b)

# get rates
# https://docs.pro.coinbase.com/#get-historic-rates
def innitPriceHistory():
    '''
    Gets innitial data needed for feature calc (20 days of minute data).
    '''
    # start
    tik = time.time()
    
    public_client = cbpro.PublicClient()  # connect to public API
    a = public_client.get_product_historic_rates('BTC-EUR', granularity=60)  # granularity=seconds
    
    maxReq = len(a)  # sets request size
    requestsNeeded = round(21*24*60/maxReq+0.5)  # calcs the number of requests needed to get innit data  
      
    startDates = [a[0][0]-((maxReq)*60)*(i+1) for i in range(requestsNeeded)]  # gets timestamps
    endDates = [i+((maxReq)*60) for i in startDates]
    
    startDateTime = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z' for i in startDates]  # turns timestamps to ISO 8601
    endDateTime = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z' for i in endDates]                
    
    for i in range(1, len(startDateTime)):  # gets data for all of the time needed
        b = public_client.get_product_historic_rates('BTC-EUR', granularity=60, start=startDateTime[i], end=endDateTime[i])
        # print(len(b))
        for k in b:
            a.append(k)
        time.sleep(2)  # due to public client limits
        
    b = public_client.get_product_historic_rates('BTC-EUR', 
                                                 granularity=60, 
                                                 start=datetime.datetime.utcfromtimestamp(endDates[0]+60).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z',
                                                 end=public_client.get_time().get('iso'))  # adds the most recent data
    
    for k in b:  # appends the most recent data
        a.append(k)
    
    a = pd.DataFrame(a, columns=['time', 
                                'low',
                                'high',
                                'open',
                                'close',
                                'volume']) 
    
    a.drop_duplicates(inplace=True)
    
    # a.drop(columns=['low',
    #                 'high',
    #                 'open',
    #                 'volume'], inplace=True)
    
    dates = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z' for i in a.time]  # converts time
    a['time'] = dates
    a = a.sort_values(by='time').reset_index(drop=True)
    
    tok = time.time()  # checks the end time
    print('Innit data time passed: ', round(tok-tik))
    return(a)

# value update
def updateValues(oldDf):
    '''
    Gets newwest prices from coinbase api
    * oldDf - the dataframe currently used, with old prices and dates. Check inniPriceHistory.
    '''
    startDate = oldDf.iloc[-1,0]
    public_client = cbpro.PublicClient()  # connect to public API
    new_list = public_client.get_product_historic_rates('BTC-EUR',
                                                  granularity=60, 
                                                  start=startDate,
                                                  end=public_client.get_time().get('iso'))
    
    df = pd.DataFrame(new_list, columns=['time', 
                                'low',
                                'high',
                                'open',
                                'close',
                                'volume']) 
    
    df.drop_duplicates(inplace=True)
    
    dates = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+'Z' for i in df.time]  # converts time
    df['time'] = dates
    df = df.sort_values(by='time').reset_index(drop=True)
    
    if len(df) > 1:
        df = pd.concat([oldDf, df], axis=0)
        didUpdate = 1
    else:
        df = oldDf
        didUpdate = 0
                
    df = df[len(df)-28800:]
    
    return(df, didUpdate)


# calc price features
def featureCalc(df, colName='close'):
    '''
    Calculates the price features. Check minute_features for the full list.
    * df - DataFrame used for feature calculation
    * colName - column name from which the features must be extracted. (Default - 'close')
    '''    
    minute_features = {
    '2min':2,
    '3min':3,
    '4min':4,
    '5min':5,
    '10min':10,
    '30min':30,
    '1h':60,
    '2h':120,
    '5h':300,
    '10h':600,
    '24h':1440,
    '2d':2880,
    '5d':7200,
    '10d':14400,
    '20d':28800
    }  # adds needed features
    
    features = []
    
    for i in minute_features:
        features.append(df.close.rolling(minute_features[i]).mean()[-1:].item())  # rolling mean
        
    # print(features)
        
    for i in range(len(features)):
        features[i] = features[i]/df.close[-1:].item()
        
    return(features)