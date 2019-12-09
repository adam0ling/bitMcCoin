'''
Trading training module for BTC bot. Deep Q learning version
'''

# check line 31 to load/start new modelling

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from public_testing.coinbase_connect import featureCalc
import keras
import random
import numpy as np
import pandas as pd
from operator import add


class TraderMan(object):
    init_BTC_wallet = 1  # BTC wallet in BTC
    init_EUR_wallet = 0  # EUR wallet in BTC
    
    init_trade_tax = 0.005  # tax ratio =0.5percent
    
    def __init__(self, BTC=init_BTC_wallet, EUR=init_EUR_wallet, trade_tax=init_trade_tax, modelName=None):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network(modelName)
        self.epsilon = 0
        self.actual = []
        self.memory = []
        
        self.BTC = BTC
        self.EUR = EUR
        self.trade_tax = trade_tax
        
        self.close_price = 1  # sets a variable to remember last closing price
        self.nextClosePrice = 1  # sets a variable for the upcoming price to evaluate current decision, at try5 made to be an average of a time period
        
        self.steps_done = 0 # calcs different steps taken
        self.BUY = 0
        self.SELL = 0 

        self.old_position = [1, 0]
        
    def last_action(self):
        '''
        Returns last action as sell or buy in binary (0,1).
        '''
        if self.EUR==0:
            return 1  # BUY
        else:
            return 0  # SELL
        
    def get_state(self, df):
        '''
        Gets input values
        '''
        state = featureCalc(df)
        trade_pos = self.last_action()  # current state of the wallet
        state.append(trade_pos)

        return np.asarray(state)
    
    def get_reward(self, in_coins=True, next_interval=False, raw=False, old_position=False):
        '''
        Returns wallet information. BTC and EUR
        
        in_coins - return wallet amount in BTC or EUR. Default BTC.
        next_interval - return wallet for next close price
        raw - return wallet in BTC, EUR or just one. see in_coins
        old_position - should the price be retuned on position before the trade or after
        '''
        # determines the price and wallet needed
        if next_interval==False:
          priceUsed = self.close_price
        else:
          priceUsed = self.nextClosePrice
          
        if old_position == False:  # checks for what to calc the prices, old position or new
          used_BTC = self.BTC
          used_EUR = self.EUR
        else:
          used_BTC = self.old_position[0]
          used_EUR = self.old_position[1]
        
        # returns wallet info
        if raw == True:
          return used_BTC, used_EUR
        else:
          if in_coins == True:
              return used_BTC, (used_EUR*(1/priceUsed))
          else:
              return (used_BTC*priceUsed), used_EUR

    def set_reward(self, trader, action, nextClosePrice):
        ''' 
        Returns positive rewards for increasement of total portfolio, 
        evaluated based on the next interval price.
        '''
        self.reward = 0
        self.nextClosePrice = nextClosePrice
        
        # sets the score even if the action can't be processed due to lack of EUR/BTC in wallets
        if np.array_equal(action, [1, 0, 0]):  # BUY
          old_score = self.close_price
          new_score = self.nextClosePrice * (1-self.trade_tax)
        elif np.array_equal(action, [0, 0, 1]):  # SELL
          new_score = self.close_price * (1-self.trade_tax)
          old_score = self.nextClosePrice
        else: 
          new_score = 1
          old_score = 1
        
        delta_score = (new_score - old_score)/old_score*100
    
        if delta_score != 0:
            self.reward = delta_score

        return self.reward
        
    def step(self, close_price, action):
        '''
        close - closing price
        action - tanh output from nnet
        '''
        self.steps_done += 1
        
        a = action
        
        # readability
        EUR = self.EUR
        BTC = self.BTC
        trade_tax = self.trade_tax
        
        # update old price
        self.close_price = close_price
        
        if np.array_equal(a, [1, 0, 0]):  # BUY
            new_wallet = EUR*(1/close_price)
            new_wallet -= trade_tax*new_wallet
            self.BTC = new_wallet
            self.EUR = 0
            self.BUY += 1
        elif np.array_equal(a, [0, 0, 1]):  # SELL
            new_wallet = BTC*close_price
            new_wallet -= trade_tax*new_wallet
            self.EUR = new_wallet
            self.BTC = 0
            self.SELL += 1
        
    def network(self, saved_model=None):
        '''
        Builds the network.
        '''
        if saved_model:
            model = keras.models.load_model(saved_model)
        else:
          model = Sequential()
          model.add(Dense(output_dim=120, activation='relu', input_dim=16))
          model.add(Dropout(0.15))
          model.add(Dense(output_dim=120, activation='relu'))
          model.add(Dropout(0.15))
          model.add(Dense(output_dim=120, activation='relu'))
          model.add(Dropout(0.15))
          model.add(Dense(output_dim=3, activation='softmax'))
          opt = Adam(self.learning_rate)
          model.compile(loss='mse', optimizer=opt)
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay_new(self, memory):
        self.BTC = 1
        self.EUR = 0
        
        self.close_price = 1  # sets a variable to remember last closing price
        self.steps_done = 0
        self.BUY = 0
        self.SELL = 0
        
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 16)))[0])
        target_f = self.model.predict(state.reshape((1, 16)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 16)), target_f, epochs=1, verbose=0)
    