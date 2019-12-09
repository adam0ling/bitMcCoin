from random import randint
from public_testing.coinbase_connect import innitPriceHistory, updateValues
from public_testing.TraderMan import TraderMan
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import time

# based on
# https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a
# https://github.com/maurock/snake-ga

# data load and setting
df, valuesUpdated = updateValues(innitPriceHistory())  # values updated is a binary variable whihc cheks if the values have been updated
modelName = './public_testing/weights_trained/try7_8-hours.hdf5'
# reward_period = (8*60)  # at which point should the nextClosePrice be taken to measure the succesess of an action, takes average of the named amount, i.e. 1440 average price for the whole day after the trade.
reward_period = 3

timeToTest = 60*3  # for how long should the test run (1440=day)

class Game:
    
    def __init__(self):
        self.score = 0
        self.crash = False
        self.steps = 0
        self.trader = Trader(self)
        
class Trader:
    
    def __init__(self, game):
        self.steps = 0
        self.trade_pos = 1
        self.last_score = 0
        
    def do_move(self, action, game, agent, closePrice):
        '''
        Checks if move is valid and if so sends it to TraderMan agent.
        '''
        print('wallet')
        print(agent.BTC, agent.EUR)
        self.trade_pos = agent.last_action()  # check witch wallet is active  
        agent.old_position = agent.get_reward(raw=True)  # updates the old position for reward calc (see set_reward())
        emptyAction = [0, 1, 0]  # do nothing action
        self.empty_move = 0 # checks if the action was empty
        
        if np.array_equal(action, [1, 0, 0]) and self.trade_pos == 1:
            game.crash = False
            # print('BUY crash')
            agent.step(closePrice, emptyAction)  # sends an empty step
            self.empty_move = 1
        elif np.array_equal(action, [0, 0, 1]) and self.trade_pos == 0:
            game.crash = False
            # print('SELL crash')
            agent.step(closePrice, emptyAction)  # send an empty step
            self.empty_move = 1
        else:
            agent.step(closePrice, action)
            self.empty_move = 0
             
        print(agent.get_reward())           
        # game.score = sum()
            
def get_record(score, record):
        if score >= record:
            return score
        else:
            return record
        
def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()
    
def initialize_game(trader, game, agent, df, closePrice, nextClosePrice):
    game.crash = False
    state_init1 = agent.get_state(df)
    agent.nextClosePrice = nextClosePrice
    do_rand_move = randint(0, 2)
    if do_rand_move == 1:
      action = [0, 0, 1]
      print('Started with EUR')
    else:
      action = [0, 1, 0]
      print('Started with BTC')
    trader.do_move(action, game, agent, closePrice)
    state_init2 = agent.get_state(df)
    reward1 = agent.set_reward(trader, game.crash, action)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory)
            
            
def run(df):
    agent = TraderMan(modelName=modelName)
    record = 0
    line = 0
    closePrice = 1
    nextClosePrice = 1
    # Initialize classes
    random_steps = 0  # random steps taken
    game = Game()
    trader1 = game.trader
    fullTimeCounter = 0
    # set game counter
    timeCounter = reward_period+1
   
    closePrice = df.close[-1:].sum()
    nextClosePrice = df.close[-1:].sum()
    initialize_game(trader1, game, agent, df, closePrice, nextClosePrice)
    
    plotDf = pd.DataFrame(index = list(range(1,timeToTest+1)), columns=['random', 'BUY', 'SELL', 'Price'], data=0)
    
    print('--------------------------------------')
    
    # selects new price state
    while fullTimeCounter <= timeToTest:
        df, valuesUpdated = updateValues(df)  # update data
        if valuesUpdated != 1:
            time.sleep(10)
        else:
            fullTimeCounter += 1
            timeCounter += 1
            
            plotDf.BUY[fullTimeCounter] = 0
            plotDf.SELL[fullTimeCounter] = 0 
            
            if timeCounter == reward_period:
                nextClosePrice = df.close[-reward_period:].mean()
                state_new = agent.get_state(df)    
                reward = agent.set_reward(trader1, final_move, nextClosePrice)  # set reward for the new state            
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)  # train short memory base on the new action and state
                agent.remember(state_old, final_move, reward, state_new, game.crash) # store the new data into a long term memory
                
            elif timeCounter > reward_period:
                state_old = agent.get_state(df)  #get old state
                prediction = agent.model.predict(state_old.reshape((1,16)))  # predict action based on the old state
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)  
                plotDf.Price[fullTimeCounter] = df.close[-1:].sum()
                closePrice = df.close[-1:].sum()  # perform new move and get new state
                trader1.do_move(final_move, game, agent, closePrice)   
                print('current state: ', state_old)
                print('final move: ', final_move)       
                if np.array_equal(final_move, [1, 0, 0]) and trader1.empty_move == 0:
                        plotDf.BUY[fullTimeCounter] = 1
                        timeCounter = 0
                elif np.array_equal(final_move, [0, 0, 1]) and trader1.empty_move == 0:
                        plotDf.SELL[fullTimeCounter] = -1
                        timeCounter = 0      
    print(
        '       EUR: %.2f' %sum(agent.get_reward(in_coins=False)), '\n',             # portfolio value at the end in EUR
        '      BTC: %.2f' %sum(agent.get_reward()), '\n',                            # portfolio value at the end in BTC
        '      Steps done: ',  agent.steps_done,                                     # total number of steps done
        '%.2f' %(agent.steps_done/fullTimeCounter), '\n',                             # proportion of steps taken compared to total amount of intervals
        '      BUYs done:  ', agent.BUY, '\n',                                       # total number of buy steps
        '      SELLs done: ', agent.SELL, '\n',                                      # total number of sell steps
        '      Total done:  %.2f' %((agent.SELL+agent.BUY)/(agent.steps_done+1)), '\n',  # proportion of sum(BUY, SELL) steps compared to total amount of steps taken
        '      Random done: ', random_steps, '\n'                                    # random steps taken (see epsilon)
        )
    
    # plot dataframe
    ax1 = plotDf[['BUY', 'SELL']].plot(legend=False)  # BUY is 1, SELL -1 on the plot
    ax1.set_ylim([-1,1])
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    plotDf['Price'].plot(ax=ax2, color='red', legend=False)
    plt.show()
    print('--------------------------------------', '\n')
    agent.replay_new(agent.memory)
    # agent.model.save('bitMcCoin_1-hour.hdf5')
    plotDf.to_csv('plotDf.csv')
    # print('Mean score: ',np.average(score_plot))
    # plot_seaborn(counter_plot, score_plot)

if __name__ == '__main__':
    run(df)