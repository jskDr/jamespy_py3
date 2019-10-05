# jrl.py based on python 3
# rl library, Sungjin Kim, (copyright) Oct 5, 2019

import numpy as np

def calc_discount_reward(reward_buff, gamma=0.99, normalize=True, done_flag=True):
    prev_dr = 0
    for ii in reversed(range(len(reward_buff))):
        if done_flag and reward_buff[ii] == 0:
            prev_dr = 0            
        else:
            reward_buff[ii] += prev_dr * gamma
            prev_dr = reward_buff[ii]
    
    if normalize:
        mean, std = np.mean(reward_buff), np.std(reward_buff)
        for ii in range(len(reward_buff)):
            reward_buff[ii] = (reward_buff[ii] - mean) / std