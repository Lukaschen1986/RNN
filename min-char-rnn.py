# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.chdir("D:/my_project/Python_Project/test/RNN")
from scipy.stats import itemfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data = open("input.txt", "r").read()
#data = pd.read_table("input.txt")

chars = list(set(data))
data_size, chars_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, chars_size))

char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

hidden_size = 100
seq_length = 50
learning_rate = 0.1

Wxh = np.random.randn(hidden_size, chars_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(chars_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((chars_size, 1))

def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev) # init on hs
    loss = 0
    # forward
    for t in range(len(inputs)):
        # t = 0
        xs[t] = np.zeros((chars_size, 1))
        xs[t][inputs[t]] = 1 # one-hot for each input
        
        hs[t] = np.tanh(np.dot(Wxh,xs[t]) + np.dot(Whh,hs[t-1]) + bh)
        ys[t] = np.dot(Why,hs[t]) + by
        
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t],0])
    # backward
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby, dhnext = np.zeros_like(bh), np.zeros_like(by), np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        # t = 0
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        dparam = np.clip(dparam, a_min=-5, a_max=5)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def generator(h, chars_size, start_char, total):
    x = np.zeros((chars_size, 1))
    x[start_char] = 1 # one-hot for init
    
    ixes = []    
    for t in range(total):
        # t = 0
        h = np.tanh(np.dot(Wxh,x) + np.dot(Whh,h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(chars_size), p=p.ravel())
        x = np.zeros((chars_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/chars_size)*seq_length # loss at iteration 0
smooth_loss_res = []
iters = 0
start = 0

while True:
    if iters == 0 or start+seq_length+1 > len(data):
        hprev = np.zeros((hidden_size, 1)) # reset RNN memory
        start = 0 # go from start of data

    inputs = [char_to_ix[char] for char in data[start : start+seq_length]]
    targets = [char_to_ix[char] for char in data[start+1 : start+seq_length+1]]
    
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    smooth_loss_res.append(smooth_loss)
    
    if iters % 100 == 0: # 每100轮预测一次
        gen_ix = generator(hprev, chars_size, inputs[0], 1000)
        txt = "".join(ix_to_char[ix] for ix in gen_ix)
        print("---- sample -----")
        print("----\n %s \n----" % txt)
        print("iter %d, loss: %f" % (iters, smooth_loss))
    
    # adagrad update
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                  [dWxh, dWhh, dWhy, dbh, dby], 
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam**2
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    
    start += seq_length # move data pointer
    iters += 1 # iteration counter 


        
