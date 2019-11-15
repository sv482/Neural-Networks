# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:15:35 2019

@author: sushv
"""

#Main function 
from RNNI import main as rnn_main
#from ffnn import main as ffnn_main
from ffnn_fixed import main as ffnn_main

#FLAG = 'RNN'
FLAG='FFNN'
    
def main():
    if FLAG == 'RNN':
        hidden_dim = 32
        number_of_epochs = 10
        n_layers=1
        rnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
        
    elif FLAG == 'FFNN':
        hidden_dim = 32
        number_of_epochs = 10
        n_layers=1
        ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)


if __name__ == '__main__':
    main()






