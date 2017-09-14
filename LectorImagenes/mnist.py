from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import scipy.io as matlabIo

from data    import load_dataset
from batch   import iterate_minibatches
from network import build_cnn, build_custom_cnn
from train   import train


# ##################  prepare the MNIST dataset ##################
# See data.py
# ##################### Build the neural network model #######################
# See network.py
# ############################# Batch iterator ###############################
# see batch.py
# ############################## Main program ################################


def main(model='cnn', input_var = T.tensor4('inputs'), target_var = T.ivector('targets'),  num_epochs=10, lrn_rate=0.00004):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    #input_var  = T.tensor4('inputs')
    #target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'cnn':
        network = build_cnn(input_var)
    elif model.startswith('custom_cnn:'):
        depth, width, drop_in, drop_hid, box_size = model.split(':', 1)[1].split(',')
        print(box_size)
        network = build_custom_cnn(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid), int(box_size))
    else:
        print("Unrecognized model type %r." % model)
        return
    
    network = train(network, num_epochs, lrn_rate, input_var, target_var,  X_train, y_train, X_val, y_val, X_test, y_test)
    
    
    
    return  network
    




   
    
    
    
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
