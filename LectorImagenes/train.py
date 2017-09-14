import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

import scipy.io as matlabIo

from batch   import iterate_minibatches

  
def train(network, num_epochs, lrn_rate, input_var, target_var, X_train, y_train, X_val, y_val, X_test, y_test):
   
    # TRAINING
    prediction = lasagne.layers.get_output(network)
    loss       = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss       = loss.mean()
    params     = lasagne.layers.get_all_params(network, trainable=True)    
    updates    = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lrn_rate, momentum=0.9)
    #train_fn   = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    train_fn   = theano.function([input_var, target_var], loss, updates=updates)
    
    
    # EVALUATION
    eval_prediction  = lasagne.layers.get_output(network, deterministic=True)
    eval_loss        = lasagne.objectives.categorical_crossentropy(eval_prediction,target_var)
    eval_loss        = eval_loss.mean()
    eval_acc         = T.mean(T.eq(T.argmax(eval_prediction, axis=1), target_var), dtype=theano.config.floatX)
    #eval_fn          = theano.function([input_var, target_var], [eval_loss, eval_acc], allow_input_downcast=True)
    eval_fn          = theano.function([input_var, target_var], [eval_loss, eval_acc])
      
    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err     = 0
        train_batches = 0
        start_time    = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            
            train_err += train_fn(inputs, targets)
            train_batches += 1
           

        # And a full pass over the evaluation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=True):
            inputs, targets = batch
            err, acc     = eval_fn(inputs, targets)
            val_err     += err
            val_acc     += acc
            val_batches += 1
            
           
            

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = eval_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("\nFinal results:")
    
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    
    return network