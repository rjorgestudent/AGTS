import lasagne

def build_custom_cnn(input_var=None, depth=2, width=30, drop_input=0,drop_hidden=.1, box_size=31):
    print("Building model and compiling functions...")
    # Setup network. Input layer, as usual (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, box_size, box_size),input_var=input_var)
    
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    
    # Stage 1 : Convolutional layer (filter bank )-> squashing
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=10,
            filter_size=(3, 3),
            nonlinearity = lasagne.nonlinearities.rectify)
    
    # Stage 2: Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        
    # Stage 3: hidden stanard depth-layers fully connected neural network with drop_hidden% dropout    
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
            
    # Finally, the 2-unit output layer with 40% dropout on its inputs, softmax output
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.4),
                                    num_units=2,
                                    nonlinearity=lasagne.nonlinearities.softmax)
    
    return network


def build_cnn(input_var=None, box_size=31):
    # Setup network. Input layer, as usual:
    net = lasagne.layers.InputLayer(shape=(None, 1, box_size, box_size),input_var=input_var)

    # Stage 1 : filter bank -> squashing
    net = lasagne.layers.Conv2DLayer(net,
            num_filters=10,
            filter_size=(3, 3),
            nonlinearity = lasagne.nonlinearities.rectify)
    
    # Stage 2: Max-pooling layer
    net = lasagne.layers.MaxPool2DLayer(net, pool_size=(2, 2))

    # Stage 3: stanard 2-layer fully connected neural network with 10% dropout
    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=.1),
            num_units=30,
            nonlinearity=lasagne.nonlinearities.rectify)
    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=.1),
            num_units=30,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Finally, the 2-unit output layer with 40% dropout on its inputs, softmax output
    net = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(net, p=.4),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return net

