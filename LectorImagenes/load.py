import cPickle as pickle
import os


import lasagne
import theano
import theano.tensor as T
import       numpy   as np

from network import build_cnn, build_custom_cnn

def cargar(nombre):
    Depth,  Width, Drop_in, Drop_hid, Box_size = leertxt(nombre)
    
    input_var, target_var = prepTheanoVar()        
    network   = build_custom_cnn(input_var, Depth, Width, Drop_in, Drop_hid, Box_size)  
    
    
    network   = loadNetwork(network, nombre)
    
    return network
    
    
def prepTheanoVar():
    input_var  = T.tensor4('inputs')
    target_var = T.ivector('targets')
       
    return input_var, target_var
    

def leertxt(nombre):
    #archi=open(nombre +'.txt','r')
    archi=open(nombre,'r')
    
    Depth    =   int(archi.readline().rstrip('\n'))
    Width    =   int(archi.readline().rstrip('\n'))
    Drop_in  = float(archi.readline().rstrip('\n'))
    Drop_hid = float(archi.readline().rstrip('\n'))
    Box_size =   int(archi.readline().rstrip('\n'))
    
    archi.close()   
    return Depth,  Width, Drop_in, Drop_hid, Box_size



def loadNetwork(model, filename):
   
    """Unpickles and loads parameters into a Lasagne model."""
    
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)
    print "Parametros cargados en la red"
    return model