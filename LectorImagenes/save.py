import cPickle as pickle
import os

import numpy as np
import lasagne

    
def creartxt(nombre):    
    archi=open(nombre + '.txt','w')
    archi.close()

def grabartxt(nombre, parameters):
    archi=open(nombre + '.txt','a')
    
    Depth, Width, Drop_in, Drop_hid, Box_size = parameters.split(',')
    
    archi.write( Depth    + '\n')
    archi.write( Width    + '\n')
    archi.write( Drop_in  + '\n')
    archi.write( Drop_hid + '\n')
    archi.write( Box_size + '\n')
    archi.close()
    
def guardar_net(model, filename, parameters):
    
    PARAM_EXTENSION = 'params'
    
    data = lasagne.layers.get_all_param_values(model)
    
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)
        
    creartxt(filename)
    grabartxt(filename, parameters)
    print("Guardado")
    
# Path included
def createTxt(filePathNameAndExtension):    
    archi=open(filePathNameAndExtension,'a')
    archi.close()
    
def dumpInfoIntoTxt(filePathNameAndExtension, parameters):
    archi=open(filePathNameAndExtension,'w')
    
    Depth, Width, Drop_in, Drop_hid, Box_size = parameters.split(',')
    
    archi.write( Depth    + '\n')
    archi.write( Width    + '\n')
    archi.write( Drop_in  + '\n')
    archi.write( Drop_hid + '\n')
    archi.write( Box_size + '\n')
    archi.close()
    
def saveNetworkParameters(network, filePathAndName):        
    data = lasagne.layers.get_all_param_values(network)    
    
    with open(filePathAndName, 'w') as f:
        pickle.dump(data, f)
        
    print("Guardado")
    
def readInfofromTxt(filePathNameAndExtension):
    archi=open(filePathNameAndExtension,'r')
        
    depth   =   int( archi.readline())
    width   =   int( archi.readline())
    dropIn  = float( archi.readline())
    dropHid = float( archi.readline())
    boxSize =   int( archi.readline())
    archi.close()
    
    
    return depth, width, dropIn, dropHid, boxSize