import sys
import os
import time
import numpy    as np
import scipy.io as matlabIo
from tkFileDialog   import askopenfilename

import re                       # sort lists numerically 

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def load_mnist_images(filename):
        archivo   = matlabIo.loadmat(filename, mdict=None)
        image    = archivo['Data']
        data    = np.uint8(image)
        data = data.reshape(-1,1,31,31)
                
        return data / np.float32(256)
    

    def load_mnist_labels(filename):
        archivo = matlabIo.loadmat(filename, mdict=None)
        t = archivo['Target']
        data = np.uint8(t)
        
        data = data.reshape(-1)
        return data
    
    root = os.path.abspath('dataset/')
    
    data_path    = root + '/Data.mat'
    targets_path = root + '/Target.mat'    
       
    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images(    data_path)
    y_train = load_mnist_labels( targets_path)
    X_test  = load_mnist_images(    data_path)
    y_test  = load_mnist_labels( targets_path)

       
    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10], X_train[-10:]
    y_train, y_val = y_train[:-10], y_train[-10:]
    
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def obtenerSetDeimagenes():    
    imagenesFolderPath = openFile()
    # Leer lista de imagenes    
    imagenesDirectory = os.path.dirname(imagenesFolderPath)    
    imagenesLista = [f for f in os.listdir(imagenesDirectory) if os.path.isfile(os.path.join(imagenesDirectory, f))]
    imagenesLista = sorted(imagenesLista, key=natural_keys)  
         
    print "Se cargaron:" + str(len(imagenesLista)) + "imagenes"    
    return  imagenesLista, imagenesDirectory 



# Auxiliar functions
def openFile():
    name = askopenfilename() 
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    ''' alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

 

