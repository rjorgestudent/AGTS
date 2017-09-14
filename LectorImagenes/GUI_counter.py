from      Tkinter   import *
from         load   import *
from tkFileDialog   import askopenfilename
from        mnist   import main  
from        train   import train

import theano.tensor          as T
import matplotlib.pyplot      as plt
import numpy                  as np
import scipy.ndimage.filters  as filters

import skimage.data
import re                       # sort lists numerically 
import os
import lasagne
import theano
import math
import sys


# PRE-ALLOCATIONS

# Interruptions
class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        
        # Buttons 
        self.salir = Button(frame,text="QUIT",fg="red",command=self.closeProgram)
        self.salir.pack(side=LEFT)
        
        self.cargar = Button(frame,text="Cargar Modelo",command=self.cargarModelo)
        self.cargar.pack(side=LEFT)        
        
        self.selecImagen = Button(frame,text="Seleccionar Folder con Imagenes",command=self.cargarImagenes)
        self.selecImagen.pack(side=LEFT) 
        
        self.contar = Button(frame,text="Analizar set de imagenes",command=self.contarFrutas)
        self.contar.pack(side=LEFT)
        
        
    def cargarModelo(self):
        global network, boxSize, inputVar, targetVar
        
        fileTxt, fileParams = openFile() 
        network, boxSize, inputVar, targetVar = initializeCNN(fileTxt)        
        network = loadNetwork(network, fileParams) 
        
    def cargarImagenes(self):
        global imagenesLista, imagenesDirectory
        
        imagenesFolderPath = askopenfilename()         
        imagenesLista, imagenesDirectory = obtenerSetDeimagenes(imagenesFolderPath)
    
    def contarFrutas(self):
               
        totalFrutas = conteoCompleto()
        
        
        
    def closeProgram(self):
        sys.exit()

# Menu
def newFile():
    print "New File!"
def openFile():
    
    filePathAndName = askopenfilename() 
    # Extracting the path and name
    pathFiles, fileParams = os.path.split(filePathAndName)    
    fileName, _ = os.path.splitext(fileParams)
    
    # Create Files name strings
    fileTxt    = pathFiles+"/"+fileName+".txt"
    fileParams = pathFiles+"/"+fileName+".params"
    
    return fileTxt, fileParams

def About():
    print "This is a simple example of a menu"

    
# Functions
def initializeCNN(fileTxt):
    depth,  width, dropIn, dropHid, boxSize = leertxt(fileTxt)
    inputVar  = T.tensor4('inputs')
    targetVar = T.ivector('targets')   
    
    network = build_custom_cnn(inputVar,depth,width,dropIn,dropHid,boxSize)
    #cls()
    
    return network, boxSize, inputVar, targetVar


def obtenerSetDeimagenes(imagenesFolderPath):    
    # Leer lista de imagenes    
    imagenesDirectory = os.path.dirname(imagenesFolderPath)    
    imagenesLista = [f for f in os.listdir(imagenesDirectory) if os.path.isfile(os.path.join(imagenesDirectory, f))]
    imagenesLista = sorted(imagenesLista, key=natural_keys)  
         
    print str(len(imagenesLista)) + " imagenes a analizar"    
    return  imagenesLista, imagenesDirectory
    
def conteoCompleto():  
   
    # Initial call to print 0% progress
    l = len(imagenesLista)
    frutasPorImagen = []
    totalFrutas = 0
       
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
      
    #for im in xrange(0,l):        
    for im in xrange(0,2):            
        # Cargar imagen    
        imPath = imagenesDirectory + "/" + imagenesLista[im]
        #img    = skimage.data.load(imPath)
        
        #img    = img.mean(axis=2)
        img    = skimage.color.rgb2gray(skimage.io.imread(imPath))
        
        # Construir funcion
        valPrediction = lasagne.layers.get_output(network, deterministic=True)
        evalFn        = theano.function([inputVar], valPrediction)
        output        = np.zeros((img.shape[0], img.shape[1], 2))
                
        # Analysis de imagen
            ### Sliding window ###
        margin = int(math.floor(boxSize / 2))
        for x in xrange(margin, int(img.shape[0] - margin)):
            for y in xrange(margin, int(img.shape[1]-margin)):
                patch = img[x-margin:x+margin+1, y-margin:y+margin+1]
                patch = patch.reshape(1,1, boxSize, boxSize)
                output[x,y,:] = evalFn(patch)
                
            ### Detect local maxima ###
                # Smooth the response map
        heatmap = filters.gaussian_filter(output[:,:,1], 3) 
                # Thresholding
        seg = heatmap>0.75
                # Non-maximum suppression
        detections = np.where(np.multiply(seg, heatmap == filters.maximum_filter(heatmap, 3)))

        # Mostrar imagen      
        fig, ax = plt.subplots()
        ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        plt.plot(detections[1], detections[0], 'b.', markersize=12, mew=3)
    
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        plt.rcParams['figure.figsize'] = 20,20
    
        
        frutasPorImagen.append(len(detections[0]))        
        totalFrutas = totalFrutas + len(detections[0]) 
        
        printProgressBar(im+1 , l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    print  "\nSe detectaron: " + str(totalFrutas) + " frutas"
    
    return totalFrutas

    
# Auxiliar functions
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    ''' alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# Clear console
def cls():  
    os.system('cls' if os.name=='nt' else 'clear')
                                 
# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
                                 
    
# INITIALIZE GUI
root = Tk()

# MAIN
# Window name
root.title("AGT counter")

# Menu
menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(    label="File",menu=filemenu)
filemenu.add_command(label="New",     command=newFile)
filemenu.add_command(label="Open...", command=openFile)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

# Buttons
app = App(root)

# FINISH
root.mainloop()

