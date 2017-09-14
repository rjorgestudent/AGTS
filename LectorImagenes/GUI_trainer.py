from      Tkinter   import *
from         load   import *
from tkFileDialog   import askopenfilename
from        mnist   import main  
from         save   import *

from        scipy   import misc 
from      network   import build_custom_cnn
from        train   import train



import theano.tensor          as T
import matplotlib.pyplot      as plt
import numpy                  as np
import scipy.ndimage.filters  as filters
import scipy.io               as matlabIo

import skimage.data
import re                       # sort lists numerically 
import os
import lasagne
import theano
import math
import sys
import time
import tkFileDialog
import tkMessageBox




# PRE-ALLOCATIONS
global depth, width, dropIn,dropHid, boxSize, iterations
filesExistent = False
depth      = 2  
width      = 30
dropIn     = 0.1
dropHid    = 0.1
boxSize    = 31
iterations = 1000
 
# Interruptions
class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        
        # Buttons 
        self.salir = Button(frame,text="QUIT",fg="red",command=self.closeProgram)
        self.salir.pack(side=LEFT) 

    
    def closeProgram(self):
        sys.exit()

    # Menu
def nuevoPatron():
    print "Nuevo"
def newFile():
    global depthSlide, widthSlide, dropInSlide, dropHidSlide, boxSizeIndicator, iterationsEntry, inputTrainDataButton, targetTrainDataButton, trainButton
    root.geometry("600x600")
    
   
    while not 'fileParams' in globals() and not 'fileTxt' in globals() :  
        saveAs()
        
        
    inputTrainDataButton   = Button(root, text="Load Input",           command =   loadInputs).pack()
    targetTrainDataButton  = Button(root, text="Load Target",          command =  loadOutputs).pack()

    # Text 
    boxSizeIndicator = Text(root, height=1, width=15, bg="gray", wrap=WORD )   
    boxSizeIndicator.pack()
    boxSizeText = "Box Size: " 
    boxSizeIndicator.insert(END, boxSizeText)
    boxSizeIndicator.config(state=DISABLED)
    
    # Sliders
    depthSlide = Scale(root, label= "Depth",  from_=1, to=depth+20, length=400,tickinterval=1, orient=HORIZONTAL)
    depthSlide.set(depth)
    depthSlide.pack()

    widthSlide = Scale(root,label= "Width", from_=10, to=width+20, length=400,tickinterval=9, orient=HORIZONTAL)
    widthSlide.set(width)
    widthSlide.pack()

    dropInSlide = Scale(root,label= "Drop In", from_=0.0, to=1.0, length=400, tickinterval=0.1,resolution=0.1, orient=HORIZONTAL)
    dropInSlide.set(dropIn)
    dropInSlide.pack()

    dropHidSlide = Scale(root,label= "Drop hidden", from_=0.0, to=1.0, length=400, tickinterval=0.1,resolution=0.1, orient=HORIZONTAL)
    dropHidSlide.set(dropHid)
    dropHidSlide.pack()
    
    # Manual entries
    labelIterationsEntry = Label(root, text="Iterations")
    labelIterationsEntry.pack()
    iterationsEntry = Entry(root)  
    iterationsEntry.insert(END, str(iterations))
    iterationsEntry.pack()
    # Buttons
    buildNetworkButton     = Button(root, text='Build Network',        command = buildNetwork)
    buildNetworkButton.pack()
    trainButton            = Button(root, text='Train', state= DISABLED,command = trainNetwork)
    trainButton.pack()
       
def openFile():
    global fileParams, fileTxt, depth, width, dropIn,dropHid, boxSize, iterations
    filePathAndName = askopenfilename() 
    # Extracting the path and name
    pathFiles, fileParams = os.path.split(filePathAndName)    
    fileName, _ = os.path.splitext(fileParams)
    
    # Create Files name strings
    fileTxt    = pathFiles+"/"+fileName+".txt"
    fileParams = pathFiles+"/"+fileName+".params"
    
    depth, width, dropIn, dropHid, boxSize = readInfofromTxt(fileTxt)
    
    print depth
    newFile()
    print "File succesfully opened" 

def About():
    print "This is a simple example of a menu"
    
# Functions
def save():
    # DUMP THE INFOR INTO THE FILES
    # Save trained network
    saveNetworkParameters(network, fileParams)
    # Save file with network characteristics
    dumpInfoIntoTxt(fileTxt, parameters)

def saveAs():
    global fileParams, fileTxt, filesExistent
    # FIRST CREATES THE FILES
    filePathAndName = tkFileDialog.asksaveasfile(mode='w', defaultextension=".params")
    
    if filePathAndName:
        
        # Extracting the path and name
        pathFiles, fileParams = os.path.split(filePathAndName.name)    
        fileName, _ = os.path.splitext(fileParams)

        # Create Files name strings
        fileTxt    = pathFiles+"/"+fileName+".txt"
        fileParams = pathFiles+"/"+fileName+".params"

        # Create aditional txt file     
        createTxt(fileTxt)

        # if the network is already trained, save the info
        if 'network' in globals() and 'parameters' in globals() :        
            save()
        else:
            print "Files succesfully created"
        filesExistent = True
        

def buildNetwork():
    global network, parameters, inputVar, targetVar
    
    depth      =   int(      depthSlide.get())
    width      =   int(      widthSlide.get())
    dropIn     = float(     dropInSlide.get())
    dropHid    = float(    dropHidSlide.get())
       
        
    parameters = str(depth) +','+ str(width) +','+ str(dropIn) +','+ str(dropHid) +','+ str(boxSize)
    inputVar =  T.tensor4('inputs')
    targetVar = T.ivector('targets')
    
    network = build_custom_cnn(inputVar,depth,width,dropIn,dropHid,boxSize)
    
    print "Model built!"
    filemenu.entryconfig(3, label="Save", command= saveAs,state=NORMAL)
    
def trainNetwork():
    global network, iterations
    iterations =   int( iterationsEntry.get())
    # Split data into Training, Validation and Test datasets
    inputDataTrain   =   inputData[                       0: int(len(inputData)* 0.7)]
    inputDataVal     =   inputData[int(len(inputData)* 0.7): int(len(inputData)* 0.9)]
    inputDataTest    =   inputData[int(len(inputData)* 0.9):     len(inputData)]
    
    outputsDataTrain = outputsData[                       0: int(len(inputData)* 0.7)]
    outputsDataVal   = outputsData[int(len(inputData)* 0.7): int(len(inputData)* 0.9)]
    outputsDataTest  = outputsData[int(len(inputData)* 0.9):     len(inputData)]
    
    lrn_rate=0.00004
    network = train(network, iterations, lrn_rate, inputVar, targetVar,  inputDataTrain, outputsDataTrain, inputDataVal, outputsDataVal, inputDataTest, outputsDataTest)
    
    print ("  TrainData:\t\t\t" + str(len(inputDataTrain)) + "\n  ValidationData:\t\t" + str(len(inputDataVal)) +"\n  TestData:\t\t\t" + str(len(inputDataTest)))
    
    filemenu.entryconfig(4, label="Save as..", command= saveAs,state=NORMAL)
    
def loadInputs():
    global inputData, boxSize
    inputsPath = askopenfilename() 
    
    archivo = matlabIo.loadmat(inputsPath, mdict=None)
    image   = archivo['allBlobs']
    #image   = archivo['Data']
    data    = np.uint8(image)
    boxSize = len(data[0][0])
    data    = data.reshape(-1,1,boxSize,boxSize)
                
    inputData = data / np.float32(256)
    
    
    
    boxSizeIndicator.config(state=NORMAL)
    boxSizeIndicator.delete(1.0, END)   
    boxSizeIndicator.insert(END, "Box Size: " + str(boxSize))
    boxSizeIndicator.config(state=DISABLED)
    
    print "Input data loaded!"

def loadOutputs():
    global outputsData, trainButton
    outputsPath = askopenfilename() 
    
    archivo = matlabIo.loadmat(outputsPath, mdict=None)
    t       = archivo['allTarge']
    #t       = archivo['Target']
    data    = np.uint8(t)        
    data    = data.reshape(-1)
    
    outputsData = data
    trainButton.config(state=NORMAL)
    print "Target data loaded!" + str(outputsData.shape)
    
    
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
root.geometry("600x600")
# MAIN
# Window name
root.title("Fruit trainer")


# Menu
menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(    label="File",menu=filemenu)
filemenu.add_command(label="New",     command=newFile)
filemenu.add_command(label="Open", command=openFile)
filemenu.add_command(label="Save", command= save, state=DISABLED)
filemenu.add_command(label="Save as..", command= saveAs,state=DISABLED)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

# Buttons
app = App(root)




# FINISH
root.mainloop()



