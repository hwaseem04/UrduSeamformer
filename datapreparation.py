'''
Given the dataset 
in the desired structure
mentioned in README .

Args: 
  > Input  : Configurations for image and dataset JSON file . 
            > Ground Truth Binary Folder.
            > Image Folder 
  > Output : Set of final folders where everything is structured as per the input to code.

'''


# Library Imports
import sys 
import os 
import cv2 
import json
import argparse 
import itertools
import numpy as np 
from empatches import EMPatches
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
from utils import generateScribble

#File Import 
sys.path.append('..') 

# Global Parameters 
THICKNESS = 5
PATCHSIZE = 256 
OVERLAP = 0.25 

# Argument Parser 
def argumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputjsonPath',type=str,default=None)
    parser.add_argument('--datafolder',type=str,default=None)
    parser.add_argument('--patchsize',type=int,default=PATCHSIZE)
    parser.add_argument('--overlap',type=float,default=OVERLAP)
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--outputfolderPath',type=str,default=None,required=True)
    args = parser.parse_args()
    return args 
      
# Helper Functions

# Make the respective folders 
def createFolders(args):
    os.makedirs(args.outputfolderPath, exist_ok=True)
    smPath = os.path.join(args.outputfolderPath,'scribbleMap/')
    imPath = os.path.join(args.outputfolderPath,'images/')
    # Prepare a key point image folder 
    try : 
        os.makedirs(smPath,exist_ok=True)
        os.makedirs(imPath,exist_ok=True)
    except FileExistsError: 
        print('Error in Folder Creation !')
    print('~Folder Creation Completed !')


# Drawing scribble on canvas
def drawScribble(canvas,scribble, thickness=THICKNESS):
    canvas=cv2.polylines(canvas,np.int32([scribble]),False,(255,255,255),thickness)
    return canvas


# Scribble Map Generation 
def get_channel_scibbles(img,scribbleList,thickness=THICKNESS):
    # blank canvas 
    h,w, _=img.shape 
    canvas_0 = np.zeros((h,w))
    for i in range(0,len(scribbleList)):
        scribble = scribbleList[i]
        canvas_0 = drawScribble(canvas_0, scribble)
    return canvas_0

def datasetPrepare(args):
    # create folders
    createFolders(args)
    # Read the json file from the arguments
    try:
        with open(args.inputjsonPath,'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('JSON File does not exist.')
        sys.exit() 
    emp = EMPatches()
    # For every datapoints 
    errors=0
    count=0

    for i,datapoint in enumerate(data):
        path = datapoint['imgPath'].replace('./',args.datafolder)
        print('Processing .. {}'.format(path))
        img = cv2.imread(path)
        imgName=os.path.basename(path)

        try:
            if  'scribbles' not in datapoint:
                H = img.shape[0]
                W = img.shape[1]
                # if the ground truth polygons are in rectangular shape set isBox=True, else isBox=False
                scribbles = [generateScribble(H, W, polygon, isBox=False) for polygon in datapoint['gdPolygons'] ]
            else:   
                scribbles = [scr for scr in  datapoint['scribbles']]
            sMap = get_channel_scibbles(img,scribbles,thickness=THICKNESS)
            if sMap is None or img is None: 
                print('Nothing to process..')
                continue

            # Get all patches 
            spatches,indices = emp.extract_patches(sMap,patchsize=args.patchsize,overlap=args.overlap)
            ipatches,indices = emp.extract_patches(img,patchsize=args.patchsize,overlap=args.overlap)

            N = len(spatches)
            for i in range(0,N,1):
                count = count + 1
                # Resizing of the patches to 256 x 256 pixels
                ipatch=  cv2.resize(ipatches[i], (args.patchsize,args.patchsize), interpolation = cv2.INTER_AREA)
                spatch=  cv2.resize(spatches[i], (args.patchsize,args.patchsize), interpolation = cv2.INTER_AREA)

                # List of indices to name the patch
                lindices = list(indices[i])
                imageName_i = imgName.split('.')[0]+'_{}_{}_{}_{}'.format(str(lindices[0]),str(lindices[1]),str(lindices[2]),str(lindices[3]))
                try:
                    # Save the image patch to respective folders
                    cv2.imwrite(os.path.join(args.outputfolderPath,'scribbleMap/sm_{}.jpg'.format(imageName_i)),spatch)
                    cv2.imwrite(os.path.join(args.outputfolderPath,'images/im_{}.jpg'.format(imageName_i)),ipatch)
                except Exception as exp:
                    print('Error : Saving the patch {}'.format(exp))
                    errors+=1
                    continue

        except Exception as exp:
            print('Error:{}-{}'.format(imgName,exp))
            errors+=1
            continue

# Main 
if __name__ == "__main__":
    args = argumentParser()
    print('Invoking dataset preparation function...')
    datasetPrepare(args)
    print('~Competed!')

