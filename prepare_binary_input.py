'''
Full Scale Inference : Stage I & II 
Input : Image Folder or Corresponding JSON File 
Output : JSON File : Image Information , Predicted Polygons & Scribbles 
'''

import json
import copy
import os 
import sys 
import csv 
import torch
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import ViT
from empatches import EMPatches
import argparse 

# Global settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# File Imports 
from seam_conditioned_scribble_generation import *
from utils import *
from network import *

# Argument Parser 
def addArgs():
    # Required to override these params
    parser = argparse.ArgumentParser(description="SeamFormer:Inference")
    parser.add_argument("--input_image_folder",type=str, help="Input Folder Path",default=None)
    parser.add_argument("--output_image_folder",type=str, help="Output Folder Path for storing bin & scr results",required=True,default=None)
    parser.add_argument("--model_weights_path",type=str,help="Seamformer Model Checkpoint Weights",default=None)
    
    # Fixed Arguments ( override in special cases only)
    parser.add_argument("--encoder_layers",type=int, help="Encoder Level Layers",default=6)
    parser.add_argument("--encoder_heads",type=int, help="Encoder Heads",default=8)
    parser.add_argument("--encoder_dims",type=int, help="Internal Encoder Dim",default=768)
    parser.add_argument("--img_size",type=int, help="Image Shape",default=256)
    parser.add_argument("--patch_size",type=int, help="Input Patch Shape",default=8)
    parser.add_argument("--split_size",type=int, help="Splitting Image Dim",default=256)
    parser.add_argument("--threshold",type=float,help="Prediction Thresholding",default=0.30)
    args_ = parser.parse_args()
    settings = vars(args_)
    return settings

'''
Takes in the default settings 
and args to create the network.
'''
# Network Configuration  
def buildModel(settings):
    print('Present here : {}'.format(settings))
    # Encoder settings
    encoder_layers = settings['encoder_layers']
    encoder_heads = settings['encoder_heads']
    encoder_dim = settings['encoder_dims']
    patch_size = settings['patch_size']
    # Encoder
    v = ViT(
        image_size = settings['img_size'],
        patch_size =  settings['patch_size'],
        num_classes = 1000,
        dim = encoder_dim,
        depth = encoder_layers,
        heads = encoder_heads,
        mlp_dim = 2048)
    
    # Full model
    network =  SeamFormer(encoder = v,
        decoder_dim = encoder_dim,      
        decoder_depth = encoder_layers,
        decoder_heads = encoder_heads,
        patch_size = patch_size)
    
    print('Model Weight Loading ...')
    # Load pre-trained network + letting the encoder network also trained in the process.
    if settings['model_weights_path'] is not None:
        if os.path.exists(settings['model_weights_path']):
            try:
                network.load_state_dict(torch.load(settings['model_weights_path'], map_location=device),strict=True)
                print('Network Weights loaded successfully!')
            except Exception as exp :
                print('Network Weights Loading Error , Exiting !: %s' % exp)
                sys.exit()
        else:
            print('Network Weights File Not Found')
            sys.exit()

    network = network.to(device)
    network.eval()
    return network


'''
Performs both binary and scribble output generation.
'''
def imageInference(network,path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True):
    emp = EMPatches()
    if not os.path.exists(path):
        print('Exiting! Invalid Image Path : {}'.format(path))
        sys.exit(0)
    else:
        weight = torch.tensor(1) #Dummy weight 
        input_patches , indices = readFullImage(path,PDIM,DIM,OVERLAP)

        patch_size=args['patch_size']
        spilt_size = args['split_size']
        image_size = (spilt_size,spilt_size)
        THRESHOLD = args['threshold']

        soutput_patches=[]
        boutput_patches=[]
        # Iterate through the resulting patches
        for i,sample in enumerate(input_patches):
            p = sample['img']
            target_shape = (sample['resized'][1],sample['resized'][0])
            with torch.no_grad():
                inputs =torch.from_numpy(p).to(device)
                # Pass through model
                loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
                pred_pixel_values_bin, _ =network(inputs,gt_bin_img=inputs,gt_scr_img=inputs,criterion=loss_criterion,strain=True,btrain=True,mode='test')

                # Send them to .cpu
                pred_pixel_values_bin = pred_pixel_values_bin.cpu()

                bpatch=reconstruct(pred_pixel_values_bin,patch_size,target_shape,image_size)

                # binarize the predicted image taking 0.5 as threshold
                bpatch = ( bpatch>THRESHOLD)*1

                # Append the net processed patch
                boutput_patches.append(255*bpatch)

        try:
            assert len(boutput_patches)==len(input_patches)
        except Exception as exp:
            print('Error in patch processing outputs : Exiting!')
            sys.exit(0)
        
        # Restich the image
        boutput = emp.merge_patches(boutput_patches,indices,mode='max')

        # Transpose
        binaryOutput=np.transpose(boutput,(1,0))
        
        return binaryOutput
    
'''
Performs Binary & Scribble 
Inference given imageFolder
'''

def Inference(args):
    # Get the model first 
    network = buildModel(args)
    print('Completed loading weight')

    # Make output directory if its not present 
    os.makedirs(args['output_image_folder'],exist_ok=True)

    if  args['input_image_folder'] is not None:
        files_ = os.listdir(args['input_image_folder'])
        jsonOutput = []
        if len(files_)>0:
            for f in files_ : 
                try:
                    print('Processing image - {}'.format(f))
                    file_path = os.path.join(args['input_image_folder'],f)
                    file_name = os.path.basename(file_path)
                    img = cv2.imread(file_path)
                    H,W,C = img.shape
                    
                    # Calling Seamformer Stage I Inference to get binarised image
                    binaryMap =  imageInference(network,file_path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True)
                    binaryMap=np.uint8(binaryMap)
                    
                    # Store the images
                    cv2.imwrite(os.path.join(args['output_image_folder'], file_name),binaryMap)
                
                except Exception as exp:
                    print('Image :{} Error :{}'.format(file_name,exp))
                    continue

        else:
            print('Empty Input Image Folder , Exiting !')
            sys.exit(0)
    
    else:
        print('Enter path for valid folder , Exiting !')
        sys.exit(0)

    print('~Binary Images Generated !')   


if __name__ == "__main__":    
    args = addArgs()
    print('Running Script...')
    Inference(args)

