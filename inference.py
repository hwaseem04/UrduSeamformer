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
import utils_urdu as pu
from network import *

# Argument Parser 
def addArgs():
    # Required to override these params
    parser = argparse.ArgumentParser(description="SeamFormer:Inference")
    parser.add_argument("--exp_name",type=str, help="Unique Experiment Name",required=True,default=None)
    parser.add_argument("--input_image_folder",type=str, help="Input Folder Path",default=None)
    parser.add_argument("--input_image_json",type=str, help="Input JSON Path",required=False,default=None)
    parser.add_argument("--output_image_folder",type=str, help="Output Folder Path for storing bin & scr results",required=True,default=None)
    parser.add_argument("--model_weights_path",type=str,help="Model Checkpoint Weights",default=None)
    parser.add_argument("--input_json", action="store_true", help="Inference Based on JSON File ")
    parser.add_argument("--input_folder", action="store_true", help="Inference Based on Image Folder")
    parser.add_argument("--vis", action="store_true", help="Visualisation Flag")
    
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
                pred_pixel_values_scr=network(inputs, gt_scr_img=inputs,criterion=loss_criterion, mode='test')

                # Send to .cpu
                pred_pixel_values_scr = pred_pixel_values_scr.cpu()
                spatch=reconstruct(pred_pixel_values_scr,patch_size,target_shape,image_size)

                # binarize the predicted image taking 0.5 as threshold
                spatch = ( spatch>THRESHOLD)*1

                # Append the net processed patch
                soutput_patches.append(255*spatch)

        try:
            assert len(soutput_patches)==len(input_patches)
        except Exception as exp:
            print('Error in patch processing outputs : Exiting!')
            sys.exit(0)
        
        # Restich the image
        soutput = emp.merge_patches(soutput_patches,indices,mode='max')

        # Transpose
        scribbleOutput=np.transpose(soutput,(1,0))
        
        return scribbleOutput
    
'''
Post Processing Function
'''
def postProcess_URDU(scr, bin, thresh=40, rectangularKernel=30):
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(bin[:, :, np.newaxis], 3, axis=2)

    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)

    tmp = pu.polygon_to_distance_mask(scr,threshold=30)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    bin[bin>=thresh]=255
    bin[bin<thresh]=0
    scr[scr>=thresh]=255
    scr[scr<thresh]=0

    # Usage of binarization results
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    scr_ = pu.dilate_image(scr_,kernel_size=3,iterations=2)
    scr_ = pu.horizontal_dilation(scr_,rectangularKernel,1)

    box_img, box, cut = pu.find_text_bounding_box(scr_*255)

    scr_ = pu.get_subset(box, scr_)

    # temporary contours
    contours = pu.cleanImageFindContours(scr_,threshold = 0.10)

    H,W,C = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # Hull operation - merge contours
    _,new_hulls = pu.combine_hulls_on_same_level(contours, threshold=30)
    genScribbles=[]

    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        canvas_copy = pu.horizontal_dilation(canvas_copy,cut + cut//2,1)

        contours = pu.cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = pu.generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        genScribbles.append(scr_lst)
    return genScribbles

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
    # Make a seperate scribble & binary image folders 
    scr_folder = os.path.join(args['output_image_folder'],'scr')
    bin_folder =  os.path.join(args['output_image_folder'],'bin')
    vis_folder =  os.path.join(args['output_image_folder'],'vis')

    os.makedirs(scr_folder,exist_ok=True)
    os.makedirs(bin_folder,exist_ok=True)
    os.makedirs(vis_folder,exist_ok=True)

    if  args['input_image_folder'] is not None and args['input_image_json'] is None:
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
                    
                    # Calling Stage I Inference 
                    scribbleMap =  imageInference(network,file_path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True)
                    scribbleMap=np.uint8(scribbleMap)

                    # Post Processing of Scribble Branch
                    scribbles = postProcess_URDU(scribbleMap,binaryMap)
                    
                    # Visualisation purpose
                    binaryMap = img.astype('uint8')
                    cv2.imwrite(os.path.join(scr_folder,'scr_'+file_name),scribbleMap)
                    cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),binaryMap)

                    # Preparation for Stage II 
                    binaryMap = cv2.imread(os.path.join(bin_folder,'bin_'+file_name))
                    scribbleMap = cv2.imread(os.path.join(scr_folder,'scr_'+file_name))

                    # Stage II Output : Text Line Polygons 
                    ppolygons = imageTask(img,binaryMap,scribbles)

                
                    # Visualise the predicted polygons and store it 
                    img2 = copy.deepcopy(img)
                    for p in ppolygons:
                        p = np.asarray(p,dtype=np.int32).reshape((-1,1,2))
                        img2 = cv2.polylines(img2, [p],True, (255, 0, 0),3)
                    cv2.imwrite(os.path.join(vis_folder,'vis_'+file_name),img2)

                    # Writing it to JSON file 
                    scrs_ = [ np.asarray(gd).reshape((-1,2)).tolist() for gd in scribbles]
                    pps_ = [ np.asarray(gd).reshape((-1,2)).tolist() for gd in ppolygons]

                    jsonOutput.append({'imgPath':file_path,'imgDims':[H,W],'predScribbles':scrs_,'predPolygons':pps_})
                
                except Exception as exp:
                    print('Image :{} Error :{}'.format(file_name,exp))
                    continue

        else:
            print('Empty Input Image Folder , Exiting !')
            sys.exit(0)


    # Save the json file 
    with open(os.path.join(args['output_image_folder'],'{}.json'.format(args['exp_name'])),'w') as f:
        json.dump(jsonOutput,f)
    f.close() 
    print('~Completed Inference !') 

if __name__ == "__main__":    
    args = addArgs()
    print('Running Inference...')
    Inference(args)

