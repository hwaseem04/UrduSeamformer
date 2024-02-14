<div align="center">

<samp>
<h2> Advancing Text Line Segmentation in Complex
Urdu Documents with Seamformer </h2>
</samp>

</div>

The preprint can be accessed here: [Advancing Text Line Segmentation in Complex
Urdu Documents with Seamformer](https://drive.google.com/file/d/1x6jDApziX_JqaxXyLXXKkozS0l05tbaZ/view?usp=drive_link)

This is a forked repository from the original [Seamformer](https://github.com/ihdia/seamformer) repository. 

## Table of contents

1. [Getting Started](#getting-started)
1. [Model Overview](#model-overview)
1. [Dataset](#dataset)
1. [Training](#training)
1. [Download Pretrained weights](#downloading-pretrained-weights)
1. [Inference](#inference)
1. [Visual Results](#visual-results)
1. [Contact](#contact)
1. [License](#license)
<!-- 1. [Citation](#citation) -->

## Getting Started

To make the code run, install the necessary libraries preferably using [conda](https://www.anaconda.com/) or else [pip](https://pip.pypa.io/en/stable/) environment manager.

```bash
conda create -n seamformer python
conda activate seamformer
pip install -r requirements.txt
```

## Model Overview

Instead of multi-task, the UrduSeamformer uses just the scribble branch of the original Seamformer. Binarised result from Seamformer turns to be good for majority of the case, hence it order to reduce training time, the model is made single branched. Remaining configuration remains same and you can get more information on the experiment setup formm original repository.

Modified Structure
<div align="center">

![Urdu Experiment Architecture](assets/Urdu_Architecture.png)

</div>

## Dataset

Dataset is originally collected from this [website](https://beratkurar.github.io/). Since this work need input images in binarised format, the downloaded images are already converted to binarised format and you can access them from this [drive link](https://drive.google.com/file/d/1FPBTJoDW_YRXlrhFD8ZRHKx1rbwmj-R_/view?usp=sharing).
This Arabic Dataset comprises of [VML-AHTE](https://beratkurar.github.io/data/ahte_dataset.zip), [VML-HD](https://www.cs.bgu.ac.il/~vml/database/VML-HD/VML-HD.zip).

## Training

The Urdu-SeamFormer is split into two parts:
- Stage-1: Scribble Generation [ Requires Training/Finetuning ]
- Stage-2: Seam generation and final segmentation prediction [ No Training ]

### Preparing the Data

Refer to [Seamformer](https://github.com/ihdia/seamformer) repository.
```
├── DATASET
│   ├── <DATASET>_Train
│   │   ├── images/
│   │   ├── binaryImages/
│   │   ├── <DATASET>_TRAIN.json
│   ├── <Dataset>_Test
│   │   ├── images/
│   │   ├── binaryImages/
│   │   ├── <DATASET>_TEST.json
│
├── ...
```

### Preparing the configuration files

  | Parameters  | Description | Default Value
  | ----------  | ----------- | ------------- |
  | dataset_code   | Codename for dataset   | I2 | 
  | data_path   | Dataset Folder  | /data/ | 
  | model_weights_path   | Location to store trained weights  | /weights/ | 
  | visualisation_folder   | Folder path to store visualisation results | /vis_results/ | 
  | learning_rate   | Initial learning rate of optimizer (scheduler applied) | 0.005-0.0009 | 
  | weight_logging_interval  | Epoch interval to store weights, i.e 3 -> Store weight every 3 epoch    | 3 | 
  | img_size   | ViT input size    | 256 x 256| 
  | patch_size   | ViT patch size   | 8 x 8 | 
  | encoder_layers   | Number of encoder layers in stage-1 multi-task transformer   | 6 | 
  | encoder_heads   | Number of heads in MHSA    | 8 | 
  | encoder_dims   | Dimension of token in encoder   | 768 | 
  | batch_size   | Batch size for training   | 4 | 
  | num_epochs   | Total epochs for training   | 30 | 
  | mode   | Flag to train or test. Either use "train"/"test"   | "train" | 
  | train_scribble   | Enables scribble branch train  | false| 
  | train_binary  | Enables binary branch train   | true | 
  | pretrained_weights_path   | Path location for pretrained weights(either for scribble/binarisation)   | /weights/ | 
  | enableWandb  | Enable it if you have wandB account, else the results are stored locally in  `visualisation_folder`  | false |
  | wid   | WandB experiment Name (optional)   | I2_V0_Train_lr_0.001 | 

### Stage-1
Preparation of binarised input and patches
```bash
# Since `Arabic` dataset already is in binarised version you dont need to run `prepare_binary_input.py` script. If you have any new dataset, you can run it.

# python prepare_binary_input.py  --input_image_folder "Arabic_train_images" --output_image_folder "data/Arabic/Arabic_Train/images" --model_weights_path "weights/I2.pt" --input_folder
# python prepare_binary_input.py  --input_image_folder "Arabic_test_images" --output_image_folder "data/Arabic/Arabic_Test/images" --model_weights_path "weights/I2.pt" --input_folder

# You can just go ahead with the preparation of patches
python datapreparation.py  --datafolder 'data/'  --outputfolderPath 'data/Arabic_train_patches'  --inputjsonPath 'data/Arabic/Arabic_Train/ARABIC_TRAIN.json'
python datapreparation.py  --datafolder 'data/'  --outputfolderPath 'data/Arabic_test_patches'  --inputjsonPath 'data/Arabic/Arabic_Test/ARABIC_TEST.json'

```


Training Scribble branch
```bash
python train.py --exp_json_path 'Arabic.json' --mode 'train' --train_scribble
```

## Downloading Pretrained Weights
Download our existing modelcheckpoints for SeamFormer network via the following commands , additionally you have to override `pretrained_weights_path` in experiment configuration file accordingly.
```bash
pip install gdown 

mkdir weights
cd weights
```
For Indiscapes2 Dataset Checkpoint (To prepare binarised input data)
```bash
gdown 1O_CtJToNUPrQzbMN38FsOJwEdxCDXqHh
```

For Urdu Checkpoint - Trained on VML-HD, VML-AHTE dataset (collectively called as `Arabic` in [Dataset](#dataset) Section)
```bash
gdown 1zieHu20iWfEse2IgcLiqWjR18pjZolev
```

## Inference : 

Provide appropriate Image Folder Path for inference
```bash
python inference.py --exp_name "PU_Urdu_images" --input_image_folder "PU_Urdu_images" --output_image_folder "data/pu_urdu_output_images" --model_weights_path "weights/BEST-MODEL-Arabic_DEC-7.pt" --input_folder
```

## Visual Results
Attached is a collated diagram. Of particular significance is the intrinsic precision exhibited by the predicted polygons depicted within, handling the presence of considerable image degradation, a complex multi-page layout, and an elevated aspect ratio, etc. 

![Visual Results](assets/GoodSamples.png)
There exist still scope of imporvement in both stage-1's scribble branch and stage-2.
More on the [Arxiv preprint]()

<!-- # Citation
Please use the following BibTeX entry for citation .
```bibtex
@inproceedings{vadlamudiniharikaSF,
    title = {SeamFormer: High Precision Text Line Segmentation for Handwritten Documents},
    author = {Vadlamudi,Niharika and Rahul,Krishna and Sarvadevabhatla, Ravi Kiran},
    booktitle = {International Conference on Document Analysis and Recognition,
            {ICDAR}},
    year = {2023},
} 
```
-->


# Contact
For any queries, please contact [Dr. Ravi Kiran Sarvadevabhatla](mailto:ravi.kiran@iiit.ac.in.)

# License
This project is open sourced under [MIT License](LICENSE).
