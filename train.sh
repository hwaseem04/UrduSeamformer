mkdir data

# Since we dont have annotated Urdu images, we are using closest possible dataset, i.e Arabic
python prepare_binary_input.py  --input_image_folder "Arabic_train_images" --output_image_folder "data/Arabic/Arabic_Train/images" --model_weights_path "weights/I2.pt" --input_folder
python prepare_binary_input.py  --input_image_folder "Arabic_test_images" --output_image_folder "data/Arabic/Arabic_Test/images" --model_weights_path "weights/I2.pt" --input_folder

python datapreparation.py  --datafolder 'data/'  --outputfolderPath 'data/Arabic_train_patches'  --inputjsonPath 'data/Arabic/Arabic_Train/ARABIC_TRAIN.json'
python datapreparation.py  --datafolder 'data/'  --outputfolderPath 'data/Arabic_test_patches'  --inputjsonPath 'data/Arabic/Arabic_Test/ARABIC_TEST.json'

python train.py --exp_json_path 'Arabic.json' --mode 'train' --train_scribble

# Load best weights, uncomment and then run
# python inference.py --exp_name "PU_Urdu_images" --input_image_folder "PU_Urdu_images" --output_image_folder "data/final_urdu_upscaled_output" --model_weights_path "weights/BEST-MODEL-Arabic_DEC-7.pt" --input_folder