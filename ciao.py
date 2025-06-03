import os

from dataset.fows_preprocessing import *
from dataset.gothca_preprocessing import *

def get_model_path(model_name, dataset, trn_strategy):
    # Check if the model is valid
    if model_name not in ['mnetv2', 'effnetb4', 'xception']:
        print(f"Model {model_name} is not supported")
        exit()
    # Check if the dataset is valid
    if dataset not in ['fows_occ', 'fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']: 
        print(f"Dataset {dataset} is not valid")
        exit()
    
    # Get the model path based on the arguments
    model_info = model_name + '_' + dataset + '_' + trn_strategy #('_FT' if ft else '_TL')
    # '_TL' if tl else 
    print("Model info: ", model_info)
    # models_folder = '../model_weights/' + trn_strategy + '/'
    models_folder = 'C:\\Users\\ricky\\Documents\\scripts\\_IWBF_scritps_\\baseline_weights\\' + trn_strategy + '/'
    # if tl:
    #     models_folder = '../model_weights/TL/'
    # else:
    #     models_folder = '../model_weights/FT/' 
    pretrained_model_path = ''

    # Check if the model path exists
    if not os.path.exists(models_folder):
        print(f"Model path {models_folder} does not exist")
        exit()
    
    # navigate all model folders subdirectories and check if the model_info is in any of them
    found = False
    for root, dirs, files in os.walk(models_folder):
        for dir in dirs:
            # print(f"Checking directory: {dir}")
            if dir.lower() == model_name.lower():  # check if the directory name is the same as the model_info
                # find the .pth file in the directory
                print(f"Found directory: {dir} with same name as {model_name}")
                model_dir = os.path.join(root, dir)
                for sub_root, sub_dirs, sub_files in os.walk(model_dir):
                    for file in sub_files:
                        if file.endswith('.pth') and model_info.lower() in file.lower():
                            found = True
                            pretrained_model_path = os.path.join(sub_root, file)
                            # print(f"Pretrained model path: {pretrained_model_path}")
                            break
                break # uncomment this if you want to stop searching after finding the first directory
                
    if not found:
        print(f"NO model {model_info} found in {models_folder}")
        exit()
    
    return pretrained_model_path

def get_model_path_2(model_weights_path, model_str):
    model_path = ''
    found = False
    for root, dirs, files in os.walk(model_weights_path):
        for file in files:
            if file.endswith('.pth') and model_str in file:
                print(os.path.join(root, file))
                model_path = os.path.join(root, file)
                found = True
    if found:
        print(model_path)
    else:
        print("no model found")
        exit()

    return model_path

model_path = 'C:\\Users\\ricky\\Documents\\scripts\\_IWBF_scritps_\\baseline_weights\\'
# fix the path
model_path = model_path.replace('\\', '/')
# print(model_path)

"""
models: mnetv2, effnetb4, xception
dataset: fows_occ, fows_no_occ, gothca_occ, gotcha_no_occ
trn_strategy: TL, FT
"""

pretrained_model_path = get_model_path('effnetb4', 'gotcha_occ', 'ft')
print(pretrained_model_path)

# path2model = get_model_path_2(model_path, 'mnetv2_fows_occ_tl')
# print(path2model)
