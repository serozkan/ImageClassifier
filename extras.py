import argparse
import numpy as np
import torch

def get_input_for_training():
    """ Process input arguments for training script
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments
    parser.add_argument('data_dir', type=str,
                        help='path to the folder containing train, test, and validation subfolders')

    # optional 

    parser.add_argument('--save_dir', type=str, default='/home/workspace/',
                        help='path to the folder to save checkpoints (default= "" (root folder))')

    parser.add_argument('--arch', type=str, default='vgg16',
                        help='CNN Model Architecture - vgg11_bn, or vgg16 (default= "vgg16_bn")')

    parser.add_argument('--learn_rate', type=float, default=0.001,
                        help='Learning rate (default=0.001)')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to execute the training')

    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='Hidden units')
    parser.add_argument('--gpu', default=False, action='store_true', help='Select between --gpu or --no_gpu')
    parser.add_argument('--no_gpu', dest='gpu', action='store_false', help='Select between --gpu or --no_gpu')

    return parser.parse_args()



def get_input_for_predict():
    """ Process input arguments for predict script
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments
    parser.add_argument('image_dir', type=str,
                        help='path to the image file to predict')    
    parser.add_argument('load_dir', type=str, default='checkpoint.pth',
                        help='path to the folder to load checkpoints (default= "" (root folder)')
    parser.add_argument('--top_k', type=int, default= 5,
                        help='number of top K most likely classes, default : 5')
    parser.add_argument('--gpu', default=False, action='store_true', help='Select between --gpu or --no_gpu')
    parser.add_argument('--no_gpu', dest='gpu', action='store_false', help='Select between --gpu or --no_gpu')
    
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='category mapping .json file (default= "cat_to_name.json" ')
    
    return parser.parse_args()



