from pathlib import Path
import argparse

import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
from skimage.registration import phase_cross_correlation
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

def read_img(path):
    return skimage.io.imread(path)

def create_filename(df, aft=True):
    if aft == False:
        filenames = [
            "_".join([str(row.Location), str(row.Z_stack), row.Cycle, row.Marker])
            for row in df.itertuples() 
        ]
    else:
        filenames = [
            "_".join(['Af', str(row.Location), str(row.Z_stack), row.Cycle, row.Marker])
            for row in df.itertuples() 
        ]
    folder = data_processed_bis
    filenames = [os.path.join(folder, f + ".tiff") for f in filenames]
    return filenames

def correct_img(img, path):
    gaussianImg = cv2.GaussianBlur(img, (1025, 1025), 256)
    img_corrected = cv2.subtract(img, gaussianImg)
    cv2.imwrite(path, img_corrected)

def joblib_loop(task, pics):
    return Parallel(n_jobs=40)(delayed(task)(i) for i in pics)

def background_correction(
    df
):
    """Function to perform background substraction for image using gaussian blurr of original image

    Args:
        df (pd DataFrame) : info dataframe for all images
        filtersize (int) : filter size of gaussian kernel
        sigma (int) : sigma of guassian blurr
        folder (str) : folder to save corrected images
        save (bool) : bool to save the image
        show (bool) : bool to show corrected image and gaussian blur of original image

    Returns:
        None
    """
    grouped = df.groupby(['Location', 'Marker', 'After_bleach'])
    for name, group in tqdm(grouped):
        paths = group.Path.tolist()
        imgs = joblib_loop(read_img, paths)

        filenames = create_filename(group, name[2])
        Parallel(n_jobs=40)(delayed(correct_img)(img=i, path=j) for i in imgs for j in filenames)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get features from each experiment')
    parser.add_argument('--cycle', required=True, type=str, dest='cyc',
                        help='cycle')

    args = parser.parse_args() 

    # Import path
    module_path = str(Path.cwd().parents[0])
    if module_path not in sys.path:
        sys.path.append(module_path)

    from config import *

    csv_file = data_meta / "info.csv"
    csv_exist = csv_file.is_file()

    df = pd.read_csv(csv_file)
    df_subset = df.query('Cycle == @args.cyc')
    background_correction(df_subset)
