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


def background_correction(
    df, folder, filtersize=1025, sigma=256
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
    # Loop through the rows of the dataframe
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        
        # Read image
        img = skimage.io.imread(row.Path)

        if len(img.shape) > 2:
            img = img[:, :, 0]

        # Define saving filename for corrected image
        if row.After_bleach == False:
            filename = "_".join([str(row.Location), str(row.Z_stack), row.Cycle, row.Marker])
        else:
            filename = "_".join(['Af', str(row.Location), str(row.Z_stack), row.Cycle, row.Marker])
        path = os.path.join(folder, filename + ".tiff")

        # Background substraction using gaussian blur channel
        gaussianImg = cv2.GaussianBlur(img, (filtersize, filtersize), sigma)
        img_corrected = cv2.subtract(img, gaussianImg)

        # Save image
        cv2.imwrite(path, img_corrected)
            

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
    background_correction(df_subset, data_processed)
