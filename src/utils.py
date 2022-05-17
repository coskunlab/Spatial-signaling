import os
import cv2
import numpy as np 
from pathlib import Path
from skimage import measure, exposure
import skimage.io
import colorsys
import pickle

DEFAULT_CHANNELS = (1, 2)

RGB_MAP = {
    1: {"rgb": np.array([255, 0, 0]), "range": [0, 50]},
    # 2: {"rgb": np.array([0, 255, 0]), "range": [0, 20]},
    2: {"rgb": np.array([0, 0, 255]), "range": [0, 60]},
}


def convert_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    dim1, dim2, _ = t.shape
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, channel - 1] / vmax) / (
            (rgb_map[channel]["range"][1] - rgb_map[channel]["range"][0]) / 255
        ) + rgb_map[channel]["range"][0] / 255
        x = np.where(x > 1.0, 1.0, x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]["rgb"]).reshape(dim1, dim2, 3), dtype=int
        )
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    im = im.astype(np.uint8)
    return im

def one_channel(t, channel, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    dim1, dim2, _ = t.shape
    colored_channels = []
    x = (t[:, :, 0] / vmax) / (
        (rgb_map[channel]["range"][1] - rgb_map[channel]["range"][0]) / 255
    ) + rgb_map[channel]["range"][0] / 255
    x = np.where(x > 1.0, 1.0, x)
    x_rgb = np.array(
        np.outer(x, rgb_map[channel]["rgb"]).reshape(dim1, dim2, 3), dtype=int
    )
    colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    im = im.astype(np.uint8)
    return im

def contrast_streching(img):
    p2, p98 = np.percentile(img, (0., 99.9))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

def get_composite_image(df, seg_cycle="Cycle1"):
    # Read condition and cycles name
    
    imgs = []

    df_seg = df[df.Cycle=='Cycle2']
    df_grouped = df_seg.groupby(['Location'])
    locations = []

    for name, df_subset in df_grouped:
        dapi_path = df_subset[df_subset.Marker == "Hoeschst"].Path.item()
        pha_path = df_subset[df_subset.Marker == "Phalloidin"].Path.item()
        wga_path = df_subset[df_subset.Marker == "WGA"].Path.item()

        img_dapi = contrast_streching(skimage.io.imread(dapi_path))
        img_pha = contrast_streching(skimage.io.imread(pha_path))
        img_pha = np.clip(img_pha - img_dapi, a_min = 0, a_max=None)
        img_wga = contrast_streching(skimage.io.imread(wga_path))

        img_seg = np.concatenate([img_pha[:, :, np.newaxis], img_wga[:, :, np.newaxis]], axis=2)
        seg_max = np.amax(img_seg, axis=2, keepdims=True)
        data = np.concatenate(
            (
                img_pha[:, :, np.newaxis], 
                # seg_max,
                img_dapi[:, :, np.newaxis]
            ),
            axis=2,
        )
        img = convert_to_rgb(data, vmax=65536)
        imgs.append(img)
        locations.append(name)

    return imgs, locations

def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)
    Parameters
    ----------------
    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]
    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels
    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range
    Returns
    ----------------
    RGB: uint8, 3D array
        array of masks overlaid on grayscale image
    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

# Read mask image
def get_masks(mask_folder):
    '''
    Function to get all mask from mask forlder
    '''
    # Read masks
    masks = {}

    for (dirpath, dirnames, filenames) in os.walk(mask_folder):
        for name in sorted(filenames):
            if "tif" in name:
                condition = '_'.join(name.split("_")[:1])
                masks[condition] = masks.get(condition, {})
                filename = os.path.join(dirpath, name)
            else:
                continue
            if "cyto" in name:
                img = skimage.io.imread(filename)
                masks[condition]["cyto"] = img
            elif "nuclei" in name:
                img = skimage.io.imread(filename)
                masks[condition]["nuclei"] = img
    return masks

# Quality control of mask
def qc_nuclei(mask_cyto, mask_nuclei):
    '''
    Function to check if cell masks contain nuclei
    '''
    cell = np.zeros((mask_cyto.shape), dtype=np.uint8)
    nuclei = np.zeros((mask_cyto.shape), dtype=np.uint8)
    cyto = np.zeros((mask_cyto.shape), dtype=np.uint8)

    for label in range(1, mask_cyto.max()):
        # Check if cell has nuclei
        cell_mask = np.where(mask_cyto == label, 1, 0).astype(np.uint8)
        maski = cv2.bitwise_and(mask_nuclei, mask_nuclei, mask=cell_mask)

        # If no nuclei detected then pass
        if maski.max() == 0:
            continue

        # Link label accross cell, nuclei, cyto
        cell = np.where(mask_cyto == label, label, cell)
        nuclei = np.where(maski > 0, label, nuclei)
        maski = cv2.subtract(cell_mask, maski)
        cyto = np.where(maski > 0, label, cyto)
    return cell, nuclei, cyto

def get_contour(mask):
    labels = [n for n in np.unique(mask) if n > 0]
    contours = {}
    for i in labels:
        temp = np.where(mask == i, mask, 0)
        contours[i] = measure.find_contours(temp, 0.1)
    return contours

def plot_contours(contours_all, ax, cells=None, linewidth=2, c='w'):
    if cells == None:
        cells = list(contours_all.keys())
    for label, contours in contours_all.items():
        if label not in cells:
            continue
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth= linewidth, c=c)


def save_pickle(path, data):
    with open(path, 'wb') as f:
        for d in data:
            pickle.dump(d, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass
