"""
autoencoder_compress.py
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import os


def get_stat_info(x1d):
    return [f(x1d) for f in [np.min, np.max, np.mean, np.std]]


def clip(x1d, nstd=3, disp=False):
    Min, Max, Mean, Std = get_stat_info(x1d)
    newmin = Mean - nstd * Std
    newmax = Mean + nstd * Std

    if disp:
        print('Original: Min, Max, Mean, Std =', Min, Max, Mean, Std)
        print('New: Min, Max =', newmin, newmax)

    return np.clip(x1d, newmin, newmax)


def get_image_array(fname, disp=False, fig=False):
    """
    open image and return as an array
    """
    im = Image.open(fname)
    im_a = np.array(im)

    if disp:
        print("Image info:", im.format, im.size, im.mode)

    if fig:
        plt.imshow(im_a, cmap='gray')
        plt.title("Original Image")
        plt.show()

    return im_a


def image_clip(fname, disp=False, fig=False):
    """
    Load image and clip it with +/- 3*std wide
    return the clipping image
    """
    im_a = get_image_array(fname, disp=disp, fig=fig)
    im_a1d = im_a.reshape(-1)
    im_a1d_clip = clip(im_a1d, disp=disp)
    im_a_clip = im_a1d_clip.reshape(im_a.shape)

    if fig:
        plt.imshow(im_a_clip, cmap='gray')
        plt.title("Clipping Image")
        plt.show()

    return im_a_clip


def listdir_files(mypath):
    """
    return files in a mypath directory
    """
    # mypath = "../data/Autoencoder_compression/images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


def images_clip(fold):
    onlyfiles = listdir_files(fold)

    fold_clip = fold + '_clip'
    # Generate a new path to save clipping images
    if not os.path.exists(fold_clip):
        os.makedirs(fold_clip)

    for fname in onlyfiles:
        pf = os.path.join(fold, fname)
        iac = image_clip(pf)

        fname_clip = os.path.join(fold_clip, fname[:-4] + '_clip.tif')
        print(fname_clip)
        plt.imsave(fname_clip, iac, cmap='gray')


def load_images(fold):
    """
    Example
    =======
    fold = "../data/Autoencoder_compression/images_clip"
    im_a = load_images(fold)
    """
    files = listdir_files(fold)

    im_l = []
    for f in files:
        im = plt.imread(os.path.join(fold, f))
        im_l.append(im)

    im_a = np.array(im_l)
    if im_a.ndim == 4:
        im_a = im_a[:, :, :, 0]

    return im_a


def gen_all_images_diff(im_a):
    im_a_diff = np.diff(im_a, n=1, axis=0)
    return im_a_diff


def gen_save_all_images_diff(fold, im_a):
    im_a_diff = gen_all_images_diff(im_a)

    print("Saving...", os.path.join(fold, 'all_images_3d.npy'))
    np.save(os.path.join(fold, 'all_images_3d'), im_a)

    print("Saving...", os.path.join(fold, 'all_images_diff_3d.npy'))
    np.save(os.path.join(fold, 'all_images_diff_3d'), im_a_diff)


def split_image_a3d(im_a_diff, ni=8, nj=8, join=True):
    """
    Originally split results are saved to a list,
    though this save to an array
    """
    im_a_diff_split_i = np.array(np.split(im_a_diff, 8, axis=1))
    # im_a_diff_split_i.shape

    im_a_diff_split_ij = np.array(np.split(im_a_diff_split_i, 8, axis=3))
    # im_a_diff_split_ij.shape

    if join:
        s = im_a_diff_split_ij.shape
        im_a_diff_split_ij = im_a_diff_split_ij.reshape(-1, s[3], s[4])

    return im_a_diff_split_ij


def concatenate_image_a3d(im_a_diff_split_ij, ni=8, nj=8, join=True):
    if join:
        s = im_a_diff_split_ij.shape
        im_a_diff_split_ij = im_a_diff_split_ij.reshape(ni, nj, int(s[0]/ni/nj), s[1], s[2])

    im_a_diff_split_i = np.concatenate(im_a_diff_split_ij, axis=3)
    im_a_diff = np.concatenate(im_a_diff_split_i, axis=1)

    return im_a_diff
