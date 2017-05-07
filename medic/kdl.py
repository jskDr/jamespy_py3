"""
KDL - deep learning for medic
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d, fftconvolve
from sklearn import preprocessing, model_selection, metrics
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks

import kkeras

from . import beads

    
def fig2array(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # ones_255 = np.ones_like( data) * 255
    # data = 255 - data
    return data


def _gen_cell_r0(bd_on=True,
                 rand_pos_cell=False,
                 r_cell=0.1,  # 0<r_cell<=1
                 r_bd=0.05,  # 0<r_bd<=1
                 max_bd=5,  # max_bd >= 1
                 bound_flag=True,
                 visible=False,
                 disp=False,
                 fig=None,
                 ax=None):
    """
    Generate cell images
    """
    if fig is None or ax is None:
        assert fig is None and ax is None
        fig, ax = plt.subplots(figsize=(2, 2))
        # set_axis_bgcolor is not working because of plt.axis('off')
        # ax.set_axis_bgcolor('red')
        close_fig_flag = True
    else:
        close_fig_flag = False
    fig.patch.set_facecolor('black')

    circle_d = {}

    if rand_pos_cell:
        if bound_flag:  # Not generate cells in the boundary
            B = r_cell + 2.0 * r_bd
            pos_cell = B + (1.0 - 2 * B) * np.random.random(2)
        else:
            pos_cell = np.random.random(2)
    else:
        pos_cell = np.array([0.5, 0.5])

    def rand_pos_bd():
        th = np.random.random() * 2 * np.pi
        pos_bd = pos_cell + (r_cell + r_bd) * \
            np.array((np.cos(th), np.sin(th)))
        return pos_bd
    #print( pos_cell, pos_bd)

    circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
    if bd_on:
        for bd_n in range(np.random.randint(max_bd)+1):
            circle_d["bd{}".format(bd_n)] = plt.Circle(rand_pos_bd(), r_bd, color='w')
            # circle_d["bd2"] = plt.Circle(rand_pos_bd(), r_bd, color='w')
    for k in circle_d.keys():
        ax.add_artist(circle_d[k])
    plt.axis('off')
    data_a = fig2array(fig)
    if disp:
        print("Image array shape = ", data_a.shape)
    if visible:
        plt.show()
    else:
        if close_fig_flag:
            plt.close()
        else:
            plt.cla()

    return data_a


def _gen_cell_r1(bd_on=True,
                 rand_pos_cell=False,
                 r_cell=0.1,  # 0<r_cell<=1
                 r_bd=0.05,  # 0<r_bd<=1
                 max_bd=5, # max_bd >= 1
                 # stat_ext_bd=None, # or {'mean':5, 'std':1}
                 stat_ext_bd={'mean':2, 'std':2},
                 bound_flag=True,
                 visible=False,
                 disp=False,
                 fig=None,
                 ax=None):
    """
    Generate cell images
    The PS bead size is 6 um and silica bead is 5 um. 
    Lymphoma cell size varies in a larger variance, but the mean value is around 9-12 um.
    """
    if fig is None or ax is None:
        assert fig is None and ax is None
        fig, ax = plt.subplots(figsize=(2, 2))
        # set_axis_bgcolor is not working because of plt.axis('off')
        # ax.set_axis_bgcolor('red')
        close_fig_flag = True
    else:
        close_fig_flag = False
    fig.patch.set_facecolor('black')

    circle_d = {}

    if rand_pos_cell:
        if bound_flag:  # Not generate cells in the boundary
            B = r_cell + 2.0 * r_bd
            pos_cell = B + (1.0 - 2 * B) * np.random.random(2)
        else:
            pos_cell = np.random.random(2)
    else:
        pos_cell = np.array([0.5, 0.5])

    def rand_pos_bd():
        th = np.random.random() * 2 * np.pi
        pos_bd = pos_cell + (r_cell + r_bd) * \
            np.array((np.cos(th), np.sin(th)))
        return pos_bd
    
    # print( pos_cell, pos_bd)

    circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
    if bd_on:
        for bd_n in range(np.random.randint(max_bd)+1):
            circle_d["bd{}".format(bd_n)] = plt.Circle(rand_pos_bd(), r_bd, color='w')
            # circle_d["bd2"] = plt.Circle(rand_pos_bd(), r_bd, color='w')
    
    if stat_ext_bd is not None:
        n_ext_bd = np.max((0, int(np.random.randn()*stat_ext_bd['std'] + stat_ext_bd['mean'])))
        for ext_bd_n in range(n_ext_bd):
            ext_bd_pos = np.random.rand(2)
            circle_d["ext_bd{}".format(ext_bd_n)] = plt.Circle(ext_bd_pos, r_bd, color='w')

    for k in circle_d.keys():
        ax.add_artist(circle_d[k])
    plt.axis('off')
    data_a = fig2array(fig)
    if disp:
        print("Image array shape = ", data_a.shape)
    if visible:
        plt.show()
    else:
        if close_fig_flag:
            plt.close()
        else:
            plt.cla()

    return data_a


def _gen_cell_r2(bd_on=True,
                 rand_pos_cell=False,
                 r_cell=0.1,  # 0<r_cell<=1
                 r_bd=0.05,  # 0<r_bd<=1
                 max_bd=3, # max_bd >= 1
                 # stat_ext_bd=None, # or {'mean':5, 'std':1}
                 stat_ext_bd={'mean':2, 'std':2},
                 bound_flag=True,
                 visible=False,
                 disp=False,
                 fig=None,
                 ax=None):
    """
    Generate cell images
    The PS bead size is 6 um and silica bead is 5 um. 
    Lymphoma cell size varies in a larger variance, but the mean value is around 9-12 um.
    
    Inputs
    ======
    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 
    """
    if fig is None or ax is None:
        assert fig is None and ax is None
        fig, ax = plt.subplots(figsize=(2, 2))
        # set_axis_bgcolor is not working because of plt.axis('off')
        # ax.set_axis_bgcolor('red')
        close_fig_flag = True
    else:
        close_fig_flag = False
    fig.patch.set_facecolor('black')

    circle_d = {}

    if rand_pos_cell:
        if bound_flag:  # Not generate cells in the boundary
            B = r_cell + 2.0 * r_bd
            pos_cell = B + (1.0 - 2 * B) * np.random.random(2)
        else:
            pos_cell = np.random.random(2)
    else:
        pos_cell = np.array([0.5, 0.5])

    def rand_pos_bd():
        th = np.random.random() * 2 * np.pi
        pos_bd = pos_cell + (r_cell + r_bd) * \
            np.array((np.cos(th), np.sin(th)))
        return pos_bd
    #print( pos_cell, pos_bd)

    circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
    if bd_on:
        for bd_n in range(np.random.randint(max_bd)+1):
            circle_d["bd{}".format(bd_n)] = plt.Circle(rand_pos_bd(), r_bd, color='w')
            # circle_d["bd2"] = plt.Circle(rand_pos_bd(), r_bd, color='w')
    
    if stat_ext_bd is not None:
        #n_ext_bd = np.max((0, int(np.random.randn()*stat_ext_bd['std'] + stat_ext_bd['mean'])))
        n_ext_bd = np.random.randint(stat_ext_bd['mean']+1)
        for ext_bd_n in range(n_ext_bd):
            ext_bd_pos = np.random.rand(2)
            circle_d["ext_bd{}".format(ext_bd_n)] = plt.Circle(ext_bd_pos, r_bd, color='w')

    for k in circle_d.keys():
        ax.add_artist(circle_d[k])
    plt.axis('off')
    data_a = fig2array(fig)
    if disp:
        print("Image array shape = ", data_a.shape)
    if visible:
        plt.show()
    else:
        if close_fig_flag:
            plt.close()
        else:
            plt.cla()

    return data_a


def _gen_cell_db_r0(N=5, rand_pos_cell=False, disp=False):
    db_l = []
    cell_img_org = gen_cell(bd_on=False, rand_pos_cell=rand_pos_cell)
    for i in range(N):
        if disp:  # 1, 2, True (not 0 or False)
            print('Iteration:', i)
        elif disp == 2:
            print(i, end=",")
        if rand_pos_cell:
            cell_img = gen_cell(bd_on=False, rand_pos_cell=rand_pos_cell)
        else:
            cell_img = cell_img_org.copy()
        cellbd_img = gen_cell(bd_on=True, rand_pos_cell=rand_pos_cell)
        db_l.append(cell_img[:, :, 0])  # No RGB Info
        db_l.append(cellbd_img[:, :, 0])  # No RGB Info
    print("The end.")
    return db_l


def gen_cell_db(N=5, rand_pos_cell=False, 
                extra_bead_on=True, 
                max_bd=3,
                disp=False):
    """
    db_l = gen_cell_db(N=5, rand_pos_cell=False, extra_bead_on=True, disp=False) 
    Generate cell_db
    
    Inputs
    ======
    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 
    """

    fig, ax = plt.subplots(figsize=(2, 2))
    # ax.set_axis_bgcolor('red')    

    if extra_bead_on:
        stat_ext_bd = {'mean': 5, 'std': 1}
    else:
        stat_ext_bd = None

    db_l = []
    cell_img_org = gen_cell(bd_on=False,
                            rand_pos_cell=rand_pos_cell,
                            fig=fig, ax=ax,
                            stat_ext_bd=stat_ext_bd)

    for i in range(N):
        if disp:
            print(i, end=",")
        if rand_pos_cell:
            cell_img = gen_cell(
                bd_on=False, rand_pos_cell=rand_pos_cell,
                max_bd=max_bd,
                fig=fig, ax=ax,
                stat_ext_bd=stat_ext_bd)
        else:
            cell_img = cell_img_org.copy()

        cellbd_img = gen_cell(
            bd_on=True, rand_pos_cell=rand_pos_cell,
            fig=fig, ax=ax,
            stat_ext_bd=stat_ext_bd)

        db_l.append(cell_img[:, :, 0])  # No RGB Info
        db_l.append(cellbd_img[:, :, 0])  # No RGB Info

    plt.close(fig)
    print("The end.")
    return db_l


def save_cell_db(db_l, fname_gz="sheet.gz/cell_db.cvs.gz"):
    df_l = []
    celltype = 0
    for i, db in enumerate(db_l):
        df_i = pd.DataFrame()
        df_i["ID"] = [i] * np.prod(db.shape)
        df_i["celltype"] = celltype
        df_i["x"] = np.repeat(np.arange(db.shape[0]), db.shape[1])
        df_i["y"] = list(range(db.shape[1])) * db.shape[0]
        df_i["image"] = db.reshape(-1)
        celltype ^= 1
        df_l.append(df_i)
    cell_df = pd.concat(df_l, ignore_index=True)
    cell_df.to_csv(fname_gz, index=False, compression='gzip')
    return cell_df


# ===================================
# Functions for the Center_Cell mode
# - gen_cell_n_beads, 
#   gen_cell_db_center_cell, 
#   save_cell_db_center_cell
# ===================================

def gen_cell(bd_on=True,
             rand_pos_cell=False,
             r_cell=0.1,  # 0<r_cell<=1
             r_bd=0.05,  # 0<r_bd<=1
             max_bd=3, # max_bd >= 1
             stat_ext_bd={'mean':2, 'std':2},
             bound_flag=True,
             visible=False,
             disp=False,
             fig=None,
             ax=None):

    """
    Generate cell images
    The PS bead size is 6 um and silica bead is 5 um. 
    Lymphoma cell size varies in a larger variance, but the mean value is around 9-12 um.
    
    Inputs
    ======
    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 
    """

    return gen_cell_n_beads(bd_on=bd_on,
             rand_pos_cell=rand_pos_cell,
             r_cell=r_cell,  # 0<r_cell<=1
             r_bd=r_bd,  # 0<r_bd<=1
             max_bd=max_bd, # max_bd >= 1
             rand_bead_flag=True, # This is onlu changed part.
             # stat_ext_bd=None, # or {'mean':5, 'std':1}
             stat_ext_bd=stat_ext_bd,
             bound_flag=bound_flag,
             visible=visible,
             disp=disp,
             fig=fig,
             ax=ax)


def gen_cell_n_beads(bd_on=True,
                     rand_pos_cell=False,
                     r_cell=0.1,  # 0<r_cell<=1
                     r_bd=0.05,  # 0<r_bd<=1
                     max_bd=3,  # max_bd >= 1
                     rand_bead_flag=False,
                     # stat_ext_bd=None, # or {'mean':5, 'std':1}
                     stat_ext_bd={'mean': 2, 'std': 2},
                     bound_flag=True,
                     visible=False,
                     disp=False,
                     fig=None,
                     ax=None):
    """
    Generate cell images
    The PS bead size is 6 um and silica bead is 5 um.
    Lymphoma cell size varies in a larger variance,
    but the mean value is around 9-12 um.
    
    Inputs
    ======
    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 
    """
    if fig is None or ax is None:
        assert fig is None and ax is None
        fig, ax = plt.subplots(figsize=(2, 2))
        # set_axis_bgcolor is not working because of plt.axis('off')
        # ax.set_axis_bgcolor('red')
        close_fig_flag = True
    else:
        close_fig_flag = False
    fig.patch.set_facecolor('black')

    circle_d = {}

    if rand_pos_cell:
        if bound_flag:  # Not generate cells in the boundary
            B = r_cell + 2.0 * r_bd
            pos_cell = B + (1.0 - 2 * B) * np.random.random(2)
        else:
            pos_cell = np.random.random(2)
    else:
        pos_cell = np.array([0.5, 0.5])

    def rand_pos_bd():
        th = np.random.random() * 2 * np.pi
        pos_bd = pos_cell + (r_cell + r_bd) * \
            np.array((np.cos(th), np.sin(th)))
        return pos_bd
    #print( pos_cell, pos_bd)

    circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
    if bd_on:
        if rand_bead_flag:
            final_max_bd = np.random.randint(max_bd) + 1
        else:
            # Now, the number of total beads attached a cell is fixed (not random).
            final_max_bd = max_bd

        for bd_n in range(final_max_bd):
            circle_d["bd{}".format(bd_n)] = \
                plt.Circle(rand_pos_bd(), r_bd, color='w')
    
    if stat_ext_bd is not None:
        #n_ext_bd = np.max((0, int(np.random.randn()*stat_ext_bd['std'] + stat_ext_bd['mean'])))
        n_ext_bd = np.random.randint(stat_ext_bd['mean']+1)
        for ext_bd_n in range(n_ext_bd):
            ext_bd_pos = np.random.rand(2)
            circle_d["ext_bd{}".format(ext_bd_n)] = \
                plt.Circle(ext_bd_pos, r_bd, color='w')

    for k in circle_d.keys():
        ax.add_artist(circle_d[k])
    plt.axis('off')
    data_a = fig2array(fig)

    if disp:
        print("Image array shape = ", data_a.shape)
    if visible:
        plt.show()
    else:
        if close_fig_flag:
            plt.close()
        else:
            plt.cla()

    return data_a


class CELL():
    def __init__(self, flag_no_overlap_beads=False):
        self.flag_no_overlap_beads = flag_no_overlap_beads

    def gen(self,
            bd_on=True,
            rand_pos_cell=False,
            r_cell=0.1,  # 0<r_cell<=1
            r_bd=0.05,  # 0<r_bd<=1
            max_bd=3,  # max_bd >= 1
            rand_bead_flag=False,
            # stat_ext_bd=None, # or {'mean':5, 'std':1}
            stat_ext_bd={'mean': 2, 'std': 2},
            bound_flag=True,
            visible=False,
            disp=False,
            fig=None,
            ax=None):

        if fig is None or ax is None:
            assert fig is None and ax is None
            fig, ax = plt.subplots(figsize=(2, 2))
            # set_axis_bgcolor is not working because of plt.axis('off')
            # ax.set_axis_bgcolor('red')
            close_fig_flag = True
        else:
            close_fig_flag = False
        fig.patch.set_facecolor('black')

        circle_d = {}

        if rand_pos_cell:
            if bound_flag:  # Not generate cells in the boundary
                B = r_cell + 2.0 * r_bd
                pos_cell = B + (1.0 - 2 * B) * np.random.random(2)
            else:
                pos_cell = np.random.random(2)
        else:
            pos_cell = np.array([0.5, 0.5])

        def get_pos_bd(th):
                return pos_cell + (r_cell + r_bd) * np.array((np.cos(th), np.sin(th)))

        def rand_pos_bd():
            th = np.random.random() * 2 * np.pi
            pos_bd = get_pos_bd(th)
            return pos_bd

        circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
        if bd_on:
            if rand_bead_flag:
                final_max_bd = np.random.randint(max_bd) + 1
            else:
                # Now, the number of total beads attached
                # a cell is fixed (not random).
                final_max_bd = max_bd

            if not self.flag_no_overlap_beads:
                rand_pos_bd_l = []
                for bd_n in range(final_max_bd):
                    rand_pos_bd_l.append(rand_pos_bd())
            else:
                bead_center_l = []
                cnt = 0
                while len(bead_center_l) < final_max_bd:
                    # generate beads until the number of it reaches the limit
                    bead_center_l = beads.BEADS(r_cell, r_bd).gen_bead_centers(final_max_bd)
                    cnt += 1
                    assert cnt < 100, 'Try to reduce the number of beads!'
                rand_pos_bd_l = [get_pos_bd(th/180*np.pi) for th in bead_center_l]

            for bd_n in range(final_max_bd):
                circle_d["bd{}".format(bd_n)] = \
                    plt.Circle(rand_pos_bd_l[bd_n], r_bd, color='w')

        if stat_ext_bd is not None:
            n_ext_bd = np.random.randint(stat_ext_bd['mean'] + 1)
            for ext_bd_n in range(n_ext_bd):
                ext_bd_pos = np.random.rand(2)
                circle_d["ext_bd{}".format(ext_bd_n)] = \
                    plt.Circle(ext_bd_pos, r_bd, color='w')

        for k in circle_d.keys():
            ax.add_artist(circle_d[k])
        plt.axis('off')
        data_a = fig2array(fig)

        if disp:
            print("Image array shape = ", data_a.shape)
        if visible:
            plt.show()
        else:
            if close_fig_flag:
                plt.close()
            else:
                plt.cla()

        return data_a


def gen_cell_n_nooverlap_beads(bd_on=True,
                               rand_pos_cell=False,
                               r_cell=0.1,  # 0<r_cell<=1
                               r_bd=0.05,  # 0<r_bd<=1
                               max_bd=3,  # max_bd >= 1
                               rand_bead_flag=False,
                               # stat_ext_bd=None, # or {'mean':5, 'std':1}
                               stat_ext_bd={'mean': 2, 'std': 2},
                               bound_flag=True,
                               visible=False,
                               disp=False,
                               fig=None,
                               ax=None):
    """
    Generate cell images
    The PS bead size is 6 um and silica bead is 5 um.
    Lymphoma cell size varies in a larger variance,
    but the mean value is around 9-12 um.
    
    Inputs
    ======
    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 
    """
    if fig is None or ax is None:
        assert fig is None and ax is None
        fig, ax = plt.subplots(figsize=(2, 2))
        # set_axis_bgcolor is not working because of plt.axis('off')
        # ax.set_axis_bgcolor('red')
        close_fig_flag = True
    else:
        close_fig_flag = False
    fig.patch.set_facecolor('black')

    circle_d = {}

    if rand_pos_cell:
        if bound_flag:  # Not generate cells in the boundary
            B = r_cell + 2.0 * r_bd
            pos_cell = B + (1.0 - 2 * B) * np.random.random(2)
        else:
            pos_cell = np.random.random(2)
    else:
        pos_cell = np.array([0.5, 0.5])

    def rand_pos_bd():
        th = np.random.random() * 2 * np.pi
        pos_bd = pos_cell + (r_cell + r_bd) * \
            np.array((np.cos(th), np.sin(th)))
        return pos_bd
    #print( pos_cell, pos_bd)

    circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
    if bd_on:
        if rand_bead_flag:
            final_max_bd = np.random.randint(max_bd) + 1
        else:
            # Now, the number of total beads attached a cell is fixed (not random).
            final_max_bd = max_bd

        for bd_n in range(final_max_bd):
            circle_d["bd{}".format(bd_n)] = \
                plt.Circle(rand_pos_bd(), r_bd, color='w')
    
    if stat_ext_bd is not None:
        #n_ext_bd = np.max((0, int(np.random.randn()*stat_ext_bd['std'] + stat_ext_bd['mean'])))
        n_ext_bd = np.random.randint(stat_ext_bd['mean']+1)
        for ext_bd_n in range(n_ext_bd):
            ext_bd_pos = np.random.rand(2)
            circle_d["ext_bd{}".format(ext_bd_n)] = \
                plt.Circle(ext_bd_pos, r_bd, color='w')

    for k in circle_d.keys():
        ax.add_artist(circle_d[k])
    plt.axis('off')
    data_a = fig2array(fig)

    if disp:
        print("Image array shape = ", data_a.shape)
    if visible:
        plt.show()
    else:
        if close_fig_flag:
            plt.close()
        else:
            plt.cla()

    return data_a


def gen_cell_db_center_cell(N=5, rand_pos_cell=False,
                            extra_bead_on=True,
                            max_bd=3,
                            flag_no_overlap_beads=False,
                            disp=False):
    """
    db_l = gen_cell_db(N=5, rand_pos_cell=False, extra_bead_on=True, disp=False) 
    Generate cell_db
    
    Inputs
    ======
    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 
    """


    cellgen = CELL(flag_no_overlap_beads=flag_no_overlap_beads)
    fig, ax = plt.subplots(figsize=(2, 2))
    # ax.set_axis_bgcolor('red')    

    if extra_bead_on:
        stat_ext_bd = {'mean': 5, 'std': 1}
    else:
        stat_ext_bd = None

    db_l = []

    for i in range(N):
        if disp:
            print(i, end=",")
        # no_beads is circulated from 0 to max_bd-1
        # Hence, gen_cell_no_beads should be prepared.
        n_beads = i % max_bd
        cellbd_img = cellgen.gen(bd_on=True,
                                 rand_pos_cell=rand_pos_cell,
                                 # max_bd is repeated from 0 to max_bd
                                 max_bd=n_beads,
                                 rand_bead_flag=False,
                                 fig=fig, ax=ax,
                                 stat_ext_bd=stat_ext_bd)

        # cellbd_img = gen_cell_n_beads(bd_on=True,
        #                              rand_pos_cell=rand_pos_cell,
        #                              # max_bd is repeated from 0 to max_bd
        #                              max_bd=n_beads,
        #                              rand_bead_flag=False,
        #                              fig=fig, ax=ax,
        #                              stat_ext_bd=stat_ext_bd)

        db_l.append(cellbd_img[:, :, 0])  # No RGB Info

    plt.close(fig)
    print("The end.")
    return db_l

def save_cell_db_center_cell(db_l, max_bd, fname_gz="sheet.gz/cell_db.cvs.gz"):
    """
    Each image include a cell at the center location and
    the numbers of beads in a cell is equally distributed
    circulately. That is, 0 beads, 1 beads, ..., max_bd are repeated 
    for all images. 
    """

    df_l = []
    celltype = 0
    for i, db in enumerate(db_l):
        df_i = pd.DataFrame()
        df_i["ID"] = [i] * np.prod(db.shape)
        df_i["n_beads"] = [i % max_bd] * np.prod(db.shape)
        df_i["x"] = np.repeat(np.arange(db.shape[0]), db.shape[1])
        df_i["y"] = list(range(db.shape[1])) * db.shape[0]
        df_i["image"] = db.reshape(-1)
        celltype ^= 1
        df_l.append(df_i)
    cell_df = pd.concat(df_l, ignore_index=True)
    cell_df.to_csv(fname_gz, index=False, compression='gzip')
    return cell_df


def _gen_save_cell_db_r0(N=5, fname_gz="sheet.gz/cell_db.cvs.gz",
                     extra_bead_on=True, rand_pos_cell=False,
                     max_bd=3,
                     classification_mode="Cancer_Normal_Cell",
                     disp=False):
    """
    - Image show without pausing is needed. (Oct 31, 2016)
    
    Parameters
    ==========
    rand_pos_cell, Default=False
    If it is True, the position of cell is varied
    Otherwise, the position is fixed to be the center (0,0). 

    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 

    classification_mode, string, default="Cancer_Normal"
    if it is "Cancer_Normal_Cell", this function classifies cancer or normal. 
    If it is "Center_Cell", this fucntion classifies numer of beads in each cell. 
    In this case, the number of beads in cells are equaly distributed 
    from 0 to max_bd. For example, if N=100 & max_bd=4, 0-beads,
    1-beads, 2-beads and 3-beads cell images are repeated 25 times.  
    """

    def save(save_fn, db_l, max_bd=None, fname_gz=None):
        fname_gz_fold, fname_gz_file = os.path.split(fname_gz)
        os.makedirs(fname_gz_fold, exist_ok=True)
        if max_bd is None:
            cell_df = save_fn(db_l, fname_gz=fname_gz)
        else:
            cell_df = save_fn(db_l, max_bd, fname_gz=fname_gz)
        return cell_df

    if classification_mode == "Cancer_Normal_Cell":
        db_l = gen_cell_db(N, rand_pos_cell=rand_pos_cell,
                           extra_bead_on=extra_bead_on,
                           max_bd=max_bd,
                           # classification_mode=classification_mode,
                           disp=disp)
        if disp:
            print("Saving...")
        cell_df = save(save_cell_db, db_l, fname_gz=fname_gz)

    elif classification_mode == "Center_Cell":
        assert int(N % max_bd) == 0, "N % max_bd should zero in the Center_Cell mode" 
        db_l = gen_cell_db_center_cell(N, rand_pos_cell=rand_pos_cell,
                                       extra_bead_on=extra_bead_on,
                                       max_bd=max_bd,
                                       disp=disp)
        if disp:
            print("Saving...")
        cell_df = save(save_cell_db_center_cell, db_l, max_bd,
                       fname_gz=fname_gz)
    else:
        raise ValueError("classification_mode = {} is not supported.".format(classification_mode))    

    return cell_df


def gen_save_cell_db(N=5, fname_gz="sheet.gz/cell_db.cvs.gz",
                     extra_bead_on=True, rand_pos_cell=False,
                     max_bd=3,
                     classification_mode="Cancer_Normal_Cell",
                     flag_no_overlap_beads=False,
                     disp=False):
    """
    - Image show without pausing is needed. (Oct 31, 2016)
    
    Parameters
    ==========
    rand_pos_cell, Default=False
    If it is True, the position of cell is varied
    Otherwise, the position is fixed to be the center (0,0). 

    max_bd, int, default=3
    The number of the maximum beads attached to a cell. 

    classification_mode, string, default="Cancer_Normal"
    if it is "Cancer_Normal_Cell", this function classifies cancer or normal. 
    If it is "Center_Cell", this fucntion classifies numer of beads in each cell. 
    In this case, the number of beads in cells are equaly distributed 
    from 0 to max_bd. For example, if N=100 & max_bd=4, 0-beads,
    1-beads, 2-beads and 3-beads cell images are repeated 25 times.  
    """

    def save(save_fn, db_l, max_bd=None, fname_gz=None):
        fname_gz_fold, fname_gz_file = os.path.split(fname_gz)
        os.makedirs(fname_gz_fold, exist_ok=True)
        if max_bd is None:
            cell_df = save_fn(db_l, fname_gz=fname_gz)
        else:
            cell_df = save_fn(db_l, max_bd, fname_gz=fname_gz)
        return cell_df

    if classification_mode == "Cancer_Normal_Cell":
        db_l = gen_cell_db(N, rand_pos_cell=rand_pos_cell,
                           extra_bead_on=extra_bead_on,
                           max_bd=max_bd,
                           # classification_mode=classification_mode,
                           disp=disp)
        if disp:
            print("Saving...")
        cell_df = save(save_cell_db, db_l, fname_gz=fname_gz)

    elif classification_mode == "Center_Cell":
        assert int(N % max_bd) == 0, "N % max_bd should zero in the Center_Cell mode" 
        db_l = gen_cell_db_center_cell(N, rand_pos_cell=rand_pos_cell,
                                       extra_bead_on=extra_bead_on,
                                       max_bd=max_bd,
                                       flag_no_overlap_beads=flag_no_overlap_beads,
                                       disp=disp)
        if disp:
            print("Saving...")
        cell_df = save(save_cell_db_center_cell, db_l, max_bd,
                       fname_gz=fname_gz)
    else:
        raise ValueError("classification_mode = {} is not supported.".format(classification_mode))    

    return cell_df


class obj:
    def __init__(self, r, L=144):
        """
        The PS bead size is 6 um and silica bead is 5 um.
        Lymphoma cell size varies in a larger variance,
        but the mean value is around 9-12 um.
        """
        # Initial values
        self.Lx, self.Ly = L, L
        self.downsamples = 4
        self.d_um = 2.2 / self.downsamples

        # Input and generated values
        self.r = r
        self.r_pixels_x = self.r * self.Lx
        self.r_pixels_y = self.r * self.Ly
        self.r_x_um = self.r_pixels_x * self.d_um
        self.r_y_um = self.r_pixels_y * self.d_um


def get_h2d(nx, ny, l=700, z=0.5, dx=0.8, dy=0.8):
    """
    1D Freznel Differaction Formulation
    Input
    =====
    x, np.array
    x position

    z, np.array
    hight

    l, float
    lambda
    """
    k = 2.0 * np.pi / l

    x_vec = (np.arange(1, nx+1).reshape(1, -1) - nx/2)*dx
    x = np.dot(np.ones((nx, 1)), x_vec)
    y_vec = (np.arange(1, ny+1).reshape(-1, 1) - ny/2)*dy
    y = y_vec * np.ones((1, ny))
    
    #k = 2.0 * np.pi / l
    return np.exp(1j * k * z) / (1j * l * z) * np.exp((1j * k / (2 * z)) * 
                    (np.power(x, 2) + np.power(y, 2)))

def get_h2d_inv(nx, ny, l=700, z=0.5, dx=0.8, dy=0.8):
    """
    1D Freznel Differaction Formulation
    Input
    =====
    x, np.array
    x position

    z, np.array
    hight

    l, float
    lambda
    """
    k = 2.0 * np.pi / l

    x_vec = (np.arange(1, nx+1).reshape(1, -1) - nx/2)*dx
    x = np.dot(np.ones((nx, 1)), x_vec)
    y_vec = (np.arange(1, ny+1).reshape(-1, 1) - ny/2)*dy
    y = y_vec * np.ones((1, ny))
    
    #k = 2.0 * np.pi / l
    return np.exp(-1j * k * z) / (1j * l * z) * np.exp((-1j * k / (2 * z)) * 
                    (np.power(x, 2) + np.power(y, 2)))


# Freznel
def get_h(ny, nx, z_mm=0.5, dx_um=2.2, dy_um=2.2, l_nm=405):
    """
    1D Freznel Differaction Formulation
    Input
    =====
    x, np.array
    x position

    z, np.array
    hight

    l, float
    lambda

    "The PS bead size is 6 um and silica bead is 5 um. 
    Lymphoma cell size varies in a larger variance, but the mean value is around 9-12 um."
    """
    # nano-meter to micro-meter transform (nm -> um)
    l_um = l_nm / 1000
    z_um = z_mm * 1000

    x_vec_um = (np.arange(1, nx+1).reshape(1, -1) - nx/2)*dx_um
    x_um = np.dot(np.ones((ny, 1)), x_vec_um)
    y_vec_um = (np.arange(1, ny+1).reshape(-1, 1) - ny/2)*dy_um
    y_um = y_vec_um * np.ones((1, nx))
    
    return np.exp((1j * np.pi) / (l_um * z_um) * 
                    (np.power(x_um, 2) + np.power(y_um, 2)))

def get_h_inv(ny, nx, z_mm=0.5, dx_um=2.2, dy_um=2.2, l_nm=405):
    """
    1D Freznel Differaction Formulation
    Input
    =====
    x, np.array
    x position

    z, np.array
    hight

    l, float
    lambda

    "The PS bead size is 6 um and silica bead is 5 um. 
    Lymphoma cell size varies in a larger variance, but the mean value is around 9-12 um."
    """
    # nano-meter to micro-meter transform (nm -> um)
    l_um = l_nm / 1000
    z_um = z_mm * 1000

    x_vec_um = (np.arange(1, nx+1).reshape(1, -1) - nx/2)*dx_um
    x_um = np.dot(np.ones((ny, 1)), x_vec_um)
    y_vec_um = (np.arange(1, ny+1).reshape(-1, 1) - ny/2)*dy_um
    y_um = y_vec_um * np.ones((1, nx))
    
    return np.exp((-1j * np.pi) / (l_um * z_um) * 
                    (np.power(x_um, 2) + np.power(y_um, 2)))

def fd_conv(Img_xy, h2d, mode ='same'):
    #return convolve2d(Img_xy, h2d, mode=mode)
    return fftconvolve(Img_xy, h2d, mode=mode)

def cell_fd_info(cell_df):
    Lx = cell_df['x'].max() + 1
    Ly = cell_df['y'].max() + 1
    Limg = cell_df['ID'].max() + 1
    #print( Lx, Ly, Limg)

    return Limg, Lx, Ly


def cell_fd_conv(cell_df, h144=None):
    Limg, Lx, Ly = cell_fd_info(cell_df)
    if h144 is None:
        h144 = get_h2d(Lx, Ly, l=405, z=0.5, dx=2.2/4, dy=2.2/4)

    cell_img_fd_l = []
    for l in range(Limg):
        cell_img = cell_df[cell_df["ID"] == l]["image"].values.reshape(Lx, Ly)
        #cell_img_fd = fd_conv(cell_img, h144)
        cell_img_fd = fftconvolve(cell_img, h144, mode='same')
        cell_img_fd_l.append(cell_img_fd)

    cell_img_fd_a = np.array(cell_img_fd_l)
    #print( cell_img_fd_a.shape)

    return cell_img_fd_a


def cell_fd_extention(fname_org='sheet.gz/cell_db.cvs.gz', camera_bit_resolution=14):
    cell_df = pd.read_csv(fname_org)
    Limg, Lx, Ly = cell_fd_info(cell_df)

    cell_df_ext = cell_df.copy()

    # Fresnel diffraction
    cell_img_fd_a = cell_fd_conv(cell_df)
    cell_df_ext['freznel image'] = cell_img_fd_a.reshape(-1)

    # max_v, min_v = np.max(cell_df["image"]), np.min(cell_df["image"])
    cell_img_fd_a_2d = cell_img_fd_a.reshape(Limg, -1)
    cell_img_fd_a_2d_scale = preprocessing.minmax_scale(
        np.abs(cell_img_fd_a_2d)) * (2**camera_bit_resolution)
    cell_img_fd_a_2d_scale_200x144x144 = cell_img_fd_a_2d_scale.reshape(
        Limg, Lx, Ly).astype(int)
    cell_df_ext[
        'mag freznel image'] = cell_img_fd_a_2d_scale_200x144x144.reshape(-1)

    return cell_df_ext


def cell_fd_ext_save(fname_org='sheet.gz/cell_db100.cvs.gz',
                     fname_ext='sheet.gz/cell_fd_db100.cvs.gz'):

    cell_df_ext = cell_fd_extention(fname_org)

    # Save data
    cell_df_ext.to_csv(fname_ext, index=False, compression='gzip')
    
    return cell_df_ext


class CELL_FD_EXT():
    def __init__(self, fname_org, h2d=None, h2d_inv=None):
        cell_df = pd.read_csv(fname_org)
        Limg, Lx, Ly = cell_fd_info(cell_df)

        if h2d is None:
            h2d = get_h(Ly, Lx, z_mm=0.5, dx_um=2.2/4, dy_um=2.2/4, l_nm=405)
        if h2d_inv is None:
            h2d_inv = get_h_inv(Ly, Lx, z_mm=0.5, dx_um=2.2/4, dy_um=2.2/4, l_nm=405)

        self.fname_org = fname_org
        self.h2d = h2d
        self.h2d_inv = h2d_inv
    
    def save(self):
        fname_org = self.fname_org
        fname_ext = fname_org[:-7] + '_fd' + fname_org[-7:]
        print('fname_ext is', fname_ext)

        cell_df_ext = self.extention()

        # Save data
        cell_df_ext.to_csv(fname_ext, index=False, compression='gzip')
        
        return cell_df_ext

    def extention(self, camera_bit_resolution=14):
        fname_org = self.fname_org
        h2d = self.h2d

        cell_df = pd.read_csv(fname_org)
        Limg, Lx, Ly = cell_fd_info(cell_df)

        cell_df_ext = cell_df.copy()

        # Fresnel diffraction
        cell_img_fd_a = cell_fd_conv(cell_df, h2d)
        cell_df_ext['freznel image'] = cell_img_fd_a.reshape(-1)

        # max_v, min_v = np.max(cell_df["image"]), np.min(cell_df["image"])
        cell_img_fd_a_2d = cell_img_fd_a.reshape(Limg, -1)
        cell_img_fd_a_2d_scale = preprocessing.minmax_scale(
            np.abs(cell_img_fd_a_2d)) * (2**camera_bit_resolution)
        cell_img_fd_a_2d_scale_200x144x144 = cell_img_fd_a_2d_scale.reshape(
            Limg, Lx, Ly).astype(int)
        cell_df_ext[
            'mag freznel image'] = cell_img_fd_a_2d_scale_200x144x144.reshape(-1)

        return cell_df_ext

      
#Deep Learning
def run_dl_mgh_params_1cl_do(X, y, Lx, Ly, nb_epoch=5000,     
                      batch_size = 128,
                      nb_classes = 2):

    # input image dimensions
    img_rows, img_cols = Lx, Ly
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (50, 50)
    # convolution kernel size
    kernel_size = (20, 20)

    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=0)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=0, validation_data=(X_test, Y_test)) #, callbacks=[earlyStopping])
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    kkeras.plot_acc( history)
    plt.show()
    kkeras.plot_loss( history)        


#Deep Learning
def run_dl_mgh_params_1cl_bn(X, y, Lx, Ly, nb_epoch=5000,
                      batch_size = 128,
                      nb_classes = 2):

    # input image dimensions
    img_rows, img_cols = Lx, Ly
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (50, 50)
    # convolution kernel size
    kernel_size = (20, 20)

    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=0)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Conv2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=0, validation_data=(X_test, Y_test)) #, callbacks=[earlyStopping])
    score = model.evaluate(X_test, Y_test, verbose=0)

    Y_test_pred = model.predict(X_test, verbose=0)
    print('Confusion metrix')
    # y_test_pred = np_utils.categorical_probas_to_classes(Y_test_pred)
    y_test_pred = np.argmax(Y_test_pred, axis=1)
    print(metrics.confusion_matrix(y_test, y_test_pred))

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    kkeras.plot_acc( history)
    plt.show()
    kkeras.plot_loss( history)

run_dl_mgh_params = run_dl_mgh_params_1cl_bn

def run_dl_mgh_params_1cl_bn_do(X, y, Lx, Ly, nb_epoch=5000,     
                      batch_size = 128,
                      nb_classes = 2):

    # input image dimensions
    img_rows, img_cols = Lx, Ly
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (50, 50)
    # convolution kernel size
    kernel_size = (20, 20)

    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=0)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=0, validation_data=(X_test, Y_test)) #, callbacks=[earlyStopping])
    score = model.evaluate(X_test, Y_test, verbose=0)

    Y_test_pred = model.predict(X_test, verbose=0)
    print('Confusion metrix')
    # y_test_pred = np_utils.categorical_probas_to_classes(Y_test_pred)
    y_test_pred = np.argmax(Y_test_pred, axis=1)
    print(metrics.confusion_matrix(y_test, y_test_pred))

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    kkeras.plot_acc( history)
    plt.show()
    kkeras.plot_loss( history)


def run_dl_mgh_params_2cl_bn(X, y, Lx, Ly, nb_epoch=5000,     
                      batch_size = 128,
                      nb_classes = 2):

    # input image dimensions
    img_rows, img_cols = Lx, Ly
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (10, 10)
    # convolution kernel size
    kernel_size = (20, 20)

    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=0)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(5, 5, 5, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(5,5)))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=0, validation_data=(X_test, Y_test)) #, callbacks=[earlyStopping])
    score = model.evaluate(X_test, Y_test, verbose=0)

    Y_test_pred = model.predict(X_test, verbose=0)
    print('Confusion metrix')
    # y_test_pred = np_utils.categorical_probas_to_classes(Y_test_pred)
    y_test_pred = np.argmax(Y_test_pred, axis=1)
    print(metrics.confusion_matrix(y_test, y_test_pred))

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    kkeras.plot_acc( history)
    plt.show()
    kkeras.plot_loss( history)

# function name alias
run_dl_mgh_params_2cl = run_dl_mgh_params_2cl_bn


def run_dl_mgh_params_2cl_bn_do(X, y, Lx, Ly, nb_epoch=5000,     
                      batch_size = 128,
                      nb_classes = 2):

    """
    Dropout is also included after batchnormalization to protect
    overfitting. 
    """

    # input image dimensions
    img_rows, img_cols = Lx, Ly
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (10, 10)
    # convolution kernel size
    kernel_size = (20, 20)

    # the data, shuffled and split between train and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=0)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.1))

    model.add(Convolution2D(5, 5, 5, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=0, validation_data=(X_test, Y_test)) #, callbacks=[earlyStopping])
    score = model.evaluate(X_test, Y_test, verbose=0)

    Y_test_pred = model.predict(X_test, verbose=0)
    print('Confusion metrix')
    # y_test_pred = np_utils.categorical_probas_to_classes(Y_test_pred)
    y_test_pred = np.argmax(Y_test_pred, axis=1)
    print(metrics.confusion_matrix(y_test, y_test_pred))

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    kkeras.plot_acc( history)
    plt.show()
    kkeras.plot_loss( history)

run_dl_mgh_params_2cl_do = run_dl_mgh_params_2cl_bn_do


"""
Fresenel Diffraction with a new approach
"""
def f(x_um, y_um, z_mm=0.5, l_nm=405):
        return np.exp(1j * np.pi * (np.power(x_um, 2) + np.power(y_um,2)) / (l_nm * z_mm))

def cimshow(f_impulse):
    plt.figure(figsize=(7,5))
    plt.subplot(2,2,1)
    plt.imshow(np.real(f_impulse))
    plt.colorbar()
    plt.title('Re{}')
    
    plt.subplot(2,2,2)
    plt.imshow(np.imag(f_impulse))
    plt.colorbar()
    plt.title('Img{}')
    
    plt.subplot(2,2,2+1)
    plt.imshow(np.abs(f_impulse))
    plt.colorbar()
    plt.title('Magnitude')

    plt.subplot(2,2,2+2)
    plt.imshow(np.angle(f_impulse))
    plt.colorbar()
    plt.title('Phase')
    
def xy(MAX_x_um = 55, pixel_um=2.2, oversample_rate=4):
    N = int(MAX_x_um / (pixel_um / oversample_rate))
    x = np.dot(np.ones((N,1)), np.linspace(-MAX_x_um,MAX_x_um,N).reshape(1,-1))
    y = np.dot(np.linspace(-MAX_x_um,MAX_x_um,N).reshape(-1,1), np.ones((1,N)))
    return x, y

def u(x, y, alpha):
    out = np.zeros_like(x)
    out[(y>=-alpha/2)&(y<=alpha/2)&(x>=-alpha/2)&(x<=alpha/2)] = 1.0
    return out

def u_circle(x,y,radius):
    xy2 = np.power(x,2) + np.power(y,2)
    # Since x is already matrix for griding, out shape is copied just from x. 
    # If x is a vector, the shape of out should be redefined to have 2-D form. 
    out = np.zeros_like(x)
    out[xy2<=np.power(radius,2)] = 1.0
    return out


# Code for generation H in frequency domain: H <--> h
# Gbp(n,m) = exp(1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2))
class GenG():
    def upsampling(self, Pow2factor, dx1):
        """
        Utility codes
        """
        dx2 = dx1 / (2**Pow2factor)
        return dx2
    
    def __init__(self, NxNy=(144, 144), Dz_mm=0.5, delta_um = 2.2, UpsampleFactor=2, lambda_nm=405):  
        """
        oversample=2^UpsampleFactor
        """
        delta_m = delta_um * 1e-6
        delta2_m = self.upsampling(UpsampleFactor, delta_m)
        
        Nx, Ny = NxNy        
        dfx = 1/(Nx*delta2_m) 
        dfy = 1/(Ny*delta2_m) 

        x = np.arange(-Ny/2, Ny/2)*dfy
        y = np.arange(-Nx/2, Nx/2)*dfx
        
        self.xv, self.yv = np.meshgrid(x, y)
        self.lambda_m = lambda_nm * 1e-9
        self.k_rad = 2*np.pi/self.lambda_m
        self.Dz_m = Dz_mm * 1e-3
    
    def bp(self):
        x, y = self.xv, self.yv
        l = self.lambda_m
        k = self.k_rad
        Dz = self.Dz_m
        return np.exp(1j * k * Dz * np.sqrt(1-np.power(l*x,2)-np.power(l*y,2)))
    
    def fp(self):
        x, y = self.xv, self.yv
        l = self.lambda_m
        k = self.k_rad
        Dz = self.Dz_m
        return np.exp(-1j * k * Dz * np.sqrt(1-np.power(l*x,2)-np.power(l*y,2)))