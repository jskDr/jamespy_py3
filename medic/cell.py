"""
Codes to generate cell images.
Jan. 30, 2016
Sungjin (James) Kim

To generate cell images and save it as a file, e.g.:
cell_db = cell.gen_save_cell_db( N = N, fname_gz = "../0000_Base/sheet.gz/cell_db{}_no_extra_beads.cvs.gz".format(N),
                                rand_pos_cell = True, extra_bead_on=False, disp=2)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn import preprocessing
from scipy.signal import fftconvolve


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

    circle_d["cell"] = plt.Circle(pos_cell, r_cell, color='w')
    if bd_on:
        for bd_n in range(np.random.randint(max_bd) + 1):
            circle_d["bd{}".format(bd_n)] = \
                plt.Circle(rand_pos_bd(), r_bd, color='w')
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
                 max_bd=5,  # max_bd >= 1
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
        for bd_n in range(np.random.randint(max_bd) + 1):
            circle_d["bd{}".format(bd_n)] = plt.Circle(rand_pos_bd(), r_bd, color='w')
            # circle_d["bd2"] = plt.Circle(rand_pos_bd(), r_bd, color='w')

    if stat_ext_bd is not None:
        n_ext_bd = np.max((0, int(np.random.randn() * stat_ext_bd['std'] + stat_ext_bd['mean'])))
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
        stat_ext_bd={'mean':5, 'std':1}
    else:
        stat_ext_bd=None

    db_l = []
    cell_img_org = gen_cell(
        bd_on=False, rand_pos_cell=rand_pos_cell,
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


def gen_bead_db(N, n_beads=3, bead_2r=5, disp=False):
    """
    In an image, totally n_beads beads will be displayed. 
    """

    fig, ax = plt.subplots(figsize=(2, 2))

    db_l = []
    cell_img_org = gen_cell(
        bd_on=False, rand_pos_cell=rand_pos_cell,
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
    if fname_gz is not None:
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
             max_bd=3, # max_bd >= 1
             rand_bead_flag=False, 
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
        if rand_bead_flag:
            final_max_bd = np.random.randint(max_bd)+1
        else:
            # Now, the number of total beads attached a cell is fixed (not random).
            final_max_bd = max_bd

        for bd_n in range(final_max_bd):
            circle_d["bd{}".format(bd_n)] = plt.Circle(rand_pos_bd(), r_bd, color='w')
    
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


def gen_cell_db_center_cell(N=5, rand_pos_cell=False,
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
        stat_ext_bd={'mean':5, 'std':1}
    else:
        stat_ext_bd=None

    db_l = []

    for i in range(N):
        if disp:
            print(i, end=",")
        
        # no_beads is circulated from 0 to max_bd-1 
        # Hence, gen_cell_no_beads should be prepared. 
        n_beads = i % max_bd
        cellbd_img = gen_cell_n_beads(
            bd_on=True, 
            rand_pos_cell=rand_pos_cell,
            max_bd=n_beads, # max_bd is repeated from 0 to max_bd
            rand_bead_flag=False,
            fig=fig, ax=ax,
            stat_ext_bd=stat_ext_bd)
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
    if fname_gz is not None:
        cell_df.to_csv(fname_gz, index=False, compression='gzip')
    return cell_df


def pd_gen_cell_db(N=5, 
                     extra_bead_on=True, rand_pos_cell=False, 
                     max_bd=3, 
                     classification_mode="Cancer_Normal_Cell",
                     disp=False,
                     fname_gz=None):
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

    if classification_mode == "Cancer_Normal_Cell":
        db_l = gen_cell_db(N, rand_pos_cell=rand_pos_cell,
                           extra_bead_on=extra_bead_on,
                           max_bd=max_bd,
                           # classification_mode=classification_mode,
                           disp=disp)
        cell_df = save_cell_db(db_l, fname_gz=fname_gz)

    elif classification_mode == "Center_Cell":
        assert int(N % max_bd) == 0, "N % max_bd should zero in the Center_Cell mode" 
        db_l = gen_cell_db_center_cell(N, rand_pos_cell=rand_pos_cell,
                                       extra_bead_on=extra_bead_on,
                                       max_bd=max_bd,
                                       disp=disp)
        cell_df = save_cell_db_center_cell(db_l, max_bd, fname_gz=fname_gz)
    else:
        raise ValueError("classification_mode = {} is not supported.".format(classification_mode))    

    return cell_df


def gen_save_cell_db(N=5, fname_gz="sheet.gz/cell_db.cvs.gz",
                     extra_bead_on=True, rand_pos_cell=False,
                     max_bd=3,
                     classification_mode="Cancer_Normal_Cell",
                     disp=False):

    cell_df = pd_gen_cell_db(N=N,
                             extra_bead_on=extra_bead_on,
                             rand_pos_cell=rand_pos_cell,
                             max_bd=max_bd,
                             classification_mode=classification_mode,
                             disp=disp,
                             fname_gz=fname_gz)
    return cell_df


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


class PD_CELL_FD_EXT():
    def __init__(self, cell_df, h2d=None, h2d_inv=None):
        # cell_df = pd.read_csv(fname_org)
        Limg, Lx, Ly = cell_fd_info(cell_df)

        if h2d is None:
            h2d = get_h(Ly, Lx, z_mm=0.5, dx_um=2.2/4, dy_um=2.2/4, l_nm=405)
        if h2d_inv is None:
            h2d_inv = get_h_inv(Ly, Lx, z_mm=0.5, dx_um=2.2/4, dy_um=2.2/4, l_nm=405)

        # self.fname_org = fname_org
        self.cell_df = cell_df
        self.h2d = h2d
        self.h2d_inv = h2d_inv
    
    def extention(self, camera_bit_resolution=14):
        #fname_org = self.fname_org
        cell_df = self.cell_df
        h2d = self.h2d

        #cell_df = pd.read_csv(fname_org)
        Limg, Lx, Ly = cell_fd_info(cell_df)

        cell_df_ext = cell_df.copy()

        # Fresnel diffraction
        cell_img_fd_a = cell_fd_conv(cell_df, h2d)
        cell_df_ext['freznel image'] = cell_img_fd_a.reshape(-1)

        return cell_df_ext