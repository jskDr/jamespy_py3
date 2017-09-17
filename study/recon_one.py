# recon.py
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from skimage import morphology
import joblib
from matplotlib.pyplot import figure, subplot, imshow, title
import pandas as pd

from j3x import jpyx
from medic import kdl


def get_Gbp(NxNy, dfxdfy, k, Dz, lambda_m, sign=1):
    Nx, Ny = NxNy
    dfx, dfy = dfxdfy
    x = np.arange(-Nx / 2, Nx / 2) * dfx
    y = np.arange(-Ny / 2, Ny / 2) * dfy

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    # Cartesian (‘xy’, default)
    fx, fy = np.meshgrid(x, y)

    lp_fx_2 = np.power(lambda_m * fx, 2)
    lp_fy_2 = np.power(lambda_m * fy, 2)
    return np.exp(sign * 1j * k * Dz * np.sqrt(1 - lp_fx_2 - lp_fy_2))


def get_Gfp(NxNy, dfxdfy, k, Dz, lambda_m):
    return get_Gbp(NxNy, dfxdfy, k, Dz, lambda_m, sign=-1)


def _interp2_r0(Data, Pow2factor, kind, disp=False):
    if disp:
        p30 = np.poly1d(np.polyfit([10, 30, 50, 60, 70, 80],
                                   np.log([0.8, 1.3, 6, 12, 28, 56]), 2))
        print('Expectd time for NxN data:', np.exp(p30(Data.shape[0])))

    x = np.arange(Data.shape[1])
    y = np.arange(Data.shape[0])
    xv, yv = np.meshgrid(x, y)
    f = interpolate.interp2d(xv, yv, Data, kind=kind)

    xnew = np.arange(0, Data.shape[1], 1 / (2**Pow2factor))
    ynew = np.arange(0, Data.shape[0], 1 / (2**Pow2factor))
    Upsampled = f(xnew, ynew)

    return Upsampled


def interp2(Data, Pow2factor, kind):
    # kind is not used.
    udata = ndimage.interpolation.zoom(Data, 2 ** Pow2factor)

    # This process is added in order to resemble Matlab interp2
    sxy = 2**Pow2factor - 1
    return udata[sxy:, sxy:]


def upsampling(Data, Pow2factor, dx1):
    """
    - Fast interpolation for a whole image
    http://stackoverflow.com/questions/16983843/fast-interpolation-of-grid-data
    """
    Upsampled = interp2(Data, Pow2factor, 'cubic')
    dx2 = dx1 / (2 ** Pow2factor)

    return Upsampled, dx2


def _update_recon_py_r0(Recon1, support):
    err1 = 1
    # This code should be implemented by Cython
    Constraint = np.ones(Recon1.shape)
    for p in range(Recon1.shape[0]):
        for q in range(Recon1.shape[1]):
            if support[p, q] == 1:
                Constraint[p, q] = np.abs(Recon1[p, q])
                err1 += np.power(np.abs(Recon1[p, q]), 2)
            if np.abs(Recon1[p, q]) > 1:
                Constraint[p, q] = 1
    Recon1_update = Constraint * np.exp(1j * np.angle(Recon1))

    return Recon1_update, err1


def update_recon_py(Recon1, support):
    err1 = 1.0
    Constraint = np.ones(Recon1.shape)
    # cdef int R1y, R1x
    Recon1_abs = np.abs(Recon1)
    Recon1_pwr2 = np.power(Recon1_abs, 2)
    R1y, R1x = Recon1.shape

    for p in range(R1y):
        for q in range(R1x):
            if support[p, q] == 1:
                Constraint[p, q] = Recon1_abs[p, q]
                err1 += Recon1_pwr2[p, q]
            if Recon1_abs[p, q] > 1:
                Constraint[p, q] = 1

    Recon1_update = Constraint * np.exp(1j * np.angle(Recon1))

    return Recon1_update, err1


def update_recon(Recon1, support):
    # return jpyx.update_recon_py(Recon1, support)
    return jpyx.update_recon_pyx(Recon1, support)


def ft2(g, delta):
    return fftshift(fft2(fftshift(g))) * (delta ** 2)


def ift2(G, dfx, dfy):
    Nx = G.shape[1]
    Ny = G.shape[0]
    return ifftshift(ifft2(ifftshift(G))) * Nx * Ny * dfx * dfy


def imshow_GfpGbp(Gfp, Gbp):
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(Gfp), cmap='Greys_r')
    plt.title('abs(Gfp)')

    plt.subplot(2, 2, 2)
    plt.imshow(np.angle(Gfp), cmap='Greys_r')
    plt.title('angle(Gfp)')

    plt.subplot(2, 2, 1 + 2)
    plt.imshow(np.abs(Gbp), cmap='Greys_r')
    plt.title('abs(Gbp)')

    plt.subplot(2, 2, 2 + 2)
    plt.imshow(np.angle(Gbp), cmap='Greys_r')
    plt.title('angle(Gbp)')


class RECON():
    def __init__(self, Dz=6e-4, disp=False, fig=False, save=False):
        """
        Reconstruct real images using Method 1.
        """
        self.disp = disp
        self.fig = fig
        self.save = save

        # set basic parameters
        # Best Dz and Original Lambda are highly important
        # Dz should be searched and lamdda should be used a real value
        self.lambda_m = 405 * 1e-9  # 405nm
        self.Dz = Dz  # mm
        self.delta2 = 2.2e-6
        self.k = 2 * np.pi / self.lambda_m
        self.UpsampleFactor = 2
        self.threshold = 0.09  # other thresholds exist

        self.error = None  

        if self.disp:
            print('lambda_m, Dz =', self.lambda_m, self.Dz)

    def imread(self, xy_range=None,
               f_data='daudi_PS_CD20_postwash.png',
               fold='../Lymphoma_optimal_Data_20161007/'):
        disp = self.disp

        # fold = '../Lymphoma_optimal_Data_20161007/'
        # Data = plt.imread(fold + 'daudi_PS_CD20_postwash.png')
        Data = plt.imread(fold + f_data)
        Ref = plt.imread(fold + 'reference_image.png')

        if xy_range is not None:
            # onvert a vector to four scalars
            minx, maxx, miny, maxy = xy_range
            Data = Data[miny:maxy, minx:maxx]
            Ref = Ref[miny:maxy, minx:maxx]

        bgData = np.mean(Data)
        bgRef = np.mean(Ref)
        NormFactor = bgRef / bgData
        if disp:
            print("NormFacter =", NormFactor)

        subNormAmp = np.sqrt(Data / Ref * NormFactor)

        if self.disp:
            print(Data.shape, Ref.shape, subNormAmp.shape)

        self.Data, self.Ref = Data, Ref
        self.subNormAmp = subNormAmp

    def imshow(self):
        Data, Ref = self.Data, self.Ref
        subNormAmp = self.subNormAmp

        print('Data - Original image')
        plt.imshow(Data, cmap='Greys_r', interpolation='none')
        plt.show()

        print('Ref - Referece image')
        plt.imshow(Ref, cmap='Greys_r', interpolation='none')
        plt.show()

        print('subNormAmp - Normalized image')
        plt.imshow(subNormAmp, cmap='Greys_r', interpolation='none')
        plt.show()

    def set_params(self):
        """
        Set parameters and expand original images using zoom
        """
        # fig = self.fig
        subNormAmp = self.subNormAmp
        delta2 = self.delta2
        k = self.k
        Dz = self.Dz
        lambda_m = self.lambda_m
        UpsampleFactor = self.UpsampleFactor

        delta_prev = delta2

        if UpsampleFactor > 0:
            subNormAmp, delta2 = upsampling(subNormAmp,
                                            UpsampleFactor, delta_prev)

        Nx, Ny = subNormAmp.shape
        delta1 = delta2

        dfx = 1 / (Nx * delta2)
        dfy = 1 / (Ny * delta2)

        Gbp = get_Gbp([Nx, Ny], [dfx, dfy], k, Dz, lambda_m)
        Gfp = get_Gfp([Nx, Ny], [dfx, dfy], k, Dz, lambda_m)

        self.Nx, self.Ny = Nx, Ny
        self.dfx, self.dfy = dfx, dfy
        self.Gbp, self.Gfp = Gbp, Gfp
        self.subNormAmp = subNormAmp
        self.delta1, self.delta2 = delta1, delta2

    def imshow_h(self):
        Gbp, Gfp = self.Gbp, self.Gfp

        plt.figure(figsize=[10, 10])
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(Gbp), cmap='Greys_r', interpolation='none')
        plt.title('Gbp - Maganitute')

        plt.subplot(2, 2, 2)
        plt.imshow(np.angle(Gbp), cmap='Greys_r', interpolation='none')
        plt.title('Gbp - Angle')

        plt.subplot(2, 2, 2 + 1)
        plt.imshow(np.abs(Gfp), cmap='Greys_r', interpolation='none')
        plt.title('Gbf - Maganitute')

        plt.subplot(2, 2, 2 + 2)
        plt.imshow(np.angle(Gfp), cmap='Greys_r', interpolation='none')
        plt.title('Gbf - Angle')


    def _imshow_GfpGbp_r0(self):
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(self.Gfp), cmap='Greys_r')
        plt.title('abs(Gfp)')

        plt.subplot(2, 2, 2)
        plt.imshow(np.angle(self.Gfp), cmap='Greys_r')
        plt.title('angle(Gfp)')

        plt.subplot(2, 2, 1 + 2)
        plt.imshow(np.abs(self.Gbp), cmap='Greys_r')
        plt.title('abs(Gbp)')

        plt.subplot(2, 2, 2 + 2)
        plt.imshow(np.angle(self.Gbp), cmap='Greys_r')
        plt.title('angle(Gbp)')

    def imshow_GfpGbp(self):
        imshow_GfpGbp(self.Gfp, self.Gbp)

    def imshow_recon(self):
        plt.subplot(1, 2, 1)
        plt.imshow(self.Modulus, cmap='rainbow_r')
        plt.title('Modulus')
        plt.subplot(1, 2, 2)
        plt.imshow(self.Phase_Revise, cmap='rainbow_r')
        plt.title(r'Revised Phase: $|\theta-E[\theta]|$')

    def imshow_recon_all(self):
        plt.subplot(1, 4, 1)
        plt.imshow(self.subNormAmp, cmap='Greys_r')
        plt.title('Expanded Input: subNormAmp')

        plt.subplot(1, 4, 2)
        plt.imshow(self.Modulus, cmap='Greys_r')
        plt.title('Modulus')

        plt.subplot(1, 4, 3)
        plt.imshow(self.Phase, cmap='Greys_r')
        plt.title('Phase')

        plt.subplot(1, 4, 4)
        plt.imshow(self.Phase_Revise, cmap='rainbow_r')
        plt.title('Phase_Revise: ABS(ph-E(ph))')

    def imshow_recon_ri(self):
        plt.subplot(1, 2, 1)
        plt.imshow(np.real(self.ReconImage), cmap='Greys_r')
        plt.title('Real')

        plt.subplot(1, 2, 2)
        plt.imshow(np.imag(self.ReconImage), cmap='Greys_r')
        plt.title('Image')

    def ready(self, xy_range=None):
        print('Read data and reference images')
        self.imread(xy_range=xy_range)
        if self.fig:
            print('Show images')
            self.imshow()

        print('Set Params')
        self.set_params()
        if self.fig:
            print('Show trannsfer functions in the freqeuncy domain')
            self.imshow_h()


class Simulator():
    def __init__(self, NxNy=(144, 144)):
        """
        Diffraction and Reconstruction using synthesized image
        """
        Nx, Ny = NxNy
        delta2 = 2.2e-6 / 4
        lambda_m = 405 * 1e-9  # 405nm
        Dz = 6e-4  # mm
        k = 2 * np.pi / lambda_m

        self.delta2 = delta2
        self.delta1 = self.delta2
        self.dfxdfy = (1 / (Nx * delta2), 1 / (Nx * delta2))

        dfxdfy = self.dfxdfy
        self.Gbp = get_Gbp(NxNy, dfxdfy, k, Dz, lambda_m)
        self.Gfp = get_Gfp(NxNy, dfxdfy, k, Dz, lambda_m)

    def diffract_full(self, Original):
        """
        Diffract an input image
        """
        # Input = Img[1]
        delta1 = self.delta1
        dfx, dfy = self.dfxdfy
        Gfp = self.Gfp

        F1 = ft2(Original, delta1)
        Input = ift2(F1 * Gfp, dfx, dfy)

        return Input

    def diffract(self, Original):
        Input = self.diffract_full(Original)
        return np.abs(Input)

    def reconstruct(self, AbsInput):
        Gbp = self.Gbp
        delta2 = self.delta2
        dfx, dfy = self.dfxdfy

        # Input = np.abs(Output)
        F2 = ft2(AbsInput, delta2)
        Recon = ift2(F2 * Gbp, dfx, dfy)

        return Recon

    def run(self, Original):
        """
        diffract original image and recon.
        ALso, the results will be shown
        """
        AbsInput = self.diffract(Original)
        Recon = self.reconstruct(AbsInput)

        figure(figsize=(3 * 3 + 2, 3))
        subplot(1, 3, 1)
        imshow(Original, 'Greys_r')
        title('Original')
        subplot(1, 3, 2)
        imshow(AbsInput, 'Greys_r')
        title('Hologram: |Diffraction|')
        subplot(1, 3, 3)
        imshow(np.abs(Recon), 'Greys_r')
        title('Modulus: |Recon|')

    def run_complex(self, Original):
        """
        diffract original image and recon.
        ALso, the results will be shown
        """
        Input = self.diffract_full(Original)
        Recon = self.reconstruct(Input)

        figure(figsize=(3 * 3 + 2, 3))
        subplot(1, 4, 1)
        imshow(Original, 'Greys_r')
        title('Original')
        subplot(1, 4, 2)
        imshow(np.abs(Input), 'Greys_r')
        title('|Diffraction|')
        subplot(1, 4, 3)
        imshow(np.angle(Input), 'Greys_r')
        title('Angle(Diffraction)')
        subplot(1, 4, 4)
        imshow(np.abs(Recon), 'Greys_r')
        title('Modulus: |Recon|')


class SimulatorAll(Simulator):
    def __init__(self, fname_org='sheet.gz/cell_db100_no_extra_beads.cvs.gz'):
        """
        Read all image and prepare for converting
        """
        self.cell_df = pd.read_csv(fname_org)
        Limg, Lx, Ly = kdl.cell_fd_info(self.cell_df)

        super().__init__((Lx, Ly))
        self.Img = self.cell_df["image"].values.reshape(Limg, Lx, Ly)

    def diffract(self):
        """
        diffract all image and save to the memory
        """
        Img = self.Img
        hologram_l = []
        for img in Img:
            hologram = super().diffract(img)
            hologram_l.append(hologram)

        return hologram_l

    def reconstruct(self, hologram_l):
        recon_l = []
        for hologram in hologram_l:
            recon = super().reconstruct(hologram)
            recon_l.append(recon)

        return recon_l

    def run_save(self,
                 fname_ext='sheet.gz/cell_fd_db100_no_extra_beads.cvs.gz'):
        hologram_l = self.diffract()
        recon_l = self.reconstruct(hologram_l)

        cell_df_ext = self.cell_df.copy()
        cell_df_ext['hologram'] = np.reshape(hologram_l, -1)
        cell_df_ext['recon_abs'] = np.abs(np.reshape(recon_l, -1))
        cell_df_ext['recon_angle'] = np.angle(np.reshape(recon_l, -1))

        print('Save to', fname_ext)
        cell_df_ext.to_csv(fname_ext, index=False, compression='gzip')


class SimulatorX(SimulatorAll):
    def __init__(self, X):
        """
        Simulate with X data instead of filename
        """
        # self.cell_df = pd.read_csv(fname_org)
        Limg, Lx, Ly = X.shape

        super(SimulatorAll, self).__init__((Lx, Ly))
        self.Img = X

    def run_return(self):
        hologram_l = self.diffract()
        recon_l = self.reconstruct(hologram_l)

        X_hologram = np.array(hologram_l)
        X_recon = np.array(recon_l)

        return X_hologram, X_recon


class SimulatorXy_abs(SimulatorX):
    def __init__(self, fname_Xy='Xy.npy'):
        """
        Load data from a file instead of a variable
        """
        Xy = np.load(fname_Xy).item()
        self.X = np.abs(Xy['X'])
        self.y = Xy['y']
        self.fname_Xy = fname_Xy

        super().__init__(self.X)

    def run_save(self):
        X_hologram, X_recon = self.run_return()

        Xy_holo = {'X': self.X, 'y': self.y,
                   'X_hologram': X_hologram,
                   'X_recon': X_recon}

        fname_out = self.fname_Xy[:-4] + '_holo_abs' + self.fname_Xy[-4:]

        print('Saving to', fname_out)
        np.save(fname_out, Xy_holo)


class SimulatorXy(SimulatorX):
    def __init__(self, fname_Xy='Xy.npy'):
        """
        Load data from a file instead of a variable
        """
        Xy = self.load_Xy(fname_Xy)
        self.X = Xy['X']
        self.y = Xy['y']
        self.fname_Xy = fname_Xy

        super().__init__(self.X)

    def load_Xy(self, fname_Xy):
        return np.load(fname_Xy).item()

    def run_save(self):
        X_hologram, X_recon = self.run_return()

        Xy_holo = {'X': self.X, 'y': self.y,
                   'X_hologram': X_hologram,
                   'X_recon': X_recon}

        fname_out = self.fname_Xy[:-4] + '_holo' + self.fname_Xy[-4:]

        print('Saving to', fname_out)
        self.save_Xy(fname_out, Xy_holo)

    def save_Xy(self, fname_out, Xy_holo):
        np.save(fname_out, Xy_holo)


class SimulatorXy_joblib(SimulatorXy):
    def __init__(self, fname_Xy='Xy_joblib.npy'):
        """
        Load data from a file instead of a variable
        We use joblib for large files.
        """
        super().__init__(fname_Xy)

    def load_Xy(self, fname_Xy):
        return joblib.load(fname_Xy)

    def save_Xy(self, fname_out, Xy_holo):
        joblib.dump(Xy_holo, fname_out)
