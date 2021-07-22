# %% Code
import math

import numpy as np
from matplotlib import pyplot as plt

from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import pretty_time, randn_c

def get_noise(num_symbols, SNR_dB=20):
    snr_linear = dB2Linear(SNR_dB)
    noise_power = 1 / snr_linear
    # Noise vector
    n = math.sqrt(noise_power) * randn_c(num_symbols)
    return n

def get_qpsk_modulated_signal(num_symbols=int(1e3)):
    qpsk = QPSK()
    # We need the data to be in the interval [0, M), where M is the
    # number of symbols in the constellation
    data_qpsk = np.random.randint(0, qpsk.M, size=num_symbols)
    modulated_data_qpsk = qpsk.modulate(data_qpsk)
    return modulated_data_qpsk

def show_tx_rx_complex_signal(s, y):
    plt.figure(figsize=(12,5))
    plt.subplot(2,1,1)
    plt.plot(s.real)
    plt.plot(y.real)
    plt.title('Real')

    plt.subplot(2,1,2)
    plt.plot(s.imag)
    plt.plot(y.imag)
    plt.title('Imag')
    plt.show()    
    
def show_qpsk_constallation(received_data_qpsk):
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(received_data_qpsk.real, received_data_qpsk.imag, "*")
    # ax.axis("equal")
    plt.plot(received_data_qpsk.real, received_data_qpsk.imag, "*")
    
def show_tx_rx_qpsk_constallation(s, y):
    plt.figure(figsize=(5,5))
    show_qpsk_constallation(s)
    show_qpsk_constallation(y)
    plt.axis("equal")
    plt.show()

def complex_gaussian_channel(s, SNR_dB=20):
    n = get_noise(len(s),SNR_dB=SNR_dB)
    ch_bs_to_ms = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    y = ch_bs_to_ms * s + n
    return y

def lis_channel(s, SNR_dB=20, num_elements_in_LIS=2):
    n = get_noise(len(s),SNR_dB=SNR_dB)
    
    ch_bs_to_ms = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    
    ch_bs_to_lis = (np.random.randn(num_elements_in_LIS) + 1j*np.random.randn(num_elements_in_LIS)) / np.sqrt(2)
    ch_lis_to_ms = (np.random.randn(num_elements_in_LIS) + 1j*np.random.randn(num_elements_in_LIS)) / np.sqrt(2)
    lis_phase = np.ones(num_elements_in_LIS)
    lis_phase_diag_matrix = np.diag(lis_phase)
    ch_eff_lis = np.dot(np.transpose(ch_lis_to_ms), np.dot(lis_phase_diag_matrix, ch_bs_to_lis))
    
    print("ch_bs_to_ms, ch_bs_to_lis, ch_lis_to_ms, lis_phase, ch_eff_lis:")
    print(ch_bs_to_ms, ch_bs_to_lis, ch_lis_to_ms, lis_phase, lis_phase_diag_matrix, ch_eff_lis)
    
    y = (ch_bs_to_ms + ch_eff_lis) * s + n
    return y

# %% Main code
if __name__ == '__main__':
    s = get_qpsk_modulated_signal(num_symbols=50)
    y = lis_channel(s)

    show_tx_rx_complex_signal(s, y)
    show_tx_rx_qpsk_constallation(s, y)
# %%
