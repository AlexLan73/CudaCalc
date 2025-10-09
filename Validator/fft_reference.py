"""
Эталонные вычисления FFT через scipy
"""
import numpy as np
from scipy import fft


def compute_reference_fft(signal, window_size=16):
    """
    Вычисляет эталонный FFT через scipy
    
    Args:
        signal: комплексный массив (4096 точек)
        window_size: размер окна FFT (16)
    
    Returns:
        windows: list из 256 массивов (каждый 16 точек)
    """
    num_windows = len(signal) // window_size
    windows = []
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_data = signal[start:end]
        
        # FFT через scipy
        spectrum = fft.fft(window_data)
        
        # fftshift (как в GPU kernel)
        spectrum_shifted = fft.fftshift(spectrum)
        
        windows.append(spectrum_shifted)
    
    return windows

