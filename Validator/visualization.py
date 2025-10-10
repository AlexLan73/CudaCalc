"""
Визуализация входного сигнала и спектров
"""
import matplotlib.pyplot as plt
import numpy as np


def visualize_results(signal, gpu_spectrum, ref_spectrum, window_id=0):
    """
    Визуализация входного сигнала и спектров
    
    Args:
        signal: входной комплексный сигнал (4096 точек)
        gpu_spectrum: GPU результат (256 окон × 16 точек)
        ref_spectrum: Reference результат (256 окон × 16 точек)
        window_id: какое окно показать (по умолчанию 0)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Subplot 1: Входной сигнал
    ax1 = axes[0]
    t = np.arange(len(signal))
    ax1.plot(t, signal.real, 'b-', label='Real', linewidth=1)
    ax1.plot(t, signal.imag, 'r-', label='Imaginary', linewidth=1)
    ax1.plot(t, np.abs(signal), 'g--', label='Envelope |signal|', linewidth=1.5)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Input Signal (4096 points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: GPU спектр
    ax2 = axes[1]
    gpu_mag = np.abs(gpu_spectrum[window_id])
    freq_bins = np.arange(16) - 8  # fftshift: [-8, -7, ..., 7]
    ax2.bar(freq_bins, gpu_mag, color='blue', alpha=0.7, label='GPU FFT')
    ax2.set_xlabel('Frequency bin')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'GPU Spectrum (Window {window_id})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Сравнение
    ax3 = axes[2]
    ref_mag = np.abs(ref_spectrum[window_id])
    diff = np.abs(gpu_mag - ref_mag)
    
    ax3.bar(freq_bins, gpu_mag, color='blue', alpha=0.5, label='GPU')
    ax3.plot(freq_bins, ref_mag, 'rx', markersize=10, label='Python (scipy)', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(freq_bins, diff, 'g-', label='Difference', linewidth=1.5)
    
    ax3.set_xlabel('Frequency bin')
    ax3.set_ylabel('Magnitude')
    ax3_twin.set_ylabel('|GPU - Python|', color='g')
    ax3.set_title('GPU vs Python Comparison')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

