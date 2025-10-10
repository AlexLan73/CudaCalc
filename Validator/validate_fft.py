#!/usr/bin/env python3
"""
FFT Validation Script

Валидирует результаты GPU FFT вычислений, сравнивая с эталоном scipy.

Usage:
    python validate_fft.py                               # Последний файл
    python validate_fft.py --file "2025-10-09_10-30_fft16_wmma_test.json"
    python validate_fft.py --visualize                   # С графиками
    python validate_fft.py --no-plot                     # Только таблица
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from fft_reference import compute_reference_fft
from comparison import compare_results
from visualization import visualize_results


def find_latest_file(data_dir):
    """
    Находит последний файл валидации по дате
    
    Args:
        data_dir: путь к ValidationData/FFT16/
    
    Returns:
        путь к последнему файлу
    """
    files = list(Path(data_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Нет JSON файлов в {data_dir}")
    
    # Сортируем по имени (формат YYYY-MM-DD_HH-MM_...)
    files.sort(reverse=True)
    return files[0]


def load_json_data(file_path):
    """
    Загружает JSON данные
    
    Args:
        file_path: путь к JSON файлу
    
    Returns:
        dict с данными
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_input_signal(data):
    """
    Парсит входной сигнал из JSON
    
    Args:
        data: dict с JSON данными
    
    Returns:
        numpy array (комплексный)
    """
    real = np.array(data['input_signal']['real'])
    imag = np.array(data['input_signal']['imag'])
    return real + 1j * imag


def parse_gpu_results(data):
    """
    Парсит GPU результаты из JSON
    
    Args:
        data: dict с JSON данными
    
    Returns:
        list из numpy arrays (256 окон × 16 точек)
    """
    windows = []
    for window in data['gpu_results']['windows']:
        real = np.array(window['spectrum_real'])
        imag = np.array(window['spectrum_imag'])
        windows.append(real + 1j * imag)
    return windows


def print_report(data, results):
    """
    Выводит отчет валидации в консоль
    
    Args:
        data: dict с JSON данными
        results: dict с результатами сравнения
    """
    print("=" * 60)
    print("=== FFT Validation Report ===")
    print("=" * 60)
    print(f"File: {data['metadata'].get('description', 'N/A')}")
    print(f"GPU: {data['metadata']['gpu_model']}")
    print(f"Algorithm: {data['metadata']['algorithm']}")
    print(f"Date: {data['metadata']['date']} {data['metadata']['time']}")
    print()
    
    config = data['test_config']
    total_points = config['ray_count'] * config['points_per_ray']
    print(f"Input Signal: {total_points} points "
          f"({config['ray_count']} rays × {config['points_per_ray']} points)")
    print(f"FFT Windows: {results['total_windows']} windows × {config['window_fft']} points")
    print()
    print("Reference: scipy.fft.fft (NumPy)")
    print()
    
    print("Comparison Results:")
    print("┌─────────┬──────────────┬──────────────┬────────┐")
    print("│ Window  │ Max Error    │ Mean Error   │ Status │")
    print("├─────────┼──────────────┼──────────────┼────────┤")
    
    # Выводим первые 5, последние 5 окон
    per_window = results['per_window']
    num_windows = len(per_window)
    
    for i in range(min(5, num_windows)):
        status = "PASS" if per_window[i]['passed'] else "FAIL"
        print(f"│ {i:<7} │ {per_window[i]['max']:<12.2e} │ "
              f"{per_window[i]['mean']:<12.2e} │ {status:<6} │")
    
    if num_windows > 10:
        print("│ ...     │ ...          │ ...          │ ...    │")
    
    for i in range(max(num_windows - 5, 5), num_windows):
        status = "PASS" if per_window[i]['passed'] else "FAIL"
        print(f"│ {i:<7} │ {per_window[i]['max']:<12.2e} │ "
              f"{per_window[i]['mean']:<12.2e} │ {status:<6} │")
    
    print("└─────────┴──────────────┴──────────────┴────────┘")
    print()
    
    print("Overall Statistics:")
    print(f"  Max Error (all windows):  {results['overall_max']:.2e}")
    print(f"  Mean Error (all windows): {results['overall_mean']:.2e}")
    print(f"  Tolerance:                {results['tolerance']:.2e}")
    print(f"  Passed Windows:           {results['passed_windows']}/{results['total_windows']} "
          f"({100 * results['passed_windows'] / results['total_windows']:.1f}%)")
    print()
    
    if results['passed_windows'] == results['total_windows']:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='FFT Validation Tool')
    parser.add_argument('--file', type=str, help='Путь к JSON файлу валидации')
    parser.add_argument('--visualize', action='store_true', help='Показать графики')
    parser.add_argument('--no-plot', action='store_true', help='Только таблица (без графиков)')
    parser.add_argument('--window', type=int, default=0, help='Номер окна для визуализации')
    
    args = parser.parse_args()
    
    # Определяем путь к данным
    if args.file:
        file_path = Path(args.file)
    else:
        # Ищем последний файл
        data_dir = Path(__file__).parent.parent / "DataContext" / "ValidationData" / "FFT16"
        file_path = find_latest_file(data_dir)
        print(f"Using latest file: {file_path.name}\n")
    
    # Загружаем данные
    data = load_json_data(file_path)
    
    # Парсим входной сигнал и GPU результаты
    signal = parse_input_signal(data)
    gpu_windows = parse_gpu_results(data)
    
    # Вычисляем эталон через scipy
    window_size = data['test_config']['window_fft']
    reference_windows = compute_reference_fft(signal, window_size)
    
    # Сравниваем результаты
    tolerance = 1e-4  # 0.01%
    results = compare_results(gpu_windows, reference_windows, tolerance)
    
    # Выводим отчет
    print_report(data, results)
    
    # Визуализация (если запрошена)
    if args.visualize and not args.no_plot:
        visualize_results(signal, gpu_windows, reference_windows, args.window)


if __name__ == '__main__':
    main()

