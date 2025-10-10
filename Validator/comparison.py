"""
Сравнение GPU результатов с эталонными
"""
import numpy as np


def compare_results(gpu_windows, reference_windows, tolerance=1e-4):
    """
    Сравнивает GPU результаты с эталонными
    
    Args:
        gpu_windows: list из 256 массивов (GPU)
        reference_windows: list из 256 массивов (scipy)
        tolerance: допустимая относительная ошибка
    
    Returns:
        dict с результатами валидации
    """
    errors = []
    
    for gpu, ref in zip(gpu_windows, reference_windows):
        # Относительная ошибка
        diff = np.abs(gpu - ref)
        ref_mag = np.abs(ref)
        
        # Избегаем деления на ноль
        rel_error = np.where(ref_mag > 1e-10, diff / ref_mag, diff)
        
        max_error = np.max(rel_error)
        mean_error = np.mean(rel_error)
        
        errors.append({
            'max': max_error,
            'mean': mean_error,
            'passed': max_error < tolerance
        })
    
    return {
        'per_window': errors,
        'overall_max': max([e['max'] for e in errors]),
        'overall_mean': np.mean([e['mean'] for e in errors]),
        'tolerance': tolerance,
        'passed_windows': sum([e['passed'] for e in errors]),
        'total_windows': len(errors)
    }

