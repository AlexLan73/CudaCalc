# FFT Validator (Python)

Валидатор корректности GPU FFT вычислений через сравнение с эталонной реализацией scipy.

## Структура

- `validate_fft.py` - главный скрипт валидации
- `fft_reference.py` - эталонные вычисления через scipy
- `comparison.py` - сравнение GPU vs scipy результатов
- `visualization.py` - визуализация (matplotlib)
- `requirements.txt` - зависимости Python

## Установка

### Windows

```powershell
# Создать виртуальное окружение
python -m venv venv

# Активировать
.\venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### Ubuntu

```bash
# Установить Python 3 и pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Создать виртуальное окружение
python3 -m venv venv

# Активировать
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

## Использование

```bash
# По умолчанию: читает последний файл из ValidationData/FFT16/
python validate_fft.py

# Указать конкретный файл
python validate_fft.py --file "2025-10-09_10-30_fft16_wmma_test.json"

# С визуализацией
python validate_fft.py --visualize

# Без графиков (только таблица)
python validate_fft.py --no-plot

# Визуализация конкретного окна
python validate_fft.py --visualize --window 5
```

## Выходные данные

### Таблица в консоль
- Сравнение для каждого окна FFT
- Максимальная и средняя ошибка
- Общая статистика
- Статус: PASS/FAIL

### Визуализация (если `--visualize`)
- Subplot 1: Входной сигнал (real, imag, огибающая)
- Subplot 2: GPU спектр
- Subplot 3: GPU vs Python сравнение

## Формат входных данных

JSON файл из `DataContext/ValidationData/FFT16/`:

```json
{
  "metadata": {
    "date": "2025-10-09",
    "time": "10:30:45",
    "gpu_model": "NVIDIA RTX 3060",
    "algorithm": "FFT16_WMMA"
  },
  "test_config": {
    "ray_count": 4,
    "points_per_ray": 1024,
    "window_fft": 16
  },
  "input_signal": {
    "real": [...],
    "imag": [...]
  },
  "gpu_results": {
    "num_windows": 256,
    "windows": [
      {
        "window_id": 0,
        "spectrum_real": [...],
        "spectrum_imag": [...]
      }
    ]
  }
}
```

## Метрика валидации

- Относительная ошибка: `|GPU - scipy| / |scipy|`
- Tolerance: `1e-4` (0.01%)
- PASS: все окна < tolerance
- FAIL: хотя бы одно окно > tolerance

