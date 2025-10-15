
## 🎯 Цели проекта
Разработка комплексной GPU-библиотеки для обработки сигналов с использованием CUDA.
**Ключевые компоненты:**
- FFT/IFFT примитивы (16-8192 точек)
- Парсер interleaved данных с перекрытием окон
- Корреляция через FFT (до 40 сигналов)
- Математическая статистика

**Метрики успеха:**
- Производительность > cuFFT на 10-20% (малые окна)
- Точность < 0.01% error
- Поддержка NVIDIA + AMD/Intel GPU

---

## 🏗️ Архитектура

### Модули

```
CudaCalc/
├── Interface/              # IGPUProcessor, signal_data.h
├── SignalGenerators/       # 5+ типов генераторов
├── DataContext/            # Управление данными
│   ├── Config/
│   ├── ValidationData/     # JSON для Python
│   └── Models/             # Архив экспериментов
├── Production/             # Production-ready код
│   ├── FFT/                # 10 размеров (16-8192)
│   ├── IFFT/               # 10 размеров (16-8192)
│   ├── Correlation/        # 64-32768, до 40 сигналов
│   ├── Parser/             # Interleaved формат
│   └── Statistics/         # Моменты, анализ, фильтры
├── ModelsFunction/         # Экспериментальные реализации
├── Tester/                 # Профилирование (CUDA Events)
├── Validator/              # Python валидация + matplotlib
└── MainProgram/            # Entry points
```

### Применяемые паттерны

**Архитектурные:** Layered Architecture, Pipes & Filters, Plugin, Repository  
**GoF:** Factory Method, Adapter, Facade, Strategy, Template Method, Observer, Command  
**GRASP:** Information Expert, Controller, Low Coupling, High Cohesion  
**Всего:** 26+ паттернов

### Workflow

```
SignalGenerator → Parser (interleaved) → Parallel Streams (4 потока) →
→ [FFT/IFFT / Correlation] → DataContext (JSON) → Python Validator → Визуализация
```

---

## 🚀 Фазы разработки

### Phase 1: FFT Primitives - Малые окна (16-512)

**Цель:** Базовые FFT для окон 16-512 точек

**Спецификации:**

#### 001: FFT16 Baseline Testing Pipeline
- 3 реализации (WMMA FP16, Shared2D FP32, cuFFT)
- Профилирование (CUDA Events)
- Python Validator + matplotlib
- ModelArchiver (версионирование)
- **Deliverables:** Baseline метрики, fastest алгоритм

#### 002-006: FFT32, FFT64, FFT128, FFT256, FFT512
- Переиспользование лучшего подхода из FFT16
- Оптимизации: butterfly unrolling, memory coalescing, bank conflicts
- **Deliverables:** 6 размеров FFT готовы

---

### Phase 2: FFT Primitives - Большие окна (1024+)

**Цель:** FFT для больших окон (1024-8192)

**Спецификации:**

#### 007: FFT1024
- Исследование: cuFFT vs custom
- Выбор оптимального подхода

#### 008: FFT2048, FFT4096, FFT8192
- Универсальный kernel для 2^n
- Динамическая оптимизация
- **Deliverables:** 4 размера FFT готовы

---

### Phase 3: IFFT - Обратные преобразования

**Цель:** Обратные FFT для всех размеров (16-8192)

**Спецификации:**

#### 009: IFFT 16-512
- Переиспользование forward FFT kernels
- Конъюгирование + нормализация (деление на N)
- Валидация: Forward→Inverse→Forward cycle
- **Deliverables:** 6 размеров IFFT

#### 010: IFFT 1024-8192
- Интеграция с большими FFT
- **Deliverables:** 4 размера IFFT, полная библиотека FFT+IFFT

---

### Phase 4: Парсер и параллельная обработка

**Цель:** Обработка interleaved данных с перекрытием окон

**Концепция:**
```
Входной формат: 4 луча × 1024 точки = 4096
Формат: [луч0_re, луч0_im, луч1_re, луч1_im, ...]

Sliding window:
Итерация 0: FFT[0..15]
Итерация 1: FFT[4..19]   ← overlap 75%
Итерация 2: FFT[8..23]

Параллелизм: 4 CUDA streams одновременно
```

**Спецификации:**

#### 011: Data Parser - Interleaved Format
- Парсинг interleaved лучей
- Преобразование в GPU формат
- Zero-padding

#### 012: Parallel FFT - Sliding Window
- 4 параллельных потока (CUDA streams)
- Смещение на 4 точки
- Асинхронные копирования

#### 013: Multi-Strobe Processing
- Batch processing стробов
- Stream management
- **Deliverables:** Полная цепочка парсинга и обработки

---

### Phase 5: Корреляция через FFT

**Цель:** Быстрая корреляция с опорными сигналами

**Концепция:**
```
Алгоритм:
1. FFT(опорный)
2. FFT(входной)
3. Перемножение: FFT_опорный * conj(FFT_входной)
4. IFFT(произведение) → корреляция
5. Циклическая свёртка
6. Повторение для следующего опорного

Параллелизм: до 3 корреляций одновременно
Batch: до 40 корреляций в запросе
Размеры окон: 64-32768 точек
```

**Спецификации:**

#### 014: Reference Signal Generator
- Генерация множества опорных сигналов
- Форматирование для параллельных вычислений

#### 015: Correlation - Small Windows (64-1024)
- FFT → умножение → IFFT
- 3 параллельных корреляции

#### 016: Correlation - Large Windows (2048-32768)
- Оптимизация под большие данные
- Memory management

#### 017: Correlation Batch Processing
- Очередь до 40 корреляций
- Автоматическое разбиение на циклы по 3

#### 018: Cyclic Convolution
- Циклическая свёртка опорного сигнала
- Caching FFT опорных
- **Deliverables:** Полная система корреляции

---

### Phase 6: Математическая статистика

**Цель:** Аналитические операции

**Спецификации:**

#### 019: Statistical Moments
- Mean, Variance, Skewness, Kurtosis

#### 020: Spectral Analysis
- Power Spectral Density (PSD)
- Энергия сигнала
- Bandwidth estimation

#### 021: Peak Detection
- Локальные максимумы
- Adaptive thresholding
- Peak clustering

#### 022: Digital Filters
- Low/High/Band-pass filters
- Notch filters
- Реализация через FFT
- **Deliverables:** Полный набор аналитических инструментов

---

## 📊 Итоговая структура

### Production Modules (готовые компоненты)

| Модуль | Количество | Описание |
|--------|------------|----------|
| **FFT** | 10 размеров | 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| **IFFT** | 10 размеров | 16-8192 (все размеры FFT) |
| **Корреляция** | 3 модуля | Small (64-1024), Large (2048-32768), Batch (до 40) |
| **Парсер** | 3 модуля | Interleaved parser, Sliding window, Multi-strobe |
| **Статистика** | 4 модуля | Moments, Spectral, Peak detection, Filters |

---

## 🎨 Ключевые реализации

### FFT/IFFT: Три подхода

**A. Tensor Cores (WMMA) - FP16**
- Максимальная производительность
- Half precision
- Для малых окон (16-512)

**B. Shared2D - FP32**
- Универсальная реализация
- Single precision
- Для всех размеров

**C. cuFFT Wrapper**
- Baseline производительности
- Adapter pattern
- Для сравнения

### Корреляция: Параллельная обработка

**3 параллельных корреляции:**
- 3 CUDA streams
- Независимые вычисления
- Максимальная утилизация GPU

**Batch до 40 сигналов:**
- Автоматическое разбиение (40 / 3 = 13.3 → 14 итераций)
- Queue management
- Priority scheduling

### Парсер: Sliding Window

**Перекрытие окон 75%:**
- FFT окно: 16 точек
- Смещение: 4 точки
- Итераций: (4096 - 16) / 4 + 1 = 1021

**4 параллельных потока:**
- Поток 0: FFT[0, 16, 32, ...]
- Поток 1: FFT[4, 20, 36, ...]
- Поток 2: FFT[8, 24, 40, ...]
- Поток 3: FFT[12, 28, 44, ...]

---

## 🔍 Python Валидация

### Зачем отдельно?
✅ Независимость от GPU vendor (работает с NVIDIA, AMD, Intel)  
✅ Гибкость (без пересборки C++)  
✅ Мощные инструменты (scipy, matplotlib)  
✅ Единый валидатор для всех модулей

### Модули валидации

```
Validator/
├── validate_fft.py         # FFT валидация
├── validate_ifft.py        # IFFT валидация
├── validate_correlation.py # Корреляция валидация
├── fft_reference.py        # scipy.fft эталон
├── comparison.py           # Метрики ошибок
├── visualization.py        # matplotlib (3 графика)
└── requirements.txt        # numpy, scipy, matplotlib
```

### Workflow

```
C++ → save JSON (input + gpu_results) → 
→ Python validate_*.py → 
→ scipy (эталон) → 
→ Сравнение + метрики → 
→ Таблица + графики matplotlib
```

### Визуализация (3 графика)
1. Входной сигнал (real, imag, огибающая)
2. GPU спектр/корреляция
3. GPU vs Python (сравнение)

---

## ✅ Критерии приёмки

### Функциональные

**FFT/IFFT:**
- [ ] 10 размеров FFT (16-8192)
- [ ] 10 размеров IFFT (16-8192)
- [ ] Forward→Inverse→Forward cycle < 0.001% error

**Парсер:**
- [ ] Interleaved формат парсится
- [ ] 4 параллельных потока
- [ ] Sliding window overlap 75%
- [ ] Multi-strobe batch processing

**Корреляция:**
- [ ] Окна 64-32768 точек
- [ ] 3 параллельных корреляции
- [ ] Batch до 40 сигналов
- [ ] Циклическая свёртка

**Статистика:**
- [ ] 4 момента (mean, variance, skewness, kurtosis)
- [ ] Spectral analysis (PSD)
- [ ] Peak detection
- [ ] 4 типа фильтров

### Производительность

| Модуль | Метрика | Целевое |
|--------|---------|---------|
| FFT 16-512 | vs cuFFT | +10-20% быстрее |
| FFT 1024+ | vs cuFFT | ±5% |
| IFFT все | vs FFT | Аналогично |
| Корреляция | Throughput | > 1000/сек |
| Парсер | Overhead | < 1% |

### Качество

- [ ] SOLID принципы
- [ ] 26+ паттернов
- [ ] Code review пройден
- [ ] Unit tests > 70%
- [ ] Integration tests
- [ ] No memory leaks
- [ ] Документация полная

---

## 🔐 Риски

| Риск | Митигация |
|------|-----------|
| WMMA не даёт ускорения | Тестируем оба (WMMA + Shared2D) |
| FP16 потеря точности | Python покажет реальную точность |
| 40 корреляций сложно | Поэтапно (3→10→40) |
| AMD/Intel поддержка | Опционально, фокус NVIDIA |
| Memory bandwidth | Pinned memory, async, streams |

---

## 📈 KPI

### Производительность

| Метрика | Целевое | Факт |
|---------|---------|------|
| FFT малые (vs cuFFT) | > 1.1x | TBD |
| FFT большие (vs cuFFT) | ≥ 1.0x | TBD |
| Корреляций/сек | > 1000 | TBD |
| Точность FFT | < 0.01% | TBD |
| Точность IFFT | < 0.01% | TBD |

### Качество

| Метрика | Целевое | Факт |
|---------|---------|------|
| Unit test coverage | > 70% | TBD |
| Linter issues | 0 | TBD |
| Memory leaks | 0 | TBD |
| GPU vendors | 2+ | TBD |

---

## 🗄️ ModelArchiver

**Проблема:** Результаты НЕ ДОЛЖНЫ затираться!

**Решение:** Repository Pattern

```
Models/
├── NVIDIA/
│   ├── FFT/
│   │   └── 16/
│   │       ├── model_v1/  ← Эксперимент 1
│   │       ├── model_v2/  ← Эксперимент 2
│   │       └── model_v3/
│   ├── IFFT/
│   ├── Correlation/
│   └── Statistics/
└── AMD/
    └── ... (аналогично)
```

**API:**
- `save_model()` - сохранить
- `load_model()` - загрузить
- `compare_models()` - сравнить
- `list_models()` - список
- `get_next_version()` - автоинкремент

---

## 👥 Команда

| Роль | Ответственность | FTE |
|------|-----------------|-----|
| Tech Lead / Architect | Архитектура, критические решения | 1.0 |
| Senior CUDA Dev | FFT/IFFT kernels, оптимизация | 1.0 |
| CUDA Dev | Корреляция, статистика | 0.5-1.0 |
| Python Dev | Validator, визуализация | 0.5 |
| DevOps | CMake, CI/CD | 0.3 |
| QA | Тестирование | 0.5 |
| Performance Engineer | Профилирование | 0.3-0.5 |

**Всего:** 3.6-4.3 FTE

---

## 📚 Документы

**Созданные:**
- ✅ spec.md (1500 строк) - Phase 1.1
- ✅ DESIGN_PATTERNS_ANALYSIS.md (932 строки)
- ✅ ROADMAP.md (373 строки)
- ✅ WORK_PLAN_FULL_NO_DATES.md (этот план)
- ✅ WORK_PLAN_COMPACT_NO_DATES.md (компактная версия)

**Создадим:**
- 🔄 specs для Phase 2-6
- 🔄 tasks.md (TODO для каждой фазы)
- 🔄 README.md
- 🔄 API Reference
- 🔄 Examples/

---

## 🚀 Scope проекта

### Phase 1: FFT 16-512
✅ 6 размеров FFT  
✅ 3 реализации (WMMA, Shared2D, cuFFT)  
✅ Python Validator  
✅ ModelArchiver

### Phase 2: FFT 1024-8192
✅ 4 размера FFT  
✅ Универсальный kernel

### Phase 3: IFFT 16-8192
✅ 10 размеров IFFT  
✅ Валидация Forward→Inverse→Forward

### Phase 4: Парсер + потоки
✅ Interleaved parser  
✅ 4 параллельных потока  
✅ Sliding window overlap  
✅ Multi-strobe batch

### Phase 5: Корреляция
✅ Окна 64-32768  
✅ 3 параллельных корреляции  
✅ Batch до 40 сигналов  
✅ Циклическая свёртка

### Phase 6: Статистика
✅ 4 момента  
✅ Spectral analysis  
✅ Peak detection  
✅ 4 типа фильтров

---

## ✅ Итого

**Modules:** 30+ production-ready модулей  
**FFT/IFFT:** 20 размеров (10 FFT + 10 IFFT)  
**Корреляция:** 3 модуля (small, large, batch)  
**Парсер:** 3 модуля  
**Статистика:** 4 модуля  
**Паттерны:** 26+  
**GPU vendors:** NVIDIA + AMD/Intel (опционально)

**Результат:**
- Production-ready библиотека для обработки сигналов
- Превосходство над cuFFT на 10-20% (малые окна)
- Универсальность (NVIDIA, AMD, Intel)
- Полная документация и тесты



