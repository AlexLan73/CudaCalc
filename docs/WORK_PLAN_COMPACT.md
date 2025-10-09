# План работ: FFT16 Baseline Pipeline (Компактная версия)

**Проект:** CudaCalc Phase 1.1  
**Дата:** 09 октября 2025  
**Автор:** AlexLan73  
**Срок:** 14-17 рабочих дней (3 недели)

---

## 🎯 Цели

Разработка базовой системы тестирования GPU-ускоренных FFT алгоритмов с профилированием и валидацией.

**Ключевые результаты:**
- 3 реализации FFT16 (WMMA FP16, Shared2D FP32, cuFFT)
- Система профилирования (CUDA Events)
- Python валидатор (scipy + matplotlib)
- Архивирование моделей (ModelArchiver)

**Метрики успеха:**
- Compute time < 1.0 ms (256 FFT16, RTX 3060)
- Точность < 0.01% error vs scipy
- Определение fastest алгоритма

---

## 🏗️ Архитектура

### Модули

```
CudaCalc/
├── Interface/              # Интерфейсы (IGPUProcessor)
├── SignalGenerators/       # Генераторы сигналов
├── DataContext/            # Управление данными
│   ├── Config/
│   ├── ValidationData/     # JSON для Python
│   └── Models/             # Архив экспериментов
├── ModelsFunction/         # 3 FFT реализации
├── Tester/                 # Профилирование
├── Validator/              # Python (scipy + matplotlib)
└── MainProgram/            # Entry point
```

### Используемые паттерны

**Архитектурные:** Layered Architecture, Pipes and Filters, Plugin  
**GoF:** Factory Method, Adapter, Facade, Strategy, Template Method  
**GRASP:** Information Expert, Controller, Low Coupling, High Cohesion  
**Всего:** 26+ паттернов

### Workflow

```
SignalGenerator → Tester/Profiler → [FFT WMMA / Shared2D / cuFFT] → 
→ DataContext (save JSON) → Python Validator → Визуализация
```

---

## 📋 План реализации

### Фаза 1: Инфраструктура (2-3 дня)
- CMake для Ubuntu (CUDA 13.x)
- Структура модулей
- Интерфейсы (IGPUProcessor, signal_data.h)
- Создание директорий (Config/, ValidationData/, Models/)

**Deliverables:** CMakeLists.txt, header файлы, структура каталогов

---

### Фаза 2: Генератор сигналов (1 день)
- BaseGenerator (Template Method)
- SineGenerator (период=8, амплитуда=1.0)
- InputSignalData с полем `return_for_validation`
- Unit тесты

**Deliverables:** SignalGenerators/ модуль, тесты

---

### Фаза 3: FFT16 реализации (4-5 дней)

#### День 1-2: FFT16 Shared2D
- Kernel: `__shared__ float2[64][16]`, 4 stages, fftshift
- Wrapper: `FFT16_Shared2D : public IGPUProcessor`

#### День 3-4: FFT16 WMMA (Tensor Cores)
- Kernel: `__shared__ __half2[64][16]`, FP16, Tensor Cores
- Wrapper: `FFT16_WMMA : public IGPUProcessor`

#### День 5: cuFFT Wrapper
- Adapter: `FFT16_cuFFT : public IGPUProcessor`
- cuFFT batch FFT для сравнения

**Deliverables:** 3 реализации FFT16, все компилируются

---

### Фаза 4: Профилирование (2 дня)
- BasicProfiler (CUDA Events: upload/compute/download)
- JSONLogger
  - `Reports/YYYY-MM-DD_HH-MM_profiling.json`
  - `ValidationData/FFT16/YYYY-MM-DD_HH-MM_<algo>_test.json`
- Версионирование по дате (НЕ перезаписывать)

**Deliverables:** Profiler + JSONLogger с 2 форматами

---

### Фаза 5: Python Validator (2-3 дня)

#### День 1: Окружение
- requirements.txt (numpy, scipy, matplotlib)
- Инструкции Ubuntu/Windows

#### День 2: Модули
- `fft_reference.py` (scipy.fft)
- `comparison.py` (метрики ошибок)
- `visualization.py` (3 графика matplotlib)

#### День 3: CLI
- `validate_fft.py` (entry point)
- Аргументы: --file, --visualize, --no-plot
- Автопоиск последнего файла по дате

**Deliverables:** Validator/ модуль (5 Python файлов), README

---

### Фаза 6: ModelArchiver (1-2 дня)
- Repository Pattern для версионирования
- API: save_model, load_model, compare_models, list_models
- Автоинкремент версий (v1 → v2 → v3)
- Сохранение исходников + результатов

**Deliverables:** ModelArchiver класс, интеграция

---

### Фаза 7: Интеграция (1-2 дня)
- Запуск всех 3 реализаций
- Профилирование RTX 3060
- Валидация Python
- Визуализация matplotlib
- Сравнение производительности
- Сохранение лучшей модели

**Deliverables:** Baseline метрики, JSON отчёты, графики

---

## 📊 График

| Фаза | Дни | Статус |
|------|-----|--------|
| 1. Инфраструктура | 2-3 | ⬜ |
| 2. Генератор | 1 | ⬜ |
| 3. FFT реализации | 4-5 | ⬜ |
| 4. Профилирование | 2 | ⬜ |
| 5. Python Validator | 2-3 | ⬜ |
| 6. ModelArchiver | 1-2 | ⬜ |
| 7. Интеграция | 1-2 | ⬜ |
| **ИТОГО** | **14-17** | ⬜ |

---

## 🎨 Три реализации FFT16

### A. Tensor Cores (WMMA) - FP16
- **Цель:** Максимальная производительность
- **Технология:** Half precision, Tensor Cores
- **Memory:** `__shared__ __half2[64][16]`
- **Ожидание:** Быстрее, но менее точно

### B. Shared2D - FP32
- **Цель:** Универсальная реализация
- **Технология:** Классический shared memory
- **Memory:** `__shared__ float2[64][16]`
- **Ожидание:** Точнее, стабильно

### C. cuFFT Wrapper
- **Цель:** Baseline производительности
- **Паттерн:** Adapter (cuFFT → IGPUProcessor)
- **Назначение:** Сравнение (НЕ валидация)

---

## 🔍 Python Валидация

### Зачем отдельно?
- ✅ Независимость от GPU vendor
- ✅ Гибкость (без пересборки C++)
- ✅ Мощные инструменты (scipy, matplotlib)

### Workflow
```
C++ → save JSON (input + gpu_results) → 
→ Python validate_fft.py → 
→ scipy.fft (эталон) → 
→ Сравнение + метрики → 
→ Таблица + графики
```

### Визуализация (3 графика)
1. Входной сигнал (real, imag, огибающая)
2. GPU спектр (magnitude)
3. GPU vs Python (сравнение)

---

## ✅ Критерии приёмки

### Функциональные
- [ ] 3 FFT реализации работают
- [ ] Профилирование измеряет времена
- [ ] Python валидация < 0.01% error
- [ ] JSON с версионированием
- [ ] ModelArchiver сохраняет модели
- [ ] Визуализация работает

### Производительность
- [ ] Compute time < 1.0 ms (RTX 3060)
- [ ] Fastest алгоритм определён
- [ ] Speedup vs cuFFT посчитан

### Качество
- [ ] SOLID принципы
- [ ] 20+ паттернов применено
- [ ] CMake на Ubuntu
- [ ] Нет memory leaks
- [ ] Code review пройден

---

## 🔐 Риски

| Риск | Митигация |
|------|-----------|
| WMMA медленнее Shared2D | Тестируем оба, выбираем лучший |
| FP16 потеря точности | Python покажет реальную точность |
| Сложность WMMA | Примеры CUDA, консультации |
| CMake проблемы | Стандартные пути, Docker |

---

## 📈 KPI

| Метрика | Цель | Факт |
|---------|------|------|
| Compute Time | < 1.0 ms | TBD |
| Точность (WMMA) | < 0.01% | TBD |
| Точность (Shared2D) | < 0.001% | TBD |
| Speedup vs cuFFT | > 1.05x | TBD |
| Срок реализации | 15.5 дней | TBD |

---

## 🗄️ ModelArchiver (Критично!)

**Проблема:** Результаты экспериментов НЕ ДОЛЖНЫ затираться!

**Решение:** Repository Pattern для версионирования

```
Models/NVIDIA/FFT/16/
├── model_2025_10_09_v1/  ← Эксперимент 1
│   ├── fft16_wmma.cu
│   ├── results.json
│   └── validation.json
├── model_2025_10_09_v2/  ← Эксперимент 2
└── model_2025_10_10_v1/  ← Эксперимент 3
```

**API:**
- `save_model()` - сохранить
- `load_model()` - загрузить
- `compare_models()` - сравнить
- `list_models()` - список
- `get_next_version()` - автоинкремент

---

## 🚀 После Phase 1.1

### Следующие FFT размеры (8 недель)
- 002-fft32 (1 неделя)
- 003-fft64 (1 неделя)
- 004-fft128 (1-2 недели)
- 005-fft256 (1-2 недели)
- 006-fft512 (1-2 недели)

**Переиспользование:**
- ✅ Python Validator работает для всех
- ✅ Profiler универсальный
- ✅ ModelArchiver универсальный
- 🔄 Только kernels новые

---

## 📚 Документы

**Созданные:**
- ✅ spec.md (1500 строк)
- ✅ DESIGN_PATTERNS_ANALYSIS.md (932 строки)
- ✅ ROADMAP.md (373 строки)
- ✅ WORK_PLAN_FULL.md (этот документ)

**Создадим:**
- 🔄 tasks.md (TODO для AI)
- 🔄 README.md (quick start)
- 🔄 examples/ (примеры)

---

## 📞 Команда

| Роль | Ответственность | FTE |
|------|-----------------|-----|
| Senior C++/CUDA Dev | FFT kernels | 1.0 |
| Python Dev | Validator | 0.5 |
| DevOps | CMake, CI/CD | 0.3 |
| QA | Тестирование | 0.3 |

---

## ✅ Итого

**Scope:**
- 7 фаз разработки
- 3 FFT реализации
- Python валидация с визуализацией
- Архивирование моделей

**Срок:** 14-17 рабочих дней (3 недели)

**Результат:**
- Baseline метрики FFT16 на RTX 3060
- Fastest алгоритм определён
- Фундамент для остальных FFT размеров

**Инновации:**
- 26+ паттернов проектирования
- Python для независимой валидации
- Система версионирования экспериментов

---

**Версия:** 1.0 (Компактная)  
**Статус:** Готов к реализации  
**Дата:** 09 октября 2025

