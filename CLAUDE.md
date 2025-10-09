# Проект: CudaCalc

**CUDA-ускоренная библиотека для высокопроизводительных тензорных вычислений**

---

## 📋 Обзор проекта

**CudaCalc** - это production-ready библиотека примитивных GPU-ускоренных функций для быстрой обработки больших сигналов.

### Основная цель
Создать набор **"примитивных"** высокопроизводительных функций для обработки сигналов на GPU, готовых к использованию в закрытом продакшен-приложении.

**Ключевые примитивы:**
- FFT (Быстрое преобразование Фурье)
- IFFT (Обратное БПФ)
- Свёртка (Convolution)
- Корреляция (Correlation)
- Матричные операции
- Другие математические примитивы

### Бизнес-задача
**Быстрая обработка длинных сигналов** - подать большой сигнал (или множество сигналов) и выполнить математические операции с максимальной скоростью.

### Целевая универсальность
Библиотека должна быть **кросс-платформенной**:
- **NVIDIA** (CUDA) - первый приоритет
- **AMD** (ROCm/HIP) - планируется
- **Другие GPU** (OpenCL) - при необходимости

Специализированные ядра под конкретные GPU для максимальной производительности.

### Целевая аудитория
- Продакшен система (закрытая, для личного использования)
- Обработка сигналов в реальном времени
- Высоконагруженные вычислительные задачи

---

## 🛠 Технологический стек

### Языки и фреймворки
- **Язык**: C++17/C++20
- **GPU APIs**:
  - **CUDA 13.x** (NVIDIA) - **PRIMARY** ⭐
  - ROCm/HIP (AMD) - планируется
  - OpenCL (универсальный) - при необходимости
- **Библиотеки**:
  - cuFFT (Fast Fourier Transform для CUDA)
  - rocFFT (для AMD)
  - cuBLAS (Linear Algebra)
  - Thrust (параллельные алгоритмы)

### Целевые платформы
- **Windows** (разработка и отладка)
- **Ubuntu Linux** (разработка, отладка, сравнение производительности)
- Cross-platform build system

### Инструменты сборки
- **Build System**: CMake 3.20+
- **Generator**: Ninja (рекомендуется) / Visual Studio (Windows)
- **Компилятор**: 
  - nvcc (CUDA Toolkit 13.x)
  - MSVC (Windows)
  - GCC/Clang (Linux)

### Целевое железо
**Текущая конфигурация:**
- **GPU**: NVIDIA RTX 3060 (Ampere, Compute Capability 8.6)
- **VRAM**: 12 GB GDDR6
- **RAM**: 32 GB системной памяти
- **CUDA**: версия 13.x

**Планируется:**
- AMD GPU (конкретная модель TBD)

### Тестирование и профилирование
- **Unit Tests**: Google Test (существующие тесты есть)
- **Benchmarking**: Google Benchmark + собственные тесты
- **Profiling**: 
  - NVIDIA Nsight Compute
  - NVIDIA Nsight Systems
  - nvprof (legacy)
  - RenderDoc (при необходимости)

### Контроль качества
- **Linter**: clang-tidy
- **Formatter**: clang-format (Google Style)
- **Static Analysis**: cppcheck
- **Memory Checker**: cuda-memcheck, valgrind

---

## 🏗 Архитектура проекта

### Структура директорий
```
CudaCalc/
├── include/              # Public headers
│   ├── Tensor.hpp
│   ├── TensorFFT.hpp
│   └── CUDAMemory.hpp
├── src/                  # Implementation
│   ├── TensorFFTKernels.cu
│   ├── OptimizedTensorFFTKernels.cu
│   └── Real2DTensorFFTKernels.cu
├── tests/                # Unit tests
├── benchmarks/           # Performance benchmarks
├── docs/                 # Documentation
├── memory/               # Spec-Kit memory (constitution)
├── specs/                # Spec-Kit specifications
└── scripts/              # Utility scripts
```

### Ключевые модули

#### 1. Tensor Module
- Управление тензорными данными
- RAII для CUDA памяти
- Shape и dtype management

#### 2. TensorFFT Module
- Forward/Inverse FFT для real и complex тензоров
- Батчированная обработка
- Plan caching для производительности

#### 3. CUDAMemory Module
- Memory pool для переиспользования аллокаций
- Zero-copy transfers где возможно
- Автоматическая синхронизация и error handling

### Паттерны проектирования
- **RAII**: Все GPU ресурсы управляются через RAII
- **Strategy**: Разные стратегии для Real/Complex FFT
- **Object Pool**: Кэширование cuFFT планов
- **Factory**: Создание тензоров разных типов

---

## 📂 Активные спецификации

<!-- Добавляйте сюда новые фичи по мере разработки -->

### В работе
- **[001-fft16-baseline-pipeline](specs/001-fft16-baseline-pipeline/spec.md)** - Базовая тестовая цепочка FFT16
  - Status: Specification completed ✅
  - Priority: Critical (первая реализация)
  - Modules: SignalGenerators, ModelsFunction (FFT16_WMMA + FFT16_Shared2D), Tester
  - Goal: Сравнить Tensor Cores vs обычный 2D, получить baseline метрики

### Планируется (см. ROADMAP.md)
- **[002-fft32](specs/002-fft32-implementation/)** - FFT окно 32 точки
- **[003-fft64](specs/003-fft64-implementation/)** - FFT окно 64 точки
- **[004-fft128](specs/004-fft128-implementation/)** - FFT окно 128 точек
- **[005-fft256](specs/005-fft256-implementation/)** - FFT окно 256 точек
- **[006-fft512](specs/006-fft512-implementation/)** - FFT окно 512 точек
- И далее по **[ROADMAP.md](ROADMAP.md)** - полный план проекта

### Завершено
- (Пока нет)

---

## 🎯 Важные замечания для AI

### Принципы кодирования

1. **Memory Safety**
   - Всегда использовать RAII для управления CUDA памятью
   - Никогда не использовать raw pointers в публичном API
   - Предпочитать `std::unique_ptr` и `std::shared_ptr`

2. **Error Handling**
   - Все CUDA API вызовы оборачивать в `CUDA_CHECK()` макрос
   - Использовать exceptions для ошибок
   - Graceful degradation где возможно

3. **Performance**
   - **Измерять перед оптимизацией** (профилирование обязательно)
   - Использовать `constexpr` для compile-time вычислений
   - Избегать ненужных синхронизаций (`cudaDeviceSynchronize`)
   - Предпочитать asynchronous API

4. **Code Style**
   - Следовать Google C++ Style Guide
   - Использовать clang-format автоматически
   - Все публичные API документировать в Doxygen стиле
   - Имена: PascalCase для классов, snake_case для функций

5. **Testing**
   - Каждый публичный метод должен иметь unit test
   - Performance regression tests для критичных операций
   - Edge cases обязательны (пустые тензоры, большие размеры)

### Специфика CUDA

```cpp
// ✅ Хорошо
auto tensor = make_tensor<float>({512, 512});
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// ❌ Плохо
float* d_ptr;
cudaMalloc(&d_ptr, size);  // Нет проверки ошибок! Утечка памяти!
```

### Оптимизация

```cpp
// ✅ Хорошо - кэшируем план
class TensorFFT {
    std::unordered_map<size_t, cufftHandle> plan_cache_;
};

// ❌ Плохо - создаем план каждый раз (2ms overhead!)
Tensor forward(const Tensor& input) {
    cufftHandle plan;
    cufftPlan2d(&plan, nx, ny, CUFFT_C2C);  // Медленно!
}
```

---

## 📊 Текущее состояние

### Статус проекта
**Фаза:** Начало разработки с нуля (clean slate)  
**Наследие:** Есть предыдущие наработки и тесты для референса

### Существующие файлы (legacy/reference)
- ✅ `src/TensorFFTKernels.cu` - базовые FFT kernels
- ✅ `src/OptimizedTensorFFTKernels.cu` - оптимизированные версии
- ✅ `src/Real2DTensorFFTKernels.cu` - 2D Real FFT
- ✅ Существующие тесты (будем использовать как референс)

**Подход:** Изучить существующие результаты → Скопировать лучшее ИЛИ переписать с нуля

### Приоритеты разработки

#### Фаза 1: Базовая архитектура (СЕЙЧАС)
- ⏳ Определить структуру библиотеки
- ⏳ Создать базовые интерфейсы
- ⏳ Настроить build system (CMake для Windows + Ubuntu)
- ⏳ Портировать/адаптировать существующие тесты

#### Фаза 2: Примитивы CUDA (Priority 1)
- ⏳ FFT (Forward Fast Fourier Transform)
- ⏳ IFFT (Inverse FFT)
- ⏳ Свёртка (Convolution)
- ⏳ Корреляция (Correlation)
- ⏳ Матричные операции

#### Фаза 3: Оптимизация
- ⏳ Профилирование на RTX 3060
- ⏳ Специализированные kernels
- ⏳ Батчированная обработка (множество сигналов)
- ⏳ Memory pooling

#### Фаза 4: Кросс-платформенность
- ⏳ AMD support (ROCm/HIP)
- ⏳ OpenCL fallback
- ⏳ Автоопределение GPU и выбор оптимального backend

### Планируется
- 🎯 Production-ready API
- 🎯 Comprehensive testing suite
- 🎯 Performance benchmarks (Windows vs Ubuntu)
- 🎯 Documentation для интеграции в продакшен

---

## 🔗 Полезные ссылки

### 📍 Ключевые документы проекта
- **🗺️ [ROADMAP.md](ROADMAP.md)** - ПОЛНЫЙ ПЛАН ПРОЕКТА (Phases 1-6, ~40-50 недель)
- **Spec-Kit Cheatsheet**: `SPEC_KIT_CHEATSHEET.md`
- **Constitution**: `memory/constitution.md`

### Документация CUDA
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **cuFFT Library**: https://docs.nvidia.com/cuda/cufft/
- **cuBLAS Library**: https://docs.nvidia.com/cuda/cublas/

### Внутренние документы
- **Spec-Kit Guide**: `docs/cursor_settings/SPEC_KIT_MEMORY_BANK_GUIDE_RU.md`
- **Quick Reference**: `docs/cursor_settings/QUICK_REFERENCE_RU.md`
- **Architecture**: `docs/Архитектура проекта GPU-вычислений на C++.md`

### Инструменты
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute
- **Nsight Systems**: https://developer.nvidia.com/nsight-systems

---

## 📝 Соглашения о коммитах

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**Types:**
- `feat`: Новая фича
- `fix`: Исправление бага
- `perf`: Оптимизация производительности
- `refactor`: Рефакторинг без изменения функциональности
- `test`: Добавление/изменение тестов
- `docs`: Документация
- `style`: Форматирование
- `chore`: Рутинные задачи (сборка, зависимости)

**Примеры:**
```
feat(fft): add batched FFT support for complex tensors
fix(memory): resolve memory leak in TensorFFT destructor
perf(fft): cache cuFFT plans for 3x speedup
```

---

## 🎓 Onboarding для новых контрибьюторов

### Быстрый старт

1. **Клонировать и собрать**
   ```bash
   git clone https://github.com/AlexLan73/CudaCalc.git
   cd CudaCalc
   mkdir build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
   ninja
   ```

2. **Запустить тесты**
   ```bash
   ./tests/run_all_tests
   ```

3. **Прочитать документацию**
   - `SPEC_KIT_CHEATSHEET.md` - основы Spec-Kit
   - `memory/constitution.md` - принципы проекта
   - Активные спецификации в `specs/`

4. **Начать работу**
   - Выбрать задачу из TODO
   - Создать spec в `specs/`
   - Следовать workflow из Spec-Kit

---

## 🔐 Производительность: Baseline метрики

### FFT Operations (RTX 3090)
- Single 512x512 FFT: ~0.5ms
- Batched 100×512x512 FFT: ~15ms
- Memory overhead: ~10%

### Memory Operations
- Host→Device copy (1MB): ~0.2ms
- Device→Host copy (1MB): ~0.3ms
- cudaMalloc overhead: ~0.1ms per call

**Цели оптимизации:**
- Batched FFT: < 10ms для 100 тензоров
- Memory pool: zero overhead после warm-up
- Plan caching: избежать 2ms overhead

---

## 🐛 Known Issues

- cuFFT может падать на non-power-of-2 размерах (workaround: padding)
- NaN в output если забыли `cudaDeviceSynchronize()` перед копированием
- Memory leak если exception кидается до `cufftDestroy()` (решено через RAII)

---

**Версия:** 1.0  
**Проект:** CudaCalc  
**Автор:** AlexLan73  
**Дата последнего обновления:** 09 октября 2025

**Этот файл является основным контекстом для AI. Обновляйте его после каждой значимой фичи!**

