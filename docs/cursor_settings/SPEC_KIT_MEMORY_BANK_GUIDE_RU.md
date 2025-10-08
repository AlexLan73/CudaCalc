# 📚 Руководство по Spec-Kit и MemoryBank

Подробное руководство по использованию GitHub Spec-Kit и MemoryBank MCP для эффективной работы с AI в Cursor.

---

## 🎯 Оглавление

1. [Spec-Kit: Spec-Driven Development](#spec-kit-spec-driven-development)
2. [MemoryBank: Управление памятью проекта](#memorybank-управление-памятью-проекта)
3. [Лучшие практики](#лучшие-практики)
4. [Примеры использования](#примеры-использования)

---

# 📋 Spec-Kit: Spec-Driven Development

Spec-Kit от GitHub - это методология разработки, где сначала создаётся подробная спецификация, а потом AI реализует код по этой спецификации.

## 🏗️ Структура Spec-Kit проекта

После инициализации (`specify init`) создаётся следующая структура:

```
project/
├── CLAUDE.md                    ← Главный файл контекста для AI
├── memory/
│   └── constitution.md          ← Конституция проекта (принципы)
├── specs/
│   └── 001-feature-name/
│       ├── spec.md              ← Спецификация фичи
│       ├── plan.md              ← План реализации
│       ├── tasks.md             ← Разбивка на задачи
│       ├── research.md          ← Результаты исследования
│       ├── quickstart.md        ← Быстрый старт
│       └── contracts/           ← API контракты
│           ├── api-spec.json
│           └── signalr-spec.md
├── scripts/                     ← Скрипты автоматизации
└── templates/                   ← Шаблоны документов
```

---

## 📝 Что писать в каждом файле Spec-Kit

### 1. `CLAUDE.md` - Главный контекст проекта

**Цель:** Предоставить AI полное понимание проекта.

**Что писать:**

```markdown
# Проект: [Название]

## Обзор проекта
- Краткое описание проекта (2-3 предложения)
- Основная цель и задачи
- Целевая аудитория

## Технологический стек
- Язык программирования: C++17/20
- Фреймворки: CUDA 12.x
- Инструменты сборки: CMake, Ninja
- Тестирование: Google Test

## Архитектура проекта
- Описание модулей
- Диаграмма компонентов (если есть)
- Ключевые паттерны проектирования

## Активные спецификации
- [001-tensor-fft-optimization](specs/001-tensor-fft-optimization/spec.md)
- [002-memory-pool](specs/002-memory-pool/spec.md)

## Важные замечания для AI
- Всегда использовать RAII для управления памятью CUDA
- Предпочитать constexpr для compile-time вычислений
- Следовать стандартам кодирования проекта
```

**Правило:** Обновляйте после каждой новой фичи или архитектурного изменения.

---

### 2. `memory/constitution.md` - Конституция проекта

**Цель:** Установить незыблемые принципы разработки.

**Что писать:**

```markdown
# Конституция проекта CudaCalc

## Принципы разработки

### 1. Производительность превыше всего
- Измерять перед оптимизацией
- Использовать профилировщики (nvprof, Nsight Compute)
- Документировать результаты бенчмарков

### 2. Безопасность памяти
- Никогда не использовать raw pointers для владения
- Всегда проверять возвращаемые значения CUDA API
- Использовать RAII-обертки для ресурсов CUDA

### 3. Тестируемость
- Минимум 80% покрытие кода тестами
- Каждая публичная функция должна иметь тест
- Использовать моки для CUDA функций в unit-тестах

### 4. Документация
- Каждый публичный API должен иметь Doxygen комментарии
- Сложные алгоритмы должны быть объяснены
- Примеры использования для каждого модуля

### 5. Обратная совместимость
- API изменения только через deprecation
- Семантическое версионирование (SemVer)
- Changelog для каждого релиза

## Запрещено
- ❌ Использовать `goto`
- ❌ Глобальные переменные (кроме constexpr)
- ❌ Множественное наследование (кроме интерфейсов)
- ❌ Коммитить без тестов
- ❌ Игнорировать предупреждения компилятора

## Обязательно
- ✅ Code review перед merge
- ✅ Проверка на memory leaks (CUDA-memcheck)
- ✅ Бенчмарки для performance-critical кода
- ✅ Документация для публичных API
```

**Правило:** Пишите только то, что **никогда** не должно нарушаться.

---

### 3. `specs/XXX-feature/spec.md` - Спецификация фичи

**Цель:** Детальное описание **ЧТО** нужно реализовать (не КАК).

**Структура:**

```markdown
# Спецификация: [Название фичи]

**Статус:** Draft | In Review | Approved | Implemented  
**Автор:** [Ваше имя]  
**Дата:** 2025-10-08  
**Версия:** 1.0

---

## 1. Обзор

### Проблема
Опишите проблему, которую решает эта фича.

Пример:
> В настоящее время FFT вычисления для 2D тензоров выполняются 
> последовательно, что приводит к неоптимальному использованию GPU.
> Тесты показывают, что текущая реализация в 3 раза медленнее 
> теоретического максимума.

### Решение
Краткое описание предлагаемого решения.

Пример:
> Реализовать батчированные FFT операции с использованием cuFFT 
> batch API, что позволит обрабатывать множество тензоров одновременно.

### Цели
- [ ] Ускорить FFT вычисления минимум в 2 раза
- [ ] Поддержка тензоров размером до 4096x4096
- [ ] Обратная совместимость с существующим API
- [ ] Memory overhead < 10%

---

## 2. Требования

### Функциональные требования

**FR-1: Батчированный FFT**
- Система должна поддерживать FFT для батча из N тензоров
- N может быть от 1 до 1024
- Размер каждого тензора: от 64x64 до 4096x4096

**FR-2: Типы данных**
- Поддержка float32 и float64
- Поддержка complex64 и complex128
- Автоматическое преобразование типов (с предупреждением)

**FR-3: Направление FFT**
- Forward FFT
- Inverse FFT
- Автоматическая нормализация

### Нефункциональные требования

**NFR-1: Производительность**
- Latency: < 5ms для батча из 100 тензоров 512x512
- Throughput: > 1000 тензоров/сек на RTX 3090
- Memory usage: < batch_size * tensor_size * 1.5

**NFR-2: Надежность**
- Валидация входных данных
- Graceful degradation при нехватке памяти
- Error recovery без утечек памяти

**NFR-3: Совместимость**
- CUDA Compute Capability >= 6.0
- cuFFT версии >= 10.0
- Обратная совместимость с существующим API

---

## 3. Пользовательские сценарии

### UC-1: Обработка одного тензора
```cpp
// Пользователь: Data Scientist
TensorFFT fft;
auto input = Tensor::random({512, 512});
auto output = fft.forward(input);
// Ожидание: output содержит FFT от input
```

### UC-2: Батчированная обработка
```cpp
// Пользователь: ML Engineer
TensorFFT fft;
std::vector<Tensor> batch = load_batch(100);
auto results = fft.forward_batch(batch);
// Ожидание: все тензоры обработаны за один kernel launch
```

### UC-3: In-place преобразование
```cpp
// Пользователь: Performance Engineer
TensorFFT fft;
auto tensor = Tensor::random({1024, 1024});
fft.forward_inplace(tensor);
// Ожидание: tensor изменен на месте, без копирования
```

---

## 4. API Дизайн

### Публичный интерфейс

```cpp
class TensorFFT {
public:
    // Конструктор
    TensorFFT(FFTConfig config = FFTConfig::default());
    
    // Forward FFT
    Tensor forward(const Tensor& input);
    std::vector<Tensor> forward_batch(const std::vector<Tensor>& batch);
    void forward_inplace(Tensor& tensor);
    
    // Inverse FFT
    Tensor inverse(const Tensor& input);
    std::vector<Tensor> inverse_batch(const std::vector<Tensor>& batch);
    void inverse_inplace(Tensor& tensor);
    
    // Утилиты
    size_t estimate_memory(const Tensor& input) const;
    FFTStats get_stats() const;
};

struct FFTConfig {
    FFTType type = FFTType::Complex;  // Real или Complex
    FFTNormalization norm = FFTNormalization::Ortho;
    bool allow_cuda_fallback = true;
    size_t max_batch_size = 1024;
};
```

---

## 5. Ограничения и предположения

### Ограничения
- Тензоры должны быть степенью двойки (64, 128, 256, ..., 4096)
- Максимальный размер батча: 1024 тензора
- Работает только на GPU с CUDA

### Предположения
- У пользователя установлен CUDA Toolkit >= 11.0
- Доступно минимум 4GB GPU памяти
- Входные тензоры уже находятся в GPU памяти

### Будущие расширения
- Поддержка CPU fallback
- Поддержка произвольных размеров (не степень 2)
- Multi-GPU поддержка

---

## 6. Критерии приёмки

### Тестирование
- [ ] Unit тесты для всех публичных методов
- [ ] Integration тесты для батчированной обработки
- [ ] Performance тесты с бенчмарками
- [ ] Memory leak тесты
- [ ] Edge case тесты (пустой батч, огромные тензоры)

### Документация
- [ ] API документация (Doxygen)
- [ ] Примеры использования
- [ ] Performance guidelines
- [ ] Migration guide от старого API

### Производительность
- [ ] Speedup >= 2x по сравнению со старой реализацией
- [ ] Throughput >= 1000 тензоров/сек
- [ ] Memory overhead <= 10%

---

## 7. Review & Acceptance Checklist

- [ ] **Clarity:** Спецификация понятна и недвусмысленна
- [ ] **Completeness:** Все требования описаны
- [ ] **Correctness:** Нет технических ошибок
- [ ] **Consistency:** Термины используются последовательно
- [ ] **Feasibility:** Реализуемо в заданные сроки
- [ ] **Testability:** Есть критерии приёмки
- [ ] **Reviewed by:** [Имена ревьюеров]
- [ ] **Approved by:** [Имя утверждающего]

---

## Приложения

### A. Ссылки
- cuFFT Documentation: https://docs.nvidia.com/cuda/cufft/
- Benchmark Results: [ссылка на Google Sheets]
- Design Discussion: [ссылка на GitHub Issue]

### B. Альтернативные решения
Рассматривались, но отклонены:
1. **Использование CPU FFT** - слишком медленно
2. **Собственная реализация FFT** - reinventing the wheel
3. **VkFFT вместо cuFFT** - нет преимуществ для нашего use case
```

**Правило:** Пишите **ЧТО**, а не **КАК**. Спецификация должна быть понятна человеку без технического бэкграунда.

---

### 4. `specs/XXX-feature/plan.md` - План реализации

**Цель:** Описать **КАК** будет реализована фича (техническая детализация).

**Структура:**

```markdown
# План реализации: [Название фичи]

**Базируется на:** [spec.md версия 1.0]  
**Дата:** 2025-10-08

---

## 1. Обзор реализации

### Архитектурный подход
Используем паттерн Strategy для различных типов FFT (Real/Complex).

```
┌─────────────────┐
│   TensorFFT     │
│   (фасад)       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│Real  │  │Complex│
│FFT   │  │FFT   │
│Impl  │  │Impl  │
└──────┘  └──────┘
```

### Ключевые компоненты

1. **TensorFFT** - публичный фасад
2. **FFTStrategy** - абстрактная стратегия
3. **RealFFTImpl** - реализация для real FFT
4. **ComplexFFTImpl** - реализация для complex FFT
5. **FFTMemoryPool** - пул памяти для переиспользования буферов

---

## 2. Детали реализации

### Модуль 1: TensorFFT Facade

**Файл:** `src/fft/TensorFFT.hpp`, `src/fft/TensorFFT.cu`

**Ответственность:**
- Валидация входных данных
- Выбор правильной стратегии (Real/Complex)
- Управление жизненным циклом ресурсов
- Статистика и логирование

**Ключевые методы:**

```cpp
class TensorFFT {
private:
    std::unique_ptr<FFTStrategy> strategy_;
    std::unique_ptr<FFTMemoryPool> memory_pool_;
    FFTConfig config_;
    mutable FFTStats stats_;
    
    // Валидация
    void validate_input(const Tensor& input) const;
    void validate_batch(const std::vector<Tensor>& batch) const;
    
    // Выбор стратегии
    FFTStrategy* select_strategy(FFTType type);
    
public:
    // Реализация см. в spec.md
};
```

**Зависимости:**
- `Tensor` класс (уже существует)
- `cuFFT` библиотека
- `FFTStrategy` (создать)

---

### Модуль 2: FFT Strategy Pattern

**Файл:** `src/fft/FFTStrategy.hpp`

**Интерфейс:**

```cpp
class FFTStrategy {
public:
    virtual ~FFTStrategy() = default;
    
    // Основные операции
    virtual Tensor forward(const Tensor& input, 
                          FFTMemoryPool& pool) = 0;
    virtual std::vector<Tensor> forward_batch(
        const std::vector<Tensor>& batch,
        FFTMemoryPool& pool) = 0;
    
    // Оценка ресурсов
    virtual size_t estimate_memory(const Tensor& input) const = 0;
    
protected:
    // Общие утилиты для наследников
    cufftHandle create_plan(int nx, int ny, int batch_size);
    void destroy_plan(cufftHandle plan);
};
```

---

### Модуль 3: Complex FFT Implementation

**Файл:** `src/fft/ComplexFFTImpl.hpp`, `src/fft/ComplexFFTImpl.cu`

**Алгоритм:**

```cpp
Tensor ComplexFFTImpl::forward(const Tensor& input, FFTMemoryPool& pool) {
    // 1. Создать plan
    auto plan = create_plan(input.rows(), input.cols(), 1);
    
    // 2. Аллоцировать output буфер из пула
    auto output = pool.allocate(input.shape());
    
    // 3. Выполнить cuFFT
    cufftExecC2C(plan, 
                 input.data<cufftComplex>(),
                 output.data<cufftComplex>(),
                 CUFFT_FORWARD);
    
    // 4. Синхронизация (опционально, если нужен async)
    cudaDeviceSynchronize();
    
    // 5. Уничтожить plan
    destroy_plan(plan);
    
    return output;
}
```

**Оптимизации:**
- Кэшировать plans для популярных размеров
- Использовать streams для параллельных FFT
- Переиспользовать буферы через memory pool

---

### Модуль 4: FFT Memory Pool

**Файл:** `src/fft/FFTMemoryPool.hpp`, `src/fft/FFTMemoryPool.cu`

**Цель:** Избежать частых аллокаций GPU памяти.

**Дизайн:**

```cpp
class FFTMemoryPool {
public:
    FFTMemoryPool(size_t initial_capacity = 1GB);
    ~FFTMemoryPool();
    
    // Аллокация из пула
    Tensor allocate(const TensorShape& shape);
    
    // Возврат в пул
    void release(Tensor&& tensor);
    
    // Статистика
    size_t total_memory() const;
    size_t used_memory() const;
    size_t available_memory() const;
    
private:
    std::unordered_map<size_t, std::vector<void*>> free_buffers_;
    std::unordered_map<void*, size_t> allocated_buffers_;
    size_t total_capacity_;
};
```

**Стратегия:**
- Bucketing по размерам (64, 128, 256, ... KB)
- LRU eviction при нехватке памяти
- Thread-safe с помощью mutex

---

## 3. Интеграция с существующим кодом

### Изменения в Tensor классе

**Файл:** `src/tensor/Tensor.hpp`

**Добавить методы:**

```cpp
class Tensor {
public:
    // ... существующие методы ...
    
    // Новые методы для FFT
    bool is_complex() const;
    Tensor to_complex() const;  // float -> complex conversion
    Tensor to_real() const;     // complex -> float (magnitude)
    
    // Metadata для FFT
    void set_fft_transformed(bool value);
    bool is_fft_transformed() const;
};
```

**Обратная совместимость:** Да, только добавляем методы.

---

## 4. План тестирования

### Unit тесты

**Файл:** `tests/fft/test_tensor_fft.cpp`

```cpp
TEST(TensorFFT, ForwardTransform) {
    TensorFFT fft;
    auto input = Tensor::random({512, 512});
    auto output = fft.forward(input);
    
    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_TRUE(output.is_complex());
}

TEST(TensorFFT, InverseRecoversOriginal) {
    TensorFFT fft;
    auto input = Tensor::random({256, 256});
    auto forward = fft.forward(input);
    auto recovered = fft.inverse(forward);
    
    EXPECT_TENSOR_NEAR(recovered, input, 1e-5);
}

TEST(TensorFFT, BatchProcessing) {
    TensorFFT fft;
    std::vector<Tensor> batch;
    for (int i = 0; i < 100; ++i) {
        batch.push_back(Tensor::random({256, 256}));
    }
    
    auto results = fft.forward_batch(batch);
    
    EXPECT_EQ(results.size(), 100);
    for (const auto& result : results) {
        EXPECT_EQ(result.shape(), TensorShape{256, 256});
    }
}
```

### Performance тесты

**Файл:** `benchmarks/fft/bench_tensor_fft.cpp`

```cpp
static void BM_FFT_Single(benchmark::State& state) {
    TensorFFT fft;
    auto input = Tensor::random({512, 512});
    
    for (auto _ : state) {
        auto output = fft.forward(input);
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FFT_Single);

static void BM_FFT_Batch(benchmark::State& state) {
    TensorFFT fft;
    int batch_size = state.range(0);
    std::vector<Tensor> batch(batch_size, Tensor::random({256, 256}));
    
    for (auto _ : state) {
        auto results = fft.forward_batch(batch);
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_FFT_Batch)->Range(1, 1024);
```

---

## 5. План внедрения

### Фаза 1: Основная инфраструктура (1 неделя)
- [ ] Создать FFTStrategy интерфейс
- [ ] Реализовать TensorFFT фасад (без реальной логики)
- [ ] Настроить build system (CMake)
- [ ] Добавить базовые тесты

### Фаза 2: Complex FFT (1 неделя)
- [ ] Реализовать ComplexFFTImpl
- [ ] Интеграция с cuFFT
- [ ] Unit тесты для complex FFT
- [ ] Базовые benchmarks

### Фаза 3: Memory Pool (3 дня)
- [ ] Реализовать FFTMemoryPool
- [ ] Интеграция с ComplexFFTImpl
- [ ] Memory leak тесты
- [ ] Performance regression тесты

### Фаза 4: Real FFT (3 дня)
- [ ] Реализовать RealFFTImpl
- [ ] Unit тесты
- [ ] Benchmarks

### Фаза 5: Оптимизация и документация (1 неделя)
- [ ] Plan caching
- [ ] Async streams
- [ ] Полная документация (Doxygen)
- [ ] Примеры использования
- [ ] Performance tuning guide

---

## 6. Риски и митигация

### Риск 1: cuFFT не справляется с большими батчами
**Вероятность:** Средняя  
**Влияние:** Высокое  
**Митигация:** Разбивать большие батчи на подбатчи

### Риск 2: Memory Pool фрагментация
**Вероятность:** Низкая  
**Влияние:** Среднее  
**Митигация:** Периодическая дефрагментация, bucketing стратегия

### Риск 3: Регрессия производительности
**Вероятность:** Низкая  
**Влияние:** Высокое  
**Митигация:** Continuous benchmarking в CI/CD

---

## 7. Зависимости и требования

### Внешние библиотеки
- cuFFT >= 10.0 (часть CUDA Toolkit)
- Google Test >= 1.10 (для тестов)
- Google Benchmark >= 1.5 (для benchmarks)

### Системные требования
- CUDA >= 11.0
- CMake >= 3.18
- C++17 компилятор
- GPU с Compute Capability >= 6.0

---

## 8. Контрольные точки (Milestones)

| Milestone | Дата | Критерий успеха |
|-----------|------|-----------------|
| M1: Infrastructure | 2025-10-15 | Фасад + интерфейсы готовы |
| M2: Complex FFT | 2025-10-22 | Работает forward/inverse |
| M3: Memory Pool | 2025-10-25 | No memory leaks |
| M4: Real FFT | 2025-10-28 | Все типы FFT работают |
| M5: Release | 2025-11-04 | Все тесты pass, docs готовы |
```

**Правило:** Plan должен быть настолько детальным, чтобы junior developer мог реализовать фичу.

---

### 5. `specs/XXX-feature/tasks.md` - Разбивка на задачи

**Цель:** Список конкретных задач для реализации (для AI или человека).

**Структура:**

```markdown
# Задачи: [Название фичи]

**Основано на:** plan.md версия 1.0

---

## Формат задач

Каждая задача имеет:
- **ID:** Уникальный идентификатор
- **Название:** Краткое описание
- **Описание:** Что нужно сделать
- **Зависимости:** От каких задач зависит
- **Оценка:** Время выполнения
- **Статус:** TODO | IN_PROGRESS | DONE | BLOCKED

---

## Задачи

### TASK-001: Создать FFTStrategy интерфейс
**Статус:** TODO  
**Оценка:** 2 часа  
**Зависимости:** Нет  
**Приоритет:** Высокий

**Описание:**
Создать файлы `src/fft/FFTStrategy.hpp` и добавить абстрактный класс FFTStrategy.

**Критерии приёмки:**
- [ ] Файл `src/fft/FFTStrategy.hpp` создан
- [ ] Класс FFTStrategy объявлен
- [ ] Все методы из plan.md добавлены
- [ ] Файл компилируется без ошибок

**Код:**
```cpp
// src/fft/FFTStrategy.hpp
#pragma once

#include "Tensor.hpp"
#include "FFTMemoryPool.hpp"
#include <cufft.h>

class FFTStrategy {
public:
    virtual ~FFTStrategy() = default;
    
    virtual Tensor forward(const Tensor& input, 
                          FFTMemoryPool& pool) = 0;
    // ... остальные методы из plan.md
};
```

---

### TASK-002: Создать FFTConfig структуру
**Статус:** TODO  
**Оценка:** 1 час  
**Зависимости:** Нет  
**Приоритет:** Высокий

**Описание:**
Создать файл `src/fft/FFTConfig.hpp` с конфигурационной структурой.

**Критерии приёмки:**
- [ ] Файл `src/fft/FFTConfig.hpp` создан
- [ ] Структура FFTConfig объявлена
- [ ] Все поля из spec.md добавлены
- [ ] Есть метод `default()`

**Код:**
```cpp
// src/fft/FFTConfig.hpp
#pragma once

enum class FFTType { Real, Complex };
enum class FFTNormalization { None, Ortho, Full };

struct FFTConfig {
    FFTType type = FFTType::Complex;
    FFTNormalization norm = FFTNormalization::Ortho;
    bool allow_cuda_fallback = true;
    size_t max_batch_size = 1024;
    
    static FFTConfig default_config() {
        return FFTConfig{};
    }
};
```

---

### TASK-003: Реализовать TensorFFT конструктор
**Статус:** TODO  
**Оценка:** 2 часа  
**Зависимости:** TASK-001, TASK-002  
**Приоритет:** Высокий

**Описание:**
Создать файлы `src/fft/TensorFFT.hpp` и `src/fft/TensorFFT.cu`, реализовать конструктор.

**Критерии приёмки:**
- [ ] Файлы созданы
- [ ] Конструктор принимает FFTConfig
- [ ] Инициализируется strategy_ (пока nullptr)
- [ ] Инициализируется memory_pool_
- [ ] Компилируется без ошибок

---

### TASK-004: Реализовать FFTMemoryPool (skeleton)
**Статус:** TODO  
**Оценка:** 3 часа  
**Зависимости:** Нет  
**Приоритет:** Средний

**Описание:**
Создать базовую структуру FFTMemoryPool с заглушками.

**Критерии приёмки:**
- [ ] Файлы `src/fft/FFTMemoryPool.hpp` и `.cu` созданы
- [ ] Все методы объявлены
- [ ] Конструктор/деструктор работают
- [ ] Методы allocate/release - заглушки (TODO)
- [ ] Компилируется

---

### TASK-005: Реализовать ComplexFFTImpl::forward (basic)
**Статус:** TODO  
**Оценка:** 4 часа  
**Зависимости:** TASK-001, TASK-004  
**Приоритет:** Высокий

**Описание:**
Реализовать базовую версию ComplexFFTImpl::forward для одного тензора.

**Критерии приёмки:**
- [ ] Файлы `src/fft/ComplexFFTImpl.hpp` и `.cu` созданы
- [ ] Метод forward() реализован
- [ ] Использует cufftExecC2C
- [ ] Проходит базовый unit тест
- [ ] Нет memory leaks (проверено CUDA-memcheck)

**Тесты:**
```cpp
TEST(ComplexFFT, BasicForward) {
    ComplexFFTImpl fft;
    FFTMemoryPool pool(1024MB);
    
    auto input = Tensor::random({256, 256}, TensorType::Complex);
    auto output = fft.forward(input, pool);
    
    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_TRUE(output.is_complex());
}
```

---

### TASK-006: Реализовать валидацию входных данных
**Статус:** TODO  
**Оценка:** 2 часа  
**Зависимости:** TASK-003  
**Приоритет:** Высокий

**Описание:**
Добавить методы validate_input() и validate_batch() в TensorFFT.

**Критерии приёмки:**
- [ ] Методы реализованы
- [ ] Проверяют размер тензора (степень 2)
- [ ] Проверяют тип данных
- [ ] Бросают исключения с понятными сообщениями
- [ ] Unit тесты для всех edge cases

---

### TASK-007: Реализовать forward_batch (parallel)
**Статус:** TODO  
**Оценка:** 6 часов  
**Зависимости:** TASK-005  
**Приоритет:** Средний

**Описание:**
Реализовать батчированную версию FFT с использованием CUDA streams.

**Критерии приёмки:**
- [ ] Использует cufftPlan2d с batch параметром
- [ ] Распараллеливание через CUDA streams
- [ ] Speedup >= 2x по сравнению с последовательной обработкой
- [ ] Performance тесты pass

---

### TASK-008: Реализовать FFTMemoryPool (полная версия)
**Статус:** TODO  
**Оценка:** 8 часов  
**Зависимости:** TASK-004  
**Приоритет:** Средний

**Описание:**
Полная реализация memory pool с bucketing и LRU eviction.

**Критерии приёмки:**
- [ ] Bucketing по размерам работает
- [ ] LRU eviction при нехватке памяти
- [ ] Thread-safe (mutex)
- [ ] Memory leak тесты pass
- [ ] Performance overhead < 5%

---

### TASK-009: Добавить Doxygen комментарии
**Статус:** TODO  
**Оценка:** 4 часа  
**Зависимости:** Все предыдущие  
**Приоритет:** Низкий

**Описание:**
Документировать все публичные API в Doxygen формате.

**Критерии приёмки:**
- [ ] Все публичные классы документированы
- [ ] Все публичные методы документированы
- [ ] Примеры использования добавлены
- [ ] Doxygen генерирует HTML без warnings

---

### TASK-010: Написать примеры использования
**Статус:** TODO  
**Оценка:** 3 часа  
**Зависимости:** TASK-009  
**Приоритет:** Низкий

**Описание:**
Создать файл `examples/fft_example.cpp` с примерами.

**Критерии приёмки:**
- [ ] Минимум 5 примеров разных use cases
- [ ] Примеры компилируются
- [ ] Примеры запускаются без ошибок
- [ ] Комментарии объясняют каждый шаг

---

## Последовательность выполнения

```
TASK-001, TASK-002 (параллельно)
    ↓
TASK-003
    ↓
TASK-004 (параллельно с TASK-006)
    ↓
TASK-005
    ↓
TASK-007
    ↓
TASK-008
    ↓
TASK-009
    ↓
TASK-010
```

---

## Оценка времени

| Фаза | Задачи | Время |
|------|--------|-------|
| Фаза 1: Infrastructure | TASK-001..004 | 8 часов |
| Фаза 2: Basic FFT | TASK-005, 006 | 6 часов |
| Фаза 3: Optimization | TASK-007, 008 | 14 часов |
| Фаза 4: Documentation | TASK-009, 010 | 7 часов |
| **Итого** | | **35 часов (~5 дней)** |

---

## Отслеживание прогресса

Обновляйте статусы задач по мере выполнения:

- ✅ TASK-001: DONE (2025-10-08)
- ✅ TASK-002: DONE (2025-10-08)
- 🔄 TASK-003: IN_PROGRESS
- ⏳ TASK-004: TODO
- ⏳ TASK-005: TODO
- ...
```

**Правило:** Одна задача = одна функция или файл. Задача должна быть выполнима за 1-8 часов.

---

### 6. `specs/XXX-feature/research.md` - Результаты исследования

**Цель:** Документировать технические исследования и выбор решений.

**Что писать:**

```markdown
# Исследование: [Тема]

**Дата:** 2025-10-08  
**Автор:** [Ваше имя]

---

## Вопрос исследования

Какой подход к батчированным FFT даст лучшую производительность?

---

## Рассмотренные варианты

### Вариант 1: cuFFT batch API
**Плюсы:**
- Нативная поддержка батчирования
- Оптимизирован NVIDIA
- Прост в использовании

**Минусы:**
- Требует все тензоры одного размера
- Ограничение batch_size по памяти

**Бенчмарк:**
```
Batch size: 100 тензоров 512x512
Время: 12.3 ms
Throughput: 8130 тензоров/сек
```

### Вариант 2: Ручное батчирование через streams
**Плюсы:**
- Гибкость (разные размеры)
- Контроль над памятью

**Минусы:**
- Больше кода
- Требует ручной синхронизации

**Бенчмарк:**
```
Batch size: 100 тензоров 512x512
Время: 15.7 ms
Throughput: 6369 тензоров/сек
```

### Вариант 3: VkFFT (Vulkan compute)
**Плюсы:**
- Кроссплатформенность
- Иногда быстрее cuFFT

**Минусы:**
- Добавляет зависимость от Vulkan
- Меньше примеров и документации

**Не тестировали** (выходит за рамки проекта)

---

## Решение

Выбран **Вариант 1: cuFFT batch API**.

**Обоснование:**
- Лучшая производительность (на 27% быстрее варианта 2)
- Проще поддерживать
- Официально поддерживается NVIDIA
- Ограничение "все тензоры одного размера" приемлемо для нашего use case

---

## Дополнительные находки

### cuFFT plan caching
Тестирование показало, что создание plan занимает ~2ms.  
**Рекомендация:** Кэшировать plans для частых размеров.

### Memory alignment
cuFFT требует 512-byte alignment для оптимальной производительности.  
**Рекомендация:** Добавить aligned allocator в FFTMemoryPool.

---

## Ссылки
- cuFFT Documentation: https://docs.nvidia.com/cuda/cufft/
- Benchmark code: [ссылка на gist]
- Discussion thread: [ссылка на GitHub Discussions]
```

**Правило:** Документируйте ВСЕ исследования, даже неудачные. Это сэкономит время в будущем.

---

# 🧠 MemoryBank: Управление памятью проекта

MemoryBank MCP - это инструмент для хранения долговременной памяти проекта, которую AI может использовать в контексте.

## 📝 Что писать в MemoryBank

### 1. Архитектурные решения

**Когда сохранять:**
- Приняли важное архитектурное решение
- Выбрали между несколькими подходами
- Отказались от какого-то решения

**Пример:**

```
Title: Выбор паттерна для FFT стратегий
Content:
Решено использовать Strategy паттерн для разных типов FFT (Real/Complex).
Альтернативы: Factory, Template specialization.
Причина: Strategy даёт flexibility и упрощает тестирование.
Дата: 2025-10-08
```

### 2. Важные ограничения

**Когда сохранять:**
- Обнаружили техническое ограничение
- Есть workaround для известной проблемы
- Нужно помнить о краевых случаях

**Пример:**

```
Title: cuFFT batch size ограничение
Content:
cuFFT batch API имеет ограничение на максимальный batch_size из-за GPU памяти.
Для RTX 3090 (24GB): max ~2000 тензоров 512x512.
Workaround: Разбивать большие батчи на под-батчи по 1024 тензора.
Источник: https://docs.nvidia.com/cuda/cufft/limitations.html
```

### 3. Производительность insights

**Когда сохранять:**
- Обнаружили bottleneck
- Оптимизировали критичный код
- Benchmarks показали неожиданные результаты

**Пример:**

```
Title: FFT plan creation overhead
Content:
Создание cufftPlan занимает ~2ms, что значительно при обработке малых батчей.
Решение: Кэшировать plans для популярных размеров (64, 128, 256, 512, 1024, 2048, 4096).
Impact: Speedup 3x для repeated small batches.
Benchmark: bench_plan_caching.cpp
```

### 4. Lessons Learned

**Когда сохранять:**
- Потратили много времени на debugging
- Нашли неочевидное решение проблемы
- Хотите, чтобы AI не повторял эту ошибку

**Пример:**

```
Title: cudaDeviceSynchronize необходим после cufftExec
Content:
Забыли добавить cudaDeviceSynchronize() после cufftExec, что привело к race conditions.
Симптомы: Случайные NaN в output, heisenbug (исчезает в debug mode).
Решение: Всегда синхронизировать после CUDA kernel launches если результат нужен немедленно.
Альтернатива: Использовать streams и async API для избежания блокировки.
```

### 5. Code conventions

**Когда сохранять:**
- Установили новый coding standard
- Решили использовать определенный стиль
- Есть специфичные требования для проекта

**Пример:**

```
Title: CUDA error checking convention
Content:
Все CUDA API calls должны быть обёрнуты в макрос CUDA_CHECK():
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw CUDAException(err, __FILE__, __LINE__); \
    } \
  } while (0)

Пример: CUDA_CHECK(cudaMalloc(&ptr, size));
Причина: Единообразный error handling, легче debug.
```

---

## 🎯 Лучшие практики

### Для Spec-Kit

1. **Начинайте с constitution.md**
   - Определите принципы ДО начала кодирования
   - Не меняйте constitution часто (это фундамент)

2. **Spec.md пишите в Present Tense**
   - "Система поддерживает...", не "Система должна поддерживать..."
   - Spec описывает конечное состояние, не процесс

3. **Plan.md - максимально детально**
   - Лучше over-document, чем under-document
   - Укажите даже очевидные вещи (для AI они не очевидны)

4. **Tasks.md - atomic задачи**
   - Одна задача = один PR = одна функция
   - Задача должна быть завершена за < 8 часов

5. **Research.md - всегда документируйте**
   - Даже если исследование не дало результата
   - Особенно если НЕ дало результата (чтобы не повторять)

### Для MemoryBank

1. **Пишите кратко, но полно**
   - Максимум 200-300 слов на запись
   - Всё самое важное в начале

2. **Используйте tags/keywords**
   - #performance, #cuda, #bug, #workaround
   - AI легче найдёт релевантную информацию

3. **Всегда указывайте источник**
   - Ссылка на документацию
   - Ссылка на benchmark
   - Ссылка на GitHub issue/PR

4. **Обновляйте устаревшую информацию**
   - Если решение изменилось - обновите запись
   - Не удаляйте старую, добавьте "Updated: ..."

5. **Пишите для будущего себя**
   - Через полгода вы забудете контекст
   - Пишите так, чтобы можно было быстро вспомнить

---

## 📊 Примеры использования

### Пример 1: Начало новой фичи с Spec-Kit

```bash
# 1. Инициализация (если еще не сделано)
specify init

# 2. Создание новой фичи
cd project/
./scripts/create-new-feature.sh "tensor-fft-optimization"

# 3. Заполнение spec.md (вручную или с помощью AI)
cursor specs/001-tensor-fft-optimization/spec.md

# 4. AI clarification
# В Cursor chat:
/speckit.clarify

# 5. Генерация плана
# В Cursor chat:
/speckit.plan

# 6. Review plan
cursor specs/001-tensor-fft-optimization/plan.md

# 7. Реализация
# В Cursor chat:
/speckit.implement

# 8. Тестирование и итерация
```

### Пример 2: Сохранение в MemoryBank во время работы

```
# Во время кодирования вы нашли важный инсайт

# В Cursor chat:
Сохрани в память: "cuFFT plan creation занимает 2ms. 
Для оптимизации нужно кэшировать plans для популярных размеров.
Impact: 3x speedup для repeated small batches."

# AI автоматически сохранит в MemoryBank с правильной структурой
```

### Пример 3: Запрос информации из MemoryBank

```
# В Cursor chat:
Что мы знаем о производительности FFT?

# AI найдёт релевантные записи в MemoryBank:
# - FFT plan creation overhead
# - cuFFT batch size ограничение
# - Benchmark results
# И использует их в контексте для ответа
```

---

## 🚨 Частые ошибки

### ❌ Ошибка 1: Слишком общая spec
```markdown
## Требования
- Система должна быть быстрой
- Код должен быть читаемым
```

### ✅ Правильно:
```markdown
## Требования
- Latency < 5ms для батча из 100 тензоров 512x512 на RTX 3090
- Cyclomatic complexity < 10 для всех функций
- Минимум 80% code coverage
```

---

### ❌ Ошибка 2: Plan без деталей
```markdown
## Модуль FFT
Реализовать FFT используя cuFFT.
```

### ✅ Правильно:
```markdown
## Модуль FFT

**Файлы:**
- `src/fft/TensorFFT.hpp` (165 строк)
- `src/fft/TensorFFT.cu` (320 строк)

**Алгоритм:**
1. Создать cufftPlan2d(nx=512, ny=512, batch_size=100)
2. Аллоцировать output буфер: cudaMalloc(batch_size * nx * ny * sizeof(cufftComplex))
3. Выполнить: cufftExecC2C(plan, input_ptr, output_ptr, CUFFT_FORWARD)
4. Синхронизация: cudaDeviceSynchronize()
5. Cleanup: cufftDestroy(plan), cudaFree(output_ptr)

**Зависимости:**
- Tensor class (src/tensor/Tensor.hpp)
- CUDAException (src/cuda/Exception.hpp)
- cuFFT library

**Error handling:**
- Проверять cufftResult != CUFFT_SUCCESS
- Проверять cudaError_t != cudaSuccess
- Бросать исключения с понятными сообщениями
```

---

### ❌ Ошибка 3: Задачи слишком большие
```markdown
TASK-001: Реализовать весь FFT модуль
Оценка: 40 часов
```

### ✅ Правильно:
```markdown
TASK-001: Создать FFTStrategy интерфейс
Оценка: 2 часа

TASK-002: Реализовать ComplexFFTImpl::forward
Оценка: 4 часа

TASK-003: Добавить unit тесты для ComplexFFTImpl
Оценка: 3 часа
```

---

### ❌ Ошибка 4: MemoryBank без контекста
```markdown
Title: FFT bug
Content: Исправил баг
```

### ✅ Правильно:
```markdown
Title: FFT NaN output bug - missing cudaDeviceSynchronize
Content:
Проблема: cufftExec возвращал NaN значения случайным образом.
Root cause: Отсутствовала синхронизация после cufftExec, что приводило к race condition.
Решение: Добавить cudaDeviceSynchronize() после каждого cufftExec или использовать streams.
Commit: abc123def
Дата: 2025-10-08
```

---

## 📚 Дополнительные ресурсы

- Spec-Kit GitHub: https://github.com/github/spec-kit
- Spec-Kit Documentation: https://github.com/github/spec-kit/blob/main/spec-driven.md
- MemoryBank MCP: https://github.com/modelcontextprotocol/memory-bank
- Примеры Spec-Driven проектов: https://github.com/topics/spec-driven-development

---

## 💡 Заключение

**Золотое правило:**
> "Если AI не может понять вашу spec за 30 секунд - она недостаточно ясна."

**Для Spec-Kit:**
1. Constitution = Принципы (никогда не меняются)
2. Spec = ЧТО (для людей)
3. Plan = КАК (для разработчиков)
4. Tasks = Список дел (для AI/Junior)

**Для MemoryBank:**
1. Сохраняйте сразу, не откладывайте
2. Кратко, но полно
3. Всегда указывайте источник
4. Обновляйте устаревшее

**Помните:**
Время, потраченное на хорошую документацию, окупается 10x при реализации и 100x при поддержке!

---

**Автор:** AlexLan73  
**Проект:** CudaCalc  
**Версия:** 1.0  
**Дата:** Октябрь 2025
