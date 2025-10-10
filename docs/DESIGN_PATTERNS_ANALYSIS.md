# Анализ паттернов проектирования в CudaCalc

**Дата:** 09 октября 2025  
**Проект:** CudaCalc - GPU-ускоренная библиотека для обработки сигналов

---

## 📚 Содержание

1. [Архитектурные паттерны](#архитектурные-паттерны)
2. [Паттерны GoF (Gang of Four)](#паттерны-gof)
3. [Паттерны ответственности (GRASP)](#паттерны-ответственности-grasp)
4. [Дополнительные паттерны](#дополнительные-паттерны)
5. [Принципы проектирования](#принципы-проектирования)

---

## 🏛️ Архитектурные паттерны

### 1. **Layered Architecture (Слоистая архитектура)**

**Применение:**
```
┌─────────────────────────────────────────┐
│ Presentation Layer                      │
│ MainProgram/ - точки входа              │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Business Logic Layer                    │
│ SignalGenerators/, Tester/              │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Domain Layer                            │
│ ModelsFunction/ - FFT реализации        │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Data Access Layer                       │
│ DataContext/ - JSONLogger, ModelArchiver│
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Infrastructure Layer                    │
│ CUDA, cuFFT, Python (scipy)             │
└─────────────────────────────────────────┘
```

**Где:**
- Каждый слой имеет четкую ответственность
- Зависимости направлены вниз
- Нет циклических зависимостей

---

### 2. **Pipes and Filters (Конвейер обработки)**

**Применение в workflow:**
```
[SignalGenerator] → [Tester/Profiler] → [FFT Processor] → 
→ [DataContext] → [Python Validator] → [Visualization]
```

**Где:**
- Каждый компонент = фильтр (обрабатывает данные)
- Данные проходят через pipe (InputSignalData → OutputSpectralData)
- Компоненты независимы и могут быть заменены

**Код:**
```cpp
// Pipe 1: Генерация
InputSignalData signal = generator.generate(window_fft);

// Pipe 2: Профилирование
profiler.start();
OutputSpectralData result = processor.process(signal);
profiler.stop();

// Pipe 3: Сохранение
dataContext.save(result);

// Pipe 4: Валидация (Python)
python validate_fft.py --file result.json
```

---

### 3. **Plugin Architecture (Плагинная архитектура)**

**Применение:**
```cpp
// Базовый интерфейс
class IGPUProcessor {
    virtual OutputSpectralData process(const InputSignalData& input) = 0;
};

// Плагины
class FFT16_WMMA : public IGPUProcessor { ... };
class FFT16_Shared2D : public IGPUProcessor { ... };
class FFT16_cuFFT : public IGPUProcessor { ... };
```

**Где:**
- Легко добавить новые FFT реализации
- Все реализации взаимозаменяемы
- MainProgram работает через интерфейс IGPUProcessor

---

### 4. **Repository Pattern (Репозиторий)**

**Применение в DataContext:**
```cpp
class ModelArchiver {
    // Сохранение
    bool save_model(const ModelInfo& info, ...);
    
    // Загрузка
    ModelInfo load_model(const std::string& version);
    
    // Поиск
    std::vector<ModelInfo> list_models(...);
    
    // Сравнение
    std::string compare_models(...);
};
```

**Где:**
- Инкапсулирует доступ к Models/
- Скрывает детали файловой системы
- Предоставляет высокоуровневый API

---

## 🎨 Паттерны GoF (Gang of Four)

### **Порождающие паттерны (Creational)**

#### 1. **Factory Method (Фабричный метод)**

**Применение в SignalGenerators:**
```cpp
class BaseGenerator {
public:
    virtual InputSignalData generate(int window_fft) = 0;
    virtual SignalType get_type() const = 0;
};

class SineGenerator : public BaseGenerator {
    InputSignalData generate(int window_fft) override;
    SignalType get_type() const override { return SignalType::SINE; }
};

// Будущие:
class QuadratureGenerator : public BaseGenerator { ... };
class ModulatedGenerator : public BaseGenerator { ... };
```

**Где:**
- BaseGenerator определяет интерфейс создания
- Конкретные генераторы (SineGenerator, ...) реализуют создание
- Клиент работает через базовый класс

---

#### 2. **Builder (Строитель)** - ПЛАНИРУЕТСЯ

**Потенциальное применение:**
```cpp
class TestConfigBuilder {
private:
    StrobeConfig config_;
    SignalType signal_type_;
    bool validation_enabled_;
    
public:
    TestConfigBuilder& setRayCount(int count) {
        config_.ray_count = count;
        return *this;
    }
    
    TestConfigBuilder& setSignalType(SignalType type) {
        signal_type_ = type;
        return *this;
    }
    
    TestConfigBuilder& enableValidation(bool enable) {
        validation_enabled_ = enable;
        return *this;
    }
    
    TestDataPackage build() {
        // Создание сложного объекта
    }
};

// Использование
auto package = TestConfigBuilder()
    .setRayCount(4)
    .setPointsPerRay(1024)
    .setSignalType(SignalType::SINE)
    .enableValidation(true)
    .build();
```

---

### **Структурные паттерны (Structural)**

#### 3. **Adapter (Адаптер)**

**Применение для cuFFT:**
```cpp
// Целевой интерфейс
class IGPUProcessor {
    virtual OutputSpectralData process(const InputSignalData& input) = 0;
};

// Адаптер для cuFFT
class FFT16_cuFFT : public IGPUProcessor {
private:
    cufftHandle plan_;  // Внешняя библиотека cuFFT
    
public:
    OutputSpectralData process(const InputSignalData& input) override {
        // Адаптация cuFFT API к нашему интерфейсу
        cufftExecC2C(plan_, input_device, output_device, CUFFT_FORWARD);
        // Конвертация результата в OutputSpectralData
        return result;
    }
};
```

**Где:**
- Адаптирует cuFFT API к IGPUProcessor
- Скрывает детали cuFFT от клиента
- Позволяет использовать cuFFT как плагин

---

#### 4. **Facade (Фасад)**

**Применение в DataContext:**
```cpp
class DataContext {
private:
    JSONLogger json_logger_;
    ModelArchiver model_archiver_;
    ConfigManager config_manager_;
    
public:
    // Упрощенный интерфейс для сложной подсистемы
    void saveTestResults(const TestResults& results) {
        // Координирует работу нескольких компонентов
        json_logger_.write_profiling(results.profiling);
        
        if (results.validation_enabled) {
            json_logger_.write_validation_data(results.data);
        }
        
        model_archiver_.save_model(results.model_info);
    }
};
```

**Где:**
- Упрощает работу с DataContext подсистемой
- Скрывает сложность JSONLogger, ModelArchiver, ConfigManager
- Предоставляет единый интерфейс

---

#### 5. **Decorator (Декоратор)** - ПОТЕНЦИАЛЬНО

**Потенциальное применение для профилирования:**
```cpp
class ProfilingDecorator : public IGPUProcessor {
private:
    std::unique_ptr<IGPUProcessor> wrapped_;
    BasicProfiler profiler_;
    
public:
    ProfilingDecorator(std::unique_ptr<IGPUProcessor> processor)
        : wrapped_(std::move(processor)) {}
    
    OutputSpectralData process(const InputSignalData& input) override {
        profiler_.start_timing();
        auto result = wrapped_->process(input);  // Делегирование
        profiler_.end_timing();
        return result;
    }
};

// Использование
auto fft = std::make_unique<FFT16_WMMA>();
auto profiled_fft = std::make_unique<ProfilingDecorator>(std::move(fft));
```

---

#### 6. **Composite (Компоновщик)** - ПОТЕНЦИАЛЬНО

**Потенциальное применение для batch обработки:**
```cpp
class BatchProcessor : public IGPUProcessor {
private:
    std::vector<std::unique_ptr<IGPUProcessor>> processors_;
    
public:
    void add(std::unique_ptr<IGPUProcessor> processor) {
        processors_.push_back(std::move(processor));
    }
    
    OutputSpectralData process(const InputSignalData& input) override {
        // Обрабатываем входные данные всеми процессорами
        for (auto& proc : processors_) {
            proc->process(input);
        }
    }
};
```

---

### **Поведенческие паттерны (Behavioral)**

#### 7. **Strategy (Стратегия)**

**Применение в FFT реализациях:**
```cpp
class FFTStrategy {
public:
    virtual ~FFTStrategy() = default;
    virtual OutputSpectralData execute(const InputSignalData& input) = 0;
};

class TensorCoresStrategy : public FFTStrategy { ... };
class Shared2DStrategy : public FFTStrategy { ... };
class cuFFTStrategy : public FFTStrategy { ... };

class FFTContext {
private:
    std::unique_ptr<FFTStrategy> strategy_;
    
public:
    void setStrategy(std::unique_ptr<FFTStrategy> strategy) {
        strategy_ = std::move(strategy);
    }
    
    OutputSpectralData process(const InputSignalData& input) {
        return strategy_->execute(input);
    }
};
```

**Где:**
- Инкапсулирует различные алгоритмы FFT
- Алгоритмы взаимозаменяемы
- Можно менять стратегию в runtime

---

#### 8. **Template Method (Шаблонный метод)**

**Применение в BaseGenerator:**
```cpp
class BaseGenerator {
public:
    // Шаблонный метод
    InputSignalData generate(int window_fft) {
        validate_parameters();           // Общий шаг
        auto signal = generate_signal(); // Переопределяемый шаг
        apply_window(signal, window_fft); // Общий шаг
        return finalize(signal);         // Общий шаг
    }
    
protected:
    virtual std::vector<complex<float>> generate_signal() = 0; // Абстрактный
    
    void validate_parameters() { /* общая реализация */ }
    void apply_window(...) { /* общая реализация */ }
    InputSignalData finalize(...) { /* общая реализация */ }
};

class SineGenerator : public BaseGenerator {
protected:
    std::vector<complex<float>> generate_signal() override {
        // Специфичная реализация для синуса
    }
};
```

**Где:**
- Определяет скелет алгоритма в базовом классе
- Подклассы переопределяют отдельные шаги
- Общая логика не дублируется

---

#### 9. **Observer (Наблюдатель)** - ПОТЕНЦИАЛЬНО

**Потенциальное применение для логирования:**
```cpp
class TestObserver {
public:
    virtual void onTestStart(const TestInfo& info) = 0;
    virtual void onTestComplete(const TestResults& results) = 0;
    virtual void onError(const std::string& error) = 0;
};

class ConsoleLogger : public TestObserver { ... };
class JSONLogger : public TestObserver { ... };
class GUIUpdater : public TestObserver { ... };

class TestRunner {
private:
    std::vector<TestObserver*> observers_;
    
public:
    void addObserver(TestObserver* observer) {
        observers_.push_back(observer);
    }
    
    void runTest() {
        notifyTestStart();
        // ... выполнение теста ...
        notifyTestComplete();
    }
    
private:
    void notifyTestStart() {
        for (auto obs : observers_) {
            obs->onTestStart(test_info_);
        }
    }
};
```

---

#### 10. **Command (Команда)** - ПОТЕНЦИАЛЬНО

**Потенциальное применение для тестовых команд:**
```cpp
class TestCommand {
public:
    virtual ~TestCommand() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

class RunFFTTestCommand : public TestCommand {
private:
    IGPUProcessor* processor_;
    InputSignalData input_;
    OutputSpectralData result_;
    
public:
    void execute() override {
        result_ = processor_->process(input_);
    }
    
    void undo() override {
        // Откат результатов
    }
};

class TestInvoker {
private:
    std::vector<std::unique_ptr<TestCommand>> history_;
    
public:
    void executeCommand(std::unique_ptr<TestCommand> cmd) {
        cmd->execute();
        history_.push_back(std::move(cmd));
    }
    
    void undoLast() {
        if (!history_.empty()) {
            history_.back()->undo();
            history_.pop_back();
        }
    }
};
```

---

## 🎯 Паттерны ответственности (GRASP)

### 1. **Information Expert (Информационный эксперт)**

**Применение:**
```cpp
// StrobeConfig знает, как вычислить total_points и num_windows
struct StrobeConfig {
    int ray_count;
    int points_per_ray;
    int window_fft;
    
    // Ответственность: вычисление производных значений
    int total_points() const {
        return ray_count * points_per_ray;
    }
    
    int num_windows() const {
        return total_points() / window_fft;
    }
};
```

**Принцип:**
- Объект, который владеет информацией, должен выполнять операции над ней
- StrobeConfig владеет ray_count, points_per_ray → он вычисляет total_points()

---

### 2. **Creator (Создатель)**

**Применение:**
```cpp
// SineGenerator создает InputSignalData, т.к.:
// - содержит необходимые данные для создания (период, амплитуда)
// - использует InputSignalData
class SineGenerator {
    InputSignalData generate(int window_fft) {
        InputSignalData data;
        data.config = /* ... */;
        data.signal = /* генерация синуса */;
        return data;
    }
};
```

**Принцип:**
- Класс B должен создавать объекты класса A, если:
  - B агрегирует A
  - B тесно использует A
  - B содержит данные для инициализации A

---

### 3. **Controller (Контроллер)**

**Применение в MainProgram:**
```cpp
class FFT16TestController {
public:
    void runTest() {
        // 1. Создание сигнала
        auto signal = signal_generator_.generate(16);
        
        // 2. Профилирование
        profiler_.start();
        auto result = processor_.process(signal);
        profiler_.stop();
        
        // 3. Сохранение
        data_context_.save(result);
        
        // 4. Валидация (опционально)
        if (signal.return_for_validation) {
            validator_.validate(result);
        }
    }
    
private:
    SineGenerator signal_generator_;
    IGPUProcessor& processor_;
    BasicProfiler profiler_;
    DataContext data_context_;
};
```

**Принцип:**
- Контроллер координирует работу системы
- Не содержит бизнес-логики (делегирует)
- Точка входа для use case

---

### 4. **Low Coupling (Низкая связанность)**

**Применение:**
```cpp
// ПЛОХО: высокая связанность
class FFTProcessor {
    JSONLogger logger_;         // Зависимость от конкретной реализации
    ModelArchiver archiver_;    // Зависимость от конкретной реализации
};

// ХОРОШО: низкая связанность
class FFTProcessor {
    ILogger* logger_;           // Зависимость от интерфейса
    IArchiver* archiver_;       // Зависимость от интерфейса
};
```

**Где достигается:**
- Использование интерфейсов (IGPUProcessor)
- Инверсия зависимостей (зависимость от абстракций)
- Python валидатор отделен от C++ (через JSON файлы)

---

### 5. **High Cohesion (Высокая связность)**

**Применение:**
```cpp
// BasicProfiler - высокая связность (делает ТОЛЬКО профилирование)
class BasicProfiler {
    void start_upload_timing();
    void end_upload_timing();
    void start_compute_timing();
    void end_compute_timing();
    void start_download_timing();
    void end_download_timing();
    BasicProfilingResult get_results();
};

// НЕ добавляем сюда:
// - валидацию (отдельный класс)
// - сохранение JSON (отдельный класс)
// - визуализацию (отдельный модуль Python)
```

**Где:**
- Каждый класс имеет одну четкую ответственность
- Методы класса тесно связаны по функциональности

---

### 6. **Polymorphism (Полиморфизм)**

**Применение:**
```cpp
// Вместо if-else для выбора алгоритма
void process_data(FFTAlgorithm algo, const InputSignalData& input) {
    IGPUProcessor* processor = nullptr;
    
    if (algo == FFTAlgorithm::WMMA) {
        processor = new FFT16_WMMA();
    } else if (algo == FFTAlgorithm::SHARED2D) {
        processor = new FFT16_Shared2D();
    } else if (algo == FFTAlgorithm::CUFFT) {
        processor = new FFT16_cuFFT();
    }
    
    // Полиморфный вызов
    auto result = processor->process(input);
}

// ЛУЧШЕ: через фабрику + полиморфизм
auto processor = ProcessorFactory::create(algo);
auto result = processor->process(input);
```

---

### 7. **Pure Fabrication (Чистая выдумка)**

**Применение в JSONLogger:**
```cpp
// JSONLogger не отражает предметную область (не "реальная" сущность)
// Это вспомогательный класс для решения технической задачи
class JSONLogger {
public:
    void write_profiling(const ProfilingResult& result);
    void write_validation_data(const ValidationData& data);
};
```

**Где:**
- ModelArchiver - чистая выдумка для управления версиями
- BasicProfiler - чистая выдумка для измерения времени
- DataContext - чистая выдумка для координации сохранения

---

### 8. **Indirection (Посредник)**

**Применение:**
```cpp
// Вместо прямой зависимости MainProgram → cuFFT
// Используем посредника IGPUProcessor

MainProgram → IGPUProcessor (интерфейс) ← FFT16_cuFFT → cuFFT
```

**Где:**
- IGPUProcessor - посредник между клиентом и реализациями
- DataContext - посредник между Tester и файловой системой

---

### 9. **Protected Variations (Устойчивость к изменениям)**

**Применение:**
```cpp
// Защита от изменений в GPU API через интерфейс
class IGPUProcessor {
    virtual OutputSpectralData process(const InputSignalData& input) = 0;
};

// Если cuFFT изменится → меняем только FFT16_cuFFT
// Если добавим AMD ROCm → добавляем новую реализацию
// Клиенты (MainProgram) не затрагиваются
```

**Где:**
- Защита от изменений CUDA API
- Защита от изменений форматов JSON (через JSONLogger)
- Python валидация защищает от изменений в C++

---

## 🔧 Дополнительные паттерны

### 1. **Dependency Injection (Внедрение зависимостей)**

**Применение:**
```cpp
class FFT16TestController {
private:
    IGPUProcessor& processor_;  // Внедряется извне
    ILogger& logger_;           // Внедряется извне
    
public:
    // Constructor Injection
    FFT16TestController(IGPUProcessor& processor, ILogger& logger)
        : processor_(processor), logger_(logger) {}
};

// Использование
auto wmma_processor = std::make_unique<FFT16_WMMA>();
auto json_logger = std::make_unique<JSONLogger>();
FFT16TestController controller(*wmma_processor, *json_logger);
```

---

### 2. **Service Locator** - ПОТЕНЦИАЛЬНО

**Потенциальное применение:**
```cpp
class ServiceLocator {
private:
    static std::map<std::string, std::any> services_;
    
public:
    template<typename T>
    static void register_service(const std::string& name, T* service) {
        services_[name] = service;
    }
    
    template<typename T>
    static T* get_service(const std::string& name) {
        return std::any_cast<T*>(services_[name]);
    }
};

// Использование
auto profiler = ServiceLocator::get_service<BasicProfiler>("profiler");
```

---

### 3. **Data Transfer Object (DTO)**

**Применение:**
```cpp
// InputSignalData - DTO для передачи данных между слоями
struct InputSignalData {
    std::vector<std::complex<float>> signal;
    StrobeConfig config;
    bool return_for_validation;
};

// OutputSpectralData - DTO для результатов
struct OutputSpectralData {
    std::vector<std::vector<std::complex<float>>> windows;
};

// ProfilingResult - DTO для профилирования
struct BasicProfilingResult {
    float upload_ms;
    float compute_ms;
    float download_ms;
    // ... metadata
};
```

---

## 📐 Принципы проектирования

### **SOLID принципы**

#### 1. **Single Responsibility Principle (SRP)**

**Применение:**
- `BasicProfiler` - ТОЛЬКО профилирование
- `JSONLogger` - ТОЛЬКО логирование
- `ModelArchiver` - ТОЛЬКО архивирование
- `SineGenerator` - ТОЛЬКО генерация синусоид

---

#### 2. **Open/Closed Principle (OCP)**

**Применение:**
```cpp
// Закрыт для модификации, открыт для расширения
class IGPUProcessor {
    virtual OutputSpectralData process(...) = 0;
};

// Добавляем новую реализацию БЕЗ изменения существующего кода
class FFT16_AMD_ROCm : public IGPUProcessor { ... };
```

---

#### 3. **Liskov Substitution Principle (LSP)**

**Применение:**
```cpp
// Любая реализация IGPUProcessor взаимозаменяема
void run_test(IGPUProcessor& processor) {
    auto result = processor.process(input);  // Работает с любой реализацией
}

// WMMA, Shared2D, cuFFT - все можно подставить
run_test(fft_wmma);
run_test(fft_shared2d);
run_test(fft_cufft);
```

---

#### 4. **Interface Segregation Principle (ISP)**

**Применение:**
```cpp
// НЕ один большой интерфейс
class IGPUProcessorAndProfilerAndValidator { ... };  // ПЛОХО

// Несколько маленьких интерфейсов
class IGPUProcessor { ... };        // Для обработки
class IProfiler { ... };            // Для профилирования
class IValidator { ... };           // Для валидации (Python)
```

---

#### 5. **Dependency Inversion Principle (DIP)**

**Применение:**
```cpp
// Высокоуровневый модуль (MainProgram) НЕ зависит от низкоуровневых (FFT16_WMMA)
// Оба зависят от абстракции (IGPUProcessor)

MainProgram → IGPUProcessor ← FFT16_WMMA
              (абстракция)    (реализация)
```

---

## 📊 Итоговая таблица паттернов

| Паттерн | Категория | Где используется | Приоритет |
|---------|-----------|------------------|-----------|
| **Layered Architecture** | Архитектурный | Вся система | ⭐⭐⭐ |
| **Pipes and Filters** | Архитектурный | Workflow обработки | ⭐⭐⭐ |
| **Plugin Architecture** | Архитектурный | ModelsFunction/ | ⭐⭐⭐ |
| **Repository** | Архитектурный | ModelArchiver | ⭐⭐ |
| **Factory Method** | GoF Creational | SignalGenerators | ⭐⭐⭐ |
| **Builder** | GoF Creational | Планируется | ⭐ |
| **Adapter** | GoF Structural | cuFFT wrapper | ⭐⭐⭐ |
| **Facade** | GoF Structural | DataContext | ⭐⭐⭐ |
| **Decorator** | GoF Structural | Потенциально | ⭐ |
| **Composite** | GoF Structural | Потенциально | ⭐ |
| **Strategy** | GoF Behavioral | FFT реализации | ⭐⭐⭐ |
| **Template Method** | GoF Behavioral | BaseGenerator | ⭐⭐ |
| **Observer** | GoF Behavioral | Потенциально | ⭐ |
| **Command** | GoF Behavioral | Потенциально | ⭐ |
| **Information Expert** | GRASP | StrobeConfig | ⭐⭐⭐ |
| **Creator** | GRASP | SineGenerator | ⭐⭐⭐ |
| **Controller** | GRASP | MainProgram | ⭐⭐⭐ |
| **Low Coupling** | GRASP | Вся архитектура | ⭐⭐⭐ |
| **High Cohesion** | GRASP | Все модули | ⭐⭐⭐ |
| **Polymorphism** | GRASP | IGPUProcessor | ⭐⭐⭐ |
| **Pure Fabrication** | GRASP | JSONLogger, Profiler | ⭐⭐⭐ |
| **Indirection** | GRASP | Интерфейсы | ⭐⭐⭐ |
| **Protected Variations** | GRASP | Интерфейсы | ⭐⭐⭐ |
| **Dependency Injection** | Дополнительный | Вся система | ⭐⭐⭐ |
| **DTO** | Дополнительный | InputSignalData, ... | ⭐⭐⭐ |

**Легенда:**
- ⭐⭐⭐ - Активно используется
- ⭐⭐ - Частично используется
- ⭐ - Планируется/потенциально

---

## 🎯 Выводы

### **Сильные стороны архитектуры:**

1. ✅ **Четкое разделение ответственностей** (High Cohesion)
2. ✅ **Низкая связанность** через интерфейсы (Low Coupling)
3. ✅ **Расширяемость** (Open/Closed Principle)
4. ✅ **Независимость от GPU vendor** (Strategy + Adapter)
5. ✅ **Тестируемость** (Dependency Injection)
6. ✅ **Переиспользование** (Factory Method, Template Method)

### **Рекомендации для развития:**

1. 🔄 Добавить **Builder** для сложных конфигураций
2. 🔄 Рассмотреть **Observer** для расширенного логирования
3. 🔄 Реализовать **Command** для undo/redo тестов
4. 🔄 Добавить **Decorator** для динамического добавления функциональности

---

**Версия:** 1.0  
**Автор:** Архитектурный анализ CudaCalc  
**Дата:** 09 октября 2025

