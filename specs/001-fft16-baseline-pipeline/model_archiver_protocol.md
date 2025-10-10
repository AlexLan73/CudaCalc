# 🔴 Протокол сохранения моделей (ModelArchiver)

**Статус:** КРИТИЧЕСКИЙ - ОБЯЗАТЕЛЕН К РЕАЛИЗАЦИИ  
**Приоритет:** Максимальный  
**Версия:** 1.0  
**Дата:** 10 октября 2025

---

## 🎯 Цель

**ПРЕДОТВРАТИТЬ ПОТЕРЮ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ!**

В прошлом теряли ценные результаты из-за перезаписи файлов. ModelArchiver решает эту проблему раз и навсегда через автоматическое версионирование.

---

## 📁 Структура хранения

```
DataContext/Models/
└── NVIDIA/                         # GPU производитель
    └── FFT/                        # Тип алгоритма
        └── 16/                     # Размер окна
            ├── model_2025_10_09_v1/    # Версия 1 (дата + номер)
            │   ├── fft16_wmma.cu       # Исходный kernel
            │   ├── fft16_wmma.cpp      # Wrapper код
            │   ├── description.txt     # Описание эксперимента
            │   ├── results.json        # Результаты профилирования
            │   └── metadata.json       # Метаданные (компилятор, флаги)
            ├── model_2025_10_09_v2/    # Версия 2 (оптимизация)
            │   ├── fft16_wmma.cu       # Улучшенный код
            │   ├── fft16_wmma.cpp
            │   ├── description.txt     # "Оптимизация twiddle factors"
            │   ├── results.json        # Новые результаты
            │   └── metadata.json
            └── model_2025_10_10_v3/    # Версия 3 (следующий день)
                └── ...
```

### Принципы:
1. ✅ **Каждый эксперимент = новая версия**
2. ✅ **НИКОГДА не перезаписывать**
3. ✅ **Исходники + результаты вместе**
4. ✅ **Автоматическое инкрементирование версий**
5. ✅ **История экспериментов сохраняется полностью**

---

## 🔧 API ModelArchiver

### Класс ModelArchiver

```cpp
#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace CudaCalc {

// Информация о модели
struct ModelInfo {
    std::string gpu_type;       // "NVIDIA", "AMD", "Intel"
    std::string algorithm;      // "FFT", "IFFT", "Correlation"
    int size;                   // 16, 32, 64, ...
    std::string version;        // "model_2025_10_09_v1"
    std::string description;    // Краткое описание
    
    std::filesystem::path get_path() const;
    std::string to_string() const;
};

// Метаданные эксперимента
struct ExperimentMetadata {
    std::string compiler;           // "nvcc 13.0"
    std::string compile_flags;      // "-O3 --use_fast_math"
    std::string date;               // "2025-10-09"
    std::string time;               // "14:30:45"
    std::string gpu_model;          // "NVIDIA RTX 3060"
    std::string cuda_version;       // "13.0"
    std::string driver_version;     // "535.104.05"
    int compute_capability;         // 86 (для RTX 3060)
};

// 🔴 КРИТИЧЕСКИЙ КЛАСС
class ModelArchiver {
private:
    std::filesystem::path base_path_;
    
    // Вспомогательные методы
    std::string get_current_date() const;
    std::string get_current_time() const;
    int parse_version_number(const std::string& version_string) const;
    
public:
    explicit ModelArchiver(const std::string& base_path = "DataContext/Models");
    
    // === ОСНОВНЫЕ ФУНКЦИИ ===
    
    /**
     * @brief Получить следующую версию (автоинкремент)
     * @return "model_YYYY_MM_DD_vN" где N = max_existing + 1
     */
    std::string get_next_version(
        const std::string& gpu_type,
        const std::string& algorithm,
        int size
    );
    
    /**
     * @brief Сохранить модель с результатами
     * @param info Информация о модели
     * @param source_files Список исходных файлов для копирования
     * @param results_json JSON с результатами профилирования
     * @param description Описание эксперимента
     * @param metadata Метаданные компиляции (опционально)
     * @return true если успешно
     */
    bool save_model(
        const ModelInfo& info,
        const std::vector<std::string>& source_files,
        const std::string& results_json,
        const std::string& description,
        const std::optional<ExperimentMetadata>& metadata = std::nullopt
    );
    
    /**
     * @brief Загрузить информацию о модели
     */
    std::optional<ModelInfo> load_model(const std::string& full_version_path);
    
    /**
     * @brief Сравнить несколько моделей по производительности
     * @return Таблица сравнения (Markdown format)
     */
    std::string compare_models(
        const std::vector<std::string>& version_paths
    );
    
    /**
     * @brief Список всех моделей для GPU/Algorithm/Size
     */
    std::vector<ModelInfo> list_models(
        const std::string& gpu_type,
        const std::string& algorithm,
        int size
    );
    
    /**
     * @brief Найти лучшую модель по метрике
     * @param metric "compute_time", "total_time", "memory_usage"
     */
    std::optional<ModelInfo> find_best_model(
        const std::string& gpu_type,
        const std::string& algorithm,
        int size,
        const std::string& metric = "compute_time"
    );
};

} // namespace CudaCalc
```

---

## 🔄 Workflow использования

### Типичный сценарий (в main_fft16_test.cpp):

```cpp
int main() {
    // 1. Генерация сигнала
    SineGenerator gen(4, 1024, 8);
    auto input = gen.generate(16, true);
    
    // 2. Тест FFT16_WMMA
    FFT16_WMMA fft;
    fft.initialize();
    
    BasicProfiler profiler;
    // ... профилирование ...
    auto prof_result = profiler.get_results();
    
    // 3. 🔴 ОБЯЗАТЕЛЬНО: Сохранение модели
    ModelArchiver archiver;
    
    ModelInfo info;
    info.gpu_type = "NVIDIA";
    info.algorithm = "FFT";
    info.size = 16;
    info.version = archiver.get_next_version("NVIDIA", "FFT", 16);  // auto: v1, v2, ...
    info.description = "Baseline FFT16 WMMA с линейной раскруткой";
    
    ExperimentMetadata metadata;
    metadata.compiler = "nvcc 13.0";
    metadata.compile_flags = "-O3 --use_fast_math -arch=sm_86";
    metadata.gpu_model = "NVIDIA RTX 3060";
    // ... остальные поля ...
    
    bool saved = archiver.save_model(
        info,
        {"ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cu",
         "ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cpp"},
        prof_result.to_json(),
        "Первая реализация FFT16 через Tensor Cores. "
        "Линейная раскрутка 4 stages. Compute time: 0.456ms",
        metadata
    );
    
    if (saved) {
        std::cout << "✅ Model saved: " << info.version << std::endl;
    } else {
        std::cerr << "❌ Failed to save model!" << std::endl;
    }
    
    // 4. То же для FFT16_Shared2D
    // ...
    
    return 0;
}
```

---

## 📋 Детали реализации

### save_model() - подробный алгоритм

```cpp
bool ModelArchiver::save_model(
    const ModelInfo& info,
    const std::vector<std::string>& source_files,
    const std::string& results_json,
    const std::string& description,
    const std::optional<ExperimentMetadata>& metadata
) {
    // === ШАГ 1: Создание директории ===
    auto model_path = base_path_ / info.gpu_type / info.algorithm / 
                     std::to_string(info.size) / info.version;
    
    try {
        std::filesystem::create_directories(model_path);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return false;
    }
    
    // === ШАГ 2: Копирование исходников ===
    for (const auto& src_file : source_files) {
        if (!std::filesystem::exists(src_file)) {
            std::cerr << "Source file not found: " << src_file << std::endl;
            continue;
        }
        
        auto dest = model_path / std::filesystem::path(src_file).filename();
        try {
            std::filesystem::copy_file(
                src_file, dest,
                std::filesystem::copy_options::overwrite_existing
            );
            std::cout << "  ✓ Copied: " << src_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Failed to copy: " << src_file << std::endl;
        }
    }
    
    // === ШАГ 3: Сохранение results.json ===
    {
        std::ofstream file(model_path / "results.json");
        file << results_json;
    }
    
    // === ШАГ 4: Сохранение description.txt ===
    {
        std::ofstream file(model_path / "description.txt");
        file << description << std::endl;
        file << std::endl;
        file << "Created: " << get_current_date() << " " << get_current_time() << std::endl;
    }
    
    // === ШАГ 5: Сохранение metadata.json (если есть) ===
    if (metadata.has_value()) {
        nlohmann::json meta_json;
        meta_json["compiler"] = metadata->compiler;
        meta_json["compile_flags"] = metadata->compile_flags;
        meta_json["date"] = metadata->date;
        meta_json["time"] = metadata->time;
        meta_json["gpu_model"] = metadata->gpu_model;
        meta_json["cuda_version"] = metadata->cuda_version;
        meta_json["driver_version"] = metadata->driver_version;
        meta_json["compute_capability"] = metadata->compute_capability;
        
        std::ofstream file(model_path / "metadata.json");
        file << meta_json.dump(2);
    }
    
    // === ШАГ 6: Создание index.md (README для модели) ===
    {
        std::ofstream file(model_path / "index.md");
        file << "# " << info.version << std::endl;
        file << std::endl;
        file << "**Algorithm:** " << info.algorithm << info.size << std::endl;
        file << "**GPU:** " << info.gpu_type << std::endl;
        file << "**Description:** " << info.description << std::endl;
        file << std::endl;
        file << "## Files" << std::endl;
        for (const auto& src : source_files) {
            file << "- `" << std::filesystem::path(src).filename().string() << "`" << std::endl;
        }
        file << "- `results.json` - profiling results" << std::endl;
        file << "- `description.txt` - experiment description" << std::endl;
        if (metadata.has_value()) {
            file << "- `metadata.json` - compilation metadata" << std::endl;
        }
    }
    
    std::cout << "✅ Model successfully saved to: " << model_path << std::endl;
    return true;
}
```

---

### get_next_version() - автоинкремент

```cpp
std::string ModelArchiver::get_next_version(
    const std::string& gpu_type,
    const std::string& algorithm,
    int size
) {
    auto path = base_path_ / gpu_type / algorithm / std::to_string(size);
    
    int max_version = 0;
    
    // Поиск существующих версий
    if (std::filesystem::exists(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (!entry.is_directory()) continue;
            
            std::string dirname = entry.path().filename().string();
            
            // Парсинг "model_2025_10_09_v3" → v3
            size_t v_pos = dirname.find("_v");
            if (v_pos != std::string::npos) {
                try {
                    int v_num = std::stoi(dirname.substr(v_pos + 2));
                    max_version = std::max(max_version, v_num);
                } catch (...) {
                    // Игнорируем некорректные имена
                }
            }
        }
    }
    
    // Следующая версия
    std::string date = get_current_date();  // "2025_10_09"
    return "model_" + date + "_v" + std::to_string(max_version + 1);
}
```

---

### compare_models() - сравнение версий

```cpp
std::string ModelArchiver::compare_models(
    const std::vector<std::string>& version_paths
) {
    std::ostringstream table;
    table << "# Model Comparison\n\n";
    table << "| Version | Compute (ms) | Upload (ms) | Download (ms) | Total (ms) |\n";
    table << "|---------|-------------|-------------|--------------|------------|\n";
    
    for (const auto& vpath : version_paths) {
        auto results_file = vpath + "/results.json";
        if (!std::filesystem::exists(results_file)) continue;
        
        // Парсинг JSON
        std::ifstream file(results_file);
        nlohmann::json j;
        file >> j;
        
        std::string version = std::filesystem::path(vpath).filename().string();
        float compute = j["profiling"]["basic"]["compute_ms"];
        float upload = j["profiling"]["basic"]["upload_ms"];
        float download = j["profiling"]["basic"]["download_ms"];
        float total = j["profiling"]["basic"]["total_ms"];
        
        table << "| " << version << " | "
              << compute << " | "
              << upload << " | "
              << download << " | "
              << total << " |\n";
    }
    
    return table.str();
}
```

---

### list_models() - список всех версий

```cpp
std::vector<ModelInfo> ModelArchiver::list_models(
    const std::string& gpu_type,
    const std::string& algorithm,
    int size
) {
    std::vector<ModelInfo> models;
    auto path = base_path_ / gpu_type / algorithm / std::to_string(size);
    
    if (!std::filesystem::exists(path)) {
        return models;  // Пустой список
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_directory()) continue;
        
        ModelInfo info;
        info.gpu_type = gpu_type;
        info.algorithm = algorithm;
        info.size = size;
        info.version = entry.path().filename().string();
        
        // Загрузка description
        auto desc_file = entry.path() / "description.txt";
        if (std::filesystem::exists(desc_file)) {
            std::ifstream file(desc_file);
            std::getline(file, info.description);
        }
        
        models.push_back(info);
    }
    
    // Сортировка по версии (v1, v2, v3...)
    std::sort(models.begin(), models.end(), [](const auto& a, const auto& b) {
        return a.version < b.version;
    });
    
    return models;
}
```

---

### find_best_model() - поиск оптимальной

```cpp
std::optional<ModelInfo> ModelArchiver::find_best_model(
    const std::string& gpu_type,
    const std::string& algorithm,
    int size,
    const std::string& metric
) {
    auto models = list_models(gpu_type, algorithm, size);
    if (models.empty()) return std::nullopt;
    
    ModelInfo best = models[0];
    float best_value = std::numeric_limits<float>::max();
    
    for (const auto& model : models) {
        auto results_path = base_path_ / gpu_type / algorithm / 
                           std::to_string(size) / model.version / "results.json";
        
        if (!std::filesystem::exists(results_path)) continue;
        
        std::ifstream file(results_path);
        nlohmann::json j;
        file >> j;
        
        float value = 0.0f;
        if (metric == "compute_time") {
            value = j["profiling"]["basic"]["compute_ms"];
        } else if (metric == "total_time") {
            value = j["profiling"]["basic"]["total_ms"];
        } else if (metric == "memory_usage") {
            if (j["profiling"]["memory"]["enabled"]) {
                value = j["profiling"]["memory"]["allocated_vram_mb"];
            }
        }
        
        if (value < best_value) {
            best_value = value;
            best = model;
        }
    }
    
    return best;
}
```

---

## 💼 Примеры использования

### Пример 1: Сохранение первого эксперимента

```cpp
ModelArchiver archiver;

// Первый эксперимент
ModelInfo info;
info.gpu_type = "NVIDIA";
info.algorithm = "FFT";
info.size = 16;
info.version = archiver.get_next_version("NVIDIA", "FFT", 16);  // → "model_2025_10_09_v1"
info.description = "Baseline FFT16 WMMA";

archiver.save_model(
    info,
    {"fft16_wmma.cu", "fft16_wmma.cpp"},
    results_json,
    "Первая реализация с линейной раскруткой"
);
```

### Пример 2: Сохранение оптимизации

```cpp
// На следующий день, после оптимизации
info.version = archiver.get_next_version("NVIDIA", "FFT", 16);  // → "model_2025_10_10_v2"
info.description = "Оптимизация: precomputed twiddle factors";

archiver.save_model(
    info,
    {"fft16_wmma_optimized.cu", "fft16_wmma.cpp"},
    new_results_json,
    "Оптимизация twiddle factors. Compute: 0.320ms (было 0.456ms)"
);

// v1 НЕ перезаписывается! Обе версии сохранены!
```

### Пример 3: Сравнение версий

```cpp
auto models = archiver.list_models("NVIDIA", "FFT", 16);

std::cout << "Found " << models.size() << " models:" << std::endl;
for (const auto& m : models) {
    std::cout << "  - " << m.version << ": " << m.description << std::endl;
}

// Сравнение производительности
auto comparison = archiver.compare_models({
    "DataContext/Models/NVIDIA/FFT/16/model_2025_10_09_v1",
    "DataContext/Models/NVIDIA/FFT/16/model_2025_10_10_v2"
});

std::cout << comparison << std::endl;
```

**Вывод:**
```
| Version | Compute (ms) | Upload (ms) | Download (ms) | Total (ms) |
|---------|-------------|-------------|--------------|------------|
| model_2025_10_09_v1 | 0.456 | 0.123 | 0.089 | 0.668 |
| model_2025_10_10_v2 | 0.320 | 0.125 | 0.091 | 0.536 |

✅ v2 faster by 19.8% (0.668ms → 0.536ms)
```

### Пример 4: Поиск лучшей модели

```cpp
auto best = archiver.find_best_model("NVIDIA", "FFT", 16, "compute_time");

if (best.has_value()) {
    std::cout << "Best model: " << best->version << std::endl;
    std::cout << "Description: " << best->description << std::endl;
    
    // Можем скопировать в Production/
    // или использовать как reference
}
```

---

## 🔒 Гарантии безопасности

### Что гарантируется:

1. ✅ **Никогда не перезаписываем**
   - Даже если запустить дважды в один день
   - v1, v2, v3... автоматически

2. ✅ **Атомарность**
   - Либо вся модель сохранена, либо ничего
   - Проверка существования файлов

3. ✅ **Полная история**
   - Все эксперименты доступны
   - Можно вернуться к любой версии
   - Сравнить производительность

4. ✅ **Метаданные**
   - Знаем при каких условиях компилировалось
   - Можем воспроизвести эксперимент

---

## 📊 Структура сохранённых данных

### results.json (пример)
```json
{
  "profiling": {
    "basic": {
      "upload_ms": 0.123,
      "compute_ms": 0.456,
      "download_ms": 0.089,
      "total_ms": 0.668
    },
    "memory": {
      "enabled": false
    }
  },
  "algorithm": "FFT16_WMMA",
  "timestamp": "2025-10-09T14:30:45"
}
```

### description.txt (пример)
```
Baseline FFT16 WMMA с линейной раскруткой

Детали:
- 4 butterfly stages развёрнуты линейно
- Shared memory: [64 FFTs][16 points]
- FP16 через __half2
- FFT shift в kernel

Результаты:
- Compute time: 0.456 ms
- Total time: 0.668 ms

Created: 2025-10-09 14:30:45
```

### metadata.json (пример)
```json
{
  "compiler": "nvcc 13.0",
  "compile_flags": "-O3 --use_fast_math -arch=sm_86 -lineinfo",
  "date": "2025-10-09",
  "time": "14:30:45",
  "gpu_model": "NVIDIA RTX 3060",
  "cuda_version": "13.0",
  "driver_version": "535.104.05",
  "compute_capability": 86
}
```

---

## ⚠️ КРИТИЧЕСКИЕ ТРЕБОВАНИЯ

### ОБЯЗАТЕЛЬНО:

1. ✅ **Вызывать после КАЖДОГО теста**
   ```cpp
   // ❌ ПЛОХО
   run_test();
   // Забыли сохранить!
   
   // ✅ ХОРОШО
   run_test();
   archiver.save_model(...);  // ВСЕГДА!
   ```

2. ✅ **Проверять успешность**
   ```cpp
   if (!archiver.save_model(...)) {
       std::cerr << "CRITICAL: Failed to save model!" << std::endl;
       // Уведомить пользователя!
   }
   ```

3. ✅ **Не полагаться на ручное копирование**
   - Автоматизация через ModelArchiver
   - Человеческий фактор = потеря данных

---

## 📝 TODO для реализации

- [ ] Реализовать ModelArchiver класс
- [ ] Unit тесты для ModelArchiver
- [ ] Интеграция в main_fft16_test.cpp
- [ ] Проверка на реальных данных
- [ ] Документирование в MemoryBank
- [ ] Добавить в tasks.md как высокоприоритетную задачу

---

## 🔗 Связанные документы

- **spec.md** - Спецификация (FR-6: ModelArchiver)
- **plan.md** - План реализации (Фаза 4)
- **ROADMAP.md** - Общий план проекта

---

**Этот протокол - КРИТИЧЕСКИЙ для успеха проекта!**

**Версия:** 1.0  
**Статус:** Готов к реализации  
**Автор:** AlexLan73  
**Дата:** 10 октября 2025

