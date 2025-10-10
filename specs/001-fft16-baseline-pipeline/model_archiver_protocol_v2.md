# 🔴 Протокол сохранения моделей v2.0 (ModelArchiver Enhanced)

**Статус:** КРИТИЧЕСКИЙ - ОБЯЗАТЕЛЕН К РЕАЛИЗАЦИИ  
**Приоритет:** Максимальный  
**Версия:** 2.0 (улучшенная на основе best practices)  
**Дата:** 10 октября 2025  
**Основано на:** Статья "Надёжный склад результатов для AI"

---

## 🎯 Цель

**ПРЕДОТВРАТИТЬ ПОТЕРЮ РЕЗУЛЬТАТОВ + АВТОМАТИЧЕСКИЙ ВЫБОР ЛУЧШИХ МОДЕЛЕЙ!**

Комбинация нашего ModelArchiver + лучшие практики из статьи.

---

## 📁 Улучшенная структура хранения

```
DataContext/
├── Experiments/                    # ⭐ НОВОЕ: Все эксперименты (immutable)
│   └── runs/
│       ├── exp-20251009-1030-fft16-wmma-baseline/
│       │   ├── source/
│       │   │   ├── fft16_wmma.cu           # Копия исходника
│       │   │   └── fft16_wmma.cpp
│       │   ├── artifacts/
│       │   │   └── gpu_output.bin          # Сырые результаты (опционально)
│       │   ├── logs/
│       │   │   ├── run.log                 # Полный лог выполнения
│       │   │   └── errors.log              # Ошибки (если были)
│       │   ├── results.json                # Профилирование
│       │   ├── validation_input.json       # Для Python
│       │   ├── config.lock.json            # ⭐ Зафиксированная конфигурация
│       │   ├── manifest.json               # ⭐ Список файлов + SHA256
│       │   └── summary.md                  # ⭐ Краткий отчёт
│       ├── exp-20251009-1430-fft16-wmma-opt-twiddle/
│       └── exp-20251010-0900-fft16-wmma-opt-v2/
│
├── Best/                           # ⭐ НОВОЕ: Витрина лучших моделей
│   └── FFT/
│       └── 16/
│           └── NVIDIA/
│               ├── best.link       # Симлинк на лучший эксперимент
│               ├── best_info.json  # Метаданные лучшей модели
│               └── README.md       # Критерии выбора, история
│
├── Registry/                       # ⭐ НОВОЕ: Реестр всех экспериментов
│   ├── experiments.csv             # Таблица всех запусков
│   ├── artifacts.csv               # Все артефакты с хешами
│   └── metrics_history.csv         # История метрик
│
├── Models/                         # СТАРОЕ: Оставляем для совместимости
│   └── NVIDIA/FFT/16/... (deprecated, можно удалить позже)
│
├── Reports/                        # JSON профилирования (как было)
└── ValidationData/                 # Данные для Python (как было)
```

---

## 🔧 API ModelArchiver v2.0

### Расширенные структуры

```cpp
#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace CudaCalc {

// Конфигурация эксперимента
struct ExperimentConfig {
    std::string name;           // "fft16-wmma-baseline"
    std::string id;             // "exp-20251009-1030-fft16-wmma-baseline" (auto)
    
    // Параметры
    std::string gpu_type;       // "NVIDIA"
    std::string algorithm;      // "FFT"
    int size;                   // 16
    int ray_count;              // 4
    int points_per_ray;         // 1024
    int window_fft;             // 16
    
    // Метрика для определения "лучший"
    std::string primary_metric; // "compute_time"
    std::string mode;           // "min" или "max"
    
    // Metadata
    std::string compiler;       // "nvcc 13.0"
    std::string compile_flags;  // "-O3 --use_fast_math"
    int seed;                   // Random seed (для воспроизводимости)
};

// Информация об эксперименте
struct ExperimentInfo {
    std::string experiment_id;      // Уникальный ID
    std::filesystem::path path;     // Полный путь
    ExperimentConfig config;        // Конфигурация
    
    // Результаты
    float primary_value;            // Значение primary метрики
    bool is_best;                   // Это лучший эксперимент?
    std::string timestamp;          // "2025-10-09T14:30:45"
    std::string git_commit;         // Git hash (опционально)
};

// Манифест артефактов
struct ArtifactManifest {
    struct File {
        std::string path;           // Относительный путь
        std::string sha256;         // Хеш для проверки
        size_t size_bytes;          // Размер
        std::string type;           // "source", "result", "log"
    };
    
    std::vector<File> files;
    std::string generated_at;       // Timestamp создания
};

// 🔴 УЛУЧШЕННЫЙ КЛАСС
class ModelArchiverV2 {
private:
    std::filesystem::path experiments_root_;  // "DataContext/Experiments"
    std::filesystem::path best_root_;         // "DataContext/Best"
    std::filesystem::path registry_root_;     // "DataContext/Registry"
    
    // Вспомогательные
    std::string generate_experiment_id(const ExperimentConfig& config);
    std::string calculate_sha256(const std::filesystem::path& file);
    ArtifactManifest create_manifest(const std::filesystem::path& exp_dir);
    
public:
    ModelArchiverV2();
    
    // === PRE-RUN: Подготовка эксперимента ===
    /**
     * @brief Создать директорию эксперимента, сохранить config.lock.json
     * @return Путь к директории эксперимента
     */
    std::filesystem::path pre_run(const ExperimentConfig& config);
    
    // === SAVE: Сохранение результатов ===
    /**
     * @brief Сохранить эксперимент (immutable)
     * @param experiment_id ID из pre_run()
     * @param source_files Исходники
     * @param results Результаты профилирования
     * @param description Описание
     * @return ExperimentInfo с заполненными данными
     */
    ExperimentInfo save_experiment(
        const std::string& experiment_id,
        const std::vector<std::string>& source_files,
        const nlohmann::json& results,
        const std::string& description
    );
    
    // === POST-RUN: Финализация ===
    /**
     * @brief Создать manifest, обновить registry, проверить best
     * @param experiment_id ID эксперимента
     * @return true если успешно
     */
    bool post_run(const std::string& experiment_id);
    
    // === BEST MANAGEMENT ===
    /**
     * @brief Обновить /best/ только если эксперимент лучше
     * @param exp_info Информация об эксперименте
     * @return true если best обновлён
     */
    bool update_best_if_improved(const ExperimentInfo& exp_info);
    
    /**
     * @brief Получить текущую лучшую модель
     */
    std::optional<ExperimentInfo> get_best_model(
        const std::string& gpu_type,
        const std::string& algorithm,
        int size
    );
    
    // === REGISTRY ===
    /**
     * @brief Добавить запись в реестр экспериментов
     */
    bool register_experiment(const ExperimentInfo& exp_info);
    
    /**
     * @brief Список всех экспериментов из реестра
     */
    std::vector<ExperimentInfo> list_all_experiments();
    
    /**
     * @brief Поиск экспериментов по критериям
     */
    std::vector<ExperimentInfo> query_experiments(
        const std::string& gpu_type,
        const std::string& algorithm,
        int size,
        const std::string& order_by = "primary_value"  // сортировка
    );
    
    // === VALIDATION ===
    /**
     * @brief Проверить целостность артефактов (SHA256)
     */
    bool verify_artifacts(const std::string& experiment_id);
    
    /**
     * @brief Проверить все эксперименты
     */
    std::vector<std::string> verify_all_artifacts();
};

} // namespace CudaCalc
```

---

## 🔄 Улучшенный Workflow (с hooks)

### Полный цикл эксперимента:

```cpp
int main() {
    ModelArchiverV2 archiver;
    
    // === 1. PRE-RUN: Подготовка ===
    ExperimentConfig config;
    config.name = "fft16-wmma-baseline";
    config.gpu_type = "NVIDIA";
    config.algorithm = "FFT";
    config.size = 16;
    config.ray_count = 4;
    config.points_per_ray = 1024;
    config.window_fft = 16;
    config.primary_metric = "compute_time";  // ⭐ Критерий "лучше"
    config.mode = "min";                     // ⭐ min = быстрее лучше
    config.compiler = "nvcc 13.0";
    config.compile_flags = "-O3 --use_fast_math -arch=sm_86";
    
    auto exp_path = archiver.pre_run(config);
    std::string exp_id = config.id;  // "exp-20251009-1030-fft16-wmma-baseline"
    
    std::cout << "Experiment ID: " << exp_id << std::endl;
    std::cout << "Path: " << exp_path << std::endl;
    
    // === 2. RUN: Выполнение эксперимента ===
    SineGenerator gen(4, 1024, 8);
    auto input = gen.generate(16, true);
    
    FFT16_WMMA fft;
    fft.initialize();
    
    BasicProfiler profiler;
    // ... профилирование ...
    
    auto results = profiler.get_results();
    
    // === 3. SAVE: Сохранение результатов ===
    nlohmann::json results_json = results.to_json();
    
    auto exp_info = archiver.save_experiment(
        exp_id,
        {"ModelsFunction/.../fft16_wmma.cu", "ModelsFunction/.../fft16_wmma.cpp"},
        results_json,
        "Baseline FFT16 WMMA. Linear unroll 4 stages."
    );
    
    std::cout << "✓ Experiment saved" << std::endl;
    std::cout << "  Primary metric (compute_time): " << exp_info.primary_value << " ms" << std::endl;
    
    // === 4. POST-RUN: Финализация ===
    bool ok = archiver.post_run(exp_id);
    
    if (ok) {
        std::cout << "✓ Post-run complete:" << std::endl;
        std::cout << "  - Manifest created (SHA256)" << std::endl;
        std::cout << "  - Registry updated" << std::endl;
        
        if (exp_info.is_best) {
            std::cout << "  - 🏆 NEW BEST MODEL! Updated /Best/" << std::endl;
        } else {
            std::cout << "  - Current best is still better" << std::endl;
        }
    }
    
    return 0;
}
```

---

## 📋 Детальная реализация

### pre_run() - Подготовка

```cpp
std::filesystem::path ModelArchiverV2::pre_run(ExperimentConfig& config) {
    // 1. Генерация ID (если не задан)
    if (config.id.empty()) {
        config.id = generate_experiment_id(config);
        // → "exp-20251009-1030-fft16-wmma-baseline"
    }
    
    // 2. Создание директории
    auto exp_path = experiments_root_ / "runs" / config.id;
    
    // ⚠️ FAIL-FAST: если существует - остановка!
    if (std::filesystem::exists(exp_path)) {
        throw std::runtime_error(
            "Experiment directory already exists! " + exp_path.string() +
            "\nThis prevents accidental overwrites."
        );
    }
    
    // 3. Создание структуры
    std::filesystem::create_directories(exp_path / "source");
    std::filesystem::create_directories(exp_path / "artifacts");
    std::filesystem::create_directories(exp_path / "logs");
    
    // 4. Сохранение config.lock.json
    nlohmann::json config_json;
    config_json["name"] = config.name;
    config_json["id"] = config.id;
    config_json["timestamp"] = get_timestamp();
    config_json["gpu_type"] = config.gpu_type;
    config_json["algorithm"] = config.algorithm;
    config_json["size"] = config.size;
    config_json["primary_metric"] = config.primary_metric;
    config_json["mode"] = config.mode;
    config_json["compiler"] = config.compiler;
    config_json["compile_flags"] = config.compile_flags;
    config_json["git_commit"] = get_git_commit();  // ⭐ Git hash!
    
    std::ofstream(exp_path / "config.lock.json") << config_json.dump(2);
    
    std::cout << "✅ Pre-run complete: " << config.id << std::endl;
    std::cout << "   Path: " << exp_path << std::endl;
    
    return exp_path;
}
```

---

### save_experiment() - Сохранение

```cpp
ExperimentInfo ModelArchiverV2::save_experiment(
    const std::string& experiment_id,
    const std::vector<std::string>& source_files,
    const nlohmann::json& results,
    const std::string& description
) {
    auto exp_path = experiments_root_ / "runs" / experiment_id;
    
    if (!std::filesystem::exists(exp_path)) {
        throw std::runtime_error("Experiment directory not found! Call pre_run() first.");
    }
    
    // 1. Копирование исходников
    for (const auto& src : source_files) {
        auto dest = exp_path / "source" / std::filesystem::path(src).filename();
        std::filesystem::copy_file(src, dest);
    }
    
    // 2. Сохранение results.json
    std::ofstream(exp_path / "results.json") << results.dump(2);
    
    // 3. Сохранение description
    std::ofstream desc_file(exp_path / "description.txt");
    desc_file << description << std::endl;
    desc_file << std::endl;
    desc_file << "Created: " << get_timestamp() << std::endl;
    desc_file << "Git commit: " << get_git_commit() << std::endl;
    
    // 4. Извлечение primary метрики
    float primary_value = extract_primary_metric(results, config.primary_metric);
    
    // 5. Создание ExperimentInfo
    ExperimentInfo info;
    info.experiment_id = experiment_id;
    info.path = exp_path;
    // config загружается из config.lock.json
    info.primary_value = primary_value;
    info.is_best = false;  // Проверим в post_run
    info.timestamp = get_timestamp();
    info.git_commit = get_git_commit();
    
    return info;
}
```

---

### post_run() - Финализация (КЛЮЧЕВАЯ ФУНКЦИЯ!)

```cpp
bool ModelArchiverV2::post_run(const std::string& experiment_id) {
    auto exp_path = experiments_root_ / "runs" / experiment_id;
    
    // === ШАГ 1: Создание manifest.json (SHA256) ===
    auto manifest = create_manifest(exp_path);
    
    nlohmann::json manifest_json;
    manifest_json["generated_at"] = manifest.generated_at;
    manifest_json["experiment_id"] = experiment_id;
    
    for (const auto& file : manifest.files) {
        manifest_json["files"].push_back({
            {"path", file.path},
            {"sha256", file.sha256},
            {"size_bytes", file.size_bytes},
            {"type", file.type}
        });
    }
    
    std::ofstream(exp_path / "manifest.json") << manifest_json.dump(2);
    std::cout << "✓ Manifest created (" << manifest.files.size() << " files)" << std::endl;
    
    // === ШАГ 2: Загрузка ExperimentInfo ===
    auto config = load_config(exp_path / "config.lock.json");
    auto results = load_results(exp_path / "results.json");
    
    ExperimentInfo exp_info;
    exp_info.experiment_id = experiment_id;
    exp_info.path = exp_path;
    exp_info.config = config;
    exp_info.primary_value = extract_primary_metric(results, config.primary_metric);
    exp_info.timestamp = get_timestamp();
    exp_info.git_commit = get_git_commit();
    
    // === ШАГ 3: Обновление реестра ===
    register_experiment(exp_info);
    std::cout << "✓ Registry updated" << std::endl;
    
    // === ШАГ 4: Проверка и обновление BEST ===
    bool is_best = update_best_if_improved(exp_info);
    exp_info.is_best = is_best;
    
    if (is_best) {
        std::cout << "🏆 NEW BEST MODEL!" << std::endl;
        std::cout << "   Previous: ..." << std::endl;
        std::cout << "   New: " << exp_info.primary_value << " " << config.primary_metric << std::endl;
    }
    
    // === ШАГ 5: Создание summary.md ===
    create_summary(exp_path, exp_info);
    std::cout << "✓ Summary created" << std::endl;
    
    return true;
}
```

---

### update_best_if_improved() - Обновление лучшей модели

```cpp
bool ModelArchiverV2::update_best_if_improved(const ExperimentInfo& exp_info) {
    // 1. Путь к best
    auto best_path = best_root_ / exp_info.config.algorithm / 
                    std::to_string(exp_info.config.size) / 
                    exp_info.config.gpu_type;
    
    std::filesystem::create_directories(best_path);
    
    // 2. Загрузка текущего best (если есть)
    auto best_info_file = best_path / "best_info.json";
    
    float current_best_value;
    bool has_current_best = false;
    
    if (std::filesystem::exists(best_info_file)) {
        std::ifstream file(best_info_file);
        nlohmann::json j;
        file >> j;
        current_best_value = j["primary_value"];
        has_current_best = true;
    }
    
    // 3. Сравнение
    bool is_better = false;
    
    if (!has_current_best) {
        is_better = true;  // Первый эксперимент = автоматически best
    } else {
        if (exp_info.config.mode == "min") {
            is_better = (exp_info.primary_value < current_best_value);
        } else {  // "max"
            is_better = (exp_info.primary_value > current_best_value);
        }
    }
    
    // 4. Обновление (только если лучше!)
    if (is_better) {
        // Удаление старого симлинка
        std::filesystem::remove(best_path / "best.link");
        
        // Создание нового симлинка
        std::filesystem::create_symlink(
            exp_info.path,
            best_path / "best.link"
        );
        
        // Обновление best_info.json
        nlohmann::json best_json;
        best_json["experiment_id"] = exp_info.experiment_id;
        best_json["primary_metric"] = exp_info.config.primary_metric;
        best_json["primary_value"] = exp_info.primary_value;
        best_json["timestamp"] = exp_info.timestamp;
        best_json["git_commit"] = exp_info.git_commit;
        best_json["path"] = exp_info.path.string();
        
        std::ofstream(best_info_file) << best_json.dump(2);
        
        // Обновление README.md в best/
        update_best_readme(best_path, exp_info, current_best_value);
        
        return true;
    }
    
    return false;  // Не лучше, best не обновлён
}
```

---

### register_experiment() - Добавление в реестр

```cpp
bool ModelArchiverV2::register_experiment(const ExperimentInfo& exp_info) {
    auto registry_csv = registry_root_ / "experiments.csv";
    
    bool write_header = !std::filesystem::exists(registry_csv);
    
    std::ofstream file(registry_csv, std::ios::app);
    
    if (write_header) {
        file << "experiment_id,gpu_type,algorithm,size,primary_metric,primary_value,"
             << "mode,is_best,timestamp,git_commit,path" << std::endl;
    }
    
    file << exp_info.experiment_id << ","
         << exp_info.config.gpu_type << ","
         << exp_info.config.algorithm << ","
         << exp_info.config.size << ","
         << exp_info.config.primary_metric << ","
         << exp_info.primary_value << ","
         << exp_info.config.mode << ","
         << (exp_info.is_best ? "true" : "false") << ","
         << exp_info.timestamp << ","
         << exp_info.git_commit << ","
         << exp_info.path.string() << std::endl;
    
    return true;
}
```

**Пример experiments.csv:**
```csv
experiment_id,gpu_type,algorithm,size,primary_metric,primary_value,mode,is_best,timestamp,git_commit,path
exp-20251009-1030-fft16-wmma-baseline,NVIDIA,FFT,16,compute_time,0.456,min,true,2025-10-09T10:30:45,abc123,/path/to/exp
exp-20251009-1430-fft16-wmma-opt-v2,NVIDIA,FFT,16,compute_time,0.320,min,true,2025-10-09T14:30:12,def456,/path/to/exp
exp-20251010-0900-fft16-shared2d,NVIDIA,FFT,16,compute_time,0.512,min,false,2025-10-10T09:00:23,ghi789,/path/to/exp
```

---

### create_manifest() - SHA256 для всех файлов

```cpp
ArtifactManifest ModelArchiverV2::create_manifest(const std::filesystem::path& exp_dir) {
    ArtifactManifest manifest;
    manifest.generated_at = get_timestamp();
    
    // Сканирование всех файлов
    for (const auto& entry : std::filesystem::recursive_directory_iterator(exp_dir)) {
        if (!entry.is_regular_file()) continue;
        
        auto rel_path = std::filesystem::relative(entry.path(), exp_dir);
        
        ArtifactManifest::File file;
        file.path = rel_path.string();
        file.sha256 = calculate_sha256(entry.path());  // ⭐ Хеш!
        file.size_bytes = std::filesystem::file_size(entry.path());
        
        // Определение типа
        if (rel_path.string().find("source/") == 0) {
            file.type = "source";
        } else if (rel_path.string().find(".json") != std::string::npos) {
            file.type = "result";
        } else if (rel_path.string().find("logs/") == 0) {
            file.type = "log";
        } else {
            file.type = "other";
        }
        
        manifest.files.push_back(file);
    }
    
    return manifest;
}
```

---

### create_summary() - Краткий отчёт

```cpp
void ModelArchiverV2::create_summary(
    const std::filesystem::path& exp_path,
    const ExperimentInfo& exp_info
) {
    std::ofstream file(exp_path / "summary.md");
    
    file << "# Experiment Summary" << std::endl;
    file << std::endl;
    file << "**ID:** `" << exp_info.experiment_id << "`" << std::endl;
    file << "**Date:** " << exp_info.timestamp << std::endl;
    file << "**Git commit:** " << exp_info.git_commit << std::endl;
    file << std::endl;
    
    file << "## Configuration" << std::endl;
    file << "- GPU: " << exp_info.config.gpu_type << std::endl;
    file << "- Algorithm: " << exp_info.config.algorithm << exp_info.config.size << std::endl;
    file << "- Compiler: " << exp_info.config.compiler << std::endl;
    file << "- Flags: `" << exp_info.config.compile_flags << "`" << std::endl;
    file << std::endl;
    
    file << "## Results" << std::endl;
    file << "- **Primary metric (" << exp_info.config.primary_metric << "):** "
         << exp_info.primary_value << " ms" << std::endl;
    file << "- **Is best:** " << (exp_info.is_best ? "✅ YES" : "❌ NO") << std::endl;
    file << std::endl;
    
    file << "## Artifacts" << std::endl;
    file << "- Source code: `source/`" << std::endl;
    file << "- Results: `results.json`" << std::endl;
    file << "- Manifest: `manifest.json` (SHA256 checksums)" << std::endl;
    file << "- Logs: `logs/run.log`" << std::endl;
    file << std::endl;
    
    if (exp_info.is_best) {
        file << "## 🏆 Best Model" << std::endl;
        file << "This is currently the BEST model for " 
             << exp_info.config.algorithm << exp_info.config.size << "!" << std::endl;
    }
}
```

---

## 📊 Структура Best/

```
Best/FFT/16/NVIDIA/
├── best.link → ../../../Experiments/runs/exp-20251009-1430-fft16-wmma-opt-v2/
├── best_info.json
└── README.md
```

**best_info.json:**
```json
{
  "experiment_id": "exp-20251009-1430-fft16-wmma-opt-v2",
  "primary_metric": "compute_time",
  "primary_value": 0.320,
  "timestamp": "2025-10-09T14:30:12",
  "git_commit": "def456",
  "path": "/path/to/exp-20251009-1430-fft16-wmma-opt-v2"
}
```

**README.md:**
```markdown
# Best FFT16 for NVIDIA

**Current best:** exp-20251009-1430-fft16-wmma-opt-v2

**Metric:** compute_time (min)  
**Value:** 0.320 ms

## History
| Date | Experiment | Value | Improvement |
|------|-----------|-------|-------------|
| 2025-10-09 10:30 | exp-...-baseline | 0.456 ms | - (first) |
| 2025-10-09 14:30 | exp-...-opt-v2 | 0.320 ms | ↓ 29.8% |

## How to use
```bash
# Симлинк указывает на лучший эксперимент
cd Best/FFT/16/NVIDIA/best.link/source/
# Исходники лучшей модели
```
```

---

## 🔒 Гарантии безопасности (улучшенные!)

### v1.0 (было):
- ✅ Не перезаписываем (v1, v2, v3)
- ✅ Сохранение исходников + результатов

### v2.0 (стало):
- ✅ **Immutable runs** (fail-fast если директория существует)
- ✅ **Manifest с SHA256** (проверка целостности)
- ✅ **Автоматический best** (обновляется только при улучшении)
- ✅ **Registry CSV** (таблица всех экспериментов)
- ✅ **Git commit** в метаданных (воспроизводимость)
- ✅ **summary.md** (читаемый отчёт)
- ✅ **config.lock.json** (зафиксированная конфигурация)
- ✅ **Структурированные логи** (logs/ директория)

---

## 📋 Сравнение v1.0 vs v2.0

| Функция | v1.0 | v2.0 |
|---------|------|------|
| Не затирать | ✅ v1,v2,v3 | ✅ Immutable runs + fail-fast |
| Исходники | ✅ Копирует | ✅ Копирует + SHA256 |
| Результаты | ✅ JSON | ✅ JSON + manifest |
| Best модель | ❌ Вручную | ✅ Автоматически |
| Реестр | ❌ Нет | ✅ CSV таблица |
| Целостность | ❌ Нет | ✅ SHA256 checksums |
| Воспроизводимость | ⚠️ Частично | ✅ Git commit + config.lock |
| Отчёты | ❌ Нет | ✅ summary.md |

---

## 🚀 Что использовать?

### Рекомендация:

**Для BASELINE (сейчас):**
- Используем **v1.0** (проще, быстрее реализовать)
- Фокус на рабочем прототипе

**Для PRODUCTION (потом):**
- Мигрируем на **v2.0** (полная система)
- Добавляем hooks, registry, best/

---

## 💡 План миграции v1.0 → v2.0

**Этап 1:** Реализуем v1.0 (простой ModelArchiver)
- TASK-021, TASK-022 из tasks.md

**Этап 2:** После успешного FFT16, улучшаем до v2.0
- Добавляем pre_run/post_run
- Registry CSV
- Best/ витрина
- Manifest SHA256

---

## ❓ Вопрос к вам:

**Что делаем?**

**A)** Реализуем v1.0 (простой, быстрее) для baseline?  
**B)** Сразу делаем v2.0 (полный, но дольше)?  
**C)** Гибрид: v1.0 сейчас + план миграции на v2.0?

**Моя рекомендация:** **C** - начинаем с v1.0, потом улучшаем! 🎯

Что выбираете? 😊
