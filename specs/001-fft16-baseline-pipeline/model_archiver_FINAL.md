# 🔴 Протокол сохранения моделей FINAL (на основе AMGpuCuda опыта)

**Версия:** 3.0 FINAL  
**Статус:** КРИТИЧЕСКИЙ - К РЕАЛИЗАЦИИ  
**Дата:** 10 октября 2025  
**Основано на:** 
- Статья "Надёжный склад результатов"
- Опыт проекта AMGpuCuda_copy (ваши лучшие решения!)

---

## ⏱️ 1. ОЦЕНКА ВРЕМЕНИ РЕАЛИЗАЦИИ

### ModelArchiver v3.0 (FINAL):

| Компонент | Оценка | Описание |
|-----------|--------|----------|
| Базовые функции | 6ч | save_experiment(), pre_run(), post_run() |
| Registry (JSON) | 3ч | experiments_registry.json вместо CSV |
| Best/ витрина | 2ч | Автообновление лучших |
| Manifest SHA256 | 2ч | Проверка целостности |
| Интеграция в main | 2ч | Hooks в workflow |
| Тестирование | 3ч | Unit тесты + проверка |
| **ИТОГО** | **18ч** | **~2-3 рабочих дня** |

**Вывод:** Реально сделать за 2-3 дня!

---

## 📁 2. ФИНАЛЬНАЯ СТРУКТУРА (на основе AMGpuCuda)

### Структура DataContext/:

```
DataContext/
├── Experiments/                        # Все эксперименты (immutable!)
│   └── runs/
│       ├── 2025-10-09_10-30__fft16_wmma_baseline/
│       │   ├── source/
│       │   │   ├── fft16_wmma.cu
│       │   │   └── fft16_wmma.cpp
│       │   ├── results/
│       │   │   ├── profiling.json          # Профилирование
│       │   │   ├── validation_input.json   # Для Python
│       │   │   └── gpu_output.json         # Результаты GPU
│       │   ├── artifacts/
│       │   │   ├── heatmaps/              # PNG графики
│       │   │   └── comparisons/           # Сравнительные графики
│       │   ├── logs/
│       │   │   ├── run.log
│       │   │   └── errors.log
│       │   ├── config.lock.json           # Конфигурация
│       │   ├── manifest.json              # SHA256 всех файлов
│       │   ├── summary.md                 # Краткий отчёт
│       │   └── README.md                  # Описание эксперимента
│       │
│       └── 2025-10-09_14-30__fft16_wmma_opt_twiddle/
│           └── ... (аналогично)
│
├── Reports/                            # ⭐ Отчёты по датам (как в AMGpuCuda!)
│   ├── 2025-10-09/                     # Дата (день)
│   │   ├── session_10-30/              # Сессия (время)
│   │   │   ├── README.md               # Описание сессии
│   │   │   ├── fft16_comparison/       # Сравнение реализаций
│   │   │   │   ├── COMPARISON_TABLE.md
│   │   │   │   ├── comparison_times.json
│   │   │   │   ├── heatmap_comparison.png
│   │   │   │   └── statistics.json
│   │   │   ├── wmma_performance/       # Производительность WMMA
│   │   │   │   ├── PERFORMANCE_REPORT.md
│   │   │   │   ├── profiling.json
│   │   │   │   └── charts/
│   │   │   └── validation/             # Результаты валидации
│   │   │       ├── python_validation_results.json
│   │   │       └── error_analysis.md
│   │   │
│   │   └── session_14-30/              # Следующая сессия
│   │       └── ...
│   │
│   └── 2025-10-10/                     # Следующий день
│       └── ...
│
├── Best/                               # Витрина лучших (автообновление)
│   └── FFT/
│       └── 16/
│           └── NVIDIA/
│               ├── current_best.link   # Симлинк на лучший
│               ├── best_info.json      # Инфо о лучшем
│               ├── history.json        # История рекордов
│               └── README.md           # Как достигнут рекорд
│
├── Registry/                           # ⭐ JSON вместо CSV!
│   ├── experiments_registry.json       # Все эксперименты
│   ├── best_records.json               # Рекорды
│   └── sessions_index.json             # Индекс сессий
│
└── ValidationData/                     # Данные для Python (как было)
    └── FFT16/
        └── ...
```

---

## 📋 3. Registry в JSON (вместо CSV)

### experiments_registry.json

```json
{
  "registry_version": "1.0",
  "last_updated": "2025-10-09T14:30:45",
  "total_experiments": 3,
  
  "experiments": [
    {
      "id": "2025-10-09_10-30__fft16_wmma_baseline",
      "date": "2025-10-09",
      "time": "10:30",
      "name": "fft16_wmma_baseline",
      "gpu_type": "NVIDIA",
      "algorithm": "FFT",
      "size": 16,
      "implementation": "WMMA",
      "primary_metric": "compute_time",
      "primary_value": 0.456,
      "metrics": {
        "upload_ms": 0.123,
        "compute_ms": 0.456,
        "download_ms": 0.089,
        "total_ms": 0.668
      },
      "validation": {
        "enabled": true,
        "max_error": 1.23e-6,
        "passed": true
      },
      "is_best": true,
      "git_commit": "abc123def",
      "path": "Experiments/runs/2025-10-09_10-30__fft16_wmma_baseline"
    },
    {
      "id": "2025-10-09_14-30__fft16_wmma_opt_twiddle",
      "date": "2025-10-09",
      "time": "14:30",
      "name": "fft16_wmma_opt_twiddle",
      "gpu_type": "NVIDIA",
      "algorithm": "FFT",
      "size": 16,
      "implementation": "WMMA",
      "primary_metric": "compute_time",
      "primary_value": 0.320,
      "metrics": {
        "upload_ms": 0.125,
        "compute_ms": 0.320,
        "download_ms": 0.091,
        "total_ms": 0.536
      },
      "validation": {
        "enabled": true,
        "max_error": 8.45e-7,
        "passed": true
      },
      "is_best": true,
      "improvement": {
        "vs_previous": "29.8% faster",
        "absolute_gain": 0.132
      },
      "git_commit": "def456ghi",
      "path": "Experiments/runs/2025-10-09_14-30__fft16_wmma_opt_twiddle"
    }
  ]
}
```

**Плюсы JSON vs CSV:**
- ✅ Вложенные структуры (metrics, validation)
- ✅ Легче парсить
- ✅ Поддержка массивов
- ✅ Комментарии (в JSON5)

---

### best_records.json

```json
{
  "last_updated": "2025-10-09T14:30:45",
  
  "records": {
    "NVIDIA": {
      "FFT": {
        "16": {
          "current_best": {
            "experiment_id": "2025-10-09_14-30__fft16_wmma_opt_twiddle",
            "metric": "compute_time",
            "value": 0.320,
            "mode": "min",
            "achieved_at": "2025-10-09T14:30:45"
          },
          "history": [
            {
              "experiment_id": "2025-10-09_10-30__fft16_wmma_baseline",
              "value": 0.456,
              "achieved_at": "2025-10-09T10:30:00",
              "note": "First baseline"
            },
            {
              "experiment_id": "2025-10-09_14-30__fft16_wmma_opt_twiddle",
              "value": 0.320,
              "achieved_at": "2025-10-09T14:30:45",
              "improvement": "29.8%",
              "note": "Optimized twiddle factors"
            }
          ]
        }
      }
    }
  }
}
```

---

### sessions_index.json

```json
{
  "sessions": [
    {
      "date": "2025-10-09",
      "sessions": [
        {
          "time": "10-30",
          "topic": "FFT16 Baseline Testing",
          "experiments": [
            "2025-10-09_10-30__fft16_wmma_baseline",
            "2025-10-09_10-45__fft16_shared2d_baseline"
          ],
          "report_path": "Reports/2025-10-09/session_10-30/",
          "summary": "First baseline tests for FFT16. WMMA vs Shared2D comparison."
        },
        {
          "time": "14-30",
          "topic": "FFT16 Twiddle Optimization",
          "experiments": [
            "2025-10-09_14-30__fft16_wmma_opt_twiddle"
          ],
          "report_path": "Reports/2025-10-09/session_14-30/",
          "summary": "Optimized twiddle factor computation. 29.8% speedup!"
        }
      ]
    }
  ]
}
```

---

## 📊 4. Структура Reports/ (на основе AMGpuCuda!)

```
Reports/
├── 2025-10-09/                         # По датам (день)
│   ├── session_10-30/                  # По времени (сессия)
│   │   ├── README.md                   # ⭐ Обязательно!
│   │   ├── fft16_comparison/           # Тематика 1
│   │   │   ├── COMPARISON_TABLE.md
│   │   │   ├── wmma_vs_shared2d.json
│   │   │   ├── comparison_chart.png
│   │   │   └── statistics.json
│   │   ├── wmma_performance/           # Тематика 2
│   │   │   ├── PERFORMANCE_REPORT.md
│   │   │   ├── profiling_detailed.json
│   │   │   └── heatmap_compute.png
│   │   ├── shared2d_performance/       # Тематика 3
│   │   │   └── ...
│   │   └── validation/                 # Тематика 4
│   │       ├── python_results.json
│   │       └── error_analysis.md
│   │
│   ├── session_14-30/                  # Другая сессия в тот же день
│   │   └── ...
│   │
│   └── README.md                       # Сводка по дню
│
├── 2025-10-10/
│   └── ...
│
├── README.md                           # Общий гид
└── TEST_REGISTRY.md                    # ⭐ Реестр рекордов (как в AMGpuCuda!)
```

---

## 🎯 5. TEST_REGISTRY.md (адаптированный)

```markdown
# 📋 РЕЕСТР ЭКСПЕРИМЕНТОВ FFT16

## 🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ (РЕКОРДЫ)

### 🥇 FFT16_WMMA - Optimized Twiddle Factors
- **Дата:** 9 октября 2025, 14:30
- **ID:** `2025-10-09_14-30__fft16_wmma_opt_twiddle`
- **Compute time:** 0.320 ms ⭐ РЕКОРД!
- **Total time:** 0.536 ms
- **Validation:** ✅ PASSED (max_error: 8.45e-7)
- **GPU:** NVIDIA RTX 3060
- **CUDA:** 13.0
- **Файлы:** `Experiments/runs/2025-10-09_14-30__fft16_wmma_opt_twiddle/`
- **Статус:** 🏆 CURRENT BEST

### 🥈 FFT16_Shared2D - Baseline
- **Дата:** 9 октября 2025, 10:45
- **Compute time:** 0.512 ms
- **Validation:** ✅ PASSED (max_error: 4.56e-7)
- **Статус:** ✅ Работает, но медленнее

---

## 📊 ВСЕ ЭКСПЕРИМЕНТЫ (хронологически)

### Сессия: 2025-10-09, 10:30 - Baseline Testing
- `fft16_wmma_baseline` - 0.456 ms
- `fft16_shared2d_baseline` - 0.512 ms
- **Вывод:** WMMA быстрее на 11%

### Сессия: 2025-10-09, 14:30 - Twiddle Optimization
- `fft16_wmma_opt_twiddle` - 0.320 ms 🏆 NEW RECORD!
- **Улучшение:** 29.8% vs baseline

---

## 🎯 РЕКОМЕНДАЦИИ

### Для production:
1. **Используйте:** FFT16_WMMA (optimized twiddle)
2. **Файл:** `Best/FFT/16/NVIDIA/current_best.link`
3. **Производительность:** 0.320 ms (compute)

### Для новых экспериментов:
1. **Сравнивайте с:** 0.320 ms (текущий рекорд)
2. **Цель:** < 0.300 ms
3. **Документируйте:** В TEST_REGISTRY.md
```

---

## 🔧 6. API ModelArchiver FINAL

```cpp
namespace CudaCalc {

// Конфигурация сессии тестирования
struct SessionConfig {
    std::string topic;          // "FFT16 Baseline Testing"
    std::vector<std::string> experiments;  // Список ID экспериментов в сессии
};

class ModelArchiverFinal {
private:
    std::filesystem::path experiments_root_;  // "DataContext/Experiments"
    std::filesystem::path reports_root_;      // "DataContext/Reports"
    std::filesystem::path best_root_;         // "DataContext/Best"
    std::filesystem::path registry_root_;     // "DataContext/Registry"
    
public:
    // === ЭКСПЕРИМЕНТЫ ===
    
    /**
     * @brief Создать новый эксперимент
     * @param name Краткое имя (без пробелов): "fft16_wmma_baseline"
     * @param config Конфигурация
     * @return ID: "YYYY-MM-DD_HH-MM__name"
     */
    std::string create_experiment(
        const std::string& name,
        const ExperimentConfig& config
    );
    
    /**
     * @brief Сохранить результаты эксперимента
     */
    ExperimentInfo save_experiment(
        const std::string& experiment_id,
        const std::vector<std::string>& source_files,
        const nlohmann::json& profiling,
        const nlohmann::json& validation_input,
        const nlohmann::json& gpu_output,
        const std::string& description
    );
    
    /**
     * @brief Финализация эксперимента (manifest, registry, best)
     */
    bool finalize_experiment(const std::string& experiment_id);
    
    // === СЕССИИ (как в AMGpuCuda!) ===
    
    /**
     * @brief Создать сессию тестирования
     * @param topic "FFT16 Baseline Testing"
     * @return Путь к сессии: "Reports/YYYY-MM-DD/session_HH-MM/"
     */
    std::filesystem::path create_session(const std::string& topic);
    
    /**
     * @brief Добавить эксперимент в текущую сессию
     */
    bool add_to_session(
        const std::filesystem::path& session_path,
        const std::string& experiment_id
    );
    
    /**
     * @brief Создать отчёт сессии (README.md + summary)
     */
    bool generate_session_report(
        const std::filesystem::path& session_path,
        const SessionConfig& config
    );
    
    // === BEST MANAGEMENT ===
    
    /**
     * @brief Обновить Best/ только при улучшении
     */
    bool update_best_if_improved(const ExperimentInfo& exp_info);
    
    /**
     * @brief Получить текущий рекорд
     */
    std::optional<ExperimentInfo> get_current_best(
        const std::string& gpu_type,
        const std::string& algorithm,
        int size
    );
    
    // === REGISTRY (JSON!) ===
    
    /**
     * @brief Добавить в experiments_registry.json
     */
    bool register_experiment(const ExperimentInfo& exp_info);
    
    /**
     * @brief Запросить эксперименты (с фильтрами)
     */
    std::vector<ExperimentInfo> query_experiments(
        const std::string& gpu_type = "",
        const std::string& algorithm = "",
        int size = -1,
        const std::string& order_by = "date"
    );
    
    /**
     * @brief Обновить TEST_REGISTRY.md
     */
    bool update_test_registry();
    
    // === VALIDATION ===
    
    /**
     * @brief Проверка целостности (SHA256)
     */
    bool verify_experiment(const std::string& experiment_id);
};

} // namespace CudaCalc
```

---

## 🔄 7. Workflow (полный цикл)

```cpp
int main() {
    ModelArchiverFinal archiver;
    
    // === СЕССИЯ ТЕСТИРОВАНИЯ ===
    auto session_path = archiver.create_session("FFT16 Baseline Testing");
    // → "Reports/2025-10-09/session_10-30/"
    
    std::cout << "Session: " << session_path << std::endl;
    
    // === ЭКСПЕРИМЕНТ 1: FFT16_WMMA ===
    {
        // 1. Создание эксперимента
        ExperimentConfig config;
        config.gpu_type = "NVIDIA";
        config.algorithm = "FFT";
        config.size = 16;
        config.implementation = "WMMA";
        config.primary_metric = "compute_time";
        config.mode = "min";
        
        auto exp_id = archiver.create_experiment("fft16_wmma_baseline", config);
        // → "2025-10-09_10-30__fft16_wmma_baseline"
        
        // 2. Генерация сигнала
        SineGenerator gen(4, 1024, 8);
        auto input = gen.generate(16, true);
        
        // 3. Выполнение FFT
        FFT16_WMMA fft;
        fft.initialize();
        
        BasicProfiler profiler;
        // ... профилирование ...
        
        auto output = fft.process(input);
        
        // 4. Сохранение
        nlohmann::json profiling = profiler.get_results_json();
        nlohmann::json validation_input = input_to_json(input);
        nlohmann::json gpu_output = output_to_json(output);
        
        auto exp_info = archiver.save_experiment(
            exp_id,
            {"ModelsFunction/.../fft16_wmma.cu", ".../fft16_wmma.cpp"},
            profiling,
            validation_input,
            gpu_output,
            "Baseline FFT16 WMMA. Linear unroll 4 stages."
        );
        
        // 5. Финализация
        archiver.finalize_experiment(exp_id);
        
        // 6. Добавление в сессию
        archiver.add_to_session(session_path, exp_id);
    }
    
    // === ЭКСПЕРИМЕНТ 2: FFT16_Shared2D ===
    {
        // ... аналогично ...
    }
    
    // === ГЕНЕРАЦИЯ ОТЧЁТА СЕССИИ ===
    SessionConfig session_config;
    session_config.topic = "FFT16 Baseline Testing";
    session_config.experiments = {exp_id_wmma, exp_id_shared2d};
    
    archiver.generate_session_report(session_path, session_config);
    
    // === ОБНОВЛЕНИЕ TEST_REGISTRY.md ===
    archiver.update_test_registry();
    
    std::cout << "✅ Session complete!" << std::endl;
    std::cout << "   Reports: " << session_path << std::endl;
    
    return 0;
}
```

---

## 📝 8. Структура отчёта сессии (README.md)

```markdown
# 📊 Сессия тестирования: FFT16 Baseline Testing

**Дата:** 2025-10-09  
**Время:** 10:30  
**Тема:** Сравнение FFT16_WMMA и FFT16_Shared2D  
**Статус:** ✅ Завершено

---

## 🎯 Цель сессии
Получить baseline метрики для двух реализаций FFT16 на RTX 3060.

## 🧪 Эксперименты

### 1. fft16_wmma_baseline
- **ID:** `2025-10-09_10-30__fft16_wmma_baseline`
- **Compute:** 0.456 ms
- **Total:** 0.668 ms
- **Validation:** ✅ PASSED
- **Статус:** ✅ Success

### 2. fft16_shared2d_baseline
- **ID:** `2025-10-09_10-45__fft16_shared2d_baseline`
- **Compute:** 0.512 ms
- **Total:** 0.728 ms
- **Validation:** ✅ PASSED
- **Статус:** ✅ Success

---

## 📊 Результаты

### Сравнение производительности
См. `fft16_comparison/COMPARISON_TABLE.md`

| Implementation | Compute (ms) | Total (ms) | Winner |
|----------------|-------------|-----------|--------|
| WMMA | 0.456 | 0.668 | ✅ |
| Shared2D | 0.512 | 0.728 | - |

**Вывод:** WMMA быстрее на 11%

### Валидация
См. `validation/`
- WMMA: max_error = 1.23e-6 ✅
- Shared2D: max_error = 4.56e-7 ✅

**Вывод:** Обе реализации корректны

---

## 🎯 Выводы

1. ✅ FFT16_WMMA - лучший выбор для производительности
2. ✅ FFT16_Shared2D - лучше по точности
3. 🏆 Для production: WMMA (приемлемая точность, лучшая скорость)

---

## 📁 Артефакты

- **Исходники:** См. эксперименты
- **Графики:** `*/heatmap_*.png`, `*/comparison_*.png`
- **Данные:** `*/*.json`
- **Отчёты:** `*/*_REPORT.md`

---

**Создано:** 2025-10-09 10:30:00  
**Обновлено:** 2025-10-09 16:00:00  
**Автор:** AlexLan73
```

---

## ⚙️ 9. Автоматизация (скрипты)

### scripts/start_session.sh
```bash
#!/bin/bash
# Создать новую сессию тестирования

TOPIC="$1"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M")

SESSION_PATH="DataContext/Reports/${DATE}/session_${TIME}"

mkdir -p "${SESSION_PATH}"/{comparison,performance,validation,artifacts}

cat > "${SESSION_PATH}/README.md" << EOF
# 📊 Сессия: ${TOPIC}

**Дата:** ${DATE}
**Время:** ${TIME}
**Статус:** В процессе

## Эксперименты
(обновляется автоматически)
EOF

echo "✅ Session created: ${SESSION_PATH}"
echo "${SESSION_PATH}" > /tmp/current_session.txt
```

### scripts/finish_session.sh
```bash
#!/bin/bash
# Финализировать сессию

SESSION_PATH=$(cat /tmp/current_session.txt)

# Генерация summary
python scripts/generate_session_summary.py "$SESSION_PATH"

# Обновление TEST_REGISTRY.md
python scripts/update_test_registry.py

echo "✅ Session finished: ${SESSION_PATH}"
rm /tmp/current_session.txt
```

---

## ⏱️ 10. ФИНАЛЬНАЯ ОЦЕНКА ВРЕМЕНИ

### Реализация ModelArchiver FINAL:

| Задача | Часы | Комментарий |
|--------|------|-------------|
| Базовые функции (create, save, finalize) | 6ч | Основной функционал |
| Registry JSON (3 файла) | 3ч | experiments, best_records, sessions |
| Best/ витрина с авто-обновлением | 2ч | update_best_if_improved() |
| Manifest SHA256 | 2ч | Проверка целостности |
| Sessions (create_session, add_to_session) | 2ч | Организация по датам |
| TEST_REGISTRY.md генерация | 2ч | Автообновление реестра |
| Интеграция в main | 2ч | Полный workflow |
| Скрипты автоматизации | 3ч | start_session.sh, etc |
| Тестирование | 3ч | Unit тесты + проверка |
| **ИТОГО** | **25ч** | **~3-4 рабочих дня** |

---

## 🤔 УТОЧНЯЮЩИЕ ВОПРОСЫ:

### 1. **Графики (PNG)** - нужны сейчас?
В AMGpuCuda есть:
- `heatmap_*.png`
- `comparison_*.png`
- `statistics_*.png`

**Вариант A:** Добавим Python скрипты для генерации графиков (+4-5ч)  
**Вариант B:** Пока только JSON, графики потом  

**Что выбираем?**

---

### 2. **Структура Reports/** - формат даты?

**Из AMGpuCuda:**
```
Reports/
└── 2025-10-09/
    └── session_10-30/
```

**Или проще:**
```
Reports/
└── 2025-10-09_10-30/
```

**Какой вариант?** (Первый гибче для нескольких сессий в день!)

---

### 3. **Тематические папки** - какие?

Для FFT16 baseline:
- `fft16_comparison/` (WMMA vs Shared2D)
- `wmma_performance/` (детали WMMA)
- `shared2d_performance/` (детали Shared2D)
- `validation/` (Python результаты)

**Нормально или что-то изменить?**

---

### 4. **JSON vs MD** - что куда?

**JSON для:**
- Данные (профилирование, результаты)
- Registry
- Конфиги

**MD для:**
- Отчёты (для людей)
- README
- Описания

**Так правильно?**

---

## 📋 Ответьте на вопросы:

1. **Графики PNG**: A (добавить сейчас) или B (потом)?
2. **Формат даты**: `2025-10-09/session_10-30/` или `2025-10-09_10-30/`?
3. **Тематические папки**: OK или что изменить?
4. **JSON vs MD**: правильное разделение?

**После ответов создам ФИНАЛЬНЫЙ протокол!** 🚀
