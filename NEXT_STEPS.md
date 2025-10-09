# 🚀 СЛЕДУЮЩИЕ ШАГИ - Продолжение работы

**Дата создания:** 09 октября 2025  
**Статус:** Готов к продолжению на другом компьютере  
**Текущая фаза:** Phase 1 - FFT16 Baseline

---

## ✅ ЧТО УЖЕ СДЕЛАНО (на этом сеансе):

### 1. **Spec-Kit настроен и готов**
- ✅ CLAUDE.md - главный контекст проекта
- ✅ constitution.md - принципы разработки
- ✅ SPEC_KIT_CHEATSHEET.md - шпаргалка по командам

### 2. **Спецификация FFT16 ГОТОВА**
- ✅ `specs/001-fft16-baseline-pipeline/spec.md` (1056 строк)
- ✅ Все требования описаны
- ✅ ModelArchiver добавлен (🔴 КРИТИЧНО!)
- ✅ Два профайлера: BasicProfiler + MemoryProfiler
- ✅ ValidationData через cuFFT
- ✅ Структуры данных исправлены

### 3. **ROADMAP проекта создан**
- ✅ ROADMAP.md - полный план на 40-50 недель
- ✅ 6 фаз развития проекта
- ✅ Все будущие модули описаны

### 4. **Базовая структура**
- ✅ Директории: memory/, specs/, scripts/, templates/
- ✅ Скрипт create-feature.sh
- ✅ Шаблон spec_template.md

---

## 🎯 ЧТО ДЕЛАТЬ ДАЛЬШЕ (на другом компьютере):

### **Шаг 1: Продолжить с plan.md** (1-2 часа)

**Команда в Cursor:**
```
"Создай plan.md для specs/001-fft16-baseline-pipeline/ на основе spec.md"
```

**Или используй Spec-Kit команду:**
```
/speckit.plan
```

**Что должно быть в plan.md:**
- Детальная архитектура каждого модуля
- Алгоритмы (псевдокод) для FFT16 kernels
- Последовательность реализации (фазы)
- Конкретные файлы и их размеры
- Зависимости между модулями
- План тестирования
- Риски и митигация

**Файл:** `specs/001-fft16-baseline-pipeline/plan.md`

---

### **Шаг 2: Настроить MemoryBank** (30 минут - 1 час)

**2.1. Проверить что MemoryBank MCP работает:**
```bash
# В терминале проверить ~/.cursor/mcp.json
cat ~/.cursor/mcp.json | grep MemoryBank
```

Должно быть:
```json
"MemoryBank": {
  "command": "npx",
  "args": ["memory-bank-mcp@latest"]
}
```

**2.2. Протестировать MemoryBank:**

В Cursor чате:
```
"Сохрани в память: Проект CudaCalc начат 09 октября 2025.
Цель: Production-ready GPU примитивы для обработки сигналов.
Первая задача: FFT16 baseline.
Ключевые решения:
- ModelArchiver для сохранения экспериментов (КРИТИЧНО!)
- Два профайлера: BasicProfiler (обязательно) + MemoryProfiler (опционально)
- ValidationData генерируется один раз в DataContext через cuFFT
- Структуры данных без device pointers в публичном API
Tags: #project-start #fft16 #architecture"
```

**2.3. Запросить из памяти:**
```
"Что мы знаем о проекте CudaCalc?"
```

Если работает - отлично! ✅

---

### **Шаг 3: 🔴 КРИТИЧНО! Протокол сохранения моделей** (1-2 часа)

**Это ОБЯЗАТЕЛЬНЫЙ шаг перед началом реализации!**

**3.1. Создать спецификацию ModelArchiver:**

Файл: `specs/001-fft16-baseline-pipeline/model_archiver_protocol.md`

**Содержание (детальное описание):**
```markdown
# Протокол сохранения моделей (ModelArchiver)

## Цель
ПРЕДОТВРАТИТЬ ПОТЕРЮ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ!

## Структура хранения
DataContext/Models/
└── NVIDIA/
    └── FFT/
        └── 16/
            ├── model_2025_10_09_v1/
            │   ├── fft16_wmma.cu
            │   ├── fft16_wmma.cpp
            │   ├── description.txt
            │   ├── results.json
            │   └── validation.json
            └── model_2025_10_09_v2/

## Workflow
1. После каждого эксперимента:
   - ModelArchiver::save_model()
   - Автоматическая версия (v1, v2, v3...)
   
2. Сохраняется:
   - Исходный код (.cu, .cpp)
   - Результаты профилирования
   - Результаты валидации
   - Описание эксперимента

3. Никогда не перезаписывается!
   - Каждый эксперимент = новая версия
   - История экспериментов сохраняется

## API
class ModelArchiver {
    bool save_model(ModelInfo, source_files, results_json, validation_json);
    ModelInfo load_model(version);
    string compare_models(versions);
    vector<ModelInfo> list_models(gpu, algorithm, size);
    string get_next_version(gpu, algorithm, size);
};

## Интеграция
После каждого теста в main_fft16_test.cpp:
1. Запускаем FFT16_WMMA
2. Профилируем + валидируем
3. ModelArchiver::save_model() ← ОБЯЗАТЕЛЬНО!
4. То же для FFT16_Shared2D
```

**3.2. Добавить в MemoryBank:**
```
"Сохрани в память: ModelArchiver - КРИТИЧЕСКИЙ компонент!
Проблема: В прошлом теряли результаты экспериментов при перезаписи.
Решение: Автоматическое версионирование (v1, v2, v3...) в Models/NVIDIA/FFT/16/
Каждый эксперимент сохраняется с исходниками + результатами.
НИКОГДА не перезаписывать!
Priority: CRITICAL
Tags: #model-archiver #critical #experiments"
```

---

### **Шаг 4: Создать tasks.md** (30 минут)

**Команда в Cursor:**
```
"Создай tasks.md для specs/001-fft16-baseline-pipeline/ на основе plan.md"
```

**Или:**
```
/speckit.tasks
```

**Что должно быть в tasks.md:**
- Конкретные задачи (TASK-001, TASK-002, ...)
- Оценки времени (1-8 часов каждая)
- Зависимости между задачами
- Критерии приёмки для каждой задачи
- Статус: TODO / IN_PROGRESS / DONE

---

## 📋 Порядок действий (пошагово):

```
1. ✅ Git pull (получить все изменения)
   └─> cd /path/to/CudaCalc && git pull

2. 📝 Создать plan.md
   └─> /speckit.plan в Cursor
   └─> Файл: specs/001-fft16-baseline-pipeline/plan.md

3. 🧠 Настроить и протестировать MemoryBank
   └─> Сохранить стартовую информацию о проекте
   └─> Протестировать запрос из памяти

4. 🔴 КРИТИЧНО! Протокол ModelArchiver
   └─> Создать model_archiver_protocol.md
   └─> Сохранить в MemoryBank важность этого компонента
   └─> Убедиться что он в приоритете реализации

5. ✅ Создать tasks.md
   └─> /speckit.tasks в Cursor
   └─> Файл: specs/001-fft16-baseline-pipeline/tasks.md

6. 🚀 Начать реализацию (или продолжить планирование)
```

---

## 📂 Важные файлы для работы:

### Читать обязательно:
- **ROADMAP.md** - полный план проекта
- **CLAUDE.md** - контекст проекта для AI
- **specs/001-fft16-baseline-pipeline/spec.md** - текущая спецификация
- **memory/constitution.md** - принципы разработки
- **SPEC_KIT_CHEATSHEET.md** - шпаргалка

### Создать на следующем шаге:
- `specs/001-fft16-baseline-pipeline/plan.md` ← СЛЕДУЮЩИЙ
- `specs/001-fft16-baseline-pipeline/model_archiver_protocol.md` ← КРИТИЧНО!
- `specs/001-fft16-baseline-pipeline/tasks.md`

---

## 🎯 Цель следующего сеанса:

**К концу сеанса должно быть:**
1. ✅ plan.md создан и проверен
2. ✅ MemoryBank работает и протестирован
3. ✅ Протокол ModelArchiver детально описан
4. ✅ tasks.md создан с конкретными задачами
5. ✅ Готовы начать реализацию Фазы 1 (CMake + базовые модули)

---

## 💡 Полезные команды Spec-Kit:

```bash
# В Cursor чате:
/speckit.clarify    # AI задаст уточняющие вопросы
/speckit.plan       # Создать plan.md
/speckit.tasks      # Создать tasks.md
/speckit.implement  # Начать реализацию

# Работа с памятью:
"Сохрани в память: [информация]"
"Что мы знаем о [тема]?"

# Создание новой фичи:
./scripts/create-feature.sh "название-фичи"
```

---

## 🔴 КРИТИЧЕСКИЕ НАПОМИНАНИЯ:

### ❗ ModelArchiver - ОБЯЗАТЕЛЕН!
Без него потеряем результаты экспериментов!
- Реализовать в Фазе 5 (перед оптимизацией)
- Интегрировать в каждый тест
- Автоматическое версионирование

### ❗ ValidationData генерируется ОДИН РАЗ
В DataContext через cuFFT, не в каждом тесте!

### ❗ Два профайлера
- BasicProfiler (обязательно) - CUDA Events
- MemoryProfiler (опционально) - Memory + GPU metrics

---

## 📞 Если что-то непонятно:

1. Прочитайте `SPEC_KIT_CHEATSHEET.md`
2. Посмотрите `ROADMAP.md` для контекста
3. Изучите `specs/001-fft16-baseline-pipeline/spec.md`
4. Используйте AI: "Объясни что делать дальше в проекте CudaCalc"

---

## 🎉 Статус:

**ВСЁ ГОТОВО ДЛЯ ПРОДОЛЖЕНИЯ!**

Спецификация полная, ROADMAP ясен, структура создана.

**Следующий шаг:** plan.md → MemoryBank → ModelArchiver protocol → tasks.md

---

**Удачи в работе! 🚀**

**Версия:** 1.0  
**Дата:** 09 октября 2025  
**Автор:** AlexLan73

