# Scripts

Утилиты для работы с Spec-Kit в проекте CudaCalc

---

## 📝 Доступные скрипты

### `create-feature.sh`
Создаёт новую фичу со всеми необходимыми файлами

**Использование:**
```bash
./scripts/create-feature.sh "feature-name"
```

**Пример:**
```bash
./scripts/create-feature.sh "memory-pool"
```

**Что создаётся:**
```
specs/002-memory-pool/
├── spec.md          # Спецификация (ЧТО делаем)
├── plan.md          # План реализации (КАК делаем)
├── tasks.md         # Задачи (TODO список)
├── research.md      # Результаты исследований
├── quickstart.md    # Быстрый старт
└── contracts/       # API контракты
```

**После создания:**
1. Отредактировать `spec.md`
2. В Cursor: `/speckit.clarify`
3. В Cursor: `/speckit.plan`
4. Проверить `plan.md`
5. В Cursor: `/speckit.implement`
6. Обновить `CLAUDE.md`

---

## 🔧 Будущие скрипты

Здесь могут появиться:
- `benchmark.sh` - автоматические бенчмарки
- `check-coverage.sh` - проверка code coverage
- `format-all.sh` - форматирование всего кода
- `run-profiler.sh` - профилирование производительности

