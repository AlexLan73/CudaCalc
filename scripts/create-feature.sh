#!/bin/bash

# Скрипт для создания новой фичи в Spec-Kit структуре
# Использование: ./scripts/create-feature.sh "feature-name"

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <feature-name>"
    echo "Example: $0 tensor-fft-optimization"
    exit 1
fi

FEATURE_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SPECS_DIR="$PROJECT_ROOT/specs"

# Найти следующий номер спецификации
LAST_NUM=0
if [ -d "$SPECS_DIR" ]; then
    for dir in "$SPECS_DIR"/[0-9][0-9][0-9]-*; do
        if [ -d "$dir" ]; then
            NUM=$(basename "$dir" | cut -d'-' -f1)
            if [ "$NUM" -gt "$LAST_NUM" ]; then
                LAST_NUM=$NUM
            fi
        fi
    done
fi

NEXT_NUM=$((LAST_NUM + 1))
SPEC_NUM=$(printf "%03d" $NEXT_NUM)
FEATURE_DIR="$SPECS_DIR/$SPEC_NUM-$FEATURE_NAME"

echo "Creating new feature specification: $SPEC_NUM-$FEATURE_NAME"

# Создать директорию
mkdir -p "$FEATURE_DIR/contracts"

# Создать файлы из шаблонов
if [ -f "$PROJECT_ROOT/templates/spec_template.md" ]; then
    sed "s/\[Название фичи\]/$FEATURE_NAME/g" "$PROJECT_ROOT/templates/spec_template.md" > "$FEATURE_DIR/spec.md"
else
    # Минимальный шаблон если templates нет
    cat > "$FEATURE_DIR/spec.md" << EOF
# Спецификация: $FEATURE_NAME

## 1. Обзор
### Проблема:
### Решение:
### Цели:

## 2. Требования
### Функциональные:
### Нефункциональные:

## 3. Сценарии использования
\`\`\`cpp
// Пример кода
\`\`\`

## 4. API Дизайн

## 5. Критерии приёмки
- [ ] Unit тесты
- [ ] Performance тесты
- [ ] Документация
EOF
fi

# Создать остальные файлы
cat > "$FEATURE_DIR/plan.md" << 'EOF'
# План реализации: [Название]

## 1. Архитектура

## 2. Модули
### Модуль 1:
- Файлы: 
- Зависимости: 
- Алгоритм:

## 3. План тестирования

## 4. Риски
EOF

cat > "$FEATURE_DIR/tasks.md" << 'EOF'
# Задачи: [Название]

### TASK-001: [Описание]
**Статус:** TODO
**Оценка:** [X] часов
**Приоритет:** Высокий

**Описание:**

**Критерии приёмки:**
- [ ] 
EOF

cat > "$FEATURE_DIR/research.md" << 'EOF'
# Исследование: [Название]

## Цель исследования

## Методология

## Результаты

## Выводы

## Ссылки
EOF

cat > "$FEATURE_DIR/quickstart.md" << 'EOF'
# Quick Start: [Название]

## Быстрый старт

```bash
# Примеры команд
```

## Основное использование

```cpp
// Примеры кода
```

## Часто задаваемые вопросы

**Q:** 
**A:** 
EOF

echo "✅ Created feature specification at: $FEATURE_DIR"
echo ""
echo "Next steps:"
echo "1. Edit $FEATURE_DIR/spec.md (ЧТО делаем)"
echo "2. In Cursor: /speckit.clarify"
echo "3. In Cursor: /speckit.plan"
echo "4. Review plan.md (КАК делаем)"
echo "5. In Cursor: /speckit.implement"
echo ""
echo "Don't forget to update CLAUDE.md with the new spec!"

