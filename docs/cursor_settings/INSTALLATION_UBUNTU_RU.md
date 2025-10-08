# 🚀 Установка настроек Cursor на Ubuntu/Linux

## 📋 Что включено в этот пакет:

1. **settings.json** - основные настройки Cursor
2. **keybindings.json** - пользовательские горячие клавиши
3. **global_rules.md** - AI правила для C++, Python, C#, MATLAB
4. **mcp.json** - конфигурация MCP серверов
5. **spec-kit** - GitHub Spec-Kit для Spec-Driven Development

---

## 🛠️ Инструкция по установке на Ubuntu/Linux

### Шаг 1: Проверка зависимостей

Убедитесь, что установлены необходимые инструменты:

```bash
# Проверка Node.js (для MCP серверов)
node --version
# Должно быть >= 18.0.0

# Проверка Python (для spec-kit)
python3 --version
# Должно быть >= 3.11

# Проверка pip
pip3 --version

# Проверка Git
git --version
```

**Установка недостающих зависимостей:**

```bash
# Node.js (если не установлен)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Python и pip (если не установлены)
sudo apt update
sudo apt install python3 python3-pip -y

# Git (если не установлен)
sudo apt install git -y
```

---

### Шаг 2: Установка файлов настроек Cursor

#### A. Основные настройки (settings.json и keybindings.json)

```bash
# Создаем папку User, если её нет
mkdir -p ~/.config/Cursor/User/

# Копируем настройки (из папки проекта)
cp docs/cursor_settings/settings.json ~/.config/Cursor/User/
cp docs/cursor_settings/keybindings.json ~/.config/Cursor/User/
```

#### B. AI Rules (global_rules.md)

```bash
# Создаем папку .cursor, если её нет
mkdir -p ~/.cursor/

# Копируем AI правила
cp docs/cursor_settings/global_rules.md ~/.cursor/
```

#### C. MCP серверы (mcp.json)

**⚠️ ВАЖНО: Сначала настройте токены!**

```bash
# Копируем шаблон mcp.json
cp docs/cursor_settings/mcp.json ~/.cursor/

# Открываем файл для редактирования
nano ~/.cursor/mcp.json
# или
vim ~/.cursor/mcp.json
# или
code ~/.cursor/mcp.json
```

**Замените плейсхолдеры на реальные значения:**

```json
"GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN_HERE"
```

Замените на ваш реальный GitHub токен (начинается с `ghp_`)

**Сохраните файл:**
- Для nano: `Ctrl+O`, Enter, `Ctrl+X`
- Для vim: `:wq`

---

### Шаг 3: Установка Spec-Kit

Spec-Kit - это инструмент от GitHub для Spec-Driven Development.

#### Вариант 1: Клонирование из GitHub (рекомендуется)

```bash
# Переходим в папку для хранения инструментов
cd ~/tools  # или любая другая папка
# Если папки нет, создаем:
mkdir -p ~/tools && cd ~/tools

# Клонируем spec-kit
git clone https://github.com/github/spec-kit.git

# Устанавливаем spec-kit в editable режиме
pip3 install -e spec-kit/
```

#### Вариант 2: Копирование из другого компьютера

Если у вас уже есть папка spec-kit:

```bash
# Копируем папку spec-kit (например, из USB или сетевого диска)
cp -r /path/to/spec-kit ~/tools/

# Устанавливаем
pip3 install -e ~/tools/spec-kit/
```

#### Проверка установки spec-kit

```bash
# Проверяем, что команда доступна
specify check

# Должен появиться красивый баннер:
# ███████╗██████╗ ███████╗ ██████╗██╗███████╗██╗   ██╗
# ...
# Specify CLI is ready to use!
```

---

### Шаг 4: Перезапустите Cursor

```bash
# Закрываем все окна Cursor
killall cursor

# Запускаем Cursor снова
cursor
# или через GUI
```

---

## 🔧 Настройка MCP серверов

### GitHub MCP

Для работы с GitHub вам нужен Personal Access Token:

1. Перейдите: https://github.com/settings/tokens
2. Нажмите "Generate new token" → "Classic"
3. Выберите права:
   - ✅ `repo` (полный доступ)
   - ✅ `workflow`
   - ✅ `gist`
4. Скопируйте токен и вставьте в `~/.cursor/mcp.json`

### Context7 (опционально)

Если нужна актуальная документация библиотек:
1. Зарегистрируйтесь на https://context7.com
2. Получите API ключ
3. Вставьте в `~/.cursor/mcp.json`

### BraveSearch (опционально)

Для веб-поиска через AI:
1. Зарегистрируйтесь на https://brave.com/search/api/
2. Получите API ключ
3. Вставьте в `~/.cursor/mcp.json`

---

## 📁 Структура файлов Linux

```
~/
├── .cursor/
│   ├── mcp.json                    ← MCP серверы
│   └── global_rules.md             ← AI правила (опционально)
│
├── .config/Cursor/
│   └── User/
│       ├── settings.json           ← Настройки
│       └── keybindings.json        ← Горячие клавиши
│
└── tools/                          ← или любая другая папка
    └── spec-kit/                   ← Spec-Kit (опционально)
```

---

## ✅ Проверка установки

### Проверка MCP серверов:

1. Откройте Cursor
2. Нажмите `Ctrl + Shift + P`
3. Введите "MCP" или "Model Context Protocol"
4. Вы должны увидеть доступные MCP серверы

Или просто напишите в чате:
```
Покажи статус этого Git репозитория
```

Если GitHub MCP работает, он покажет информацию.

### Проверка AI Rules:

1. Откройте любой `.cpp` файл
2. AI должен автоматически следовать правилам C++ (RAII, smart pointers, etc.)
3. Спросите в чате: "Какие правила ты используешь для C++?"

### Проверка Spec-Kit:

```bash
specify check

# Должен показать список установленных инструментов:
# ✅ Git version control (available)
# ✅ Cursor IDE agent
# ...
```

---

## 🆘 Решение проблем

### Проблема: MCP серверы не работают

**Решение:**

1. Проверьте, установлен ли Node.js:
   ```bash
   node --version
   ```
   Должно быть >= 18.0.0

2. Установите Node.js:
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

3. Перезапустите Cursor

### Проблема: GitHub MCP выдает ошибку аутентификации

**Решение:**

1. Проверьте, что токен правильно скопирован в `~/.cursor/mcp.json`
2. Проверьте права токена на GitHub
3. Создайте новый токен, если старый истек

### Проблема: Команда `specify` не найдена

**Решение:**

1. Проверьте установку:
   ```bash
   pip3 show specify-cli
   ```

2. Проверьте PATH:
   ```bash
   echo $PATH | grep -o "[^:]*\.local/bin[^:]*"
   ```

3. Добавьте в PATH (если нужно):
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

### Проблема: Настройки не применяются

**Решение:**

1. Убедитесь, что файлы скопированы в правильные папки:
   ```bash
   ls -la ~/.config/Cursor/User/
   ls -la ~/.cursor/
   ```

2. Проверьте права доступа:
   ```bash
   chmod 644 ~/.config/Cursor/User/settings.json
   chmod 644 ~/.config/Cursor/User/keybindings.json
   chmod 644 ~/.cursor/mcp.json
   chmod 644 ~/.cursor/global_rules.md
   ```

3. Полностью закройте Cursor и перезапустите:
   ```bash
   killall cursor
   cursor
   ```

---

## 📝 Автоматическая установка (скрипт)

Создайте файл `install-cursor-settings.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Установка настроек Cursor для Ubuntu/Linux"
echo ""

# Проверка зависимостей
echo "📦 Проверка зависимостей..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js не установлен. Устанавливаю..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не установлен. Устанавливаю..."
    sudo apt update
    sudo apt install python3 python3-pip -y
fi

echo "✅ Зависимости проверены"
echo ""

# Создание папок
echo "📁 Создание структуры папок..."
mkdir -p ~/.config/Cursor/User/
mkdir -p ~/.cursor/
mkdir -p ~/tools/

# Копирование настроек
echo "📄 Копирование настроек..."
cp docs/cursor_settings/settings.json ~/.config/Cursor/User/
cp docs/cursor_settings/keybindings.json ~/.config/Cursor/User/
cp docs/cursor_settings/global_rules.md ~/.cursor/
cp docs/cursor_settings/mcp.json ~/.cursor/

echo "✅ Настройки скопированы"
echo ""

# Установка spec-kit
echo "🔧 Установка Spec-Kit..."
if [ ! -d ~/tools/spec-kit ]; then
    cd ~/tools
    git clone https://github.com/github/spec-kit.git
    pip3 install -e spec-kit/
    echo "✅ Spec-Kit установлен"
else
    echo "⚠️  Spec-Kit уже установлен"
fi

echo ""
echo "🎉 Установка завершена!"
echo ""
echo "⚠️  ВАЖНО:"
echo "1. Откройте ~/.cursor/mcp.json и замените YOUR_GITHUB_TOKEN_HERE на ваш токен"
echo "2. Перезапустите Cursor: killall cursor && cursor"
echo ""
echo "Проверьте установку:"
echo "  - specify check    # для проверки spec-kit"
echo "  - node --version   # должно быть >= 18.0.0"
```

Сделайте скрипт исполняемым и запустите:

```bash
chmod +x install-cursor-settings.sh
./install-cursor-settings.sh
```

---

## 🎯 Быстрая установка (одной командой)

```bash
# Создание папок
mkdir -p ~/.config/Cursor/User/ ~/.cursor/ ~/tools/

# Копирование настроек
cp docs/cursor_settings/{settings.json,keybindings.json} ~/.config/Cursor/User/
cp docs/cursor_settings/global_rules.md ~/.cursor/
cp docs/cursor_settings/mcp.json ~/.cursor/

# Установка spec-kit (если еще не установлен)
[ ! -d ~/tools/spec-kit ] && cd ~/tools && git clone https://github.com/github/spec-kit.git && pip3 install -e spec-kit/ && cd -

# Готово!
echo "✅ Установка завершена! Не забудьте настроить GitHub токен в ~/.cursor/mcp.json"
```

---

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте логи Cursor: `Help → Toggle Developer Tools → Console`
2. Проверьте документацию MCP: https://docs.cursor.com/context/mcp
3. Проверьте GitHub Issues выбранных MCP серверов
4. Проверьте документацию Spec-Kit: https://github.com/github/spec-kit

---

## 🔗 Полезные ссылки

- **Spec-Kit GitHub**: https://github.com/github/spec-kit
- **Cursor Docs**: https://docs.cursor.com
- **MCP Servers**: https://github.com/modelcontextprotocol
- **GitHub Tokens**: https://github.com/settings/tokens

---

**Автор конфигурации:** AlexLan73  
**Проект:** CudaCalc  
**Дата:** Октябрь 2025  
**Платформа:** Ubuntu/Linux

