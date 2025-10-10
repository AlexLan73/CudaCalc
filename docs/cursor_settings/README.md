# 📦 Настройки Cursor для разработки C++/Python/C#/MATLAB

Пакет настроек для IDE Cursor с конфигурацией AI правил и MCP серверов.

## 🎯 Что включено:

- ✅ **settings.json** - основные настройки редактора
- ✅ **keybindings.json** - пользовательские горячие клавиши  
- ✅ **global_rules.md** - AI правила для C++, Python, C#, MATLAB
- ✅ **mcp.json** - конфигурация MCP серверов (Context7, GitHub, Sequential Thinking, Playwright, MemoryBank, BraveSearch, Firecrawl)

## 📚 Документация:

- 📖 **[SPEC_KIT_MEMORY_BANK_GUIDE_RU.md](SPEC_KIT_MEMORY_BANK_GUIDE_RU.md)** - Полное руководство по Spec-Kit и MemoryBank
- ⚡ **[QUICK_REFERENCE_RU.md](QUICK_REFERENCE_RU.md)** - Шпаргалка на одну страницу (распечатай и держи под рукой!)
- 🪟 **[INSTALLATION_WINDOWS_RU.md](INSTALLATION_WINDOWS_RU.md)** - Установка на Windows
- 🐧 **[INSTALLATION_UBUNTU_RU.md](INSTALLATION_UBUNTU_RU.md)** - Установка на Ubuntu/Linux

## 📖 Установка

### Linux/Ubuntu:
См. подробную инструкцию в файле **INSTALLATION_UBUNTU_RU.md**

Краткая версия:
```bash
# Основные настройки
cp settings.json keybindings.json ~/.config/Cursor/User/

# AI Rules
cp global_rules.md ~/.cursor/

# MCP серверы (сначала настройте токены!)
cp mcp.json ~/.cursor/

# Spec-Kit
git clone https://github.com/github/spec-kit.git ~/tools/spec-kit
pip3 install -e ~/tools/spec-kit/
```

### Windows:
См. подробную инструкцию в файле **INSTALLATION_WINDOWS_RU.md**

Краткая версия:
```
settings.json, keybindings.json → %APPDATA%\Cursor\User\
global_rules.md → %APPDATA%\Cursor\
mcp.json → C:\Users\ВашеИмя\.cursor\
```

## ⚠️ Важно!

Перед использованием `mcp.json`:
1. Замените `YOUR_GITHUB_TOKEN_HERE` на ваш реальный GitHub токен
2. Настройте другие API ключи по необходимости (Context7, BraveSearch, Firecrawl)

## 🔧 MCP Серверы

| Сервер | Описание | Требует API ключ |
|--------|----------|------------------|
| **sequential-thinking** | Анализ сложных задач пошагово | ❌ Нет |
| **playwright** | Автоматизация браузера | ❌ Нет |
| **github** | Управление Git репозиториями | ✅ GitHub Token |
| **MemoryBank** | Управление знаниями проекта | ❌ Нет |
| **context7** | Актуальная документация библиотек | ✅ Context7 API |
| **BraveSearch** | Веб-поиск | ✅ Brave API |
| **Firecrawl** | Веб-скрапинг | ✅ Firecrawl API |

## 📝 AI Rules

Включены правила для:
- **C++**: Modern C++ (C++17/20/23), RAII, Smart Pointers, Performance
- **Python**: PEP 8, Type Hints, Async, Testing
- **C#**: .NET 6+, Clean Architecture, CQRS, Performance
- **MATLAB**: Vectorization, Numerical Stability, Scientific Computing

## 🚀 Расширения Cursor

Рекомендуемые расширения (уже установлены в исходной конфигурации):
- `anysphere.cursorpyright` - Python type checking
- `kylinideteam.cppdebug` - C++ debugger
- `ms-python.python` - Python поддержка
- `ms-python.debugpy` - Python отладка

## 📞 Поддержка

- **Проект:** CudaCalc
- **GitHub:** https://github.com/AlexLan73/CudaCalc
- **Автор:** AlexLan73

---

**Лицензия:** Свободное использование  
**Версия:** 1.0  
**Дата:** Октябрь 2025
