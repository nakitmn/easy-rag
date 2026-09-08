#!/usr/bin/env bash
# Упаковывает указанную папку в RAG/repomix/output.xml через repomix.
# repomix запускается из RAG/repomix — оттуда же автоматически читается repomix.config.json.
#
# Использование:
#   repomix-pack.sh [путь_к_папке] [доп. аргументы repomix...]
# Если папка не указана — пакуется корень проекта.
# Пример:
#   repomix-pack.sh src/Launcher/Assets/Scripts
#   repomix-pack.sh src/Launcher/Assets/Scripts --compress

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOMIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # RAG/repomix
PROJECT_ROOT="$(cd "$REPOMIX_DIR/../.." && pwd)"      # корень проекта (launcher)

usage() {
    cat <<EOF
Использование: $(basename "$0") [путь_к_папке] [доп. аргументы repomix...]

Упаковывает папку в $REPOMIX_DIR/output.xml
(repomix запускается из $REPOMIX_DIR — конфиг берётся оттуда же).
Если папка не указана — пакуется корень проекта ($PROJECT_ROOT).
Доп. аргументы передаются repomix как есть (например --compress, --style markdown).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
elif [[ $# -ge 1 && "${1:0:1}" != "-" ]]; then
    target="$1"
    shift
else
    target="$PROJECT_ROOT"
fi

if [[ "$target" == /* ]]; then
    abs_target="$target"
else
    abs_target="$PWD/$target"
fi

if [[ ! -d "$abs_target" ]]; then
    echo "Ошибка: папка не найдена — $abs_target" >&2
    exit 1
fi

if ! command -v repomix >/dev/null 2>&1; then
    echo "Ошибка: repomix не установлен или нет в PATH" >&2
    exit 1
fi

echo "→ Запаковываю: $abs_target"
echo "→ Рабочая папка repomix (откуда читается repomix.config.json): $REPOMIX_DIR"

cd "$REPOMIX_DIR"
repomix "$abs_target" "$@"

echo "✓ Готово: $REPOMIX_DIR/output.xml"
