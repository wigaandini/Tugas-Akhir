#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${1:-repo_structure.txt}"
TARGET_DIR="${2:-.}"

# Normalize TARGET_DIR supaya path matching lebih aman
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"
OUTPUT_FILE_ABS="$(realpath "$OUTPUT_FILE" 2>/dev/null || echo "$PWD/$OUTPUT_FILE")"

IGNORE_DIRS=(
  ".git"
  "node_modules"
  "vendor"
  "dist"
  "build"
  ".next"
  ".nuxt"
  "coverage"
  ".cache"
  ".idea"
  ".vscode"
  "__pycache__"
  ".venv"
  "venv"
  "tmp"
  "logs"
)

IGNORE_FILES=(
  "*.log"
  "*.lock"
  "*.pyc"
  "*.class"
  "*.o"
  "*.so"
  "*.dll"
  "*.exe"
  "*.DS_Store"
)

SPECIAL_SUMMARY_DIR="data/processed/windows"

should_ignore_dir() {
  local name="$1"

  for d in "${IGNORE_DIRS[@]}"; do
    if [[ "$name" == "$d" ]]; then
      return 0
    fi
  done

  return 1
}

should_ignore_file() {
  local name="$1"

  for p in "${IGNORE_FILES[@]}"; do
    if [[ "$name" == $p ]]; then
      return 0
    fi
  done

  return 1
}

is_inside_ignored_dir() {
  local path="$1"
  local rel_path="${path#"$TARGET_DIR"/}"
  local part

  IFS='/' read -ra parts <<< "$rel_path"

  for part in "${parts[@]}"; do
    if should_ignore_dir "$part"; then
      return 0
    fi
  done

  return 1
}

is_inside_special_dir() {
  local path="$1"
  local rel_path="${path#"$TARGET_DIR"/}"

  if [[ "$rel_path" == "$SPECIAL_SUMMARY_DIR" || "$rel_path" == "$SPECIAL_SUMMARY_DIR/"* ]]; then
    return 0
  fi

  return 1
}

print_tree() {
  local current="$1"
  local prefix="$2"
  local rel="${current#"$TARGET_DIR"/}"

  if [[ "$current" == "$TARGET_DIR" ]]; then
    echo "."
  fi

  # Kalau ketemu folder special, ringkas saja dan jangan masuk ke file-file di dalamnya
  if [[ "$rel" == "$SPECIAL_SUMMARY_DIR" ]]; then
    echo "${prefix}$(basename "$current")/"
    summarize_windows_dir "$current" "$prefix  "
    return
  fi

  local entries=()
  while IFS= read -r -d '' entry; do
    entries+=("$entry")
  done < <(find "$current" -mindepth 1 -maxdepth 1 -print0 | sort -z)

  local filtered=()
  local entry base

  for entry in "${entries[@]}"; do
    base="$(basename "$entry")"

    if [[ -d "$entry" ]]; then
      should_ignore_dir "$base" && continue
    else
      should_ignore_file "$base" && continue

      # Jangan tulis output file ke report
      if [[ "$(realpath "$entry" 2>/dev/null || echo "$entry")" == "$OUTPUT_FILE_ABS" ]]; then
        continue
      fi
    fi

    filtered+=("$entry")
  done

  local total="${#filtered[@]}"
  local i=0
  local branch
  local next_prefix

  for entry in "${filtered[@]}"; do
    i=$((i + 1))
    base="$(basename "$entry")"

    if [[ $i -eq $total ]]; then
      branch="└── "
      next_prefix="${prefix}    "
    else
      branch="├── "
      next_prefix="${prefix}│   "
    fi

    if [[ -d "$entry" ]]; then
      echo "${prefix}${branch}${base}/"
      print_tree "$entry" "$next_prefix"
    else
      echo "${prefix}${branch}${base}"
    fi
  done
}

summarize_windows_dir() {
  local win_dir="$1"
  local prefix="$2"

  if [[ ! -d "$win_dir" ]]; then
    echo "${prefix}[missing]"
    return
  fi

  local h
  local hdir
  local level1_count
  local level2_count
  local npz_count
  local total_files
  local branch

  for h in $(seq 0 35); do
    hdir="$win_dir/h$h"

    if [[ ! -d "$hdir" ]]; then
      if [[ "$h" -eq 35 ]]; then
        branch="└── "
      else
        branch="├── "
      fi

      echo "${prefix}${branch}h$h/ [missing]"
      continue
    fi

    level1_count=$(find "$hdir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    level2_count=$(find "$hdir" -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')
    npz_count=$(find "$hdir" -type f -name "*.npz" | wc -l | tr -d ' ')
    total_files=$(find "$hdir" -type f | wc -l | tr -d ' ')

    if [[ "$h" -eq 35 ]]; then
      branch="└── "
    else
      branch="├── "
    fi

    echo "${prefix}${branch}h$h/ [summary: level1_dirs=${level1_count}, level2_dirs=${level2_count}, npz_files=${npz_count}, total_files=${total_files}]"
  done
}

generate_metadata() {
  echo "REPOSITORY STRUCTURE REPORT"
  echo "Generated at : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Target dir   : $TARGET_DIR"
  echo "Special mode : summarize ${SPECIAL_SUMMARY_DIR}/h0 ... h35"
  echo
  echo "Ignored dirs : ${IGNORE_DIRS[*]}"
  echo "Ignored files: ${IGNORE_FILES[*]}"
  echo
  echo "=================================================="
  echo "DIRECTORY TREE"
  echo "=================================================="
}

generate_file_summary() {
  echo
  echo "=================================================="
  echo "FILE SUMMARY"
  echo "=================================================="

  while IFS= read -r -d '' file; do
    local rel_path
    local base
    local file_abs

    rel_path="${file#"$TARGET_DIR"/}"
    base="$(basename "$file")"
    file_abs="$(realpath "$file" 2>/dev/null || echo "$file")"

    # Skip output report file itu sendiri
    if [[ "$file_abs" == "$OUTPUT_FILE_ABS" ]]; then
      continue
    fi

    # Skip file dengan pattern ignore
    if should_ignore_file "$base"; then
      continue
    fi

    # Skip file yang ada di dalam folder ignore, misalnya .git/, .venv/, venv/
    if is_inside_ignored_dir "$file"; then
      continue
    fi

    # Skip semua file detail di data/processed/windows
    if is_inside_special_dir "$file"; then
      continue
    fi

    local line_count
    local file_type
    local size_bytes

    line_count=$(wc -l < "$file" 2>/dev/null || echo 0)
    file_type=$(file -b "$file" 2>/dev/null || echo "unknown")
    size_bytes=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "N/A")

    printf -- "- %s | lines: %s | size: %s bytes | type: %s\n" \
      "$rel_path" "$line_count" "$size_bytes" "$file_type"

  done < <(
    find "$TARGET_DIR" \
      \( -type d \( \
        -name ".git" -o \
        -name "node_modules" -o \
        -name "vendor" -o \
        -name "dist" -o \
        -name "build" -o \
        -name ".next" -o \
        -name ".nuxt" -o \
        -name "coverage" -o \
        -name ".cache" -o \
        -name ".idea" -o \
        -name ".vscode" -o \
        -name "__pycache__" -o \
        -name ".venv" -o \
        -name "venv" -o \
        -name "tmp" -o \
        -name "logs" \
      \) -prune \) \
      -o \( -path "$TARGET_DIR/$SPECIAL_SUMMARY_DIR" -prune \) \
      -o \( -type f -print0 \) | sort -z
  )
}

generate_special_section() {
  local win_dir="$TARGET_DIR/$SPECIAL_SUMMARY_DIR"

  echo
  echo "=================================================="
  echo "SPECIAL SUMMARY: $SPECIAL_SUMMARY_DIR"
  echo "=================================================="

  if [[ ! -d "$win_dir" ]]; then
    echo "[missing]"
    return
  fi

  local total_npz
  local total_files

  total_npz=$(find "$win_dir" -type f -name "*.npz" | wc -l | tr -d ' ')
  total_files=$(find "$win_dir" -type f | wc -l | tr -d ' ')

  echo "total_npz_files  : $total_npz"
  echo "total_files      : $total_files"
  echo

  local h
  local hdir
  local npz_count
  local total_count
  local sample_first
  local sample_last

  for h in $(seq 0 35); do
    hdir="$win_dir/h$h"

    if [[ ! -d "$hdir" ]]; then
      echo "- h$h: missing"
      continue
    fi

    npz_count=$(find "$hdir" -type f -name "*.npz" | wc -l | tr -d ' ')
    total_count=$(find "$hdir" -type f | wc -l | tr -d ' ')

    sample_first=$(find "$hdir" -type f | sort | head -n 1 | sed "s#^$TARGET_DIR/##" || true)
    sample_last=$(find "$hdir" -type f | sort | tail -n 1 | sed "s#^$TARGET_DIR/##" || true)

    echo "- h$h: npz_files=$npz_count, total_files=$total_count"

    if [[ -n "$sample_first" ]]; then
      echo "  first_sample: $sample_first"
    fi

    if [[ -n "$sample_last" ]]; then
      echo "  last_sample : $sample_last"
    fi
  done
}

{
  generate_metadata
  print_tree "$TARGET_DIR" ""
  generate_special_section
  generate_file_summary
} > "$OUTPUT_FILE"

echo "Selesai. Output disimpan ke: $OUTPUT_FILE"