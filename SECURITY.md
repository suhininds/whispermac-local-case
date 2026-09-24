# Security & Privacy

## 1. Data flow

- Движок `groq` (по умолчанию, если найден ключ): аудио записи сжимается в m4a и отправляется в Groq API (`api.groq.com`), в ответ приходит текст.
- Движок `local` (`WHISPERMAC_ENGINE=local`, нет ключа или Groq не ответил): аудио обрабатывается на устройстве (MLX Whisper).
- Транскрипт вставляется в активное приложение через системный буфер обмена.
- Опционально создаются локальные файлы:
  - `~/whisper_log.txt` (транскрипт)
  - `~/whisper_perf.log` (метрики скорости)

## 2. Network behavior

- Groq-ключ читается из `GROQ_API_KEY` или из файла `~/.whispermac_groq_key` (вне репозитория). В код и в репозиторий ключ не попадает.
- С ключом каждая запись уходит в Groq. Полностью без сети: `WHISPERMAC_ENGINE=local`.
- На первом запуске локальная модель может быть скачана из Hugging Face.
- После кэширования можно принудительно включить offline-only:
  - `WHISPERMAC_STRICT_LOCAL=1`
  - это активирует `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`.

## 3. Recommended secure mode

Для приватных сессий:

```bash
# после того как модель уже была скачана хотя бы один раз
WHISPERMAC_ENGINE=local \
WHISPERMAC_STRICT_LOCAL=1 \
WHISPERMAC_SAVE_TRANSCRIPTS=0 \
WHISPERMAC_SAVE_PERF_LOG=1 \
python whisper_mac.py
```

## 4. Before publishing repository

1. Убедись, что в репо не попали `venv/`, `WhisperMac.app/`, `*.dSYM`.
2. Удали/не добавляй пользовательские логи и транскрипты.
3. Прогони:

```bash
./scripts/preflight_share.sh
```

## 5. Threat model (short)

- В режиме `local` защищает от утечки через облачные ASR API. В режиме `groq` аудио уходит в Groq — используй `local` для приватных записей.
- Не защищает, если устройство уже скомпрометировано локально.
- Не защищает от лог-утечек, если включено сохранение транскрипта.
