# WhisperMac

Голосовой ввод для macOS: зажал правый Option, говоришь, отпустил — текст вставляется в активное поле любого приложения.

Автор материала: [t.me/ei_ai_channel](https://t.me/ei_ai_channel)

## Что это

`WhisperMac` записывает голос с микрофона, превращает его в текст и вставляет в активное приложение.

Два движка распознавания:
- **Groq** (по умолчанию, если есть ключ) — запись целиком уходит в Groq API (`whisper-large-v3-turbo`), ответ приходит примерно за секунду даже на минутную запись;
- **локальный** `mlx-whisper` — работает без сети и без ключа. Включается сам, если ключа нет или Groq не ответил, либо принудительно через `WHISPERMAC_ENGINE=local`.

## Виджет

Компактная «пилюля» 112×30 поверх всех окон, её можно таскать мышью по экрану:
- серые точки — готов к записи;
- красный эквалайзер — идёт запись;
- жёлтая волна — распознавание;
- при наведении по краям появляются `≡` (логи) и `×` (выход);
- клик по пилюле — старт/стоп записи.

В простое виджет не перерисовывается и почти не нагружает процессор.

## Groq-ключ

Ключ берётся в [console.groq.com](https://console.groq.com/keys) и кладётся в файл (не в репозиторий):

```bash
echo 'gsk_...' > ~/.whispermac_groq_key
chmod 600 ~/.whispermac_groq_key
```

Либо через переменную окружения `GROQ_API_KEY`. Без ключа приложение работает локально.

## Почему локальный режим быстрый

В проекте реализованы 4 принципа:
- streaming-обработка чанков без повторной конкатенации всего буфера;
- backlog-flush после stop (без одного гигантского хвоста);
- адаптивный final-pass только при проблемных случаях;
- фильтр тишины/галлюцинаций на `no_speech_prob`.

## Приватность

- С Groq-ключом аудио каждой записи отправляется в Groq API. Если это не подходит — `WHISPERMAC_ENGINE=local` или просто не класть ключ: тогда всё остаётся на устройстве.
- `WHISPERMAC_STRICT_LOCAL` отключает только сетевую загрузку локальной модели и не отключает Groq.
- По умолчанию отключена телеметрия Hugging Face (`HF_HUB_DISABLE_TELEMETRY=1`).
- Для офлайн режима после первичного кэша модели:
  - `WHISPERMAC_STRICT_LOCAL=1`
  - это включает `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`.

Подробно: [`SECURITY.md`](SECURITY.md)

## Быстрый старт (macOS)

```bash
git clone https://github.com/ranlywood/whispermac-local-case.git
cd whispermac-local-case
./setup.sh
open /Applications/WhisperMac.app
```

Для быстрого режима положи Groq-ключ (см. выше) и перезапусти приложение.

`setup.sh` делает всё необходимое:
- проверяет Homebrew Python + Tk (>= 8.6);
- пересоздаёт `venv` на Homebrew Python;
- ставит зависимости;
- предзагружает модель;
- собирает `dist/WhisperMac.app`.
- обновляет `/Applications/WhisperMac.app` (с backup старой версии).

Разрешения в macOS (важно: именно для `WhisperMac.app`):
- `System Settings -> Privacy & Security -> Microphone -> WhisperMac ✅`
- `System Settings -> Privacy & Security -> Accessibility -> WhisperMac ✅`
- `System Settings -> Privacy & Security -> Input Monitoring -> WhisperMac ✅`
- если выданы после запуска: перезапусти `WhisperMac.app`

## Платформы

- macOS: поддерживается.
- Windows: сейчас не поддерживается, так как приложение использует macOS-специфичные `Quartz` и `AppKit`.
- Linux: не поддерживается в текущей реализации UI и горячих клавиш.

## Безопасный запуск (рекомендуется)

```bash
./scripts/launch_secure.sh
```

Скрипт запускает приложение с:
- `WHISPERMAC_STRICT_LOCAL=auto` (включается только если модель уже в кэше)
- `WHISPERMAC_SAVE_TRANSCRIPTS=1` (логи пишутся в `~/whisper_log.txt`)
- `WHISPERMAC_SAVE_PERF_LOG=1`

После того как модель уже скачана, можно включить full offline:

```bash
WHISPERMAC_STRICT_LOCAL=1 ./scripts/launch_secure.sh
```

По умолчанию hotkey-режим:
- `Right Option` зажат -> запись;
- `Right Option` отпущен -> остановка и вставка текста.

Если нужно вернуть старое поведение (только клик по виджету):

```bash
WHISPERMAC_HOLD_KEY=off ./scripts/launch_secure.sh
```

Логи транскрибаций:
- кнопка `≡` появляется слева на пилюле при наведении, клик открывает/скрывает окно логов;
- горячая клавиша: `Cmd+Shift+E` (также поддерживается `Cmd+Shift+H`);
- источник данных: `~/whisper_log.txt`.
- диагностический лог вставки/фокуса: `~/whisper_runtime.log` (кнопка `Диагностика` в окне логов).

## Сборка .app bundle (иконка в Dock)

```bash
./scripts/build_app.sh
open ./dist/WhisperMac.app
```

Сборка создаёт native macOS executable внутри `.app`, чтобы macOS показывала `WhisperMac`, а не `bash`, в Privacy & Security permissions.

Если запускать через `.app`, в Dock будет имя и иконка WhisperMac (не Python).

## Запуск без .app (dev mode)

```bash
./scripts/launch_secure.sh
```

## Тюнинг

```bash
export WHISPERMAC_CHUNK_SEC=10
export WHISPERMAC_WORKER_POLL_SEC=0.20
export WHISPERMAC_FINAL_PASS_MIN_SEC=15
export WHISPERMAC_FINAL_PASS_MAX_SEC=95
python whisper_mac.py
```

Полезные env:
- `WHISPERMAC_MODEL_REPO` - HF repo или локальный путь к модели.
- `WHISPERMAC_LANGUAGE` - язык (по умолчанию `ru`).
- `WHISPERMAC_PY_FORMULA` - Homebrew Python formula для `setup.sh` (по умолчанию `python@3.12`).
- `WHISPERMAC_TK_FORMULA` - Homebrew Tk formula для `setup.sh` (по умолчанию `python-tk@3.12`).
- `WHISPERMAC_STRICT_LOCAL=1` - только локальный режим после кэша.
- `WHISPERMAC_DOCK_MODE=regular|accessory` - отображение в Dock.
- `WHISPERMAC_ENGINE=groq|local` - движок распознавания (по умолчанию `groq`, без ключа — локально).
- `GROQ_API_KEY` - ключ Groq (альтернатива файлу `~/.whispermac_groq_key`).
- `WHISPERMAC_GROQ_MODEL` - модель Groq (по умолчанию `whisper-large-v3-turbo`).
- `WHISPERMAC_HOLD_KEY=right_option|off` - режим удержания: зажал `Right Option` -> запись, отпустил -> вставка.
- `WHISPERMAC_SAVE_TRANSCRIPTS=0` - не писать `~/whisper_log.txt`.
- `WHISPERMAC_SAVE_PERF_LOG=0` - не писать `~/whisper_perf.log`.
- `WHISPERMAC_PASTE_SHORTCUT_MODE=auto|osascript|pynput|session|cgevent` - способ отправки `Cmd+V` (по умолчанию `auto`).
- `WHISPERMAC_RUNTIME_LOG=0` - отключить `~/whisper_runtime.log`.

## Публичный релиз-чек

Перед публикацией прогоняй:

```bash
./scripts/preflight_share.sh
```

Скрипт проверяет:
- потенциальные секреты;
- персональные абсолютные пути;
- крупные бинарники, которые не должны попасть в репо.

## Материалы кейса

- Кейс: [`docs/CASE_STUDY_RU.md`](docs/CASE_STUDY_RU.md)
- Драфт поста: [`docs/POST_DRAFT_RU.md`](docs/POST_DRAFT_RU.md)
