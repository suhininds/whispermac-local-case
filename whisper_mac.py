#!/usr/bin/env python3
"""
WhisperMac — голосовой ввод для macOS (streaming + real-time EQ)
"""

import math
import os
import subprocess
import threading
import time
import sys
import atexit
from pathlib import Path
import fcntl
import io
import wave
import shutil
import tempfile

import numpy as np
import sounddevice as sd
import mlx_whisper
import tkinter as tk
from ApplicationServices import (
    AXIsProcessTrusted,
    AXIsProcessTrustedWithOptions,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXUIElementIsAttributeSettable,
    AXUIElementSetAttributeValue,
    AXUIElementSetMessagingTimeout,
    kAXFocusedUIElementAttribute,
    kAXRoleAttribute,
    kAXSelectedTextAttribute,
    kAXTrustedCheckOptionPrompt,
)

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
    kCGSessionEventTap,
)
from AppKit import NSWorkspace, NSPasteboard, NSPasteboardTypeString


# ═══════════════════════════════════════════════════
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_config_file() -> None:
    """Личные настройки из ~/.config/whispermac/env (KEY=VALUE, # — комментарий).

    Файл вне репозитория, его значения перекрывают умолчания лаунчера:
    .app не видит переменные окружения оболочки, а словарь терминов и
    режим приватности не должны лежать в публичном форке.
    """
    path = Path(os.getenv("WHISPERMAC_CONFIG", str(Path.home() / ".config" / "whispermac" / "env")))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


_load_config_file()

# Отключаем телеметрию Hugging Face по умолчанию.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Для строгого локального режима после первичного кэша модели:
# WHISPERMAC_STRICT_LOCAL=1 -> без сетевых запросов.
STRICT_LOCAL_MODE = _env_bool("WHISPERMAC_STRICT_LOCAL", False)
if STRICT_LOCAL_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_REPO   = os.getenv("WHISPERMAC_MODEL_REPO", "mlx-community/whisper-large-v3-mlx-4bit")
LANGUAGE     = os.getenv("WHISPERMAC_LANGUAGE", "ru")
SAMPLE_RATE  = 16000
MIN_DURATION = 0.3
SAVE_TRANSCRIPTS = _env_bool("WHISPERMAC_SAVE_TRANSCRIPTS", True)
SAVE_PERF_LOG = _env_bool("WHISPERMAC_SAVE_PERF_LOG", True)

CHUNK_SEC    = max(5.0, _env_float("WHISPERMAC_CHUNK_SEC", 10.0))
WORKER_POLL_SEC = max(0.05, _env_float("WHISPERMAC_WORKER_POLL_SEC", 0.20))
FINAL_PASS_MIN_SEC = max(5.0, _env_float("WHISPERMAC_FINAL_PASS_MIN_SEC", 15.0))
FINAL_PASS_MAX_SEC = max(
    FINAL_PASS_MIN_SEC,
    _env_float("WHISPERMAC_FINAL_PASS_MAX_SEC", 95.0),
)
LOW_CONF_LOGPROB = _env_float("WHISPERMAC_LOW_CONF_LOGPROB", -1.15)
SILENCE_SKIP_NO_SPEECH = min(
    0.99,
    max(0.5, _env_float("WHISPERMAC_SILENCE_SKIP_NO_SPEECH", 0.83)),
)
SILENCE_SKIP_MAX_CHARS = int(max(8, _env_float("WHISPERMAC_SILENCE_SKIP_MAX_CHARS", 36)))
FINAL_TEMPERATURES = (0.0, 0.2, 0.4, 0.6)
HOTWORDS_PROMPT = os.getenv(
    "WHISPERMAC_HOTWORDS",
    "WhisperMac, Whisper Flow, Miro, Zoom, Claude Code, ChatGPT.",
)
PASTE_SHORTCUT_MODE = os.getenv("WHISPERMAC_PASTE_SHORTCUT_MODE", "auto").strip().lower()

# ── Движок транскрипции ─────────────────────────────
# ENGINE=groq (по умолчанию) — быстрый облачный путь через Groq API,
# локальный mlx-whisper остаётся фоллбэком при ошибке/отсутствии сети/ключа.
# ENGINE=local — только локальная модель (как раньше).
ENGINE       = os.getenv("WHISPERMAC_ENGINE", "groq").strip().lower()
GROQ_MODEL   = os.getenv("WHISPERMAC_GROQ_MODEL", "whisper-large-v3-turbo")
GROQ_TIMEOUT = max(3.0, _env_float("WHISPERMAC_GROQ_TIMEOUT", 120.0))
GROQ_CONNECT_TIMEOUT = max(3.0, _env_float("WHISPERMAC_GROQ_CONNECT_TIMEOUT", 10.0))
GROQ_API_URL = os.getenv(
    "WHISPERMAC_GROQ_URL",
    "https://api.groq.com/openai/v1/audio/transcriptions",
)
# strict_local относится только к загрузке ЛОКАЛЬНОЙ модели (offline от HuggingFace)
# и НЕ отключает Groq. Чтобы работать полностью локально — задай WHISPERMAC_ENGINE=local.


def _load_groq_key() -> str:
    """Ключ Groq: сначала env, затем локальный файл вне репозитория."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    for path in (Path.home() / ".whispermac_groq_key",
                 Path.home() / ".config" / "whispermac" / "groq_key"):
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


GROQ_API_KEY = _load_groq_key()


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1..1] → 16-bit PCM WAV в памяти (для отправки в Groq)."""
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _encode_for_groq(audio: np.ndarray) -> tuple:
    """
    Готовит payload для Groq. Длинные записи в несжатом WAV грузятся долго
    и упираются в таймаут, поэтому сжимаем речь в AAC/m4a нативным afconvert
    (всегда есть в macOS): ~6 МБ WAV → ~1 МБ. При любой ошибке — обычный WAV.
    Возвращает (filename, bytes, mime).
    """
    wav = _audio_to_wav_bytes(audio)
    afconvert = shutil.which("afconvert")
    if not afconvert:
        return "audio.wav", wav, "audio/wav"
    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.wav"
            dst = Path(td) / "out.m4a"
            src.write_bytes(wav)
            proc = subprocess.run(
                [afconvert, "-f", "m4af", "-d", "aac", "-b", "48000",
                 str(src), str(dst)],
                capture_output=True, timeout=30,
            )
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return "audio.m4a", dst.read_bytes(), "audio/mp4"
            log(f"[groq] afconvert rc={proc.returncode} — шлю WAV")
    except Exception as ex:  # noqa: BLE001
        log(f"[groq] сжатие не удалось ({ex}) — шлю WAV")
    return "audio.wav", wav, "audio/wav"


def groq_transcribe(audio: np.ndarray, *, prompt: str = "", api_key: str = "") -> str:
    """
    Одним запросом отправляет всю запись в Groq и возвращает текст.
    При любой ошибке возвращает "" — вызывающий код падает на локальный фоллбэк.
    """
    key = api_key or GROQ_API_KEY
    if not key:
        log("[groq] ключ не найден — фоллбэк на локальную модель")
        return ""
    try:
        import requests  # ленивый импорт: локальный режим не требует requests
    except Exception as ex:  # noqa: BLE001
        log(f"[groq] requests недоступен ({ex}) — фоллбэк")
        return ""
    try:
        fname, payload, mime = _encode_for_groq(audio)
        files = {"file": (fname, payload, mime)}
        data = {
            "model": GROQ_MODEL,
            "language": LANGUAGE,
            "response_format": "json",
            "temperature": "0",
        }
        if prompt:
            data["prompt"] = prompt
        started = time.perf_counter()
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {key}"},
            files=files,
            data=data,
            timeout=(GROQ_CONNECT_TIMEOUT, GROQ_TIMEOUT),
        )
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            log(f"[groq] HTTP {resp.status_code}: {resp.text[:160]} — фоллбэк")
            return ""
        text = (resp.json().get("text") or "").strip()
        audio_sec = len(audio) / SAMPLE_RATE
        kb = len(payload) / 1024
        log(f"[groq] {audio_sec:.1f}s аудио ({fname}, {kb:.0f}КБ) → {elapsed:.2f}s: {text}")
        return text
    except Exception as ex:  # noqa: BLE001
        log(f"[groq] ошибка запроса ({ex}) — фоллбэк на локальную модель")
        return ""


# ═══════════════════════════════════════════════════

# Компактная «пилюля» в духе Wispr Flow: бары по центру, кнопки ≡ и ×
# появляются по краям при наведении.
W, H   = 112, 30
WIN_ALPHA = 0.97

PILL          = "#0B0B0D"
C_BORDER      = "#3A3A40"   # обводка в покое
C_BORDER_REC  = "#7A2638"   # обводка при записи
C_BORDER_PROC = "#7A5214"   # обводка при обработке
C_IDLE        = "#5E5E66"
C_REC         = "#FF375F"
C_PROC        = "#FF9F0A"
C_SPINNER     = "#8E8E93"
C_BTN_BG      = "#232327"
C_BTN_HV      = "#3A3A40"
C_BTN_FG      = "#A1A1A8"

BTN_R    = 9
LOGS_CX  = 16
CLOSE_CX = W - 16

BAR_COUNT = 9
BAR_W     = 2.5
BAR_STEP  = 5
BARS_X    = (W - (BAR_COUNT - 1) * BAR_STEP) / 2   # центр первого бара
BAR_MIN   = 1.25
BAR_MAX   = 8.5

# Полосы спектра: лог-шкала, низкие частоты в центре, зеркально к краям.
# Наклон поднимает тихие высокие полосы, огибающая держит центр выше краёв
# (подобрано на живом голосе: центр ~0.65, края ~0.15 от максимума).
EQ_BANDS         = (BAR_COUNT + 1) // 2          # 5 полос на 9 баров
EQ_BAND_EDGES_HZ = (200.0, 5000.0)
EQ_BAND_TILT     = np.array([1.0, 1.2, 1.6, 2.4, 3.4], dtype=np.float32)
EQ_ENVELOPE      = np.array([1.0, 0.9, 0.78, 0.66, 0.55], dtype=np.float32)
EQ_MIRROR        = np.array(
    [min(EQ_BANDS - 1, abs(i - BAR_COUNT // 2)) for i in range(BAR_COUNT)]
)

# Окно логов
LOG_BG    = "#111114"
LOG_PANEL = "#18181C"
LOG_FG    = "#ECECF1"
LOG_DIM   = "#8E8E96"
LOG_SEL   = "#34343C"
LOG_BTN   = "#26262C"
LOG_BTN_HV = "#34343C"
UI_FONT   = "Helvetica Neue"

# EQ smoothing
EQ_ATTACK        = 0.86
EQ_DECAY         = 0.28    # плавнее спад, меньше "рваности"
EQ_RMS_THRESHOLD = 0.0038  # ниже этого — тишина
EQ_RMS_ALPHA     = 0.42    # сглаживание RMS для плавного live-ответа
EQ_RMS_FULL      = 0.028   # rms, при котором бары считаются "полными"
EQ_VISUAL_GAMMA  = 0.62    # усиливает видимую реакцию на среднюю громкость
EQ_WOBBLE_MAX    = 0.16    # добавляет "живость" баров при речи

V_KEY = 9
COMMAND_KEY = 55
INSTANCE_LOCK_PATH = os.getenv("WHISPERMAC_INSTANCE_LOCK", "/tmp/whispermac-app.lock")


def log(msg):
    line = f"  {msg}"
    print(line, flush=True)
    if not _env_bool("WHISPERMAC_RUNTIME_LOG", True):
        return
    try:
        from datetime import datetime
        with open(Path.home() / "whisper_runtime.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    except Exception:
        pass


_PYNPUT_TSM_PATCHED = False


def _patch_pynput_darwin_tsm_crash():
    """Чинит краш WhisperMac на свежих macOS.

    Глобальный слушатель pynput на каждый клавишный ивент зовёт
    CGEventKeyboardGetUnicodeString (перевод клавиши в символ), а та лезет в
    Text Services Manager за текущей раскладкой. На новых macOS этот вызов
    из фонового потока слушателя ловит main-thread-ассерт libdispatch ->
    SIGTRAP, и приложение падает при любом наборе текста.

    WhisperMac нужен только модификатор (правый option, vk 61), символ не
    нужен. Поэтому подменяем _event_to_key на версию без перевода в Unicode:
    спец-клавиши берём из таблицы pynput, всё остальное отдаём как vk. TSM
    больше не дёргается -> краша нет, распознавание hold-key не меняется.
    """
    global _PYNPUT_TSM_PATCHED
    if _PYNPUT_TSM_PATCHED:
        return
    try:
        import contextlib
        from pynput.keyboard import _darwin as d
        from pynput.keyboard import KeyCode

        # (1) На каждый клавишный ивент — без перевода в символ (не зовём TSM).
        def _event_to_key_no_tsm(self, event):
            vk = d.CGEventGetIntegerValueField(event, d.kCGKeyboardEventKeycode)
            event_type = d.CGEventGetType(event)
            is_media = True if event_type == d.NSSystemDefined else None
            key = (vk, is_media)
            if key in self._SPECIAL_KEYS:
                return self._SPECIAL_KEYS[key]
            return KeyCode.from_vk(vk)

        d.Listener._event_to_key = _event_to_key_no_tsm

        # (2) При старте потока-слушателя pynput грузит раскладку через
        #     TISCopyCurrentKeyboardInputSource/TISGetInputSourceProperty
        #     (keycode_context) — это и есть TSM-вызов на фоновом потоке,
        #     который валит приложение ЕЩЁ ДО ввода. Раскладка нам не нужна
        #     (см. пункт 1), поэтому отдаём пустой контекст без обращения к TSM.
        @contextlib.contextmanager
        def _keycode_context_no_tsm():
            yield (None, None)

        d.keycode_context = _keycode_context_no_tsm

        _PYNPUT_TSM_PATCHED = True
        log("pynput TSM-фикс применён (перевод клавиш и загрузка раскладки отключены)")
    except Exception as ex:
        log(f"pynput TSM-фикс не применён: {ex}")


def acquire_instance_lock():
    lock_file = Path(INSTANCE_LOCK_PATH).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_file, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def frontmost_bundle():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.bundleIdentifier() if app else None


def activate_bundle(bid):
    NSApplicationActivateAllWindows = 1
    NSApplicationActivateIgnoringOtherApps = 2
    options = NSApplicationActivateAllWindows | NSApplicationActivateIgnoringOtherApps
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.bundleIdentifier() == bid:
            try:
                app.activateWithOptions_(options)
                if _wait_for_frontmost_bundle(bid):
                    return True
            except Exception as ex:
                log(f"NSWorkspace activate failed for {bid}: {ex}")
            break
    return activate_bundle_osascript(bid)


def _wait_for_frontmost_bundle(bid, timeout: float = 0.45) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if frontmost_bundle() == bid:
            return True
        time.sleep(0.03)
    return False


def activate_bundle_osascript(bid) -> bool:
    if not bid:
        return False
    script = f'tell application id "{bid}" to activate'
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except subprocess.TimeoutExpired:
        log(f"osascript activate timed out: {bid}")
        return False
    except Exception as ex:
        log(f"osascript activate failed for {bid}: {ex}")
        return False

    if proc.returncode == 0 and _wait_for_frontmost_bundle(bid, timeout=0.65):
        return True

    err = (proc.stderr or proc.stdout or "").strip()
    log(f"osascript activate did not focus {bid}: {err or f'code={proc.returncode}'}")
    return False


def running_app_for_bundle(bid):
    if not bid:
        return None
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.bundleIdentifier() == bid:
            return app
    return None


def request_accessibility_permission(prompt: bool = False) -> bool:
    try:
        if prompt:
            return bool(
                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
            )
        return bool(AXIsProcessTrusted())
    except Exception as ex:
        log(f"Accessibility check failed: {ex}")
        return False


def ax_insert_text(text: str, target_bid: str = "") -> bool:
    if not text:
        return False
    if not request_accessibility_permission(prompt=False):
        log("AX insert skipped: Accessibility permission missing")
        return False

    app = None
    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    if front and (not target_bid or front.bundleIdentifier() == target_bid):
        app = front
    if app is None:
        app = running_app_for_bundle(target_bid)
    if app is None:
        log(f"AX insert failed: target app not found ({target_bid or '-'})")
        return False

    try:
        ax_app = AXUIElementCreateApplication(app.processIdentifier())
        AXUIElementSetMessagingTimeout(ax_app, 0.35)
        err, focused = AXUIElementCopyAttributeValue(
            ax_app,
            kAXFocusedUIElementAttribute,
            None,
        )
        if err != 0 or focused is None:
            log(f"AX insert failed: no focused element err={err}")
            return False

        try:
            _, role = AXUIElementCopyAttributeValue(focused, kAXRoleAttribute, None)
        except Exception:
            role = "?"

        settable_err, settable = AXUIElementIsAttributeSettable(
            focused,
            kAXSelectedTextAttribute,
            None,
        )
        if settable_err != 0 or not settable:
            log(f"AX insert unavailable: role={role} settable_err={settable_err}")
            return False

        err = AXUIElementSetAttributeValue(focused, kAXSelectedTextAttribute, text)
        if err == 0:
            log(f"AX insert succeeded: role={role}")
            return True
        log(f"AX insert failed: role={role} err={err}")
        return False
    except Exception as ex:
        log(f"AX insert exception: {ex}")
        return False


def _post_key(key_code: int, pressed: bool, flags: int = 0, tap: int = kCGHIDEventTap):
    event = CGEventCreateKeyboardEvent(None, key_code, pressed)
    CGEventSetFlags(event, flags)
    CGEventPost(tap, event)


def cmd_v(tap: int = kCGHIDEventTap):
    _post_key(COMMAND_KEY, True, kCGEventFlagMaskCommand, tap)
    time.sleep(0.015)
    _post_key(V_KEY, True, kCGEventFlagMaskCommand, tap)
    time.sleep(0.015)
    _post_key(V_KEY, False, kCGEventFlagMaskCommand, tap)
    time.sleep(0.015)
    _post_key(COMMAND_KEY, False, 0, tap)


def cmd_v_osascript(target_bid: str = "") -> bool:
    script_lines = []
    if target_bid:
        script_lines.extend([
            f'tell application id "{target_bid}" to activate',
            "delay 0.18",
        ])
    script_lines.append('tell application "System Events" to keystroke "v" using command down')
    args = ["/usr/bin/osascript"]
    for line in script_lines:
        args.extend(["-e", line])
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except subprocess.TimeoutExpired:
        log("osascript Cmd+V timed out")
        return False
    except Exception as ex:
        log(f"osascript Cmd+V failed: {ex}")
        return False

    if proc.returncode == 0:
        return True

    err = (proc.stderr or proc.stdout or "").strip()
    log(f"osascript Cmd+V failed: {err or f'code={proc.returncode}'}")
    return False


def cmd_v_osascript_keycode(target_bid: str = "") -> bool:
    script_lines = []
    if target_bid:
        script_lines.extend([
            f'tell application id "{target_bid}" to activate',
            "delay 0.18",
        ])
    script_lines.append('tell application "System Events" to key code 9 using command down')
    args = ["/usr/bin/osascript"]
    for line in script_lines:
        args.extend(["-e", line])
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except subprocess.TimeoutExpired:
        log("osascript keycode Cmd+V timed out")
        return False
    except Exception as ex:
        log(f"osascript keycode Cmd+V failed: {ex}")
        return False

    if proc.returncode == 0:
        return True

    err = (proc.stderr or proc.stdout or "").strip()
    log(f"osascript keycode Cmd+V failed: {err or f'code={proc.returncode}'}")
    return False


def cmd_v_pynput() -> bool:
    try:
        from pynput import keyboard
        ctl = keyboard.Controller()
        ctl.press(keyboard.Key.cmd)
        time.sleep(0.015)
        ctl.press("v")
        time.sleep(0.015)
        ctl.release("v")
        time.sleep(0.015)
        ctl.release(keyboard.Key.cmd)
        return True
    except Exception as ex:
        log(f"pynput Cmd+V failed: {ex}")
        return False


def _clean_chunk(text: str) -> str:
    """Убирает артефакты Whisper на границах чанков."""
    import re
    text = text.strip()
    # Убираем ведущие/замыкающие точки, многоточия, пробелы
    text = re.sub(r'^[\s\.…]+', '', text)
    text = re.sub(r'[\s\.…]+$', '', text)
    return text.strip()


def _join_chunks(parts: list) -> str:
    cleaned = [_clean_chunk(p) for p in parts]
    cleaned = [p for p in cleaned if p]
    return " ".join(cleaned)


def _prompt_from_parts(parts: list) -> str:
    """Короткий prompt для смешанной русско-английской речи."""
    tail = _join_chunks(parts)[-180:] if parts else ""
    return f"{HOTWORDS_PROMPT}\n{tail}" if tail else HOTWORDS_PROMPT


def _is_repetition_loop(text: str) -> bool:
    """Детектирует типичный whisper-loop c многократным повтором одной фразы."""
    import re
    words = re.findall(r"[\w$]+", text.lower())
    if len(words) < 12:
        return False

    def max_consecutive_repeat(n: int) -> int:
        best = 1
        i = 0
        end = len(words) - 2 * n
        while i <= end:
            gram = words[i:i+n]
            j = i + n
            run = 1
            while j + n <= len(words) and words[j:j+n] == gram:
                run += 1
                j += n
            if run > best:
                best = run
            i = i + 1 if run == 1 else j
        return best

    # Главный сигнал: подряд много раз повторяется одна и та же короткая фраза.
    for n in (2, 3):
        if max_consecutive_repeat(n) >= 5:
            return True

    def max_ngram_count(n: int) -> tuple:
        from collections import Counter
        grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not grams:
            return (), 0
        gram, count = Counter(grams).most_common(1)[0]
        return gram, count

    # Если один и тот же биграм/триграм покрывает заметную часть текста — это loop.
    for n in (2, 3):
        gram, count = max_ngram_count(n)
        if count >= 10 and (count * n) / max(1, len(words)) >= 0.08:
            return True

    # Отдельный сигнал: многократно повторяется короткая конструкция, напр. "выиграли $0".
    m = re.search(r"(\b[\w$]+(?:\s+[\w$]+){0,2}\b)(?:[\s,.;:!?-]+\1){6,}", text.lower())
    if m:
        return True

    # Еще один сильный индикатор артефакта: ненормально много "$0".
    zero_dollars = sum(1 for w in words if w == "$0" or w.endswith("$0"))
    if zero_dollars >= 6 and zero_dollars / len(words) >= 0.04:
        return True

    return False


def _collapse_repetition_loop(text: str) -> str:
    """Последняя защита: схлопывает подряд идущие повторы короткой фразы."""
    import re
    prev = None
    cur = text
    pattern = re.compile(
        r"(\b[\w$]+(?:\s+[\w$]+){0,2}\b)(?:[\s,.;:!?-]+\1){3,}",
        re.IGNORECASE,
    )
    while prev != cur:
        prev = cur
        cur = pattern.sub(r"\1", cur)
    return cur


def _segment_quality(result: dict) -> tuple:
    segments = result.get("segments") or []
    if not segments:
        return 0.0, 0.0
    avg_logprob = sum(float(s.get("avg_logprob", 0.0)) for s in segments) / len(segments)
    avg_no_speech = sum(float(s.get("no_speech_prob", 0.0)) for s in segments) / len(segments)
    return avg_logprob, avg_no_speech


def _likely_silence_hallucination(text: str, avg_no_speech: float) -> bool:
    if not text:
        return False
    return avg_no_speech >= SILENCE_SKIP_NO_SPEECH and len(text.strip()) <= SILENCE_SKIP_MAX_CHARS


def _eq_band_bins(n: int) -> list:
    """Границы полос эквалайзера в бинах rfft для кадра длиной n."""
    edges_hz = np.geomspace(EQ_BAND_EDGES_HZ[0], EQ_BAND_EDGES_HZ[1], EQ_BANDS + 1)
    n_bins = n // 2 + 1
    edges = np.clip(np.round(edges_hz * n / SAMPLE_RATE).astype(int), 1, n_bins - 1)
    return [(int(a), int(max(b, a + 1))) for a, b in zip(edges[:-1], edges[1:])]


def _plural_records(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} запись"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} записи"
    return f"{n} записей"


class App:
    EXCLUDED = {"python", "whisper-mac", "whispermac", "com.apple.finder"}

    def __init__(self):
        self.ready      = False
        self.recording  = False
        self.processing = False
        self.chunks     = []
        self._chunks_lock = threading.Lock()
        self.stream     = None
        self.target     = None
        self._recording_started_at = None
        self._frame     = 0
        self._drag_ox   = 0
        self._drag_oy   = 0
        self._dragging  = False
        self._hold_key_down = False
        self._hold_started_recording = False
        self._keyboard_listener = None
        self._keyboard_mod = None
        self._hold_key_mode = self._normalize_hold_key_mode(
            os.getenv("WHISPERMAC_HOLD_KEY", "off")
        )

        # Real-time EQ levels (driven by FFT in audio callback)
        self._eq_levels = np.zeros(BAR_COUNT, dtype=np.float32)
        self._eq_smooth = np.zeros(BAR_COUNT, dtype=np.float32)
        self._rms_smooth = 0.0
        self._eq_window = None      # окно Хэннинга, считается один раз
        self._eq_band_idx = []      # границы полос в бинах rfft

        self._anim_running     = False
        self._bar_color        = None
        self._logs_win         = None
        self._logs_list        = None
        self._logs_text        = None
        self._logs_count       = None
        self._copy_btn         = None
        self._log_records      = []
        self._logs_bounds      = (0, 0, 0, 0)
        self._close_bounds     = (0, 0, 0, 0)
        self._suppress_next_toggle = False

        # ── Окно ────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        try:
            self.root.createcommand("tk::mac::Quit", self._quit)
        except tk.TclError:
            pass
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", WIN_ALPHA)
        self._make_borderless()
        self.root.configure(bg=PILL)

        sx = self.root.winfo_screenwidth()  - W - 28
        sy = self.root.winfo_screenheight() - H - 96
        self.root.geometry(f"{W}x{H}+{sx}+{sy}")

        # ── Canvas ───────────────────────────────────────────────
        self.cv = tk.Canvas(self.root, width=W, height=H,
                            bg=PILL, highlightthickness=0)
        self.cv.pack()

        # Пилюля: толстая линия со скруглёнными концами даёт ровные полукруглые
        # торцы (вершины многоугольника Tk округляет до целых точек — выходят
        # грани). Нижняя линия — обводка, верхняя на 2pt уже — заливка.
        r = H / 2
        self._pill_border = self.cv.create_line(
            r, r, W - r, r, width=H, capstyle="round", fill=C_BORDER
        )
        self.cv.create_line(
            r, r, W - r, r, width=H - 2, capstyle="round", fill=PILL
        )

        # Спиннер загрузки (виден, пока модель не готова): вращающаяся дуга
        sr = 6
        self.spinner = self.cv.create_arc(
            W / 2 - sr, H / 2 - sr, W / 2 + sr, H / 2 + sr,
            start=90, extent=270, style="arc", outline=C_SPINNER, width=1.6
        )

        # Эквалайзер: бары-линии со скруглёнными концами
        cy = H / 2
        self.bars = []
        for i in range(BAR_COUNT):
            x = BARS_X + i * BAR_STEP
            b = self.cv.create_line(
                x, cy - BAR_MIN, x, cy + BAR_MIN,
                fill=C_IDLE, width=BAR_W, capstyle="round",
                tags="bar", state="hidden"
            )
            self.bars.append((b, x))

        # Кнопки ≡ (логи) и × (выход): скрыты, показываются при наведении
        self._draw_logs_button()
        self._draw_close()
        self.cv.itemconfig("ctl", state="hidden")

        # ── Биндинги ────────────────────────────────────────────
        self.cv.tag_bind("close", "<Button-1>", self._quit)
        self.cv.tag_bind("close", "<Enter>",
                         lambda e: self.cv.itemconfig(self._close_bg, fill=C_BTN_HV))
        self.cv.tag_bind("close", "<Leave>",
                         lambda e: self.cv.itemconfig(self._close_bg, fill=C_BTN_BG))
        self.cv.tag_bind("logs", "<Button-1>", self._toggle_logs_window)
        self.cv.tag_bind("logs", "<ButtonRelease-1>", lambda _e: "break")
        self.cv.tag_bind("logs", "<Enter>",
                         lambda e: self.cv.itemconfig(self._logs_bg, fill=C_BTN_HV))
        self.cv.tag_bind("logs", "<Leave>",
                         lambda e: self.cv.itemconfig(self._logs_bg, fill=C_BTN_BG))

        self.cv.bind("<Enter>", self._show_controls)
        self.cv.bind("<Leave>", self._hide_controls)
        self.cv.bind("<ButtonPress-1>",  self._press)
        self.cv.bind("<B1-Motion>",       self._motion)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self.root.bind_all("<Command-Shift-E>", self._toggle_logs_window)
        self.root.bind_all("<Command-Shift-H>", self._toggle_logs_window)

        self.root.bind("<Destroy>", self._on_destroy)

        self._setup_hold_key_listener()
        if not request_accessibility_permission(prompt=True):
            log("Accessibility not granted: auto-paste and hold-key may not work")
            self._open_privacy_panel("Accessibility")
        self._track_app()
        self._kick_anim()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _normalize_hold_key_mode(self, raw: str) -> str:
        mode = (raw or "").strip().lower()
        if mode in {"right_option", "option_r", "alt_r"}:
            return "right_option"
        return "off"

    def _is_excluded_bundle(self, bundle_id: str) -> bool:
        bid = (bundle_id or "").lower()
        return not bid or any(ex in bid for ex in self.EXCLUDED)

    def _setup_hold_key_listener(self):
        if self._hold_key_mode == "off":
            log("Hold-to-talk: off")
            return
        try:
            from pynput import keyboard
        except Exception as ex:
            log(f"Hold-to-talk отключен: не удалось загрузить pynput ({ex})")
            self._hold_key_mode = "off"
            return

        self._keyboard_mod = keyboard
        _patch_pynput_darwin_tsm_crash()
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_global_key_press,
            on_release=self._on_global_key_release,
        )
        self._keyboard_listener.daemon = True
        self._keyboard_listener.start()
        log("Hold-to-talk: right option (нажал -> запись, отпустил -> вставка)")

    def _on_destroy(self, event):
        if event.widget is self.root:
            self._close_logs_window()
            try:
                if self._keyboard_listener:
                    self._keyboard_listener.stop()
            except Exception:
                pass

    def _quit(self, _event=None):
        self.recording = False
        self.processing = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as ex:
            log(f"Stream close during quit failed: {ex}")
        try:
            if self._keyboard_listener:
                self._keyboard_listener.stop()
        except Exception as ex:
            log(f"Keyboard listener stop during quit failed: {ex}")
        self._close_logs_window()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)
        return "break"

    def _is_hold_key(self, key) -> bool:
        if self._hold_key_mode != "right_option" or self._keyboard_mod is None:
            return False
        alt_right = getattr(self._keyboard_mod.Key, "alt_r", None)
        alt_any = getattr(self._keyboard_mod.Key, "alt", None)
        alt_graph = getattr(self._keyboard_mod.Key, "alt_gr", None)
        if key == alt_right or key == alt_graph or key == alt_any:
            return True
        # fallback for macOS: right option often arrives as virtual keycode 61
        return getattr(key, "vk", None) == 61

    def _on_global_key_press(self, key):
        if not self._is_hold_key(key):
            return
        if self._hold_key_down:
            return
        self._hold_key_down = True
        try:
            self.root.after(0, self._handle_hold_key_down)
        except Exception:
            pass

    def _on_global_key_release(self, key):
        if not self._is_hold_key(key):
            return
        if not self._hold_key_down:
            return
        self._hold_key_down = False
        try:
            self.root.after(0, self._handle_hold_key_up)
        except Exception:
            pass

    def _handle_hold_key_down(self):
        if not self.ready or self.processing or self.recording:
            self._hold_started_recording = False
            return
        self._hold_started_recording = True
        self._start_rec()

    def _handle_hold_key_up(self):
        if self._hold_started_recording and self.recording:
            self._stop_rec()
        self._hold_started_recording = False

    # ── Окно без рамки ──────────────────────────────────────────
    def _make_borderless(self):
        """
        Tk 9 на macOS игнорирует overrideredirect: у окна остаётся системная
        шапка «tk» со светофором (+28pt по высоте). Снимаем её явным
        -stylemask {}.

        Прозрачный фон средствами Tk 9 не получить: окно рисуется в битмап
        без альфа-канала, systemTransparent выходит чёрным. Поэтому форму
        пилюли даёт маска слоя окна со скруглением H/2, а -transparent
        убирает непрозрачный фон окна, чтобы срезанные углы стали прозрачными.
        """
        try:
            self.root.attributes("-stylemask", "")
        except tk.TclError:
            pass  # Tk 8.6: -stylemask нет, там overrideredirect работает сам
        try:
            self.root.attributes("-transparent", True)
        except tk.TclError as ex:
            log(f"Прозрачный фон окна недоступен: {ex}")
            return
        try:
            from AppKit import NSApplication
            # Кроме нашего окна у Tk есть служебное скрытое TKWindow с рамкой —
            # наше узнаём по снятой рамке и прозрачности.
            for win in NSApplication.sharedApplication().windows():
                if (win.className() != "TKWindow" or win.styleMask() != 0
                        or win.isOpaque()):
                    continue
                view = win.contentView()
                view.setWantsLayer_(True)
                view.layer().setCornerRadius_(H / 2)
                view.layer().setMasksToBounds_(True)
                return
            log("Скругление окна: окно пилюли не найдено")
        except Exception as ex:  # noqa: BLE001
            log(f"Скругление окна недоступно: {ex}")

    def _open_privacy_panel(self, key: str):
        try:
            subprocess.run(
                ["open", f"x-apple.systempreferences:com.apple.preference.security?Privacy_{key}"],
                check=False,
            )
        except Exception as ex:
            log(f"Не удалось открыть настройки Privacy_{key}: {ex}")

    # ── Рисование кнопок и состояния ────────────────────────────
    def _apply_state_style(self):
        """Цвет баров и обводки: серый — покой, красный — запись, жёлтый — обработка."""
        if self.recording:
            bar, border = C_REC, C_BORDER_REC
        elif self.processing:
            bar, border = C_PROC, C_BORDER_PROC
        else:
            bar, border = C_IDLE, C_BORDER
        if bar != self._bar_color:
            self.cv.itemconfig("bar", fill=bar)
            self._bar_color = bar
        self.cv.itemconfig(self._pill_border, fill=border)

    def _show_controls(self, _event=None):
        self.cv.itemconfig("ctl", state="normal")

    def _hide_controls(self, _event=None):
        self.cv.itemconfig("ctl", state="hidden")
        self.cv.itemconfig(self._logs_bg, fill=C_BTN_BG)
        self.cv.itemconfig(self._close_bg, fill=C_BTN_BG)

    def _draw_close(self):
        """Круглая кнопка выхода с крестиком (правый край)."""
        cx, cy_c = CLOSE_CX, H / 2
        hit = BTN_R + 2
        self._close_bounds = (cx - hit, cy_c - hit, cx + hit, cy_c + hit)
        self._close_bg = self.cv.create_oval(
            cx - BTN_R, cy_c - BTN_R, cx + BTN_R, cy_c + BTN_R,
            fill=C_BTN_BG, outline="", tags=("close", "ctl")
        )
        d = 3   # целые координаты: Tk округляет их до точек, дробные дают перекос
        for x1, y1, x2, y2 in ((cx - d, cy_c - d, cx + d, cy_c + d),
                               (cx - d, cy_c + d, cx + d, cy_c - d)):
            self.cv.create_line(x1, y1, x2, y2, fill=C_BTN_FG, width=1.4,
                                capstyle="round", tags=("close", "ctl"))

    def _draw_logs_button(self):
        """Круглая кнопка логов со значком «список» (левый край)."""
        cx, cy_c = LOGS_CX, H / 2
        hit = BTN_R + 2
        self._logs_bounds = (cx - hit, cy_c - hit, cx + hit, cy_c + hit)
        self._logs_bg = self.cv.create_oval(
            cx - BTN_R, cy_c - BTN_R, cx + BTN_R, cy_c + BTN_R,
            fill=C_BTN_BG, outline="", tags=("logs", "ctl")
        )
        for dy in (-3, 0, 3):
            self.cv.create_line(cx - 4, cy_c + dy, cx + 4, cy_c + dy,
                                fill=C_BTN_FG, width=1.3, capstyle="round",
                                tags=("logs", "ctl"))

    # ── Логи (UI) ───────────────────────────────────────────────
    def _toggle_logs_window(self, _event=None):
        self._suppress_next_toggle = True
        if self._logs_win and self._logs_win.winfo_exists():
            self._close_logs_window()
        else:
            self._open_logs_window()
        return "break"

    def _flat_button(self, parent, text: str, command):
        """Плоская тёмная кнопка: tk.Button на macOS игнорирует цвета фона."""
        btn = tk.Label(
            parent, text=text, bg=LOG_BTN, fg=LOG_FG,
            font=(UI_FONT, 12), padx=12, pady=5, cursor="hand2",
        )
        btn.bind("<Enter>", lambda _e: btn.config(bg=LOG_BTN_HV))
        btn.bind("<Leave>", lambda _e: btn.config(bg=LOG_BTN))
        btn.bind("<ButtonRelease-1>", lambda _e: command())
        return btn

    def _open_logs_window(self):
        win = tk.Toplevel(self.root)
        self._logs_win = win
        win.title("WhisperMac — диктовки")
        win.geometry("820x500")
        win.minsize(640, 360)
        win.configure(bg=LOG_BG)
        win.protocol("WM_DELETE_WINDOW", self._close_logs_window)

        # Шапка: заголовок, счётчик, кнопки
        header = tk.Frame(win, bg=LOG_BG)
        header.pack(fill="x", padx=16, pady=(14, 10))
        tk.Label(header, text="Диктовки", bg=LOG_BG, fg=LOG_FG,
                 font=(UI_FONT, 17, "bold")).pack(side="left")
        self._logs_count = tk.Label(header, text="", bg=LOG_BG, fg=LOG_DIM,
                                    font=(UI_FONT, 12))
        self._logs_count.pack(side="left", padx=(10, 0), pady=(4, 0))
        self._flat_button(header, "Диагностика", self._open_runtime_log_file).pack(side="right")
        self._flat_button(header, "Открыть файл", self._open_logs_file).pack(side="right", padx=(0, 6))
        self._copy_btn = self._flat_button(header, "Копировать", self._copy_selected_log)
        self._copy_btn.pack(side="right", padx=(0, 6))
        self._flat_button(header, "Обновить", self._refresh_logs).pack(side="right", padx=(0, 6))

        body = tk.Frame(win, bg=LOG_BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Слева — список диктовок
        left = tk.Frame(body, bg=LOG_PANEL, padx=6, pady=6)
        left.pack(side="left", fill="y")
        self._logs_list = tk.Listbox(
            left,
            width=38,
            bg=LOG_PANEL,
            fg=LOG_FG,
            font=(UI_FONT, 12),
            selectbackground=LOG_SEL,
            selectforeground="#FFFFFF",
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
        )
        list_scroll = tk.Scrollbar(left, orient="vertical", command=self._logs_list.yview)
        self._logs_list.config(yscrollcommand=list_scroll.set)
        self._logs_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self._logs_list.bind("<<ListboxSelect>>", self._on_log_select)

        # Справа — полный текст выбранной
        right = tk.Frame(body, bg=LOG_PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._logs_text = tk.Text(
            right,
            wrap="word",
            bg=LOG_PANEL,
            fg=LOG_FG,
            font=(UI_FONT, 14),
            insertbackground=LOG_FG,
            borderwidth=0,
            highlightthickness=0,
            padx=16,
            pady=14,
            spacing2=3,
        )
        self._logs_text.tag_configure("ts", foreground=LOG_DIM, font=(UI_FONT, 11))
        text_scroll = tk.Scrollbar(right, orient="vertical", command=self._logs_text.yview)
        self._logs_text.config(yscrollcommand=text_scroll.set)
        self._logs_text.pack(side="left", fill="both", expand=True)
        text_scroll.pack(side="right", fill="y")

        self._refresh_logs()

    def _close_logs_window(self):
        if self._logs_win and self._logs_win.winfo_exists():
            self._logs_win.destroy()
        self._logs_win = None
        self._logs_list = None
        self._logs_text = None
        self._logs_count = None
        self._copy_btn = None
        self._log_records = []

    def _log_file_path(self) -> Path:
        return Path.home() / "whisper_log.txt"

    def _runtime_log_file_path(self) -> Path:
        return Path.home() / "whisper_runtime.log"

    def _read_log_records(self) -> list:
        path = self._log_file_path()
        if not path.exists():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    ts = ""
                    text = raw
                    if raw.startswith("[") and "] " in raw:
                        ts, text = raw[1:].split("] ", 1)
                    text = text.strip()
                    if not text:
                        continue
                    records.append({"ts": ts, "text": text})
        except Exception as ex:
            log(f"Не удалось прочитать лог: {ex}")
            return []
        return list(reversed(records[-1000:]))

    def _log_preview(self, text: str, max_len: int = 72) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 1] + "…"

    def _short_ts(self, ts: str) -> str:
        """«2026-09-24 13:56:19» → «24.09 13:56»."""
        try:
            date, clock = ts.split(" ", 1)
            _, month, day = date.split("-")
            return f"{day}.{month} {clock[:5]}"
        except ValueError:
            return ts or "--"

    def _refresh_logs(self):
        if not self._logs_list:
            return
        prev_idx = self._selected_log_index()
        self._log_records = self._read_log_records()
        self._logs_list.delete(0, tk.END)
        for item in self._log_records:
            ts = self._short_ts(item["ts"])
            self._logs_list.insert(tk.END, f" {ts}   {self._log_preview(item['text'], 60)}")
        if self._logs_count:
            self._logs_count.config(text=_plural_records(len(self._log_records)))
        if self._log_records:
            idx = prev_idx if prev_idx is not None and prev_idx < len(self._log_records) else 0
            self._logs_list.selection_set(idx)
            self._logs_list.activate(idx)
            self._on_log_select()
        elif self._logs_text:
            self._logs_text.delete("1.0", tk.END)

    def _selected_log_index(self):
        if not self._logs_list:
            return None
        cur = self._logs_list.curselection()
        if not cur:
            return None
        idx = int(cur[0])
        if idx < 0 or idx >= len(self._log_records):
            return None
        return idx

    def _on_log_select(self, _event=None):
        if not self._logs_text:
            return
        idx = self._selected_log_index()
        self._logs_text.delete("1.0", tk.END)
        if idx is None:
            return
        item = self._log_records[idx]
        if item["ts"]:
            self._logs_text.insert("1.0", f"{item['ts']}\n\n", "ts")
        self._logs_text.insert(tk.END, item["text"])

    def _copy_selected_log(self):
        idx = self._selected_log_index()
        if idx is None:
            return
        item = self._log_records[idx]
        if self._copy_to_clipboard(item["text"]):
            log("Лог скопирован в буфер")
            btn = self._copy_btn
            if btn:
                btn.config(text="Скопировано")
                btn.after(1200, lambda: btn.winfo_exists() and btn.config(text="Копировать"))

    def _open_logs_file(self):
        subprocess.run(["open", str(self._log_file_path())], check=False)

    def _open_runtime_log_file(self):
        subprocess.run(["open", str(self._runtime_log_file_path())], check=False)

    # ── Drag ────────────────────────────────────────────────────
    def _point_in_bounds(self, x: int, y: int, bounds: tuple) -> bool:
        x1, y1, x2, y2 = bounds
        return x1 <= x <= x2 and y1 <= y <= y2

    def _is_control_hit(self, e) -> bool:
        return (
            self._point_in_bounds(e.x, e.y, self._close_bounds)
            or self._point_in_bounds(e.x, e.y, self._logs_bounds)
        )

    def _press(self, e):
        if self._is_control_hit(e):
            self._suppress_next_toggle = True
            return "break"
        self._drag_ox = e.x
        self._drag_oy = e.y
        self._dragging = False

    def _motion(self, e):
        if self._is_control_hit(e):
            return "break"
        if abs(e.x - self._drag_ox) + abs(e.y - self._drag_oy) > 4:
            self._dragging = True
        if self._dragging:
            self.root.geometry(
                f"+{self.root.winfo_x() + e.x - self._drag_ox}"
                f"+{self.root.winfo_y() + e.y - self._drag_oy}"
            )

    def _release(self, e):
        if self._suppress_next_toggle or self._is_control_hit(e):
            self._suppress_next_toggle = False
            self._dragging = False
            return "break"
        if not self._dragging:
            self._toggle()
        self._dragging = False

    # ── Запись ──────────────────────────────────────────────────
    def _toggle(self):
        if not self.ready or self.processing:
            return
        if not self.recording:
            self._start_rec()
        else:
            self._stop_rec()

    def _start_rec(self):
        with self._chunks_lock:
            self.chunks = []
        self._eq_levels[:] = 0
        self._eq_smooth[:] = 0
        self._rms_smooth = 0.0
        self._recording_started_at = time.perf_counter()
        current_bundle = frontmost_bundle()
        if current_bundle and not self._is_excluded_bundle(current_bundle):
            self.target = current_bundle
        self.recording = True
        self._apply_state_style()
        self._kick_anim()
        log(
            f"Конфиг: chunk={CHUNK_SEC:.1f}s, poll={WORKER_POLL_SEC:.2f}s, "
            f"final-pass={FINAL_PASS_MIN_SEC:.0f}-{FINAL_PASS_MAX_SEC:.0f}s"
        )
        log(
            f"Privacy: strict_local={'on' if STRICT_LOCAL_MODE else 'off'}, "
            f"save_transcripts={'on' if SAVE_TRANSCRIPTS else 'off'}, "
            f"save_perf={'on' if SAVE_PERF_LOG else 'off'}"
        )
        log(f"Запись... ({self.target})")
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=1024, latency="low", callback=self._audio_cb
            )
            self.stream.start()
        except Exception as ex:
            log(f"Ошибка: {ex}")
            err = str(ex).lower()
            if any(k in err for k in ("permission", "not permitted", "unauthorized", "access")):
                self._open_privacy_panel("Microphone")
            # Без этого флаг записи оставался включённым: пилюля зависала
            # красной, а следующий клик уводил её в вечную «обработку».
            self.recording = False
            if self.stream:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self._reset()
            return
        if ENGINE == "groq" and GROQ_API_KEY:
            threading.Thread(target=self._groq_worker, daemon=True).start()
        else:
            threading.Thread(target=self._streaming_worker, daemon=True).start()

    def _stop_rec(self):
        self.recording = False
        current_bundle = frontmost_bundle()
        if current_bundle and not self._is_excluded_bundle(current_bundle):
            self.target = current_bundle
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.processing = True
        self._apply_state_style()
        self._kick_anim()

    # ── Аудио-коллбэк (real-time FFT для эквалайзера) ───────────
    def _audio_cb(self, indata, frames, time_info, status):
        frame = indata.flatten()
        with self._chunks_lock:
            self.chunks.append(indata.copy())

        if len(frame) < 64:
            return

        # Дальше только визуализация: её сбой не должен ронять поток записи.
        try:
            self._update_eq_levels(frame)
        except Exception:  # noqa: BLE001
            self._eq_levels[:] = 0

    def _update_eq_levels(self, frame: np.ndarray):
        # Сначала проверяем реальную громкость
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self._rms_smooth = (
            (1.0 - EQ_RMS_ALPHA) * self._rms_smooth + EQ_RMS_ALPHA * rms
        )
        gate = EQ_RMS_THRESHOLD

        if self._rms_smooth <= gate:
            self._eq_levels[:] = 0
            return

        # FFT → полосы по лог-шкале (окно и границы считаются один раз)
        n = len(frame)
        if self._eq_window is None or len(self._eq_window) != n:
            self._eq_window = np.hanning(n).astype(np.float32)
            self._eq_band_idx = _eq_band_bins(n)
        fft_vals = np.abs(np.fft.rfft(frame * self._eq_window))

        # Высокие полосы тише низких — поднимаем их наклоном, чтобы края жили
        levels = np.array(
            [fft_vals[a:b].mean() for a, b in self._eq_band_idx],
            dtype=np.float32,
        ) * EQ_BAND_TILT

        peak = levels.max()
        if peak > 1e-6:
            # Нормализуем форму (0–1), затем масштабируем по реальной громкости
            shape = levels / peak
            denom = max(1e-6, EQ_RMS_FULL - EQ_RMS_THRESHOLD)
            amplitude = min(
                1.0,
                max(0.0, (self._rms_smooth - EQ_RMS_THRESHOLD) / denom),
            )
            # Чуть поднимаем средние уровни, чтобы анимация читалась живее.
            amplitude = amplitude ** 0.72
            # Зеркально: низкие частоты в центре, высокие к краям
            self._eq_levels[:] = (shape * EQ_ENVELOPE * amplitude)[EQ_MIRROR]
        else:
            self._eq_levels[:] = 0

    def _transcribe_audio(
        self,
        audio,
        *,
        prompt=None,
        final=False,
        condition_on_previous_text=True,
        temperature=None,
    ):
        opts = dict(
            path_or_hf_repo=MODEL_REPO,
            language=LANGUAGE,
            initial_prompt=prompt,
            condition_on_previous_text=condition_on_previous_text,
        )
        # Beam search в mlx_whisper пока не реализован.
        opts["temperature"] = (
            temperature if temperature is not None
            else (FINAL_TEMPERATURES if final else 0.0)
        )
        return mlx_whisper.transcribe(audio, **opts)

    def _take_new_audio(self, chunk_idx: int) -> tuple:
        with self._chunks_lock:
            total = len(self.chunks)
            if chunk_idx >= total:
                return chunk_idx, None
            new_chunks = self.chunks[chunk_idx:total]
            chunk_idx = total
        if not new_chunks:
            return chunk_idx, None
        new_audio = np.concatenate([c.flatten() for c in new_chunks])
        return chunk_idx, new_audio

    def _decode_piece(self, audio: np.ndarray, parts: list, label: str) -> tuple:
        prompt = _prompt_from_parts(parts)
        started = time.perf_counter()
        result = self._transcribe_audio(audio, prompt=prompt, final=False)
        elapsed = time.perf_counter() - started
        text = result.get("text", "").strip()
        avg_logprob, avg_no_speech = _segment_quality(result)
        if _likely_silence_hallucination(text, avg_no_speech):
            log(
                f"[{label}] пропуск (тишина): no_speech={avg_no_speech:.2f}, "
                f"text='{text[:24]}'"
            )
            return "", elapsed, avg_logprob, avg_no_speech
        if text:
            log(f"[{label}] {text}")
        return text, elapsed, avg_logprob, avg_no_speech

    # ── Groq воркер (основной путь) ─────────────────────────────
    def _groq_worker(self):
        """
        Быстрый облачный путь: во время записи только копим аудио,
        на стопе одним запросом отправляем всё в Groq и мгновенно вставляем.
        При ошибке/пустом ответе — фоллбэк на локальную модель.
        """
        while self.recording:
            time.sleep(WORKER_POLL_SEC)

        with self._chunks_lock:
            all_audio = (
                np.concatenate([c.flatten() for c in self.chunks])
                if self.chunks else np.array([], dtype=np.float32)
            )

        audio_sec = len(all_audio) / SAMPLE_RATE
        amp = float(np.max(np.abs(all_audio))) if len(all_audio) else 0.0
        if audio_sec < MIN_DURATION or amp <= 0.001:
            log("[groq] слишком короткая/тихая запись — пропуск")
            self.root.after(0, self._reset)
            return

        full = groq_transcribe(all_audio, prompt=HOTWORDS_PROMPT)

        if not full:
            log("[groq] пустой результат — фоллбэк на локальную модель")
            full = self._local_full_transcribe(all_audio)

        if full and _is_repetition_loop(full):
            collapsed = _collapse_repetition_loop(full).strip()
            if collapsed and collapsed != full:
                log("[post] схлопнул повторяющийся loop-текст")
                full = collapsed

        log(f"→ {full}")
        if full:
            self._save(full)
            self.root.after(0, lambda t=full: self._paste_and_reset(t))
        else:
            self.root.after(0, self._reset)

    def _local_full_transcribe(self, all_audio: np.ndarray) -> str:
        """Локальный фоллбэк: единый проход mlx-whisper по всей записи."""
        if not len(all_audio):
            return ""
        try:
            res = self._transcribe_audio(
                all_audio,
                prompt=HOTWORDS_PROMPT,
                final=True,
                condition_on_previous_text=False,
            )
            text = res.get("text", "").strip()
            if text and _is_repetition_loop(text):
                safe = self._transcribe_audio(
                    all_audio, prompt=None, final=False,
                    condition_on_previous_text=False, temperature=0.0,
                )
                safe_text = safe.get("text", "").strip()
                if safe_text and not _is_repetition_loop(safe_text):
                    text = safe_text
            return text
        except Exception as ex:  # noqa: BLE001
            log(f"[local-fallback] ошибка: {ex}")
            return ""

    # ── Streaming воркер ────────────────────────────────────────
    def _streaming_worker(self):
        """
        Эффективный воркер для длинных записей.
        Хранит pending-буфер, потребляет только НОВЫЕ чанки —
        не конкатенирует весь массив каждую итерацию.
        """
        CHUNK      = int(CHUNK_SEC * SAMPLE_RATE)
        parts      = []
        pending    = np.array([], dtype=np.float32)   # необработанный буфер
        chunk_idx  = 0                                 # сколько чанков уже взяли
        decode_time_sec = 0.0
        processed_audio_sec = 0.0
        low_conf_chunks = 0
        decoded_chunks = 0

        while self.recording:
            time.sleep(WORKER_POLL_SEC)

            # Берём только новые чанки с момента последней итерации
            chunk_idx, new_audio = self._take_new_audio(chunk_idx)
            if new_audio is None:
                continue
            pending   = np.concatenate([pending, new_audio]) if len(pending) else new_audio

            # Обрабатываем все полные чанки из буфера
            # (если модель отстала — догоняем в цикле)
            while len(pending) >= CHUNK:
                segment = pending[:CHUNK]
                pending = pending[CHUNK:]

                text, elapsed, avg_logprob, _ = self._decode_piece(
                    segment, parts, "chunk"
                )
                decode_time_sec += elapsed
                processed_audio_sec += len(segment) / SAMPLE_RATE
                decoded_chunks += 1
                if avg_logprob <= LOW_CONF_LOGPROB:
                    low_conf_chunks += 1
                if text:
                    parts.append(text)

        # Запись остановлена — добираем остаток
        chunk_idx, new_audio = self._take_new_audio(chunk_idx)
        if new_audio is not None:
            pending   = np.concatenate([pending, new_audio]) if len(pending) else new_audio

        # Если во время записи модель отстала, догоняем backlog кусками.
        while len(pending) >= CHUNK:
            segment = pending[:CHUNK]
            pending = pending[CHUNK:]
            text, elapsed, avg_logprob, _ = self._decode_piece(segment, parts, "flush")
            decode_time_sec += elapsed
            processed_audio_sec += len(segment) / SAMPLE_RATE
            decoded_chunks += 1
            if avg_logprob <= LOW_CONF_LOGPROB:
                low_conf_chunks += 1
            if text:
                parts.append(text)

        amp = float(np.max(np.abs(pending))) if len(pending) else 0
        if len(pending) / SAMPLE_RATE >= MIN_DURATION and amp > 0.001:
            text, elapsed, avg_logprob, _ = self._decode_piece(pending, parts, "tail")
            decode_time_sec += elapsed
            processed_audio_sec += len(pending) / SAMPLE_RATE
            decoded_chunks += 1
            if avg_logprob <= LOW_CONF_LOGPROB:
                low_conf_chunks += 1
            if text:
                parts.append(text)

        chunk_full = _join_chunks(parts)
        full = chunk_full

        # Финальный quality-pass по всей записи: выше точность на длинных фразах.
        with self._chunks_lock:
            all_audio = (
                np.concatenate([c.flatten() for c in self.chunks])
                if self.chunks else np.array([], dtype=np.float32)
            )
        if len(all_audio):
            audio_sec = len(all_audio) / SAMPLE_RATE
            low_conf_ratio = (
                (low_conf_chunks / decoded_chunks)
                if decoded_chunks else 0.0
            )
            need_final_pass = (
                FINAL_PASS_MIN_SEC <= audio_sec <= FINAL_PASS_MAX_SEC
                and (
                    _is_repetition_loop(chunk_full)
                    or not chunk_full
                    or low_conf_ratio >= 0.35
                )
            )
            if need_final_pass and audio_sec >= MIN_DURATION:
                try:
                    final_res = self._transcribe_audio(
                        all_audio,
                        prompt=HOTWORDS_PROMPT,
                        final=True,
                        # Этот режим в Whisper меньше зацикливается на повторах.
                        condition_on_previous_text=False,
                    )
                    final_text = final_res.get("text", "").strip()
                    if final_text:
                        if _is_repetition_loop(final_text):
                            log("[final] обнаружен loop-повтор, пробую safe-pass")
                            safe_res = self._transcribe_audio(
                                all_audio,
                                prompt=None,
                                final=False,
                                condition_on_previous_text=False,
                                temperature=0.0,
                            )
                            safe_text = safe_res.get("text", "").strip()
                            if safe_text and not _is_repetition_loop(safe_text):
                                log(f"[final-safe] {safe_text}")
                                full = safe_text
                            else:
                                log("[final] loop остался, fallback на chunk-текст")
                                full = chunk_full
                        else:
                            log(f"[final] {final_text}")
                            full = final_text
                except Exception as ex:
                    log(f"[final] fallback на чанки: {ex}")
            elif audio_sec > FINAL_PASS_MAX_SEC:
                skip_line = (
                    f"[final] пропуск полного pass: запись {audio_sec:.1f}s > "
                    f"{FINAL_PASS_MAX_SEC:.0f}s"
                )
                log(skip_line)
                self._save_perf(skip_line)

        if full and _is_repetition_loop(full):
            collapsed = _collapse_repetition_loop(full).strip()
            if collapsed and collapsed != full:
                log("[post] схлопнул повторяющийся loop-текст")
                full = collapsed

        record_wall_sec = 0.0
        if self._recording_started_at is not None:
            record_wall_sec = max(0.0, time.perf_counter() - self._recording_started_at)
        if processed_audio_sec > 0:
            rtf = decode_time_sec / processed_audio_sec
            perf_line = (
                f"[perf] обработано {processed_audio_sec:.1f}s аудио за "
                f"{decode_time_sec:.2f}s (RTF={rtf:.2f}x), запись шла {record_wall_sec:.1f}s"
            )
            log(perf_line)
            self._save_perf(perf_line)

        log(f"→ {full}")
        if full:
            self._save(full)
            self.root.after(0, lambda t=full: self._paste_and_reset(t))
        else:
            self.root.after(0, self._reset)

    # ── Вспомогательные ─────────────────────────────────────────
    def _save(self, text):
        if not SAVE_TRANSCRIPTS:
            return
        from datetime import datetime
        from pathlib import Path
        with open(Path.home() / "whisper_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")
        if self._logs_win and self._logs_win.winfo_exists():
            self.root.after(0, self._refresh_logs)

    def _save_perf(self, text):
        if not SAVE_PERF_LOG:
            return
        from datetime import datetime
        from pathlib import Path
        with open(Path.home() / "whisper_perf.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")

    def _send_paste_shortcut(self, target: str = "", text: str = "") -> bool:
        mode = PASTE_SHORTCUT_MODE
        modes = {
            "ax": (("ax", lambda: ax_insert_text(text, target)),),
            "keycode": (("osascript-keycode", lambda: cmd_v_osascript_keycode(target)),),
            "osascript-keycode": (("osascript-keycode", lambda: cmd_v_osascript_keycode(target)),),
            "osascript": (("osascript", lambda: cmd_v_osascript(target)),),
            "pynput": (("pynput", cmd_v_pynput),),
            "session": (("session", lambda: self._send_cgevent(kCGSessionEventTap)),),
            "cgevent": (("cgevent", lambda: self._send_cgevent(kCGHIDEventTap)),),
            "auto": (
                ("ax", lambda: ax_insert_text(text, target)),
                ("osascript-keycode", lambda: cmd_v_osascript_keycode(target)),
                ("osascript", lambda: cmd_v_osascript(target)),
                ("pynput", cmd_v_pynput),
                ("session", lambda: self._send_cgevent(kCGSessionEventTap)),
                ("cgevent", lambda: self._send_cgevent(kCGHIDEventTap)),
            ),
        }
        for name, sender in modes.get(mode, modes["auto"]):
            if name not in {"osascript", "osascript-keycode", "ax"} and not request_accessibility_permission(prompt=False):
                log("Accessibility permission missing for simulated paste")
                return False
            if sender():
                log(f"Paste sent via {name}")
                return True
        return False

    def _send_cgevent(self, tap: int) -> bool:
        try:
            cmd_v(tap)
            return True
        except Exception as ex:
            log(f"CGEvent Cmd+V failed: {ex}")
            return False

    def _temporarily_move_self_away(self) -> bool:
        # Лог-архив во время вставки НЕ трогаем. Раньше он сворачивался, а затем
        # разворачивался обратно — из-за чего всплывал поверх после каждой вставки.
        # Оставляем его как есть (в фоне): фокус получает целевое приложение.
        return False

    def _restore_after_paste(self, logs_hidden: bool):
        def restore():
            try:
                self.root.attributes("-topmost", True)
                self.root.attributes("-alpha", WIN_ALPHA)
                self.root.lift()
            except Exception:
                pass
            try:
                if logs_hidden and self._logs_win and self._logs_win.winfo_exists():
                    self._logs_win.deiconify()
            except Exception:
                pass
        self.root.after(350, restore)

    def _copy_to_clipboard(self, text: str) -> bool:
        env = os.environ.copy()
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_CTYPE", "C.UTF-8")

        # Основной путь для macOS GUI-приложений: нативный pasteboard (Unicode).
        try:
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            if pb.setString_forType_(text, NSPasteboardTypeString):
                return True
        except Exception as ex:
            log(f"NSPasteboard failed, fallback to pbcopy: {ex}")

        try:
            subprocess.run(
                ["/usr/bin/pbcopy"],
                input=text.encode("utf-8"),
                check=True,
                env=env,
            )
            pasted = subprocess.check_output(
                ["/usr/bin/pbpaste"],
                env=env,
                text=True,
            )
            if pasted != text:
                raise RuntimeError("pbcopy roundtrip mismatch")
            return True
        except Exception as ex:
            log(f"pbcopy failed, fallback to Tk clipboard: {ex}")
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update_idletasks()
                return True
            except Exception as ex2:
                log(f"clipboard fallback failed: {ex2}")
                return False

    def _paste_and_reset(self, text):
        if not self._copy_to_clipboard(text):
            self._reset()
            return
        target = self.target or ""
        if self._is_excluded_bundle(target):
            target = ""
        front = frontmost_bundle() or ""
        if not target and front and not self._is_excluded_bundle(front):
            target = front

        if not target and self._is_excluded_bundle(front):
            log("Paste skipped: no external target; text is in clipboard")
            self._reset()
            return

        logs_hidden = self._temporarily_move_self_away()
        if target:
            if activate_bundle(target):
                front = frontmost_bundle() or ""
            else:
                log(f"Target activate failed: {target}")
        time.sleep(0.08)
        front = frontmost_bundle() or ""
        paste_target = target
        if front and not self._is_excluded_bundle(front):
            if not paste_target or front != paste_target:
                if paste_target:
                    log(f"Paste target changed: saved={paste_target} actual={front}")
                paste_target = front
        if not paste_target and self._is_excluded_bundle(front):
            log("Paste skipped after focus restore: no external target; text is in clipboard")
            self._restore_after_paste(logs_hidden)
            self._reset()
            return
        log(
            f"Paste target={paste_target or '-'} frontmost={front or '-'} "
            f"mode={PASTE_SHORTCUT_MODE or 'auto'}"
        )
        if not self._send_paste_shortcut(paste_target, text):
            self._open_privacy_panel("Accessibility")
        self._restore_after_paste(logs_hidden)
        self._reset()

    def _reset(self):
        self.processing  = False
        self._eq_levels[:] = 0
        self._apply_state_style()
        self._kick_anim()

    # ── Анимация ────────────────────────────────────────────────
    def _kick_anim(self):
        """Запускает цикл анимации, если он стоит. В покое цикл не крутится."""
        if not self._anim_running:
            self._anim_running = True
            self.root.after(0, self._tick)

    def _tick(self):
        self._frame += 1
        t  = self._frame * 0.12
        cy = H / 2

        if self.recording:
            # Attack/decay smoothing для ощущения VU-метра
            target = np.power(np.clip(self._eq_levels, 0.0, 1.0), EQ_VISUAL_GAMMA)
            rising = target > self._eq_smooth
            self._eq_smooth = np.where(
                rising,
                self._eq_smooth + (target - self._eq_smooth) * EQ_ATTACK,
                self._eq_smooth * (1.0 - EQ_DECAY)
            )
            energy = float(np.mean(target))
            for i, (b, bx) in enumerate(self.bars):
                wobble = 0.0
                if energy > 0.03:
                    wobble = (
                        0.5 + 0.5 * math.sin(t * (5.2 + i * 0.15) + i * 0.9)
                    ) * EQ_WOBBLE_MAX * energy
                level = min(1.0, self._eq_smooth[i] + wobble)
                h = BAR_MIN + level * BAR_MAX
                self.cv.coords(b, bx, cy - h, bx, cy + h)

        elif self.processing:
            # Волна расходится от центра к краям
            mid = (BAR_COUNT - 1) / 2
            for i, (b, bx) in enumerate(self.bars):
                h = BAR_MIN + abs(math.sin(t * 4.0 - abs(i - mid) * 0.7)) * BAR_MAX * 0.55
                self.cv.coords(b, bx, cy - h, bx, cy + h)

        elif self.ready:
            # Покой: бары-точки рисуем один раз и останавливаем цикл.
            # Раньше он перерисовывал неподвижные бары 30 раз в секунду.
            for b, bx in self.bars:
                self.cv.coords(b, bx, cy - BAR_MIN, bx, cy + BAR_MIN)
            self._anim_running = False
            return

        if not self.ready:
            self.cv.itemconfig(self.spinner, start=(90 - self._frame * 12) % 360)

        self.root.after(33, self._tick)

    def _track_app(self):
        try:
            if not self.recording and not self.processing:
                b = frontmost_bundle()
                if b and not self._is_excluded_bundle(b):
                    self.target = b
        except Exception:
            pass
        self.root.after(300, self._track_app)

    def _load_model(self):
        if ENGINE == "groq" and GROQ_API_KEY:
            log(f"Движок: Groq ({GROQ_MODEL}), локальная модель — фоллбэк")
            log("Готово")
            self.root.after(0, self._on_ready)
            return
        if ENGINE == "groq" and not GROQ_API_KEY:
            log("[groq] ключ не найден — работаю только на локальной модели")
        log("Загружаю модель...")
        log(f"Model: {MODEL_REPO}")
        if STRICT_LOCAL_MODE:
            log("Strict local mode: offline-only")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._transcribe_audio(dummy, prompt=HOTWORDS_PROMPT, final=False)
        log("Готово")
        self.root.after(0, self._on_ready)

    def _on_ready(self):
        self.ready = True
        self.cv.itemconfig(self.spinner, state="hidden")
        self.cv.itemconfig("bar", state="normal")
        self._apply_state_style()
        self._kick_anim()

    def run(self):
        log("WhisperMac запущен")
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSApplicationActivationPolicyRegular,
        )

        dock_mode = os.getenv("WHISPERMAC_DOCK_MODE", "regular").strip().lower()
        app = NSApplication.sharedApplication()
        if dock_mode == "accessory":
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            log("Dock mode: accessory")
        else:
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            log("Dock mode: regular")

        self.root.mainloop()


if __name__ == "__main__":
    _instance_lock = acquire_instance_lock()
    if _instance_lock is None:
        log("WhisperMac уже запущен, второй экземпляр остановлен")
        sys.exit(0)
    atexit.register(_instance_lock.close)
    App().run()
