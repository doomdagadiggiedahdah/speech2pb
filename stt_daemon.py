#!/usr/bin/env python3
"""
stt_daemon — lightweight daemon that manages GPU model loading and transcription.

The daemon itself uses no GPU memory. When a client sends "preload", it loads the
model into GPU. When "transcribe" arrives, it uses the (now-loaded) model. If the
model is already loading from a prior request, new preload calls are no-ops.
After an idle timeout the model is unloaded to free VRAM.

Requests are serialized: if a transcription is in progress and another arrives,
it queues behind the first.

Usage:
    ./stt_daemon.py              # foreground (for testing)
    ./stt_daemon.py --daemon     # daemonize (detach from terminal)

Client protocol (one JSON line each way):
    → {"cmd": "preload"}
    ← {"status": "loading"}  or  {"status": "already_loaded"}

    → {"cmd": "transcribe", "audio_path": "/path/to/file.wav"}
    ← {"text": "...", "elapsed": 1.23}
    or
    ← {"error": "description"}

    → {"cmd": "status"}
    ← {"status": "idle|loading|ready", "pid": 123}

    → {"cmd": "unload"}
    ← {"status": "unloaded"}
"""
import gc, json, os, socketserver, sys, threading, time
from datetime import datetime
from pathlib import Path

SPOT        = Path(__file__).parent
SOCKET_PATH = SPOT / ".output" / "stt.sock"
LOG_FILE    = SPOT / ".output" / "stt_daemon.log"
PID_FILE    = SPOT / ".output" / "stt_daemon.pid"

FW_MODEL   = os.environ.get("FW_MODEL", "distil-large-v3")
FW_DEVICE  = os.environ.get("FW_DEVICE", "cuda")
FW_COMPUTE = os.environ.get("FW_COMPUTE", "int8_float16")

IDLE_TIMEOUT = float(os.environ.get("STT_IDLE_TIMEOUT", "120"))  # seconds before auto-unload

# ── logging ──────────────────────────────────────────────────────────────────
_daemonized = False

def _log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if not _daemonized:
        print(line, flush=True)

# ── model lifecycle ──────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()       # serializes transcription
_model_ready = threading.Event()     # set once model is loaded
_model_state = "idle"                # idle | loading | ready
_state_lock = threading.Lock()
_last_use = 0.0                      # monotonic timestamp of last transcription
_idle_timer = None
_load_gen = 0                        # bumped on timeout to invalidate hung loads

def _set_state(s):
    global _model_state
    with _state_lock:
        _model_state = s

def _get_state():
    with _state_lock:
        return _model_state

def _do_load(gen):
    global _model
    _set_state("loading")
    _model_ready.clear()
    t0 = time.monotonic()
    try:
        # wake GPU from power-save before ctranslate2 touches it
        import torch
        torch.cuda.synchronize()
        from faster_whisper import WhisperModel
        m = WhisperModel(FW_MODEL, device=FW_DEVICE, compute_type=FW_COMPUTE)
        elapsed = time.monotonic() - t0
        if _load_gen != gen:
            _log(f"model load finished but gen {gen} is stale (now {_load_gen}), discarding")
            del m
            return
        _model = m
        _set_state("ready")
        _log(f"model loaded ({FW_MODEL}/{FW_DEVICE}/{FW_COMPUTE}) in {elapsed:.2f}s")
    except Exception as e:
        if _load_gen != gen:
            return
        _set_state("idle")
        _log(f"model load error: {e}")
    _model_ready.set()

def _abandon_load():
    """Invalidate a hung load so the next request retries fresh."""
    global _load_gen
    _load_gen += 1
    _set_state("idle")
    _model_ready.set()  # unblock any other waiters
    _log(f"abandoned hung load, bumped gen to {_load_gen}")

def start_preload():
    """Kick off model loading in a background thread if not already loaded/loading."""
    global _last_use, _idle_timer
    _last_use = time.monotonic()  # prevent idle timer from firing during load
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None
    state = _get_state()
    if state in ("loading", "ready"):
        return state
    gen = _load_gen
    threading.Thread(target=_do_load, args=(gen,), daemon=True).start()
    return "loading"

def _do_unload():
    """Unload model. Caller must hold _model_lock."""
    global _model
    _model = None
    _model_ready.clear()
    _set_state("idle")
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    _log("model unloaded")

def unload_model():
    with _model_lock:
        _do_unload()

def _schedule_idle_unload():
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
    if IDLE_TIMEOUT <= 0:
        return
    def _check():
        if _get_state() != "ready":
            return
        if (time.monotonic() - _last_use) < IDLE_TIMEOUT:
            return
        # try-acquire: if a transcription holds the lock, skip unload
        if not _model_lock.acquire(blocking=False):
            _log("idle timer fired but transcription in progress, skipping unload")
            _schedule_idle_unload()  # reschedule
            return
        try:
            # re-check after acquiring lock
            if _get_state() == "ready" and (time.monotonic() - _last_use) >= IDLE_TIMEOUT:
                _log(f"idle for {IDLE_TIMEOUT}s, unloading model")
                _do_unload()
        finally:
            _model_lock.release()
    _idle_timer = threading.Timer(IDLE_TIMEOUT, _check)
    _idle_timer.daemon = True
    _idle_timer.start()

def transcribe(audio_path: str) -> str:
    import local_transcribe as lt
    return "".join(lt.transcribe_stream(audio_path, model=_model)).strip()

# ── socket server ────────────────────────────────────────────────────────────
class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        global _last_use
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            self._respond(error=f"bad json: {e}")
            return

        cmd = req.get("cmd", "")

        if cmd == "status":
            self._respond(status=_get_state(), pid=os.getpid())

        elif cmd == "preload":
            result = start_preload()
            status = "already_loaded" if result == "ready" else result
            self._respond(status=status)

        elif cmd == "unload":
            unload_model()
            self._respond(status="unloaded")

        elif cmd == "transcribe":
            audio_path = req.get("audio_path")
            if not audio_path or not os.path.isfile(audio_path):
                self._respond(error=f"file not found: {audio_path}")
                return

            # ensure model is being loaded
            start_preload()
            _log(f"request queued: {audio_path}")

            with _model_lock:
                # wait for model — if it was unloaded while queued, reload
                if not _model_ready.wait(timeout=60):
                    _log("model load timed out after 60s — restarting daemon")
                    self._respond(error="model load timed out")
                    os._exit(1)  # systemd restarts us clean
                if _get_state() != "ready":
                    _log("model not ready after wait, reloading...")
                    _do_load(_load_gen)
                if _get_state() != "ready":
                    self._respond(error="model failed to load")
                    return
                _log(f"transcribing: {audio_path}")
                t0 = time.monotonic()
                try:
                    text = transcribe(audio_path)
                    elapsed = time.monotonic() - t0
                    _last_use = time.monotonic()
                    _schedule_idle_unload()
                    _log(f"done in {elapsed:.2f}s: {text[:60]}...")
                    self._respond(text=text, elapsed=round(elapsed, 3))
                except Exception as e:
                    _log(f"transcribe error: {e}")
                    self._respond(error=str(e))
        else:
            self._respond(error=f"unknown cmd: {cmd}")

    def _respond(self, **kwargs):
        self.wfile.write(json.dumps(kwargs).encode() + b"\n")
        self.wfile.flush()

class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

# ── lifecycle ────────────────────────────────────────────────────────────────
def cleanup_socket():
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

def write_pid():
    PID_FILE.write_text(str(os.getpid()))

def daemonize():
    """Double-fork to detach from terminal."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = sys.stdout
    sys.stdin = open(os.devnull)

def main():
    SPOT_OUTPUT = SPOT / ".output"
    SPOT_OUTPUT.mkdir(exist_ok=True)

    global _daemonized
    if "--daemon" in sys.argv:
        _daemonized = True
        daemonize()

    cleanup_socket()
    write_pid()

    # No model loading here — daemon starts lightweight
    server = ThreadedUnixServer(str(SOCKET_PATH), RequestHandler)
    _log(f"listening on {SOCKET_PATH} (pid {os.getpid()}) — model will load on first preload request")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("shutting down")
    finally:
        server.shutdown()
        cleanup_socket()
        PID_FILE.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
