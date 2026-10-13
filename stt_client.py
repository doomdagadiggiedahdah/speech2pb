#!/usr/bin/env python3
"""
stt_client — send commands to gpu-queued (../gpu-queued), the GPU owner.
Hotkey jobs go in at "interactive" priority, ahead of batch work.

Usage:
    ./stt_client.py preload                # tell daemon to start loading model
    ./stt_client.py transcribe audio.wav   # transcribe (waits for model + queue)
    ./stt_client.py status                 # check daemon state
    ./stt_client.py unload                 # free GPU memory
"""
import contextlib, json, os, socket, sys, wave
from pathlib import Path

SOCKET_PATH = Path(os.environ.get("GPUQ_SOCKET", f"/run/user/{os.getuid()}/gpu-queued.sock"))
TIMEOUT = 120  # floor, for short clips and non-transcribe commands

# Transcription time scales with audio length (~0.3-0.4x RTF observed), so a fixed
# ceiling silently truncates long recordings: the daemon finishes, finds the client
# gone, and throws the transcript away. Budget 2x realtime plus slack for queueing.
RTF_BUDGET = 2.0
QUEUE_SLACK = 60.0


def _send(req: dict, timeout: float = TIMEOUT) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(SOCKET_PATH))
    try:
        sock.sendall(json.dumps(req).encode() + b"\n")
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return json.loads(data.strip())
    finally:
        sock.close()


def send_preload() -> dict:
    return _send({"cmd": "preload"}, timeout=5)


def _audio_secs(audio_path: str) -> float:
    """Duration of a wav, or 0.0 if it can't be read (caller falls back to TIMEOUT)."""
    with contextlib.suppress(Exception):
        with wave.open(str(audio_path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    return 0.0


def _transcribe_timeout(audio_path: str) -> float:
    return max(TIMEOUT, _audio_secs(audio_path) * RTF_BUDGET + QUEUE_SLACK)


# Decoding settings that local_transcribe.py used when stt_daemon ran it; gpu-queued
# is model-agnostic, so the client now sends them with each job.
TRANSCRIBE_OPTIONS = {
    "beam_size": int(os.environ.get("BEAM_SIZE", "3")),
    "temperature": 0.0,
    "vad_filter": os.environ.get("DISABLE_VAD", "0") != "1",
    "vad_parameters": {
        "min_silence_duration_ms": int(os.environ.get("VAD_MIN_SILENCE_MS", "150")),
        "speech_pad_ms": int(os.environ.get("VAD_SPEECH_PAD_MS", "400")),
    },
}


def send_transcribe(audio_path: str) -> dict:
    path = str(Path(audio_path).resolve())
    return _send({"cmd": "transcribe", "audio_path": path, "priority": "interactive",
                  "options": TRANSCRIBE_OPTIONS},
                 timeout=_transcribe_timeout(path))


def send_status() -> dict:
    return _send({"cmd": "status"}, timeout=5)


def send_unload() -> dict:
    return _send({"cmd": "unload"}, timeout=10)


def is_running() -> bool:
    if not SOCKET_PATH.exists():
        return False
    try:
        send_status()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def main():
    if len(sys.argv) < 2:
        print("usage: stt_client.py <preload|transcribe|status|unload> [audio_file]", file=sys.stderr)
        sys.exit(2)

    cmd = sys.argv[1]

    try:
        if cmd == "status":
            resp = send_status()
            pid = resp.get("pid", "?")
            print(f"daemon running (pid {pid}), model: {resp.get('status', '?')}")

        elif cmd == "preload":
            resp = send_preload()
            print(f"preload: {resp.get('status', resp)}")

        elif cmd == "unload":
            resp = send_unload()
            print(f"unload: {resp.get('status', resp)}")

        elif cmd == "transcribe":
            if len(sys.argv) < 3:
                print("usage: stt_client.py transcribe <audio_file>", file=sys.stderr)
                sys.exit(2)
            audio_path = sys.argv[2]
            if not Path(audio_path).exists():
                print(f"file not found: {audio_path}", file=sys.stderr)
                sys.exit(1)
            resp = send_transcribe(audio_path)
            if "error" in resp:
                print(f"error: {resp['error']}", file=sys.stderr)
                sys.exit(1)
            print(resp["text"])

        else:
            # backwards compat: bare audio path → transcribe
            if Path(cmd).exists():
                resp = send_transcribe(cmd)
                if "error" in resp:
                    print(f"error: {resp['error']}", file=sys.stderr)
                    sys.exit(1)
                print(resp["text"])
            else:
                print(f"unknown command: {cmd}", file=sys.stderr)
                sys.exit(2)

    except (ConnectionRefusedError, FileNotFoundError):
        print("error: gpu-queued not running — systemctl --user start gpu-queued", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
