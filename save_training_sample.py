#!/usr/bin/env python3
# WAYLAND-BROKEN (2026-10-05): synthetic keypresses (pyautogui / xdotool) never reach native-Wayland
# windows, so the selection / URL grab silently fails. Fix pattern = quote.py in .setup/hotkeys
# (wl-paste for selection, ydotool for keys, shared helper planned as .setup/util/waykeys.py).
# Full context + plan: claude -r 6e6d343a-9535-4386-9d46-ef4e954bc3a6   (session 'wayland port: hotkeys, dmenu, xclip, ydotool')
"""
Save a training sample: pairs the most recent STT audio with user-corrected text.

Triggered by hotkey after stt_hk transcription:
  1. stt_hk transcribes → text to clipboard → user pastes somewhere
  2. User edits the pasted text (or leaves it)
  3. User selects the final text, hits hotkey
  4. This script: grabs selection, pairs with most recent audio,
     appends to manifest, puts text on clipboard, pastes back over selection
"""
import json
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / ".output"
AUDIO_DIR = OUTPUT_DIR / "audio"
TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
TRAINING_DIR = OUTPUT_DIR / "training"
MANIFEST = TRAINING_DIR / "manifest.jsonl"


def get_x11_text(selection):
    r = subprocess.run(
        ["wl-paste", "-n"] + (["--primary"] if selection == "primary" else []),
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def most_recent_audio():
    files = sorted(AUDIO_DIR.glob("original_*"), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_raw_transcript(audio_path):
    """Find transcript matching an audio file's timestamp."""
    ts = "_".join(audio_path.stem.split("_")[1:])  # YYYYMMDD_HHMMSS
    t = TRANSCRIPT_DIR / f"raw_transcript_{ts}.txt"
    return t.read_text().strip() if t.exists() else ""


def notify(msg):
    subprocess.Popen(["notify-send", "-t", "3000", "STT Training", msg])


def main():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    corrected = get_x11_text("primary") or get_x11_text("clipboard")
    if not corrected:
        notify("No text selected or on clipboard")
        sys.exit(1)

    audio = most_recent_audio()
    if not audio:
        notify("No audio files found")
        sys.exit(1)

    raw_text = find_raw_transcript(audio)

    entry = {"model_text": raw_text} if raw_text else {}
    if corrected != raw_text and raw_text:
        entry["human_edit"] = corrected
    entry["audio_filepath"] = str(audio)

    with open(MANIFEST, "a") as f:
        f.write(json.dumps(entry, indent=2) + "\n")

    # Put corrected text on clipboard and paste back over selection
    subprocess.run(["wl-copy"],
                   input=corrected.encode(), check=True)
    subprocess.run(["xdotool", "key", "--delay", "50", "ctrl+v"])

    label = "corrected" if "raw_text" in entry else "accepted"
    notify(f"Saved {label} sample")


if __name__ == "__main__":
    main()
