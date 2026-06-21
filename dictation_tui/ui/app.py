"""Main Textual app for the dictation TUI."""
from __future__ import annotations

import asyncio
import subprocess
from functools import partial
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Center
from textual.screen import ModalScreen
from textual.widgets import Static, LoadingIndicator

from ..buffers import RawBuffer, CleanBuffer, Segment
from ..cleanup import cleanup
from .raw_pane import RawPane
from .clean_pane import CleanPane
from .status_bar import StatusBar


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def _copy_to_clipboard(text: str):
    try:
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        proc.communicate(text.encode(), timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


HELP_TEXT = """\
[bold]Dictation TUI — Hotkeys[/bold]

[bold cyan]Enter[/]        Commit block, clean, copy to clipboard
[bold cyan]Ctrl+L[/]       Clear both panes
[bold cyan]c[/]            Copy cleaned text to clipboard
[bold cyan]r[/]            Copy raw text to clipboard
[bold cyan]p[/]            Pause / resume recording
[bold cyan]l[/]            Toggle LLM polish
[bold cyan]?[/]            Show / hide this help
[bold cyan]Ctrl+C[/]        Quit (auto-copies clean text)
"""


class HelpScreen(ModalScreen):
    BINDINGS = [
        Binding("question_mark", "dismiss", "Close"),
        Binding("escape", "dismiss", "Close"),
    ]
    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-box {
        width: 56;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT, id="help-box")


class LoadingScreen(ModalScreen):
    DEFAULT_CSS = """
    LoadingScreen {
        align: center middle;
    }
    #loading-box {
        width: 60;
        height: 7;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
        content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="loading-box")

    def on_mount(self):
        import random
        from loading_anim import ANIMATIONS

        anim_name = random.choice(list(ANIMATIONS))
        self._frames = ANIMATIONS[anim_name]
        self._frame_idx = 0
        self._label = self.query_one("#loading-box")
        self._label.update(
            f"[bold]Loading model...[/bold]  (may take a minute)\n\n"
            f"{self._frames[0]}"
        )
        self.set_interval(0.75, self._tick)

    def _tick(self):
        self._frame_idx += 1
        frame = self._frames[self._frame_idx % len(self._frames)]
        self._label.update(
            f"[bold]Loading model...[/bold]  (may take a minute)\n\n"
            f"{frame}"
        )


class DictationApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("c", "copy_clean", "Copy clean", show=False),
        Binding("r", "copy_raw", "Copy raw", show=False),
        Binding("p", "toggle_pause", "Pause/Resume", show=False),
        Binding("l", "toggle_llm", "Toggle LLM", show=False),
        Binding("ctrl+l", "clear_all", "Clear", show=False),
        Binding("enter", "commit_block", "Commit", show=False),
        Binding("question_mark", "show_help", "Help"),
    ]

    def __init__(self, model: str | None = None, device: str | None = None,
                 chunk_ms: int | None = None, verbose: bool = False):
        super().__init__()
        self.config = _load_config()
        asr_cfg = self.config.get("asr", {})
        self._model = model or asr_cfg.get("model", "nvidia/parakeet-tdt-0.6b-v3")
        self._device = device or asr_cfg.get("device", "cuda")
        self._chunk_ms = chunk_ms or asr_cfg.get("chunk_ms", 640)
        self._verbose = verbose

        cleanup_cfg = self.config.get("cleanup", {})
        self._remove_fillers = cleanup_cfg.get("remove_fillers", False)
        self._llm_polish = cleanup_cfg.get("enable_llm_polish", False)

        stability_cfg = self.config.get("stability", {})
        self._paragraph_gap_s = stability_cfg.get("paragraph_gap_s", 1.5)

        llm_cfg = self.config.get("llm", {})
        self._llm_endpoint = llm_cfg.get("endpoint", "http://localhost:11434/api/generate")
        self._llm_model = llm_cfg.get("model", "granite4:micro")

        self.raw_buffer = RawBuffer()
        self.clean_buffer = CleanBuffer()

        self._pause_event = asyncio.Event()
        self._pause_event.set()  # start recording
        self._stop_event = asyncio.Event()
        self._reset_event = asyncio.Event()
        self._asr_task = None
        self._loading_screen = None

    def compose(self) -> ComposeResult:
        yield RawPane()
        yield CleanPane()
        yield StatusBar()

    async def on_mount(self):
        status = self.query_one(StatusBar)
        status.llm_enabled = self._llm_polish
        status.model_loading = True

        self._loading_screen = LoadingScreen()
        self.push_screen(self._loading_screen)

        from ..asr import asr_task
        self._asr_task = asyncio.create_task(
            asr_task(
                raw_buffer=self.raw_buffer,
                model_name=self._model,
                device=self._device,
                chunk_ms=self._chunk_ms,
                verbose=self._verbose,
                on_update=self._on_asr_update,
                on_model_loaded=self._on_model_loaded,
                pause_event=self._pause_event,
                stop_event=self._stop_event,
                reset_event=self._reset_event,
                paragraph_gap_s=self._paragraph_gap_s,
            )
        )

    async def _on_model_loaded(self):
        """Called once ASR model is loaded and mic is open."""
        if self._loading_screen:
            self.pop_screen()
            self._loading_screen = None
        self.query_one(StatusBar).model_loading = False
        self.notify("Listening — press ? for help")

    async def _on_asr_update(self, text: str):
        """Called from ASR task when raw text changes."""
        raw_pane = self.query_one(RawPane)
        raw_pane.clear()
        for seg in self.raw_buffer.segments:
            raw_pane.write(f"[bold]{seg.text}[/bold]")
        if text.strip():
            raw_pane.write(f"[dim]{text}[/dim]")

    async def _process_segment(self, segment: Segment):
        """Clean a segment and update the clean pane."""
        loop = asyncio.get_event_loop()
        prev = self.clean_buffer.segments[-1] if self.clean_buffer.segments else None

        cleaned_seg = await loop.run_in_executor(
            None,
            partial(
                cleanup, segment,
                prev_segment=prev,
                remove_fillers=self._remove_fillers,
                llm_polish=self._llm_polish,
                llm_endpoint=self._llm_endpoint,
                llm_model=self._llm_model,
            ),
        )
        self.clean_buffer.add(cleaned_seg)

        clean_pane = self.query_one(CleanPane)
        clean_pane.cleaned_text = self.clean_buffer.get_full_text()

        status = self.query_one(StatusBar)
        status.segment_count = len(self.raw_buffer.segments)

    def action_commit_block(self):
        """Enter key: commit pending, clean it, copy to clipboard, reset ASR."""
        seg = self.raw_buffer.commit_pending()
        if seg:
            asyncio.create_task(self._commit_and_copy(seg))
        else:
            self.notify("Nothing to commit")

    async def _commit_and_copy(self, segment: Segment):
        """Process segment, copy cleaned text, signal ASR reset."""
        await self._process_segment(segment)

        # Copy the just-committed segment's cleaned text
        cleaned = segment.cleaned or segment.text
        _copy_to_clipboard(cleaned)
        self.notify(f"Copied {len(cleaned)} chars")

        # Save transcript
        from nemotron_stream import save_transcript
        save_transcript(cleaned)

        # Reset ASR decoder for fresh start
        self._reset_event.set()

        # Refresh raw pane
        raw_pane = self.query_one(RawPane)
        raw_pane.clear()
        for seg in self.raw_buffer.segments:
            raw_pane.write(f"[bold]{seg.text}[/bold]")

    def action_show_help(self):
        self.push_screen(HelpScreen())

    def action_copy_clean(self):
        text = self.clean_buffer.get_full_text()
        if text:
            _copy_to_clipboard(text)
            self.notify(f"Copied {len(text)} chars (clean)")

    def action_copy_raw(self):
        text = self.raw_buffer.get_full_text()
        if text:
            _copy_to_clipboard(text)
            self.notify(f"Copied {len(text)} chars (raw)")

    def action_toggle_pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.query_one(StatusBar).is_recording = False
            self.notify("Paused")
        else:
            self._pause_event.set()
            self.query_one(StatusBar).is_recording = True
            self.notify("Recording")

    def action_toggle_llm(self):
        self._llm_polish = not self._llm_polish
        self.query_one(StatusBar).llm_enabled = self._llm_polish
        self.notify(f"LLM polish: {'on' if self._llm_polish else 'off'}")

    def action_clear_all(self):
        self.raw_buffer.clear()
        self.clean_buffer.clear()
        self.query_one(RawPane).clear()
        self.query_one(CleanPane).cleaned_text = ""
        self.query_one(StatusBar).segment_count = 0
        self._reset_event.set()
        self.notify("Cleared")

    async def action_quit(self):
        ui_cfg = self.config.get("ui", {})
        if ui_cfg.get("auto_copy_on_quit", True):
            text = self.clean_buffer.get_full_text()
            if text:
                _copy_to_clipboard(text)
        self._stop_event.set()
        if self._asr_task:
            self._asr_task.cancel()
        self.exit()
