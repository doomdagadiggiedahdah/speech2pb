from textual.reactive import reactive
from textual.widgets import Static
from rich.text import Text


class CleanPane(Static):
    """Middle pane: cleaned transcript with brief diff highlights."""

    DEFAULT_CSS = """
    CleanPane {
        height: 1fr;
        border: solid $success;
        border-title-color: $text;
        overflow-y: auto;
    }
    """

    cleaned_text: reactive[str] = reactive("", layout=True)

    def __init__(self):
        super().__init__("")
        self.border_title = "Cleaned"

    def watch_cleaned_text(self, _old: str, new: str):
        if not new:
            self.update("")
            return

        chunks = new.split("\n\n")
        if len(chunks) <= 1:
            self.update(Text(new, style="bold cyan"))
            return

        combined = Text()
        # First chunk is the newest (reversed order from buffer)
        combined.append(chunks[0], style="bold cyan")
        for chunk in chunks[1:]:
            combined.append("\n\n")
            combined.append(chunk, style="dim")
        self.update(combined)
