#!/usr/bin/python3
# Uses system python for tkinter; pulls llmpop from .venv via site.addsitedir
"""
stt_chat - open a chat window pre-loaded with a speech-to-text transcription.
Usage: stt_chat.py "transcribed text"
"""

import sys
import os
import tkinter as tk

_venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".venv/lib/python3.12/site-packages")
if _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)
    import site
    site.addsitedir(_venv_site)


from llmpop import ChatWindow

SYSTEM_PROMPT = (
    "You are a conversational assistant. The user has just spoken a thought aloud "
    "and it has been transcribed for you. Respond naturally and helpfully.\n\n"
    "If the input is a question, answer it directly and concisely.\n"
    "If the input is a statement or idea, engage with it — ask a clarifying question, "
    "offer a perspective, or expand on it as appropriate.\n"
    "If the input is a task or instruction (e.g. 'write me X', 'explain Y'), do it.\n\n"
    "Use markdown for structure when the response benefits from it, but keep it "
    "conversational. No unnecessary preamble."
)


def main():
    query = " ".join(sys.argv[1:]).strip()
    root = tk.Tk()
    ChatWindow(root, query, system_prompt=SYSTEM_PROMPT)
    root.mainloop()


if __name__ == "__main__":
    main()
