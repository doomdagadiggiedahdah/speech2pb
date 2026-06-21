# Dictation TUI — TODO

## Jetson offload for block cleanup
- On Enter (block finalize), send the raw chunk to Jetson for LLM cleanup
- While user dictates the next block, Jetson processes the previous one in parallel
- Cleaned text streams back and replaces raw in the clean pane
- End-of-session: export full cleaned document as markdown
- Auto-sync to Obsidian note (user provides note name somehow — arg? prompt? TUI input?)

## BERT fine-tuning experiments
- Train a custom BERT model for a fun task (sentiment? domain-specific NER?)
- BERT = encoder half of the transformer (2018, Google) — bidirectional context understanding
- Fine-tuning workflow is well-documented, runs on T1000
- Good entry point for understanding transformer internals

## Research: SOTA STT post-processing
- Investigate how Anthropic's mobile app STT achieves clean output
- Look into what stack/pipeline modern voice-to-text tools use for formatting
- Determine whether downstream LLM cleanup is the right move vs better ASR models
