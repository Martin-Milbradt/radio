# CLAUDE.md

Guidance for Claude Code working in this repo.

## Overview

Endless generative radio: each channel synthesizes music one bar at a time with
pure numpy and streams raw PCM to `ffplay` (live) or a wav file. No disk writes
during playback, flat memory, and an unseeded RNG so output never repeats.

## Commands

```bash
pip install -r requirements.txt                       # numpy only
python radio.py                                        # play default channel (fable) forever
python radio.py <channel> --seconds 60                 # play a channel for 60 s
python radio.py <channel> --wav out.wav --seconds 60   # render to wav instead of playing
ruff check . && ruff format .                          # lint and format
basedpyright .                                         # type check
```

There is no test suite. The `samples/` previews are rendered to wav and then
encoded, for example:

```bash
python radio.py garden --wav g.wav --seconds 60
ffmpeg -y -i g.wav -c:a libmp3lame -q:a 0 samples/garden.mp3
```

## Architecture

- `radio.py` - CLI entry point. The `CHANNELS` dict is the channel registry.
  `--wav` swaps `FfplaySink` for `WavSink`; `--seconds` bounds playback.
- `radio_core.py` - the engine, with no music logic. DSP helpers, instruments
  (percussion, pitched, fx), the two sinks, and `BarStreamer`, which overlaps
  each bar's ring-over tail into the next bar and soft-clips. `SR = 48000`.
- `channel_*.py` - one module per channel, each exposing `stream(sink, seconds)`
  and rendering one bar at a time into a `bar + tail` buffer. Channels own all
  musical decisions; the core owns how things sound. The `*_old` modules are
  frozen archives kept for comparison: do not add features to them.

## Conventions

- Keep the synthesis invariants intact (see README, "Invariants worth keeping"):
  band-limited oscillators, `bp_noise` percussion, every percussive envelope
  through `declick`, pitched material diatonic to the current scale or chord,
  motif-based melodies rather than random walks.
- Buffers pushed to `BarStreamer.push` must be `bar + tail` samples long, and
  the tail must cover the channel's longest ring-over (garden uses a 6 s tail).
- RNG is the shared `radio_core.RNG`, unseeded by design. Do not seed it.
- Pure numpy only. Do not add other audio libraries.
