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

Browser build (Pyodide + Web Audio, deploys to GitHub Pages). The page fetches
the channel sources as siblings, so preview from a flat assembled folder:

```bash
mkdir -p _site && cp web/index.html web/radio.js _site/
cp radio_core.py channel_*.py _site/
python -m http.server -d _site                         # open http://localhost:8000
```

`serve.ps1` does the same assemble-and-serve on Windows (port 41001).

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
- `channel_*.py` - one module per channel. Each exposes a `bars(seconds)`
  generator (the single source of the play loop; yields mastered bars from
  `BarStreamer.process`) and a thin `stream(sink, seconds)` that encodes those
  bars to the sink. Channels own all musical decisions; the core owns how things
  sound. `channel_fable` and `channel_banger` are the MAIN, clean channels;
  `channel_fable_experimental` and `channel_banger_experimental` are the
  feature-rich variants (canon, extra voices, arps, blips). The mains import
  shared chord/section/layer definitions from the experimental modules, so don't
  break those exports.
- `web/` + `.github/workflows/pages.yml` - the browser front-end runs the engine
  via Pyodide and pulls `bars()` through Web Audio. The pull model is required:
  GitHub Pages can't send COOP/COEP, so no `SharedArrayBuffer` and no way to
  backpressure an infinite loop. Each channel plays as a "voice" with its own gain
  node; switching channels crossfades two voices (a longer beat-overlapping blend
  within a family, a plain crossfade across families), Pause suspends the context,
  Stop closes it. Per-bar audio crosses as raw float32 bytes (not a WASM view,
  which heap growth can detach). See [[web-build-cross-browser]] for the
  cross-browser gotchas and the Playwright test method.

## Conventions

- Keep the synthesis invariants intact (see README, "Invariants worth keeping"):
  band-limited oscillators, `bp_noise` percussion, every percussive envelope
  through `declick`, pitched material diatonic to the current scale or chord,
  motif-based melodies rather than random walks.
- Buffers pushed to `BarStreamer.process` must be `bar + tail` samples long, and
  the tail must cover the channel's longest ring-over (garden uses a 6 s tail).
- RNG is the shared `radio_core.RNG`, unseeded by design. Channel code must not
  seed it. The web glue is the one exception: it swaps `RNG.bit_generator.state`
  per voice per bar (bar rendering is atomic on the single JS thread) so each
  voice has an independent, fresh seed and concurrent voices never share a stream.
- Pure numpy only. Do not add other audio libraries.
