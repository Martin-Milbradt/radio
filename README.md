# radio

Endless generative music, synthesized bar-by-bar with numpy and streamed as
raw PCM to ffplay. Nothing is written to disk during playback and memory
stays flat no matter how long it runs. Output never loops: an unseeded RNG
drives all musical choices, and the garden channel is built on coprime loop
lengths that structurally cannot realign.

## Install

```bash
pip install -r requirements.txt
```

Live playback needs ffmpeg 7+ on PATH (`ffplay` with `-ch_layout`). Rendering
to a wav file (`--wav`) uses Python's `wave` module and needs no ffmpeg.

## Usage

```bash
python radio.py                  # default channel (fable), plays forever
python radio.py garden           # pick a channel
python radio.py fable --wav out.wav --seconds 120
```

## Channels

The three main channels:

- `fable` - one infinite, continuously evolving melodic track
- `garden` - process music: coprime loops, Reich phasing, modal drift
- `banger` - endless dance-music playlist (EDM, house, breaks, trance,
  downtempo)

The `fable-experimental` and `banger-experimental` channels are busier variants
of the same engines, adding a canon twin, kalimba/bell voices, polymetric arps,
square blips and a euclidean shaker bed.

## Run in the browser

The same numpy engine runs client-side via [Pyodide](https://pyodide.org)
(CPython + numpy compiled to WebAssembly), so it deploys to any static host
including GitHub Pages. No server, no ffplay: channels yield one bar at a time
through their `bars()` generator and the page schedules them through the Web
Audio API. Output stays fully generative and never repeats. Pause holds the
current track, Stop resets it (the next start is a fresh seed), and switching
channels crossfades (a longer beat-overlapping blend within a family, a plain
crossfade across families); Stop then a channel is a hard cut.

Local preview (the page fetches the channel sources as siblings, so assemble a
flat folder first, exactly like the deploy workflow does). On Windows, `serve.ps1`
does this for you:

```bash
mkdir -p _site && cp web/index.html web/radio.js _site/
cp radio_core.py channel_*.py _site/
python -m http.server -d _site        # open http://localhost:8000
```

Deploy: push to a GitHub repo with Pages set to "Source: GitHub Actions". The
workflow in `.github/workflows/pages.yml` assembles `web/` plus the channel
sources and publishes them. First visit downloads Pyodide and numpy (~10-15 MB),
cached afterward; playback starts on the first channel click (browser autoplay
policy).

## Layout

- `radio.py` - CLI entry point; channel registry lives in its `CHANNELS` dict.
- `radio_core.py` - shared engine: DSP helpers, instruments, sinks, and the
  `BarStreamer` that carries ring-over tails between bars. Channels own all
  music logic; the core owns how things sound and how they reach the speakers.
- `channel_*.py` - one module per channel. Each exposes a `bars(seconds)`
  generator (the single source of the play loop, yielding mastered bars) and a
  `stream(sink, seconds)` that writes those bars to a sink. `channel_fable` and
  `channel_banger` are the main channels; `channel_fable_experimental` and
  `channel_banger_experimental` are the busier variants the mains borrow shared
  definitions from.
- `web/` - the browser front-end (`index.html` + `radio.js`) that runs the
  engine via Pyodide and plays it through Web Audio.
- `.github/workflows/pages.yml` - assembles and deploys the web build to Pages.
- `samples/` - a roughly 1 min MP3 preview per channel, safe to delete.

## Invariants worth keeping

- All oscillators are band-limited (polyBLEP saws, tanh-softened squares);
  percussion noise comes from `bp_noise` (no near-Nyquist energy, no
  convolution edge artifacts). Every percussive envelope ends in `declick`.
- Buffers handed to `BarStreamer.push` must be `bar + tail` samples long, with
  the tail long enough for the channel's longest ring-over (the garden uses a
  6 s tail for its 2-bar drones).
- All pitched material in a bar must be diatonic to the channel's current
  scale/chord; melodies are motif-based (repeat with small mutations), not
  random walks.
