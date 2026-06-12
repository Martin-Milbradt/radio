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

- `banger` - endless dance-music playlist (EDM, house, breaks, trance, downtempo)
- `fable` - one infinite, continuously evolving melodic track
- `garden` - process music: coprime loops, Reich phasing, modal drift

The `banger-old` and `fable-old` channels are frozen archives of superseded
versions, kept for comparison.

## Layout

- `radio.py` - CLI entry point; channel registry lives in its `CHANNELS` dict.
- `radio_core.py` - shared engine: DSP helpers, instruments, sinks, and the
  `BarStreamer` that carries ring-over tails between bars. Channels own all
  music logic; the core owns how things sound and how they reach the speakers.
- `channel_*.py` - one module per channel, each exposing
  `stream(sink, seconds)`. The `*_old` channels are frozen archives of
  superseded versions.
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
