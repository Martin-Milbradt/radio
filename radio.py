"""Endless generative radio: pick a channel, get music forever.

Each channel synthesizes one bar at a time with numpy and streams raw PCM to
ffplay (nothing touches disk, memory stays flat) or to a wav file. See the
channel_*.py modules for what each channel sounds like.

Usage:
    python radio.py                              # default channel, forever
    python radio.py garden                       # pick a channel
    python radio.py fable --wav out.wav --seconds 120
"""

from __future__ import annotations

import argparse
import importlib

from radio_core import FfplaySink, WavSink

CHANNELS = {
    "banger": "experimental banger: adds polymetric arps, kalimba/bell, canon, shaker bed",
    "banger-old": "endless dance-music playlist (EDM, house, breaks, trance, downtempo)",
    "fable": "experimental fable: adds canon twin, kalimba/bell voices, square blips",
    "fable-old": "one infinite, continuously evolving melodic track",
    "garden": "process music: coprime loops, Reich phasing, modal drift",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Endless generative radio.",
        epilog="channels: " + "; ".join(f"{k}: {v}" for k, v in CHANNELS.items()),
    )
    ap.add_argument("channel", nargs="?", default="fable", choices=sorted(CHANNELS))
    ap.add_argument("--wav", help="render to a wav file instead of playing")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds")
    args = ap.parse_args()

    mod = importlib.import_module(f"channel_{args.channel.replace('-', '_')}")
    if args.wav:
        sink: WavSink | FfplaySink = WavSink(args.wav)
        seconds = args.seconds or 120.0
    else:
        sink = FfplaySink()
        seconds = args.seconds
    try:
        mod.stream(sink, seconds)
    except KeyboardInterrupt:
        print("\nstopped.")
    except (BrokenPipeError, OSError):
        print("\nplayback closed.")
    finally:
        try:
            sink.close()
        except (BrokenPipeError, OSError):
            pass


if __name__ == "__main__":
    main()
