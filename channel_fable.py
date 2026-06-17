"""Fable channel: one infinite, continuously evolving melodic track.

The clean melodic fable: a soft 42 Hz-tail kick, a plain saw bassline with
per-chord diatonic contours and a tanh crush at full tilt, offbeat noise ticks,
and a soft chord pad that swells in to carry the level-0 breakdowns. An intensity
random-walk shapes the energy; evolution changes at most one thing per 8-bar
phrase. The bright extras (canon twin, kalimba/bell voices, coprime
tick cycles, square blips) live in the experimental version,
channel_fable_experimental, from which this module borrows its shared chord
definitions.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from channel_fable_experimental import (
    BASS_OFFSETS,
    BASS_ROOT,
    CHORDS,
    PHRASE,
    PROGS,
    Track,
)
from radio_core import (
    RNG,
    SR,
    TAIL,
    BarStreamer,
    FfplaySink,
    WavSink,
    kick_soft,
    pad_chord,
    pick,
    place,
    rise_fx,
    saw_note,
    semi,
    sweep,
    tick,
)


def make_track() -> Track:
    # start every track from the seed: a random progression, key and energy, so
    # no two sessions open the same way
    intensity = int(pick((1, 2, 2, 3)))
    return Track(
        bpm=float(RNG.uniform(136, 144)),
        prog=list(pick(PROGS)),
        intensity=intensity,
        next_intensity=int(np.clip(intensity + pick((0, 1)), 1, 4)),
        transpose=int(RNG.integers(-3, 4)),
        bars_to_transpose=int(RNG.integers(64, 160)),
    )


def evolve(tr: Track, gbar: int) -> None:
    """Phrase boundary: walk the intensity, then mutate at most ONE thing."""
    tr.intensity = tr.next_intensity
    r = RNG.uniform()
    if tr.intensity == 0:
        tr.next_intensity = int(RNG.integers(2, 4))  # breakdowns last one phrase
    elif r < 0.05:
        tr.next_intensity = 0
    elif r < 0.15:
        tr.next_intensity = 4
    else:
        # upward-biased walk so quiet stretches don't overstay; level 4 cools off
        step = pick((-1, -1, 0, 1)) if tr.intensity == 4 else pick((-1, 0, 1, 1))
        tr.next_intensity = int(np.clip(tr.intensity + step, 1, 4))
        if tr.intensity == 1 and tr.next_intensity == 1:
            tr.next_intensity = 2

    tr.bars_to_transpose -= PHRASE
    r2 = RNG.uniform()
    if tr.bars_to_transpose <= 0:
        tr.bars_to_transpose = int(RNG.integers(64, 160))
        tr.transpose = int(np.clip(tr.transpose + pick((-2, -1, 1, 2)), -3, 3))
        print(f"   transpose -> {tr.transpose:+d} st")
    elif r2 < 0.12:
        tr.prog = list(PROGS[int(RNG.integers(0, len(PROGS)))])
        print(f"   chords -> {'-'.join(tr.prog)}")

    if tr.intensity != tr.next_intensity or gbar % (PHRASE * 4) == 0:
        print(
            f"   intensity {tr.intensity} -> {tr.next_intensity}"
            f" (bar {gbar}, {'-'.join(tr.prog)})"
        )


# ---------------------------------------------------------------- bar render


def render_bar(tr: Track, gbar: int) -> np.ndarray:
    beat = 60 / tr.bpm
    bar = 4 * beat
    barlen = int(bar * SR)
    buf = np.zeros((barlen + int(TAIL * SR), 2))
    pbar = gbar % PHRASE
    chord_name = tr.prog[(gbar // 2) % len(tr.prog)]
    shift = 2 ** (tr.transpose / 12)
    freqs = tuple(f * shift for f in CHORDS[chord_name])
    root = BASS_ROOT[chord_name] * shift
    level = tr.intensity

    # breakdown pad: a soft chord bed that swells in over the two bars before a
    # level-0 breakdown, holds through it, and tails out as the music returns
    leadin = level >= 1 and tr.next_intensity == 0 and pbar == PHRASE - 2
    if gbar % 2 == 0 and (level == 0 or leadin):
        pad = pad_chord(freqs, 2 * bar + 1.2)
        if leadin:
            pad = pad * np.linspace(0, 1, len(pad))  # swell in just before the drop
        place(buf, pad, 0, 0.15)

    # bassline
    if level == 1:
        for s in (0, 2):
            place(buf, saw_note(root, beat * 1.9), s * beat, 0.20)
        for s in range(4):
            place(buf, tick(), s * beat + beat / 2, 0.06, pan=0.2)
    elif level >= 2:
        crush = level >= 4
        offsets = BASS_OFFSETS[chord_name]
        for s in range(8):
            note = semi(root, offsets[s % 4])
            place(buf, saw_note(note, beat / 2 * 0.9, crush=crush), s * beat / 2, 0.34)

    # drums
    if level >= 2:
        for s in range(4):
            place(buf, kick_soft(), s * beat, 0.78 + 0.05 * level)
        for s in range(4):
            place(buf, tick(), s * beat + beat / 2, 0.10 + 0.02 * level, pan=0.2)
        if level >= 3:
            for s in range(8):
                if s % 2:
                    place(buf, tick(), s * beat / 2 + beat / 4, 0.05, pan=-0.3)
        if pbar == PHRASE - 1:
            place(buf, kick_soft(), 3.5 * beat, 0.55)

    if tr.next_intensity > tr.intensity and pbar == PHRASE - 2:
        place(buf, rise_fx(2 * bar), 0, 0.35)
    if level == 0 and pbar == 4 and RNG.uniform() < 0.3:
        place(buf, sweep(980, 1020, 0.07), 2 * beat, 0.15)  # the terminal bell

    return buf


# ---------------------------------------------------------------- streaming


def bars(seconds: float | None) -> Iterator[np.ndarray]:
    """Yield each mastered stereo bar; single source of the play loop (CLI and web)."""
    tr = make_track()
    print(f">> endless fable | {tr.bpm:.0f} BPM | {'-'.join(tr.prog)}")
    bs = BarStreamer()
    gbar = 0
    while True:
        if gbar % PHRASE == 0 and gbar > 0:
            evolve(tr, gbar)
        yield bs.process(render_bar(tr, gbar))
        gbar += 1
        if seconds is not None and bs.played >= seconds:
            return


def stream(sink: WavSink | FfplaySink, seconds: float | None) -> None:
    for out in bars(seconds):
        sink.write((out * 32767).astype(np.int16).tobytes())
