"""Fable channel, superseded v1 (kept as an archive).

The original endless fable track: sine melody only, no canon twin, no
add9 color, no coprime tick cycles, no bell accents. See channel_fable for
the current version.
"""

from __future__ import annotations

import numpy as np

from channel_fable import (
    BASS_OFFSETS,
    BASS_ROOT,
    CHORDS,
    PHRASE,
    PROGS,
    Track,
    make_motif,
)
from radio_core import (
    RNG,
    SR,
    TAIL,
    BarStreamer,
    FfplaySink,
    WavSink,
    blip,
    echo,
    kick_soft,
    melody_note,
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
    return Track(
        bpm=float(RNG.uniform(136, 144)),
        prog=list(PROGS[0]),
        motif=make_motif(),
        bars_to_transpose=int(RNG.integers(64, 160)),
    )


def evolve(tr: Track, gbar: int) -> None:
    tr.intensity = tr.next_intensity
    r = RNG.uniform()
    if tr.intensity == 0:
        tr.next_intensity = int(RNG.integers(2, 4))
    elif r < 0.05:
        tr.next_intensity = 0
    elif r < 0.15:
        tr.next_intensity = 4
    else:
        step = pick((-1, -1, 0, 1)) if tr.intensity == 4 else pick((-1, 0, 1, 1))
        tr.next_intensity = int(np.clip(tr.intensity + step, 1, 4))
        if tr.intensity == 1 and tr.next_intensity == 1:
            tr.next_intensity = 2
    if RNG.uniform() < 0.15:
        tr.prog = list(PROGS[int(RNG.integers(0, len(PROGS)))])
        print(f"   chords -> {'-'.join(tr.prog)}")
    if RNG.uniform() < 0.20:
        tr.motif = make_motif()
        print("   new motif")
    elif RNG.uniform() < 0.5 and tr.motif:
        i = int(RNG.integers(0, len(tr.motif)))
        s, tone, ln = tr.motif[i]
        tr.motif[i] = (s, int(np.clip(tone + RNG.integers(-2, 3), 0, 7)), ln)
    tr.blips_on = tr.intensity >= 4 and RNG.uniform() < 0.5
    tr.bars_to_transpose -= PHRASE
    if tr.bars_to_transpose <= 0:
        tr.bars_to_transpose = int(RNG.integers(64, 160))
        tr.transpose = int(np.clip(tr.transpose + RNG.choice([-2, -1, 1, 2]), -3, 3))
        print(f"   transpose -> {tr.transpose:+d} st")
    if tr.intensity != tr.next_intensity or gbar % (PHRASE * 4) == 0:
        print(f"   intensity {tr.intensity} -> {tr.next_intensity}"
              f" (bar {gbar}, {'-'.join(tr.prog)})")


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

    if gbar % 2 == 0:
        place(buf, pad_chord(freqs, 2 * bar + 1.2), 0, 0.13 + 0.02 * level)

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

    if tr.motif:
        half = (gbar % 2) * 8
        for s, tone, ln in tr.motif:
            if not half <= s < half + 8:
                continue
            if s % 8 >= 6:
                ln = min(ln, 1.0)
            f = freqs[tone % 4] * (2 if tone >= 4 else 1)
            snd = echo(melody_note(f, beat * ln), 0.18, 0.4, 2)
            place(buf, snd, (s - half) * beat / 2, 0.09 + 0.01 * level,
                  pan=float(RNG.uniform(-0.3, 0.3)))

    if tr.blips_on and pbar in (2, 6):
        for i, tone in enumerate((2, 1, 2, 3)):
            place(buf, blip(freqs[tone] * 2), i * beat + beat / 4, 0.05,
                  pan=0.35 if i % 2 else -0.35)

    if tr.next_intensity > tr.intensity and pbar == PHRASE - 2:
        place(buf, rise_fx(2 * bar), 0, 0.35)
    if level == 0 and pbar == 4 and RNG.uniform() < 0.3:
        place(buf, sweep(980, 1020, 0.07), 2 * beat, 0.15)

    return buf


def stream(sink: WavSink | FfplaySink, seconds: float | None) -> None:
    tr = make_track()
    print(f">> endless fable (old) | {tr.bpm:.0f} BPM | {'-'.join(tr.prog)}")
    bs = BarStreamer(sink)
    gbar = 0
    while True:
        if gbar % PHRASE == 0 and gbar > 0:
            evolve(tr, gbar)
        bs.push(render_bar(tr, gbar))
        gbar += 1
        if seconds is not None and bs.played >= seconds:
            return
