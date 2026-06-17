"""Fable channel: one infinite, continuously evolving melodic track.

The soft 42 Hz-tail kick, plain saw bassline with per-chord diatonic contours
and the tanh crush at full tilt, offbeat noise ticks plus coprime 5- and
7-pulse tick cycles that never realign, breathing vibrato-sine pad chords
with an add9 color that blooms and fades, a chord-locked melody motif on
sine/kalimba/bell voices with an octave-down canon twin, and bell accents in
the quiet phrases. An intensity random-walk shapes the energy; evolution
changes exactly one thing per 8-bar phrase.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from radio_core import (
    RNG,
    SR,
    TAIL,
    BarStreamer,
    FfplaySink,
    WavSink,
    bell,
    blip,
    echo,
    kalimba,
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

# chord voicings and bass roots in the A minor family
CHORDS = {
    "Am": (110.0, 164.81, 220.0, 261.63),
    "C": (130.81, 196.0, 261.63, 329.63),
    "Dm": (146.83, 174.61, 220.0, 293.66),
    "Em": (164.81, 196.0, 246.94, 329.63),
    "F": (87.31, 130.81, 174.61, 220.0),
    "G": (98.0, 146.83, 196.0, 246.94),
}
BASS_ROOT = {"Am": 55.0, "C": 65.41, "Dm": 73.42, "Em": 82.41, "F": 43.65, "G": 49.0}
# bassline contour per chord, in semitones from its root: the ytp [0,0,3,-2]
# shape works for minor chords; major chords get their major third and a
# diatonic step up so every note stays in the key
BASS_OFFSETS = {
    "Am": (0, 0, 3, -2),
    "C": (0, 0, 4, 2),
    "Dm": (0, 0, 3, -2),
    "Em": (0, 0, 3, -2),
    "F": (0, 0, 4, 2),
    "G": (0, 0, 4, 2),
}
PROGS = (
    ["Am", "F", "C", "G"],
    ["Am", "F", "G", "C"],
    ["Am", "C", "F", "G"],
    ["Am", "G", "F", "Em"],
    ["Am", "Dm", "F", "G"],
    ["Am", "Dm", "C", "G"],
    ["F", "G", "Am", "Em"],
)
MELODY_VOICES = {"bell": bell, "kalimba": kalimba, "sine": melody_note}
PHRASE = 8  # bars; chords change every 2 bars, one progression pass per phrase


@dataclass
class Track:
    bpm: float
    prog: list[str]
    intensity: int = 1
    next_intensity: int = 2
    transpose: int = 0
    motif: list[tuple[int, int, float]] = field(default_factory=list)
    blips_on: bool = False
    bars_to_transpose: int = 0
    mel_instr: str = "sine"
    canon: bool = True
    color: float = 0.0  # how much add9 sits in the pad bed
    breath_ph: float = 0.0


def make_motif() -> list[tuple[int, int, float]]:
    """Chord-locked hook over 2 bars: (8th slot, chord-tone index, length in beats)."""
    slots = sorted(RNG.choice(16, size=int(RNG.integers(4, 8)), replace=False).tolist())
    tone = int(RNG.integers(2, 6))
    out = []
    for s in slots:
        tone = int(np.clip(tone + RNG.integers(-2, 3), 0, 7))
        out.append((int(s), tone, float(RNG.choice([0.5, 1.0, 1.0, 2.0]))))
    return out


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
        motif=make_motif(),
        bars_to_transpose=int(RNG.integers(64, 160)),
        mel_instr=pick(("sine", "kalimba", "kalimba", "bell")),
        canon=RNG.uniform() < 0.7,
        breath_ph=float(RNG.uniform(0, 6.28)),
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
    tr.blips_on = tr.intensity >= 4 and RNG.uniform() < 0.5

    tr.bars_to_transpose -= PHRASE
    r2 = RNG.uniform()
    if tr.bars_to_transpose <= 0:
        tr.bars_to_transpose = int(RNG.integers(64, 160))
        tr.transpose = int(np.clip(tr.transpose + pick((-2, -1, 1, 2)), -3, 3))
        print(f"   transpose -> {tr.transpose:+d} st")
    elif r2 < 0.12:
        tr.prog = list(PROGS[int(RNG.integers(0, len(PROGS)))])
        print(f"   chords -> {'-'.join(tr.prog)}")
    elif r2 < 0.24:
        tr.motif = make_motif()
        print("   new motif")
    elif r2 < 0.45 and tr.motif:
        i = int(RNG.integers(0, len(tr.motif)))
        s, tone, ln = tr.motif[i]
        tr.motif[i] = (s, int(np.clip(tone + pick((-1, 1)), 0, 7)), ln)
        print("   motif shifts a note")
    elif r2 < 0.62:
        old = tr.color
        tr.color = float(np.clip(tr.color + RNG.uniform(-0.3, 0.35), 0, 1))
        if (old < 0.3) != (tr.color < 0.3):
            print(f"   color {'blooms' if tr.color >= 0.3 else 'fades'}")
    elif r2 < 0.72:
        tr.canon = not tr.canon
        print(f"   canon {'joins' if tr.canon else 'rests'}")

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

    # pad bed: breathing gain, add9 color blooming in and out
    if gbar % 2 == 0:
        breath = 0.85 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * gbar / 19 + tr.breath_ph))
        voicing = freqs
        if tr.color >= 0.3:
            voicing = freqs + (freqs[0] * 2 ** (14 / 12),)  # 9th above the root
        place(buf, pad_chord(voicing, 2 * bar + 1.2), 0, (0.13 + 0.02 * level) * breath)

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

    # drums, plus coprime 5- and 7-pulse tick cycles that drift forever
    if level >= 2:
        for s in range(4):
            place(buf, kick_soft(), s * beat, 0.78 + 0.05 * level)
        for s in range(4):
            place(buf, tick(), s * beat + beat / 2, 0.10 + 0.02 * level, pan=0.2)
        gpulse = gbar * 8
        for p in range(8):
            if (gpulse + p) % 5 == 0:
                place(buf, tick(), p * beat / 2, 0.05, pan=-0.4)
            if (gpulse + p) % 7 == 0:
                place(buf, tick(), p * beat / 2, 0.04, pan=0.45)
        if pbar == PHRASE - 1:
            place(buf, kick_soft(), 3.5 * beat, 0.55)

    # bell accent at the start of quiet phrases
    if level <= 1 and pbar == 0 and gbar % 2 == 0:
        place(buf, bell(freqs[2], 2.5), 0, 0.07, pan=0.1)

    # melody motif, chord-locked, with an octave-down canon twin
    if tr.motif:
        half = (gbar % 2) * 8
        voice = MELODY_VOICES[tr.mel_instr]
        for s, tone, ln in tr.motif:
            if not half <= s < half + 8:
                continue
            if s % 8 >= 6:
                ln = min(ln, 1.0)  # don't let late notes ring into the next chord
            f = freqs[tone % 4] * (2 if tone >= 4 else 1)
            snd = echo(voice(f, beat * ln), 0.18, 0.4, 2)
            pan = float(RNG.uniform(-0.3, 0.3))
            place(buf, snd, (s - half) * beat / 2, 0.09 + 0.01 * level, pan)
            if tr.canon and s % 8 < 5:
                twin = echo(voice(f / 2, beat * ln), 0.18, 0.4, 2)
                place(
                    buf,
                    twin,
                    (s - half) * beat / 2 + 1.5 * beat,
                    (0.09 + 0.01 * level) * 0.45,
                    -pan,
                )

    # square blips on chord tones, rare, full tilt only
    if tr.blips_on and pbar in (2, 6):
        for i, tone in enumerate((2, 1, 2, 3)):
            place(
                buf,
                blip(freqs[tone] * 2),
                i * beat + beat / 4,
                0.05,
                pan=0.35 if i % 2 else -0.35,
            )

    if tr.next_intensity > tr.intensity and pbar == PHRASE - 2:
        place(buf, rise_fx(2 * bar), 0, 0.35)
    if level == 0 and pbar == 4 and RNG.uniform() < 0.3:
        place(buf, sweep(980, 1020, 0.07), 2 * beat, 0.15)  # the terminal bell

    return buf


# ---------------------------------------------------------------- streaming


def bars(seconds: float | None) -> Iterator[np.ndarray]:
    """Yield each mastered stereo bar; single source of the play loop (CLI and web)."""
    tr = make_track()
    print(
        f">> endless fable | {tr.bpm:.0f} BPM | {'-'.join(tr.prog)}"
        f" | melody:{tr.mel_instr}{' + canon' if tr.canon else ''}"
    )
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
