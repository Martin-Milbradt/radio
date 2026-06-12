"""Banger channel: endless generative dance-music playlist.

Style presets (peak EDM, house/disco, breakbeat, trance, downtempo) pick
tempo, groove, swing, scale, bass pattern, chord instrument, build type and
song structure per track. Polymetric arps (12/24-step patterns over 16-step
bars), kalimba/bell voices, a canon twin answering the melody an octave down,
a rotating euclidean shaker layer, and 8-bar micro-variations that toggle one
groove element at a time. DJ-style transitions keep the kick running between
songs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radio_core import (
    NOTE_NAMES,
    RNG,
    SR,
    TAIL,
    BarStreamer,
    FfplaySink,
    WavSink,
    bell,
    chip,
    clap,
    duck_curve,
    euclid,
    hat,
    impact,
    kalimba,
    keys,
    kick,
    lead_saw,
    midi_freq,
    pad_note,
    pick,
    place,
    pluck,
    riser,
    shaker,
    snare,
    stab,
    sub_bass,
    t16,
)

SCALES = {
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "harm_minor": (0, 2, 3, 5, 7, 8, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
}
PROGRESSIONS = (
    [0, 0, 3, 4],
    [0, 2, 5, 6],
    [0, 3, 0, 4],
    [0, 3, 5, 6],
    [0, 4, 0, 5],
    [0, 4, 5, 3],
    [0, 5, 0, 6],
    [0, 5, 2, 6],
    [0, 6, 5, 4],
)
CHORD_RHYTHMS = {
    "keys": ([0.0, 2.5], [0.0, 1.75, 2.5], [0.0, 2.0, 3.5]),
    "pluck": ([0.5, 1.5, 2.5, 3.5], [0.5, 1.75, 2.5, 3.75], [0.0, 0.75, 1.5, 2.5, 3.25]),
    "stab": ([0.0, 0.75, 1.5, 2.5, 3.25], [0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 2.75, 3.5], [0.5, 1.5, 2.5, 3.5]),
}
ARP_VOICES = {"chip": chip, "kalimba": kalimba}
LEAD_VOICES = {"bell": bell, "chip": chip, "kalimba": kalimba, "keys": keys, "saw": lead_saw}
VARIATIONS = ("none", "shimmer", "octarp", "canon", "perc")


@dataclass(frozen=True)
class Style:
    name: str
    bpm: tuple[float, float]
    grooves: tuple[str, ...]
    basses: tuple[str, ...]
    chords: tuple[str, ...]
    scales: tuple[str, ...]
    swing: tuple[float, float]
    builds: tuple[str, ...]
    structure: str  # 'peak' (build/drop arcs) or 'groove' (block-based)
    seventh_p: float


STYLES = (
    Style("peak edm", (126, 150), ("four",), ("eights", "offbeat", "rolling16"),
          ("stab",), ("harm_minor", "minor", "phrygian"), (0.0, 0.08),
          ("hats", "riser", "roll"), "peak", 0.25),
    Style("house/disco", (112, 126), ("four",), ("disco", "funk16", "offbeat"),
          ("pluck", "stab"), ("dorian", "major", "minor"), (0.08, 0.25),
          ("hats", "riser"), "groove", 0.7),
    Style("breakbeat", (100, 114), ("breaks",), ("disco", "eights", "funk16"),
          ("keys", "pluck"), ("dorian", "major", "minor"), (0.1, 0.3),
          ("hats", "roll"), "groove", 0.6),
    Style("trance", (136, 144), ("four",), ("eights", "rolling16"),
          ("pad", "stab"), ("harm_minor", "minor"), (0.0, 0.05),
          ("riser", "roll"), "peak", 0.2),
    Style("downtempo", (80, 96), ("halftime",), ("funk16", "halftime"),
          ("keys", "pad"), ("dorian", "minor"), (0.05, 0.2),
          ("riser",), "groove", 0.6),
)


@dataclass
class Sect:
    kind: str  # intro | build | lift | drop | groove | break | trans
    bars: int
    drums: float
    bass: bool
    chords: bool
    arp: bool
    lead: bool
    pad: bool
    slam: bool = False  # impact + open hat at section start
    keyup: bool = False


@dataclass
class Song:
    style: Style
    bpm: float
    root: int
    scale: tuple[int, ...]
    scale_name: str
    prog: list[int]
    seventh: bool
    groove: str
    bass_style: str
    chord_instr: str
    chord_rhythm: list[float]
    build_kind: str
    lead_instr: str
    swing: float
    hat16: bool
    sections: list[Sect]
    kick_snd: np.ndarray
    snare_snd: np.ndarray
    arp_len: int
    arp_instr: str
    canon: bool
    eu_k: int
    motif: list[tuple[int, int, float]] = field(default_factory=list)
    arp: list[tuple[int, int]] = field(default_factory=list)
    arp_key: int = -1
    bass_pat: list[tuple[int, int]] = field(default_factory=list)
    bass_key: int = -1
    var: str = "none"
    var_key: int = -1


def make_motif() -> list[tuple[int, int, float]]:
    """A 2-bar hook: (16th slot, scale index, length multiplier)."""
    slots = sorted(RNG.choice(32, size=int(RNG.integers(5, 9)), replace=False).tolist())
    deg = int(RNG.integers(3, 7))
    out = []
    for s in slots:
        deg = int(np.clip(deg + RNG.integers(-2, 3), 0, 9))
        out.append((int(s), deg, float(pick((1.0, 1.0, 2.0)))))
    return out


def make_sections(style: Style) -> list[Sect]:
    if style.structure == "peak":
        out = []
        for cycle in range(2):
            arp, lead = RNG.uniform() < 0.7, RNG.uniform() < 0.8
            out.append(Sect("build", int(pick((4, 8))), 0.85, True, True, False, False, True))
            out.append(Sect("drop", int(pick((8, 16, 16, 24))), 1.0, True, True, arp, lead, False,
                            slam=True, keyup=cycle == 1))
            out.append(Sect("break", int(pick((4, 8))), 0.0, False, RNG.uniform() < 0.4,
                            RNG.uniform() < 0.6, RNG.uniform() < 0.5, True, slam=True))
        out.pop()  # no break after the final drop
        out.append(Sect("trans", 2, 1.0, True, False, False, False, False))
        return out
    # groove structure: blocks with toggled layers, one breakdown in the middle
    out = []
    for block in range(3):
        if block:
            out.append(Sect("lift", int(pick((2, 4))), 0.7, True, False, False, False, False))
        out.append(Sect("groove", int(pick((8, 12, 16))), 1.0,
                        True, RNG.uniform() < 0.85, RNG.uniform() < 0.35,
                        block % 2 == 1 or RNG.uniform() < 0.4, RNG.uniform() < 0.3,
                        slam=block > 0, keyup=block == 2))
        if block == 1:
            out.append(Sect("break", int(pick((4, 8))), 0.15, RNG.uniform() < 0.5, True,
                            RNG.uniform() < 0.4, RNG.uniform() < 0.7, True))
    out.append(Sect("trans", 2, 1.0, True, False, False, False, False))
    return out


def make_song(first: bool) -> Song:
    style: Style = pick(STYLES)
    scale_name = pick(style.scales)
    prog = list(pick(PROGRESSIONS))
    if RNG.uniform() < 0.35:
        prog = prog + list(pick(PROGRESSIONS))  # 8-chord progression
    chord_instr = pick(style.chords)
    sections = make_sections(style)
    if first:
        sections.insert(0, Sect("intro", 4, 0.0, False, False, False, False, True))
    return Song(
        style=style,
        bpm=float(RNG.uniform(*style.bpm)),
        root=int(RNG.integers(33, 45)),
        scale=SCALES[scale_name],
        scale_name=scale_name,
        prog=prog,
        seventh=RNG.uniform() < style.seventh_p,
        groove=pick(style.grooves),
        bass_style=pick(style.basses),
        chord_instr=chord_instr,
        chord_rhythm=list(pick(CHORD_RHYTHMS.get(chord_instr, CHORD_RHYTHMS["stab"]))),
        build_kind=pick(style.builds),
        lead_instr=pick(("saw", "saw", "chip", "keys", "kalimba", "bell")),
        swing=float(RNG.uniform(*style.swing)),
        hat16=bool(RNG.integers(0, 2)),
        sections=sections,
        kick_snd=kick(f0=float(RNG.uniform(250, 480)), punch=float(RNG.uniform(45, 80)),
                      body_decay=float(RNG.uniform(5.5, 9.0))),
        snare_snd=snare(body_freq=float(RNG.uniform(165, 205))),
        arp_len=int(pick((12, 16, 24))),
        arp_instr=pick(("chip", "kalimba", "kalimba")),
        canon=RNG.uniform() < 0.5,
        eu_k=int(pick((3, 5, 7))),
        motif=make_motif(),
    )


def chord_notes(song: Song, gbar: int) -> list[int]:
    """Diatonic triad on the progression degree, topped with 7th or octave."""
    deg = song.prog[gbar % len(song.prog)]
    sc = song.scale
    notes = []
    for off in (0, 2, 4, 6 if song.seventh else 7):
        idx = deg + off
        notes.append(song.root + sc[idx % 7] + 12 * (idx // 7))
    return notes


def scale_note(song: Song, idx: int, base_oct: int = 36) -> int:
    return song.root + base_oct + song.scale[idx % 7] + 12 * (idx // 7)


def update_variation(song: Song, gbar: int) -> None:
    """Every 8 bars, exactly one groove element toggles."""
    if gbar // 8 == song.var_key:
        return
    song.var_key = gbar // 8
    old = song.var
    song.var = pick(VARIATIONS)
    if song.var != old:
        print(f"   ~ {song.var}")


# ---------------------------------------------------------------- bar layers


def drums_into(buf: np.ndarray, song: Song, sect: Sect, beat: float, gbar: int) -> None:
    g = sect.drums
    if g <= 0:
        return
    sw = song.swing
    backbeat = g > 0.85 and sect.kind in ("drop", "groove", "trans")
    if song.groove == "four":
        for s in range(4):
            place(buf, song.kick_snd, s * beat, 0.95 * g)
        if backbeat:
            place(buf, song.snare_snd, beat, 0.6)
            place(buf, clap(), beat, 0.45)
            place(buf, song.snare_snd, 3 * beat, 0.6)
            place(buf, clap(), 3 * beat, 0.45)
        for s in range(4):
            place(buf, hat(), t16(s * 4 + 2, beat, sw), 0.3 * g, pan=0.25)
        if song.hat16 and g > 0.85:
            for s in range(16):
                place(buf, hat(), t16(s, beat, sw), 0.09, pan=-0.3 if s % 2 else 0.3)
    elif song.groove == "breaks":
        place(buf, song.kick_snd, 0, 0.95 * g)
        place(buf, song.kick_snd, 2.5 * beat, 0.85 * g)
        if RNG.uniform() < 0.4:
            place(buf, song.kick_snd, 1.75 * beat, 0.6 * g)
        if backbeat:
            place(buf, song.snare_snd, beat, 0.65)
            place(buf, song.snare_snd, 3 * beat, 0.65)
        for s in range(8):
            place(buf, hat(), t16(s * 2, beat, sw), (0.3 if s % 2 else 0.2) * g, pan=0.25)
        if gbar % 2 == 1:
            place(buf, hat(open_hat=True), 3.5 * beat, 0.25 * g)
    elif song.groove == "halftime":
        place(buf, song.kick_snd, 0, 0.95 * g)
        if RNG.uniform() < 0.5:
            place(buf, song.kick_snd, 2.25 * beat, 0.6 * g)
        if backbeat:
            place(buf, song.snare_snd, 2 * beat, 0.65)
        for s in range(8):
            place(buf, hat(), t16(s * 2, beat, sw), 0.16 * g, pan=0.2)
        for s in range(16):
            place(buf, shaker(), t16(s, beat, sw), 0.5 * g, pan=-0.35 if s % 2 else 0.1)
    # euclidean shaker bed, rotating one slot per phrase: never the same twice
    if g > 0.85 and song.var != "perc":
        rot = (gbar // 8) % 16
        for s in euclid(song.eu_k, 16, rot):
            place(buf, shaker(), t16(s, beat, sw), 0.32 * g, pan=-0.2)
    if g > 0.85 and song.var == "shimmer":
        for s in range(16):
            if s % 2:
                place(buf, hat(), t16(s, beat, sw), 0.07, pan=0.35)
    # fill into every 8-bar phrase boundary
    if gbar % 8 == 7 and g > 0.5:
        for s in range(8):
            place(buf, song.snare_snd, 2 * beat + t16(s, beat, sw) / 2, (0.18 + 0.04 * s) * g)


def bass_into(layer: np.ndarray, song: Song, chord: list[int], beat: float, gbar: int) -> None:
    root = chord[0]
    st = song.bass_style
    if st == "eights":
        for s in range(8):
            note = root + (12 if s in (3, 7) and RNG.uniform() < 0.35 else 0)
            place(layer, sub_bass(midi_freq(note), beat * 0.48), s * beat / 2, 0.36)
    elif st == "offbeat":
        for s in (1, 3, 5, 7):
            place(layer, sub_bass(midi_freq(root), beat * 0.38), s * beat / 2, 0.4)
    elif st == "disco":
        for s in range(8):
            note = root + (12 if s % 2 else 0)
            place(layer, sub_bass(midi_freq(note), beat * 0.42), s * beat / 2, 0.34)
    elif st == "rolling16":
        for s in range(16):
            place(layer, sub_bass(midi_freq(root), beat * 0.21), t16(s, beat, song.swing), 0.3)
    elif st == "funk16":
        if gbar // 8 != song.bass_key:
            song.bass_key = gbar // 8
            slots = sorted(RNG.choice(16, size=int(RNG.integers(6, 10)), replace=False).tolist())
            song.bass_pat = [(s, int(pick((0, 0, 0, 7, 12)))) for s in slots]
        for s, off in song.bass_pat:
            place(layer, sub_bass(midi_freq(root + off), beat * 0.22), t16(s, beat, song.swing), 0.36)
    elif st == "halftime":
        place(layer, sub_bass(midi_freq(root), beat * 1.6), 0, 0.38)
        place(layer, sub_bass(midi_freq(root), beat * 1.2), 2.5 * beat, 0.34)


def chords_into(song: Song, layer: np.ndarray, chord: list[int], beat: float) -> None:
    instr = {"keys": keys, "pluck": pluck, "stab": stab}.get(song.chord_instr)
    if instr is None:  # 'pad' as the chord instrument
        pad_into(layer, chord, beat, 0.16)
        return
    dur = beat * (1.4 if song.chord_instr == "keys" else 0.6)
    for j, b in enumerate(song.chord_rhythm):
        pan = 0.3 if j % 2 else -0.3
        for note, g in zip(chord, (0.2, 0.2, 0.2, 0.12)):
            place(layer, instr(midi_freq(note + 12), dur), b * beat, g, pan)


def pad_into(layer: np.ndarray, chord: list[int], beat: float, gain: float) -> None:
    pans = (-0.4, 0.4, -0.2, 0.2)
    for note, pan in zip(chord, pans):
        place(layer, pad_note(midi_freq(note + 12), 4 * beat + 0.7), 0, gain, pan)


def arp_into(song: Song, layer: np.ndarray, chord: list[int], beat: float,
             gbar: int, gain: float) -> None:
    """Polymetric arp: an arp_len-step pattern read across 16-step bars."""
    if gbar // 8 != song.arp_key:
        song.arp_key = gbar // 8
        song.arp = [(int(RNG.integers(0, 4)), int(RNG.integers(0, 2)) * 12)
                    for _ in range(song.arp_len)]
    voice = ARP_VOICES[song.arp_instr]
    lift = 12 if song.var == "octarp" else 0
    if song.arp_instr == "kalimba":
        gain *= 0.6  # it rings much longer than the chip
    for s in range(16):
        idx, octave = song.arp[(gbar * 16 + s) % song.arp_len]
        f = midi_freq(chord[idx] + 12 + octave + lift)
        place(layer, voice(f, 0.14), t16(s, beat, song.swing), gain,
              pan=0.5 if s % 2 else -0.5)


def melody_into(song: Song, layer: np.ndarray, beat: float, gbar: int, gain: float) -> None:
    """The 2-bar motif with mutation, plus an octave-down canon twin."""
    half = (gbar % 2) * 16
    voice = LEAD_VOICES[song.lead_instr]
    canon = song.canon or song.var == "canon"
    mutate = gbar % 4 >= 2
    for s, deg, lenmult in song.motif:
        if not half <= s < half + 16:
            continue
        if mutate and RNG.uniform() < 0.25:
            deg = int(np.clip(deg + RNG.integers(-2, 3), 0, 9))
        f = midi_freq(scale_note(song, deg))
        pan = float(RNG.uniform(-0.25, 0.25))
        place(layer, voice(f, beat * 0.4 * lenmult), t16(s - half, beat, song.swing), gain, pan)
        if canon and s - half < 12:
            place(layer, voice(f / 2, beat * 0.4 * lenmult),
                  t16(s - half + 3, beat, song.swing), gain * 0.45, -pan)


def build_fx(buf: np.ndarray, song: Song, beat: float, bi: int, nbars: int) -> None:
    p0, p1 = bi / nbars, (bi + 1) / nbars
    place(buf, riser(p0, p1, 4 * beat), 0, 0.28)
    if song.build_kind == "roll":
        div = (4, 4, 8, 8, 16, 16, 16, 16)[min(int(p0 * 8), 7)]
        for s in range(div):
            place(buf, song.snare_snd, s * 4 * beat / div, 0.2 + 0.4 * p0)
    elif song.build_kind == "hats":
        div = 8 if p0 < 0.5 else 16
        for s in range(div):
            place(buf, hat(), s * 4 * beat / div, 0.12 + 0.3 * p0, pan=0.2)


# ---------------------------------------------------------------- bar render


def render_bar(song: Song, sect: Sect, bi: int, gbar: int) -> np.ndarray:
    beat = 60 / song.bpm
    barlen = int(4 * beat * SR)
    buf = np.zeros((barlen + int(TAIL * SR), 2))
    ducked = np.zeros_like(buf)
    chord = chord_notes(song, gbar)
    update_variation(song, gbar)

    drums_into(buf, song, sect, beat, gbar)
    if sect.bass:
        bass_into(ducked, song, chord, beat, gbar)
    if sect.chords:
        chords_into(song, ducked, chord, beat)
    if sect.pad:
        pad_into(ducked, chord, beat, 0.16)
    if sect.arp:
        arp_into(song, ducked, chord, beat, gbar, 0.11)
    if sect.lead:
        melody_into(song, ducked, beat, gbar, 0.24)
    if sect.kind in ("build", "lift"):
        build_fx(buf, song, beat, bi, sect.bars)
    if sect.kind == "trans":
        for s in range(16):
            place(buf, song.snare_snd, s * beat / 4, 0.2 + 0.35 * (bi * 16 + s) / (sect.bars * 16))
        place(buf, riser(0.4 + 0.6 * bi / sect.bars, 0.4 + 0.6 * (bi + 1) / sect.bars, 4 * beat), 0, 0.3)
    if bi == 0 and sect.slam:
        place(buf, impact(), 0, 0.7)
        place(buf, hat(open_hat=True), 0, 0.55)

    pump = 1.0
    if song.groove == "four" and sect.drums > 0.8:
        depth = 0.78 if song.style.structure == "peak" else 0.5
        pump = duck_curve(len(buf), beat, depth)[:, None]
    buf += ducked * pump
    return buf


# ---------------------------------------------------------------- streaming


def announce(song: Song) -> None:
    key = NOTE_NAMES[song.root % 12]
    print(f"\n>> {song.style.name} | {key} {song.scale_name} | {song.bpm:.0f} BPM"
          f" | bass:{song.bass_style} chords:{song.chord_instr} lead:{song.lead_instr}"
          f" | arp:{song.arp_instr}x{song.arp_len}{' | canon' if song.canon else ''}"
          f" | E({song.eu_k},16)")


def stream(sink: WavSink | FfplaySink, seconds: float | None) -> None:
    bs = BarStreamer(sink)
    first = True
    while True:
        song = make_song(first)
        first = False
        announce(song)
        gbar = 0
        for sect in song.sections:
            if sect.keyup:
                song.root += int(pick((1, 2)))
                print("   key change up")
            print(f"   [{sect.kind}]")
            for bi in range(sect.bars):
                bs.push(render_bar(song, sect, bi, gbar))
                gbar += 1
                if seconds is not None and bs.played >= seconds:
                    return
