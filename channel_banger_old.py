"""Banger channel, superseded v1 (kept as an archive).

The original style-preset playlist: no polymetric arps, no canon twin, no
euclidean shaker layer, no micro-variations. See channel_banger for the
current version.
"""

from __future__ import annotations

import numpy as np

from channel_banger import (
    CHORD_RHYTHMS,
    PROGRESSIONS,
    SCALES,
    STYLES,
    Sect,
    Song,
    Style,
    bass_into,
    build_fx,
    chord_notes,
    chords_into,
    make_motif,
    make_sections,
    pad_into,
    scale_note,
)
from radio_core import (
    NOTE_NAMES,
    RNG,
    SR,
    TAIL,
    BarStreamer,
    FfplaySink,
    WavSink,
    chip,
    clap,
    duck_curve,
    hat,
    impact,
    keys,
    kick,
    lead_saw,
    midi_freq,
    pick,
    place,
    riser,
    shaker,
    snare,
    t16,
)

LEAD_VOICES = {"chip": chip, "keys": keys, "saw": lead_saw}


def make_song(first: bool) -> Song:
    style: Style = pick(STYLES)
    scale_name = pick(style.scales)
    prog = list(pick(PROGRESSIONS))
    if RNG.uniform() < 0.35:
        prog = prog + list(pick(PROGRESSIONS))
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
        lead_instr=pick(("saw", "saw", "chip", "keys")),
        swing=float(RNG.uniform(*style.swing)),
        hat16=bool(RNG.integers(0, 2)),
        sections=sections,
        kick_snd=kick(f0=float(RNG.uniform(250, 480)), punch=float(RNG.uniform(45, 80)),
                      body_decay=float(RNG.uniform(5.5, 9.0))),
        snare_snd=snare(body_freq=float(RNG.uniform(165, 205))),
        arp_len=16,
        arp_instr="chip",
        canon=False,
        eu_k=5,
        motif=make_motif(),
    )


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
    if gbar % 8 == 7 and g > 0.5:
        for s in range(8):
            place(buf, song.snare_snd, 2 * beat + t16(s, beat, sw) / 2, (0.18 + 0.04 * s) * g)


def arp_into(song: Song, layer: np.ndarray, chord: list[int], beat: float,
             gbar: int, gain: float) -> None:
    if gbar // 8 != song.arp_key:
        song.arp_key = gbar // 8
        song.arp = [(int(RNG.integers(0, 4)), int(RNG.integers(0, 2)) * 12) for _ in range(16)]
    for s, (idx, octave) in enumerate(song.arp):
        f = midi_freq(chord[idx] + 12 + octave)
        place(layer, chip(f, 0.14), t16(s, beat, song.swing), gain, pan=0.5 if s % 2 else -0.5)


def melody_into(song: Song, layer: np.ndarray, beat: float, gbar: int, gain: float) -> None:
    half = (gbar % 2) * 16
    voice = LEAD_VOICES[song.lead_instr]
    mutate = gbar % 4 >= 2
    for s, deg, lenmult in song.motif:
        if not half <= s < half + 16:
            continue
        if mutate and RNG.uniform() < 0.25:
            deg = int(np.clip(deg + RNG.integers(-2, 3), 0, 9))
        f = midi_freq(scale_note(song, deg))
        place(layer, voice(f, beat * 0.4 * lenmult), t16(s - half, beat, song.swing),
              gain, pan=float(RNG.uniform(-0.25, 0.25)))


def render_bar(song: Song, sect: Sect, bi: int, gbar: int) -> np.ndarray:
    beat = 60 / song.bpm
    barlen = int(4 * beat * SR)
    buf = np.zeros((barlen + int(TAIL * SR), 2))
    ducked = np.zeros_like(buf)
    chord = chord_notes(song, gbar)

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


def stream(sink: WavSink | FfplaySink, seconds: float | None) -> None:
    bs = BarStreamer(sink)
    first = True
    while True:
        song = make_song(first)
        first = False
        key = NOTE_NAMES[song.root % 12]
        print(f"\n>> {song.style.name} | {key} {song.scale_name} | {song.bpm:.0f} BPM"
              f" | bass:{song.bass_style} chords:{song.chord_instr} lead:{song.lead_instr}")
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
