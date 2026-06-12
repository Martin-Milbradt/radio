"""Garden channel: endless process music (phase garden).

Melodic cells looping over coprime pulse counts (7/11/13/17/19) so their
combined pattern never realigns - Eno's incommensurate tape loops. The first
cell has a Reich-style phasing twin that drifts one pulse ahead and locks at
each new offset. Pitches come from a modal field walking the brightness axis
(lydian to aeolian), over a tonic-fifth drone and a near-subliminal Shepard
tone that ascends forever and never arrives. Cells wake, sleep, and mutate
one note at a time. There are no drops and no sections: the form is the
interference pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radio_core import (
    NOTE_NAMES,
    RNG,
    SR,
    BarStreamer,
    FfplaySink,
    WavSink,
    bell,
    droplet,
    echo,
    euclid,
    flute,
    kalimba,
    midi_freq,
    pad_chord,
    pick,
    place,
    soft_tick,
    thump,
)

GTAIL = 6.0  # long ring-over: slow tempo plus 2-bar drones need room to decay

# brightness axis, lydian (most major) down to aeolian (most minor)
MODES = (
    ("lydian", (0, 2, 4, 6, 7, 9, 11)),
    ("ionian", (0, 2, 4, 5, 7, 9, 11)),
    ("mixolydian", (0, 2, 4, 5, 7, 9, 10)),
    ("dorian", (0, 2, 3, 5, 7, 9, 10)),
    ("aeolian", (0, 2, 3, 5, 7, 8, 10)),
)
CELL_LENGTHS = (7, 11, 13, 17, 19)
TIMBRES = {"bell": bell, "droplet": droplet, "flute": flute, "kalimba": kalimba}


@dataclass
class Cell:
    name: str
    length: int  # loop length in pulses; coprime lengths never realign
    notes: list[tuple[int, int, float]]  # (pulse slot, scale degree, length in pulses)
    timbre: str
    base_oct: int
    gain: float
    pan: float
    breath: float  # bars per wake/sleep cycle
    breath_ph: float
    awake: bool = True


@dataclass
class Garden:
    bpm: float
    tonic: int  # midi
    mode_i: int
    cells: list[Cell]
    twin_offset: int = 0  # pulses the phasing twin is ahead
    twin_frac: float = 0.0
    twin_locked_bars: int = 12
    shep_x: np.ndarray = field(default_factory=lambda: np.arange(6, dtype=float))
    shep_ph: np.ndarray = field(default_factory=lambda: np.zeros(6))
    perc_on: bool = False
    perc_rot: int = 0


def make_cell(name: str, length: int, timbre: str, base_oct: int, gain: float,
              pan: float) -> Cell:
    n_notes = int(RNG.integers(3, max(4, length // 2 + 1)))
    slots = sorted(RNG.choice(length, size=n_notes, replace=False).tolist())
    deg = int(RNG.integers(3, 7))
    notes = []
    for s in slots:
        deg = int(np.clip(deg + RNG.integers(-2, 3), 0, 9))
        ln = float(pick((2.0, 3.0, 4.0)) if timbre == "flute" else pick((1.0, 1.0, 2.0)))
        notes.append((int(s), deg, ln))
    return Cell(name, length, notes, timbre, base_oct, gain, pan,
                breath=float(RNG.uniform(21, 55)), breath_ph=float(RNG.uniform(0, 6.28)))


def make_garden() -> Garden:
    lengths = list(RNG.permutation(CELL_LENGTHS))
    cells = [
        make_cell("kalimba", lengths[0], "kalimba", 1, 0.12, -0.35),
        make_cell("bell", lengths[1], "bell", 2, 0.07, 0.4),
        make_cell("flute", lengths[2], "flute", 1, 0.07, -0.1),
        make_cell("drops", lengths[3], "droplet", 2, 0.06, 0.15),
    ]
    return Garden(
        bpm=float(RNG.uniform(63, 78)),
        tonic=int(RNG.integers(43, 53)),
        mode_i=int(RNG.integers(0, 3)),  # start somewhere bright
        cells=cells,
    )


def activity(cell: Cell, gbar: int) -> float:
    """Each cell breathes on its own cycle; below the floor it sleeps."""
    a = 0.5 + 0.5 * np.sin(2 * np.pi * gbar / cell.breath + cell.breath_ph)
    return 0.0 if a < 0.25 else float((a - 0.25) / 0.75)


def degree_freq(g: Garden, deg: int, base_oct: int) -> float:
    scale = MODES[g.mode_i][1]
    return midi_freq(g.tonic + 12 * base_oct + scale[deg % 7] + 12 * (deg // 7))


def shepard_into(buf: np.ndarray, g: Garden, n: int) -> None:
    """Six octave-spaced partials glide upward forever under a gaussian window.

    Phase is carried across bars so the bed is seamless; a partial fading out
    at the top is the same partial fading in at the bottom.
    """
    rate = 1 / 95  # octaves per second: one full octave every ~95 s
    for i in range(6):
        x0 = g.shep_x[i]
        xs = (x0 + rate * np.arange(n) / SR) % 6.0
        f = 34.0 * 2**xs
        amp = np.exp(-((xs - 3.0) ** 2) / (2 * 1.1**2))
        phase = g.shep_ph[i] + 2 * np.pi * np.cumsum(f) / SR
        sig = np.sin(phase) * amp * 0.04
        buf[:n, 0] += sig
        buf[:n, 1] += sig
        g.shep_ph[i] = float(phase[-1] % (2 * np.pi))
        g.shep_x[i] = float((x0 + rate * n / SR) % 6.0)


def evolve(g: Garden, gbar: int) -> None:
    """Every 8 bars: nudge the weather, never more than one change at a time."""
    r = RNG.uniform()
    if r < 0.12 and gbar % 32 == 0:
        old_name = MODES[g.mode_i][0]
        g.mode_i = int(np.clip(g.mode_i + pick((-1, 1)), 0, len(MODES) - 1))
        if MODES[g.mode_i][0] != old_name:
            print(f"   mode -> {NOTE_NAMES[g.tonic % 12]} {MODES[g.mode_i][0]}")
    elif r < 0.17 and gbar % 32 == 0:
        g.tonic += int(pick((-5, 7)))
        g.tonic = int(np.clip(g.tonic, 41, 55))
        print(f"   tonic -> {NOTE_NAMES[g.tonic % 12]} {MODES[g.mode_i][0]}")
    elif r < 0.45:
        cell = g.cells[int(RNG.integers(0, len(g.cells)))]
        i = int(RNG.integers(0, len(cell.notes)))
        s, deg, ln = cell.notes[i]
        cell.notes[i] = (s, int(np.clip(deg + pick((-1, 1)), 0, 9)), ln)
        print(f"   {cell.name} shifts a note")
    # percussion drifts in and out on a long arc
    want = np.sin(2 * np.pi * gbar / 89) > -0.1
    if want != g.perc_on:
        g.perc_on = bool(want)
        g.perc_rot = int(RNG.integers(0, 8))
        print(f"   pulse {'rises' if want else 'recedes'}")


# ---------------------------------------------------------------- bar render


def cell_into(buf: np.ndarray, g: Garden, cell: Cell, gbar: int, pulse: float,
              offset_pulses: int = 0, frac: float = 0.0, mirror: bool = False) -> None:
    act = activity(cell, gbar)
    was = cell.awake
    cell.awake = act > 0
    if cell.awake != was and not mirror:
        print(f"   {cell.name} {'wakes' if cell.awake else 'rests'}")
    if act <= 0:
        return
    gpulse = gbar * 8
    fn = TIMBRES[cell.timbre]
    pan = -cell.pan if mirror else cell.pan
    for p in range(8):
        k = (gpulse + p + offset_pulses) % cell.length
        for s, deg, ln in cell.notes:
            if s != k:
                continue
            f = degree_freq(g, deg, cell.base_oct)
            place(buf, fn(f, ln * pulse), (p + frac) * pulse,
                  cell.gain * (0.45 + 0.55 * act), pan)


def render_bar(g: Garden, gbar: int) -> np.ndarray:
    beat = 60 / g.bpm
    bar = 4 * beat
    pulse = beat / 2
    barlen = int(bar * SR)
    buf = np.zeros((barlen + int(GTAIL * SR), 2))
    scale = MODES[g.mode_i][1]

    # drone: tonic, fifth, octave, crossfading every two bars
    if gbar % 2 == 0:
        freqs = tuple(midi_freq(g.tonic + o) for o in (0, 7, 12))
        place(buf, pad_chord(freqs, 2 * bar + 1.5), 0, 0.12)

    shepard_into(buf, g, barlen)

    # the cells; the first one carries its Reich-style phasing twin
    for ci, cell in enumerate(g.cells):
        cell_into(buf, g, cell, gbar, pulse)
        if ci == 0:
            cell_into(buf, g, cell, gbar, pulse, offset_pulses=g.twin_offset,
                      frac=g.twin_frac, mirror=True)

    # twin drift: hold the current offset, then slide one pulse ahead
    if g.twin_locked_bars > 0:
        g.twin_locked_bars -= 1
    else:
        g.twin_frac += 0.125
        if g.twin_frac >= 1.0:
            g.twin_frac = 0.0
            g.twin_offset = (g.twin_offset + 1) % g.cells[0].length
            g.twin_locked_bars = int(RNG.integers(10, 22))
            print(f"   phase {g.twin_offset}/{g.cells[0].length}")

    # soft euclidean pulse, when the weather allows it
    if g.perc_on:
        for p in euclid(3, 8, g.perc_rot):
            place(buf, thump(), p * pulse, 0.22)
        for p in euclid(5, 8, g.perc_rot + 3):
            place(buf, soft_tick(), p * pulse, 0.18, pan=0.3)

    # rain: a stray high drop now and then, echoing
    if RNG.uniform() < 0.18:
        deg = int(pick((0, 2, 4, 7, 9)))
        f = midi_freq(g.tonic + 36 + scale[deg % 7] + 12 * (deg // 7))
        snd = echo(droplet(f, 1.2), 0.21, 0.45, 3)
        place(buf, snd, float(RNG.uniform(0, 3)) * beat, 0.05, pan=float(RNG.uniform(-0.5, 0.5)))

    return buf


# ---------------------------------------------------------------- streaming


def stream(sink: WavSink | FfplaySink, seconds: float | None) -> None:
    g = make_garden()
    lengths = "/".join(str(c.length) for c in g.cells)
    print(f">> phase garden | {NOTE_NAMES[g.tonic % 12]} {MODES[g.mode_i][0]}"
          f" | {g.bpm:.0f} BPM | loops {lengths} | twin on {g.cells[0].length}")
    bs = BarStreamer(sink, tail=GTAIL, drive=1.8)
    gbar = 0
    while True:
        if gbar % 8 == 0 and gbar > 0:
            evolve(g, gbar)
        bs.push(render_bar(g, gbar))
        gbar += 1
        if seconds is not None and bs.played >= seconds:
            return
