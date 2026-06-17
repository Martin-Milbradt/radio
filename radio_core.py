"""Shared engine for all radio channels: DSP helpers, instruments, sinks.

Everything here is pure numpy synthesis. Oscillators are band-limited
(polyBLEP saws, tanh-softened squares), percussion noise is band-passed with
no near-Nyquist junk, every envelope ends in a declick fade, and the
BarStreamer carries a ring-over tail between bars so notes decay across bar
lines while memory stays flat. Channels own all music logic; this module owns
how things sound and how they reach the speakers.

Requires ffmpeg 7+ (ffplay with -ch_layout) in PATH for live playback.
"""

from __future__ import annotations

import wave

import numpy as np

SR = 48000
TAIL = 3.0  # default seconds of ring-over carried between bars
RNG = np.random.default_rng()
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# ---------------------------------------------------------------- helpers


def midi_freq(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def semi(freq: float, n: float) -> float:
    return freq * 2 ** (n / 12)


def trange(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def pick(seq: tuple | list):  # noqa: ANN201 - heterogeneous pools
    return seq[int(RNG.integers(0, len(seq)))]


def euclid(k: int, n: int, rot: int = 0) -> set[int]:
    """k onsets spread as evenly as possible over n slots."""
    return {(round(i * n / k) + rot) % n for i in range(k)}


def lowpass(snd: np.ndarray, k: int) -> np.ndarray:
    """Triangular-kernel FIR: steeper rolloff and far less leakage than a boxcar."""
    if k <= 1:
        return snd
    kern = np.convolve(np.ones(k), np.ones(k))
    return np.convolve(snd, kern / kern.sum(), mode="same")


def bp_noise(n: int, lo_k: int, hi_k: int) -> np.ndarray:
    """Unit-RMS band-passed noise: difference of two triangle-FIR lowpasses.

    Smooth spectrum with no near-Nyquist junk. Smaller k = higher cutoff, so
    lo_k sets the top of the band, hi_k the bottom. Generated with padding so
    convolution edge effects never reach the output.
    """
    pad = hi_k * 2
    nz = RNG.uniform(-1, 1, n + 2 * pad)
    out = lowpass(nz, lo_k) - lowpass(nz, hi_k)
    return out[pad : pad + n] / max(out.std(), 1e-9)


def declick(snd: np.ndarray, fade: float = 0.004) -> np.ndarray:
    """Ramp the last few ms to zero so truncated envelopes can't click."""
    n = min(len(snd), int(fade * SR))
    snd[-n:] *= np.linspace(1, 0, n)
    return snd


def edge_fade(snd: np.ndarray, fade: float = 0.008) -> np.ndarray:
    n = min(len(snd) // 2, int(fade * SR))
    snd[:n] *= np.linspace(0, 1, n)
    snd[-n:] *= np.linspace(1, 0, n)
    return snd


def polyblep(ph: np.ndarray, dt: float) -> np.ndarray:
    """Band-limited saw from a phase ramp: smooths the wrap discontinuity."""
    out = 2 * ph - 1
    m = ph < dt
    v = ph[m] / dt
    out[m] -= v + v - v * v - 1
    m = ph > 1 - dt
    v = (ph[m] - 1) / dt
    out[m] -= v * v + v + v + 1
    return out


def saw(freq: float, n: int) -> np.ndarray:
    dt = freq / SR
    ph = (RNG.uniform() + np.arange(n) * dt) % 1.0
    return polyblep(ph, dt)


def place(
    buf: np.ndarray, snd: np.ndarray, t: float, gain: float, pan: float = 0.0
) -> None:
    start = int(t * SR)
    if start >= len(buf) or start < 0:
        return
    end = min(start + len(snd), len(buf))
    seg = snd[: end - start]
    buf[start:end, 0] += seg * gain * min(1.0, 1.0 - pan)
    buf[start:end, 1] += seg * gain * min(1.0, 1.0 + pan)


def t16(slot: int, beat: float, swing: float) -> float:
    """Time of a 16th slot, with swing delaying the off-16ths."""
    return (slot + (swing if slot % 2 else 0.0)) * beat / 4


def duck_curve(n: int, beat: float, depth: float) -> np.ndarray:
    """Sidechain pump: 8ms dip at every beat, recovers over ~0.28s."""
    phase = (np.arange(n) / SR) % beat
    dip = np.clip(phase / 0.008, 0, 1)
    recovery = np.clip((phase - 0.008) / 0.28, 0, 1) ** 1.4
    return 1 - depth * dip * (1 - recovery)


def echo(
    snd: np.ndarray, delay: float = 0.18, decay: float = 0.45, taps: int = 3
) -> np.ndarray:
    pad = int(delay * SR * taps)
    out = np.concatenate([snd, np.zeros(pad)])
    for i in range(1, taps + 1):
        off = int(delay * SR * i)
        out[off : off + len(snd)] += snd * decay**i
    return out


# ---------------------------------------------------------------- percussion


def kick(f0: float = 350.0, punch: float = 60.0, body_decay: float = 7.0) -> np.ndarray:
    """Club kick: the fast extra sweep term gives the attack punch, no noise click."""
    t = trange(0.42)
    f = f0 * np.exp(-t * punch) + 150 * np.exp(-t * 20) + 45
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * body_decay)
    return declick(body)


def kick_soft() -> np.ndarray:
    """The fable kick: rounder, 42 Hz tail."""
    t = trange(0.35)
    f = 120 * np.exp(-t * 18) + 42
    return declick(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9))


def thump() -> np.ndarray:
    """Felt-piano-ish low tap."""
    t = trange(0.5)
    f = 90 * np.exp(-t * 12) + 48
    return declick(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7))


def hat(open_hat: bool = False, bright: int = 2) -> np.ndarray:
    dur = 0.4 if open_hat else 0.06
    t = trange(dur)
    env = np.exp(-t * (14 if open_hat else 90)) * np.clip(t / 0.0015, 0, 1)
    return declick(bp_noise(len(t), 3 if open_hat else bright, 8) * env * 0.32)


def snare(body_freq: float = 185.0) -> np.ndarray:
    t = trange(0.22)
    attack = np.clip(t / 0.0015, 0, 1)
    nz = bp_noise(len(t), 5, 16) * np.exp(-t * 22) * attack
    body = np.sin(2 * np.pi * body_freq * t) * np.exp(-t * 28)
    return declick(nz * 0.34 + body * 0.7)


def clap() -> np.ndarray:
    n = int(0.3 * SR)
    out = np.zeros(n)
    for off in (0.0, 0.012, 0.026):
        i = int(off * SR)
        tt = np.arange(n - i) / SR
        out[i:] += (
            bp_noise(n - i, 4, 14) * np.exp(-tt * 60) * np.clip(tt / 0.0015, 0, 1)
        )
    out += bp_noise(n, 4, 14) * np.exp(-np.arange(n) / SR * 14) * 0.4
    return declick(out * 0.26)


def shaker() -> np.ndarray:
    t = trange(0.07)
    env = np.sin(np.pi * np.clip(t / 0.07, 0, 1)) ** 2
    return declick(bp_noise(len(t), 4, 9) * env * 0.2)


def tick() -> np.ndarray:
    t = trange(0.04)
    env = np.exp(-t * 40) * np.clip(t / 0.001, 0, 1)
    return declick(bp_noise(len(t), 3, 10) * env * 0.4)


def soft_tick() -> np.ndarray:
    t = trange(0.05)
    env = np.exp(-t * 50) * np.clip(t / 0.0015, 0, 1)
    return declick(bp_noise(len(t), 4, 12) * env * 0.3)


# ---------------------------------------------------------------- pitched


def sub_bass(freq: float, dur: float) -> np.ndarray:
    t = trange(dur)
    env = np.clip(t * 60, 0, 1) * np.clip((dur - t) * 18, 0, 1)
    return np.tanh(np.sin(2 * np.pi * freq * t) * 2.2) * env


def saw_note(freq: float, dur: float, crush: bool = False) -> np.ndarray:
    """The fable bass voice; crush is its tanh distortion moment."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = saw(freq, n) * np.minimum(1, t * 200) * np.exp(-t * 6)
    if crush:
        out = np.tanh(out * 4.0)
    return declick(out)


def supersaw(
    freq: float, dur: float, voices: int, detune: float, cutoff: float
) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    spread = np.linspace(-1, 1, voices) if voices > 1 else np.zeros(1)
    for d in spread:
        out += saw(freq * (1 + detune * d), n)
    return lowpass(out / voices, max(1, int(SR / cutoff)))


def stab(freq: float, dur: float) -> np.ndarray:
    t = trange(dur)
    env = np.clip(t * 300, 0, 1) * np.exp(-t * 7.5)
    return declick(supersaw(freq, dur, voices=5, detune=0.012, cutoff=5200) * env)


def pluck(freq: float, dur: float) -> np.ndarray:
    """Rounder, shorter chord hit: house-piano stand-in."""
    t = trange(dur)
    env = np.clip(t * 600, 0, 1) * np.exp(-t * 9)
    return declick(supersaw(freq, dur, voices=2, detune=0.004, cutoff=2400) * env)


def keys(freq: float, dur: float) -> np.ndarray:
    """Soft EP-ish tone: sine plus gentle harmonics, slow decay."""
    t = trange(dur)
    out = (
        np.sin(2 * np.pi * freq * t)
        + 0.4 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.12 * np.sin(2 * np.pi * 3 * freq * t)
    )
    env = np.clip(t * 250, 0, 1) * np.exp(-t * 3.0)
    return declick(out / 1.5 * env)


def pad_note(freq: float, dur: float) -> np.ndarray:
    """Sustains through the bar; the long release crossfades into the next chord."""
    t = trange(dur)
    env = np.clip(t / 0.35, 0, 1) * np.clip((dur - t) / 0.7, 0, 1)
    return supersaw(freq, dur, voices=3, detune=0.008, cutoff=2600) * env


def pad_chord(freqs: tuple[float, ...], dur: float) -> np.ndarray:
    """The fable pad: vibrato sines with a faint octave shimmer."""
    t = trange(dur)
    out = np.zeros_like(t)
    for k, f in enumerate(freqs):
        vib = 0.35 * np.sin(2 * np.pi * (0.23 + 0.07 * k) * t)
        out += np.sin(2 * np.pi * f * t + vib)
        out += 0.18 * np.sin(2 * np.pi * f * 2 * t + vib)
    env = np.clip(np.minimum(t / 1.6, (dur - t) / 2.2), 0, 1)
    return out / (len(freqs) * 1.2) * env


def chip(freq: float, dur: float) -> np.ndarray:
    t = trange(dur)
    env = np.clip(t * 400, 0, 1) * np.exp(-t * 11)
    soft_square = np.tanh(np.sin(2 * np.pi * freq * t) * 3.5)
    return declick(lowpass(soft_square * env, 6))


def lead_saw(freq: float, dur: float) -> np.ndarray:
    t = trange(dur)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip(t * 3, 0, 1)
    out = np.zeros_like(t)
    dt = freq / SR
    for d in (-0.006, 0.006):
        ph = (RNG.uniform() + np.cumsum(freq * (1 + d) * vib) / SR) % 1.0
        out += polyblep(ph, dt)
    env = np.clip(t * 200, 0, 1) * np.clip((dur - t) * 30, 0, 1)
    return lowpass(out / 2, 8) * env


def melody_note(freq: float, dur: float) -> np.ndarray:
    """The fable melody voice: vibrato sine with octave shimmer."""
    t = trange(dur)
    vib = 0.3 * np.sin(2 * np.pi * 4.7 * t) * np.clip(t * 2, 0, 1)
    out = np.sin(2 * np.pi * freq * t + vib) + 0.18 * np.sin(
        2 * np.pi * 2 * freq * t + vib
    )
    env = np.clip(t / 0.012, 0, 1) * np.exp(-t * 2.2)
    return declick(out / 1.2 * env)


def kalimba(freq: float, dur: float) -> np.ndarray:
    """FM pluck with a slightly inharmonic modulator."""
    t = trange(max(dur, 1.0))
    mod = np.sin(2 * np.pi * freq * 2.41 * t) * 1.7 * np.exp(-t * 8)
    out = np.sin(2 * np.pi * freq * t + mod)
    env = np.clip(t / 0.003, 0, 1) * np.exp(-t * 4.2)
    return declick(out * env)


def bell(freq: float, dur: float) -> np.ndarray:
    t = trange(max(dur, 2.4))
    out = np.zeros_like(t)
    for ratio, amp, dec in (
        (1.0, 1.0, 1.9),
        (2.0, 0.45, 2.6),
        (2.756, 0.3, 3.4),
        (4.07, 0.12, 5.0),
    ):
        out += amp * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t * dec)
    env = np.clip(t / 0.004, 0, 1)
    return declick(out / 1.5 * env)


def flute(freq: float, dur: float) -> np.ndarray:
    t = trange(dur + 0.15)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 4.6 * t) * np.clip(t * 2, 0, 1)
    phase = 2 * np.pi * np.cumsum(freq * vib) / SR
    tone = np.tanh(np.sin(phase) * 2.2)
    breath = bp_noise(len(t), 3, 10) * 0.05
    env = np.clip(t / 0.06, 0, 1) * np.clip((t[-1] - t) / 0.12, 0, 1)
    return lowpass((tone + breath) * env, 8)


def droplet(freq: float, dur: float) -> np.ndarray:
    t = trange(max(dur, 1.6))
    out = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    env = np.clip(t / 0.002, 0, 1) * np.exp(-t * 3)
    return declick(out / 1.3 * env)


# ---------------------------------------------------------------- fx


def riser(p0: float, p1: float, dur: float) -> np.ndarray:
    t = trange(dur)
    p = p0 + (p1 - p0) * t / dur
    nz = bp_noise(len(t), 2, 12) * (0.15 + 0.85 * p**2)
    f = 180 + 1400 * p**2
    swp = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.4 * p
    return edge_fade(nz * 0.3 + swp)


def rise_fx(dur: float) -> np.ndarray:
    t = trange(dur)
    p = t / dur
    nz = bp_noise(len(t), 3, 14) * p**2 * 0.35
    return edge_fade(sweep(160, 720, dur) * 0.5 * p + nz)


def impact() -> np.ndarray:
    t = trange(1.2)
    f = 300 * np.exp(-t * 6) + 38
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.5)
    nz = bp_noise(len(t), 5, 18) * np.exp(-t * 8) * np.clip(t / 0.002, 0, 1)
    return declick(boom + nz * 0.25)


def sweep(f0: float, f1: float, dur: float) -> np.ndarray:
    t = trange(dur)
    f = f0 + (f1 - f0) * t / dur
    env = np.sin(np.pi * np.minimum(t / dur, 1)) ** 0.5
    return edge_fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * env)


def blip(freq: float, dur: float = 0.18) -> np.ndarray:
    t = trange(dur)
    out = np.tanh(np.sin(2 * np.pi * freq * t) * 6) * np.clip(t * 400, 0, 1)
    return declick(out * np.exp(-t * 8))


# ---------------------------------------------------------------- streaming


class WavSink:
    def __init__(self, path: str) -> None:
        self.w = wave.open(path, "wb")
        self.w.setnchannels(2)
        self.w.setsampwidth(2)
        self.w.setframerate(SR)

    def write(self, data: bytes) -> None:
        self.w.writeframes(data)

    def close(self) -> None:
        self.w.close()


class FfplaySink:
    def __init__(self) -> None:
        import shutil  # lazy: keeps radio_core importable under Pyodide, which lacks subprocess
        import subprocess

        exe = shutil.which("ffplay")
        if exe is None:
            raise RuntimeError("ffplay not found in PATH (it ships with ffmpeg)")
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nodisp",
            "-autoexit",
            "-f",
            "s16le",
            "-ar",
            str(SR),
            "-ch_layout",
            "stereo",
            "-i",
            "-",
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            self.proc.wait(timeout=0.6)
        except subprocess.TimeoutExpired:
            print(f"playing via {exe}")
        else:
            raise RuntimeError(
                f"ffplay at {exe} exited immediately: it predates -ch_layout. "
                "Update to ffmpeg 7+ or fix PATH order so the new build wins."
            )
        assert self.proc.stdin is not None

    def write(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.wait()


class BarStreamer:
    """Carries the ring-over tail between bars, soft-clips, writes PCM.

    Channels render one bar at a time into a buffer of (bar + tail) samples;
    push() overlaps the previous tail, masters the bar portion, keeps the new
    tail, and tracks how many seconds have been emitted.
    """

    def __init__(
        self,
        sink: WavSink | FfplaySink | None = None,
        tail: float = TAIL,
        drive: float = 1.1,
    ) -> None:
        self.sink = sink
        self.carry = np.zeros((int(tail * SR), 2))
        self.drive = drive
        self.played = 0.0

    def process(self, buf: np.ndarray) -> np.ndarray:
        """Overlap the previous tail, master the bar, keep the new tail; return the bar."""
        buf[: len(self.carry)] += self.carry
        barlen = len(buf) - len(self.carry)
        out = np.tanh(buf[:barlen] * self.drive) * 0.95
        self.carry = buf[barlen:].copy()
        self.played += barlen / SR
        return out

    def push(self, buf: np.ndarray) -> np.ndarray:
        out = self.process(buf)
        if self.sink is not None:
            self.sink.write((out * 32767).astype(np.int16).tobytes())
        return out
