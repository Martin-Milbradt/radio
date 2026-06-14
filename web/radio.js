// Browser front-end for the radio engine.
//
// Pyodide runs the real numpy synthesis (radio_core.py + the channel modules)
// compiled to WebAssembly. Each channel exposes a bars(seconds) generator that
// yields one mastered stereo bar at a time; we pull bars on demand and schedule
// them through the Web Audio API. The pull model is deliberate: GitHub Pages
// can't send the COOP/COEP headers that SharedArrayBuffer needs, so we can't
// pause an infinite Python loop for backpressure. A generator gives us
// backpressure for free.
//
// Each bar crosses the JS boundary as raw float32 bytes, not a numpy view onto
// WASM memory: a later numpy allocation can grow (and detach) the WASM heap,
// which would invalidate any live view mid-copy. Bytes are a native JS buffer.
//
// Switching channels crossfades two "voices" (a longer, beat-overlapping blend
// within a family, a plain equal-power crossfade across families). Each voice
// owns an independent RNG seed, so concurrent voices never share a random
// stream and every start is a fresh, never-before-heard take.

const SR = 48000;
const LOOKAHEAD = 1.2; // seconds of audio kept scheduled ahead of the clock
const TICK_MS = 200; // how often we top up the schedule
const START_DELAY = 0.12; // seconds before a voice's first bar
const CROSSFADE = 2.0; // cross-family crossfade, seconds
const CROSSFADE_FAMILY = 4.5; // same-family blend, seconds (kicks overlap)
const RATE_LO = 0.94; // tempo-nudge clamp (limits pitch shift during a family blend)
const RATE_HI = 1.06;

const CHANNELS = [
  { id: "fable-old", title: "fable", blurb: "one infinite, evolving melodic track" },
  { id: "garden", title: "garden", blurb: "process music: coprime loops, modal drift" },
  { id: "banger-old", title: "banger", blurb: "endless dance-music playlist" },
  { id: "fable", title: "fable experimental", blurb: "fable + canon, extra voices, blips" },
  { id: "banger", title: "banger experimental", blurb: "banger + arps, shaker, canon" },
];

const BUILD = Date.now(); // cache-buster so a stale channel source is never served
const TITLES = Object.fromEntries(CHANNELS.map((ch) => [ch.id, ch.title]));

const statusEl = document.getElementById("status");
const pauseBtn = document.getElementById("pause");
const stopBtn = document.getElementById("stop");
const channelsEl = document.getElementById("channels");

let pyodide = null;
let startVoice = null; // python: (id, name) -> creates a generator with a fresh seed
let nextBar = null; // python: id -> float32 bytes (planar L,R) or None
let endVoice = null; // python: id -> drops the generator

let audioCtx = null;
let voices = []; // active voices; 1 normally, 2 (or more) mid-transition
let paused = false;
let nextVoiceId = 1;
let activeId = null;

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const familyOf = (id) => id.replace(/-old$/, "");

function freshSeed() {
  // 128 bits of browser entropy so every voice seeds independently, regardless
  // of how Pyodide's WASM entropy source behaves
  if (globalThis.crypto && crypto.getRandomValues) {
    return Array.from(crypto.getRandomValues(new Uint32Array(4))).join(",");
  }
  return [Date.now() >>> 0, (Math.random() * 2 ** 32) >>> 0,
    (Math.random() * 2 ** 32) >>> 0, nextVoiceId].join(",");
}

function status(text) {
  statusEl.textContent = text;
}

function buildChannelButtons() {
  for (const ch of CHANNELS) {
    const btn = document.createElement("button");
    btn.dataset.id = ch.id;
    btn.disabled = true;
    btn.className =
      "flex flex-col gap-1 rounded-md border border-neutral-800 bg-neutral-900 p-4 text-left transition hover:border-neutral-600 disabled:cursor-not-allowed disabled:opacity-40";
    btn.innerHTML =
      `<span class="text-sm font-semibold text-neutral-100">${ch.title}</span>` +
      `<span class="text-xs text-neutral-500">${ch.blurb}</span>`;
    btn.addEventListener("click", () => play(ch.id));
    channelsEl.appendChild(btn);
  }
}

function highlightActive() {
  for (const btn of channelsEl.children) {
    const on = btn.dataset.id === activeId;
    btn.classList.toggle("border-emerald-500", on);
    btn.classList.toggle("bg-neutral-800", on);
  }
}

function enableChannels(enabled) {
  for (const btn of channelsEl.children) btn.disabled = !enabled;
}

function updateControls() {
  const active = voices.length > 0;
  pauseBtn.disabled = !active;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  stopBtn.disabled = !active;
}

async function boot() {
  status("loading python runtime...");
  pyodide = await loadPyodide();
  status("loading numpy...");
  await pyodide.loadPackage("numpy");

  status("loading channels...");
  const files = ["radio_core.py"].concat(
    CHANNELS.map((ch) => `channel_${ch.id.replace(/-/g, "_")}.py`),
  );
  for (const file of files) {
    const resp = await fetch(`./${file}?v=${BUILD}`);
    if (!resp.ok) throw new Error(`failed to fetch ${file}: ${resp.status}`);
    pyodide.FS.writeFile(file, await resp.text());
  }

  // The channels print() as they evolve. Pyodide's default stdout is an
  // Emscripten fd device that raises OSError on write in some browsers, so we
  // replace sys.stdout/stderr with a writer that forwards to JS and can never
  // itself raise (a throw on the stderr path turns a normal traceback into a
  // SystemError). With the on-page log gone, those lines go to the console.
  globalThis.radioLog = (line) => console.debug("[radio]", line);

  // Each voice carries its own RNG state. We load it into the shared
  // radio_core.RNG (which the channels and DSP all reference) just for the
  // duration of one bar render, then save it back. Bar rendering is atomic on
  // this single JS thread, so two concurrent voices never corrupt each other's
  // stream, and every voice starts from fresh OS entropy: distinct per channel,
  // fresh on every start.
  pyodide.runPython(`
import sys
import traceback

import js
import importlib

import numpy as np
import radio_core


class Ticker:
    """Line-buffered stdout/stderr that forwards to JS and never raises."""

    def __init__(self):
        self.buf = ""

    def write(self, s):
        try:
            self.buf += s
            while "\\n" in self.buf:
                line, self.buf = self.buf.split("\\n", 1)
                js.radioLog(line)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            if self.buf:
                js.radioLog(self.buf)
                self.buf = ""
        except Exception:
            pass

    def isatty(self):
        return False


sys.stdout = Ticker()
sys.stderr = Ticker()


def report_exception(exc_type, exc, tb):
    try:
        js.radioLog("".join(traceback.format_exception(exc_type, exc, tb)))
    except Exception:
        pass


sys.excepthook = report_exception

voices = {}


def start_voice(vid, name, seed):
    rng = np.random.default_rng([int(x) for x in seed.split(",")])
    voices[vid] = {
        "gen": importlib.import_module("channel_" + name.replace("-", "_")).bars(None),
        "state": rng.bit_generator.state,
    }


def next_bar(vid):
    v = voices.get(vid)
    if v is None:
        return None
    radio_core.RNG.bit_generator.state = v["state"]
    out = next(v["gen"], None)
    v["state"] = radio_core.RNG.bit_generator.state
    if out is None:
        voices.pop(vid, None)
        return None
    # planar float32 [L0..Ln, R0..Rn] as raw bytes: a native JS buffer, decoupled
    # from WASM memory so heap growth can never detach it mid-copy
    return np.ascontiguousarray(out.T.astype(np.float32)).tobytes()


def end_voice(vid):
    voices.pop(vid, None)
`);
  startVoice = pyodide.globals.get("start_voice");
  nextBar = pyodide.globals.get("next_bar");
  endVoice = pyodide.globals.get("end_voice");

  enableChannels(true);
  status("ready - pick a channel");
}

function toAudioBuffer(u8) {
  // u8: planar float32 bytes [left..., right...]. Build at 48 kHz (the data's
  // real rate); if the context runs at another rate the source node resamples
  // on playback, which is correct and preserves duration.
  const f32 = new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
  const n = f32.length / 2;
  const ab = audioCtx.createBuffer(2, n, SR);
  ab.copyToChannel(f32.subarray(0, n), 0);
  ab.copyToChannel(f32.subarray(n, 2 * n), 1);
  return ab;
}

function fadeCurve(into) {
  // equal-power cosine: into=true ramps 0->1, false ramps 1->0
  const N = 33;
  const c = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = i / (N - 1);
    c[i] = into ? Math.sin((x * Math.PI) / 2) : Math.cos((x * Math.PI) / 2);
  }
  return c;
}

function ensureContext() {
  if (audioCtx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) throw new Error("Web Audio API is not available in this browser");
  try {
    audioCtx = new AC({ sampleRate: SR });
  } catch {
    audioCtx = new AC();
  }
}

function makeVoice(name) {
  const id = nextVoiceId++;
  startVoice(id, name, freshSeed());
  const gain = audioCtx.createGain();
  gain.connect(audioCtx.destination);
  const v = {
    id,
    name,
    gain,
    nextTime: audioCtx.currentTime + START_DELAY,
    timer: null,
    rate: 1,
    barDur: 0,
    matchOld: null, // old voice whose tempo this one nudges toward during a blend
    blendUntil: 0,
  };
  voices.push(v);
  return v;
}

function pumpVoice(v) {
  if (paused || !audioCtx || audioCtx.state !== "running" || !voices.includes(v)) return;
  // a fading-out voice stops generating once its blend window closes; its
  // already-scheduled (now near-silent) tail plays out and reapVoice clears it
  if (v.stopPumpAt && audioCtx.currentTime >= v.stopPumpAt) return;
  try {
    while (v.nextTime < audioCtx.currentTime + LOOKAHEAD) {
      const res = nextBar(v.id);
      if (!res) {
        teardownVoice(v); // bounded generator ended (never happens for endless channels)
        return;
      }
      let u8 = res;
      if (typeof res.toJs === "function") {
        u8 = res.toJs();
        res.destroy();
      }
      const ab = toAudioBuffer(u8);
      const natBarDur = ab.duration; // duration at playbackRate 1
      // tempo nudge: during a family blend, bend this voice toward the outgoing
      // tempo so the beats lock, then settle back to its own tempo
      if (v.matchOld && v.matchOld.barDur && audioCtx.currentTime < v.blendUntil) {
        v.rate = clamp(v.matchOld.barDur / natBarDur, RATE_LO, RATE_HI);
      } else if (v.matchOld) {
        v.rate = 1;
        v.matchOld = null;
      }
      v.barDur = natBarDur;
      const src = audioCtx.createBufferSource();
      src.buffer = ab;
      src.playbackRate.value = v.rate;
      src.connect(v.gain);
      v.nextTime = Math.max(v.nextTime, audioCtx.currentTime + 0.005); // never in the past
      src.start(v.nextTime);
      v.nextTime += natBarDur / v.rate;
    }
  } catch (err) {
    console.error(err);
    status(`playback error: ${err.message}`);
    teardownVoice(v);
    return;
  }
  v.timer = setTimeout(() => pumpVoice(v), TICK_MS);
}

function teardownVoice(v) {
  if (v.timer) {
    clearTimeout(v.timer);
    v.timer = null;
  }
  endVoice(v.id);
  try {
    v.gain.disconnect();
  } catch {
    /* already disconnected */
  }
  voices = voices.filter((x) => x !== v);
  updateControls();
}

function reapVoice(v) {
  // tear down a faded-out voice once the fade has finished in CONTEXT time
  // (not wall-clock), so a pause mid-transition holds it instead of cutting it
  if (!voices.includes(v)) return;
  if (audioCtx && audioCtx.currentTime >= v.fadeOutAt) {
    teardownVoice(v);
    return;
  }
  setTimeout(() => reapVoice(v), 120);
}

async function play(name) {
  if (name === activeId) {
    if (paused) pauseResume(); // clicking the live channel while paused resumes it
    return; // otherwise a no-op; Stop then a channel rerolls the seed
  }
  try {
    ensureContext();
    if (audioCtx.state !== "running") {
      try {
        await audioCtx.resume();
      } catch (e) {
        console.error(e);
      }
      for (let i = 0; i < 10 && audioCtx.state !== "running"; i++) {
        await new Promise((r) => setTimeout(r, 10));
      }
    }
    if (audioCtx.state !== "running") {
      status("audio is blocked by the browser - click a channel again");
      return;
    }
    paused = false;

    const now = audioCtx.currentTime;
    const old = voices.slice();
    const sameFamily = old.length > 0 && old.every((o) => familyOf(o.name) === familyOf(name));
    const dur = old.length === 0 ? 0 : sameFamily ? CROSSFADE_FAMILY : CROSSFADE;

    const nv = makeVoice(name);
    if (dur > 0) {
      nv.gain.gain.setValueCurveAtTime(fadeCurve(true), now, dur);
      if (sameFamily) {
        nv.matchOld = old[0];
        nv.blendUntil = now + dur;
      }
    } else {
      nv.gain.gain.setValueAtTime(1, now);
    }

    for (const ov of old) {
      try {
        ov.gain.gain.cancelScheduledValues(now);
      } catch {
        /* nothing scheduled */
      }
      ov.gain.gain.setValueCurveAtTime(fadeCurve(false), now, dur);
      ov.stopPumpAt = now + dur; // keep its groove running through the blend
      ov.fadeOutAt = now + dur + 0.05;
      reapVoice(ov);
    }

    activeId = name;
    highlightActive();
    updateControls();
    status(`playing ${TITLES[name]}`);
    pumpVoice(nv);
    for (const ov of old) pumpVoice(ov);
  } catch (err) {
    console.error(err);
    status(`could not start ${TITLES[name] || name}: ${err.message}`);
  }
}

async function pauseResume() {
  if (!audioCtx || voices.length === 0) return;
  if (paused) {
    paused = false;
    try {
      await audioCtx.resume();
    } catch (e) {
      console.error(e);
    }
    for (let i = 0; i < 10 && audioCtx.state !== "running"; i++) {
      await new Promise((r) => setTimeout(r, 10));
    }
    for (const v of voices) pumpVoice(v);
    updateControls();
    status(`playing ${TITLES[activeId]}`);
  } else {
    paused = true;
    audioCtx.suspend();
    for (const v of voices) {
      if (v.timer) {
        clearTimeout(v.timer);
        v.timer = null;
      }
    }
    updateControls();
    status("paused");
  }
}

function stop() {
  paused = false;
  for (const v of voices.slice()) teardownVoice(v);
  voices = [];
  if (audioCtx) {
    audioCtx.close(); // hard cut: stops every scheduled bar immediately
    audioCtx = null;
  }
  activeId = null;
  highlightActive();
  updateControls();
  if (pyodide) status("stopped - pick a channel");
}

pauseBtn.addEventListener("click", pauseResume);
stopBtn.addEventListener("click", stop);

buildChannelButtons();
boot().catch((err) => {
  console.error(err);
  status(`failed to start: ${err.message}`);
});
