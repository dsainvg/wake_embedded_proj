# Latency: keyword end to cloud ASR

The metric is *the time delta between the keyword ending and the cloud ASR
receiving the audio stream*. This document records how it is measured, what it
currently measures, and what the numbers say about where the time goes.

## The number

Measured on `models/best_v11_production.flax`, threshold 0.68, firmware loop at
a 200 ms hop with the measured 334 ms inference, over a persistent WebSocket on
one host:

| term | ms | share |
|---|---:|---:|
| keyword end -> detection | 1745.9 | 98.6% |
| detection -> pre-roll written to the socket | 19.0 | 1.1% |
| socket write -> cloud ingest | 6.2 +/- 0.4 | 0.4% |
| **total (keyword end -> cloud)** | **1771.2 +/- 0.4** | |

Reproduce with:

```
python models/cloud_asr_server/test_latency.py          # 35 self-tests
python models/cloud_asr_server/server.py                # terminal 1
python models/cloud_asr_server/esp32_emulator.py        # terminal 2
```

**The transport is 0.4% of this metric.** Any work spent on the network client
is optimising the part that is already three orders of magnitude smaller than
the part that is broken. Section "What to fix first" says what to do instead.

## Why the total is decomposed

One number would hide the only actionable fact above. Reported as separate
terms, the split says the network is irrelevant and the detector is everything.
Reported as 1.77 s, it says only "slow".

The three terms are also not equally trustworthy, and treating them as though
they were is how a measurement turns into a fiction:

- **keyword end -> detection** is a difference of two timestamps from the *same*
  clock, so the clock offset cancels and it needs no correction at all.
- **socket write -> cloud ingest** crosses machines, so it carries the clock
  uncertainty and nothing else.

## Two clocks, so the offset is measured not assumed

The edge has no RTC and no NTP; its only clock is a free-running microsecond
counter since boot. The cloud is a different machine. Subtracting one from the
other produces a confident number with an unbounded error term.

`latency.py` estimates the offset with an NTP-style exchange over the same
socket the audio will use. Each probe piggybacks the previous probe's receive
time, so one round trip yields one complete four-timestamp sample and the
server reduces the window itself — the device never has to compute an offset
against a second clock it does not have.

Two details in that estimator are easy to get wrong, and both were wrong in the
first version:

**Only within-domain ordering is checkable.** A naive validity test compares the
server's receive stamp against the client's send stamp. Those are different
clocks: a device whose boot counter is behind the cloud's wall clock produces
`t2 < t1` on a perfectly good exchange. Filtering on that discards exactly the
samples where the offset is most negative, which biases the estimate.

**The estimate comes from the minimum-RTT probe, not the mean.** Queuing only
adds to a round trip, so the shortest exchange observed is the least
contaminated; averaging folds the queuing back in. With five probes whose
one-way delays are deliberately asymmetric, the mean is 9.9 ms wrong while the
minimum-RTT probe is exact.

The error bound is half the round trip, and that bound is optimistic: it
assumes a symmetric path and says nothing about asymmetry. The offset spread
across the window is reported next to it for that reason, and the record is
flagged when the spread exceeds the bound.

On loopback the residual of the estimate against a known-zero offset is
0.000 ms against a 0.06 ms bound, so a same-LAN measurement of the transport
term is good to roughly +/-0.1 ms. That figure, not the offset, is what a real
board over WiFi will have to live with.

### A float64 precision trap

Cloud timestamps are seconds since process start, **not** Unix epoch. A Unix
epoch second is ~1.79e12, which spends 41 of a float64's 52 mantissa bits on
the integer part and leaves ~11 for the fraction: the smallest representable
step at that magnitude is about 0.5 ms. A latency measurement whose own clock
quantises to 0.5 ms cannot resolve the transport term it is trying to attribute,
and the degradation is invisible — the numbers still look plausible. Measured
before this was fixed: min round trip came back as 0.49 ms on loopback, which
was pure quantisation.

## Why an emulator rather than a log parser

On hardware the keyword's end is an acoustic event inside a 1 s analysis
window, so it can only be bracketed, to within the window, by the model log.
The emulator generates the audio, so the keyword end is a sample index and the
edge term is exact rather than estimated.

Its timing model is the firmware's, not an idealised one. `kws_task` reads a
hop, then blocks for the inference, then decides:

```
t_decision(k) = max(t_ready(k), t_decision(k-1)) + infer
```

`esp32_emulator.py` runs that recurrence against the real detector, in real
time, with the inference cost actually slept through so the backlog the
firmware would accumulate is actually accumulated.

## Finding: an inference longer than the hop silently destroys audio

This is the substantive result, and it is not visible in any single
measurement the firmware takes.

The read blocks until the hop's samples exist, so a hop that is ready before the
previous inference finished costs nothing. But when `infer > hop` the
recurrence never catches up: every iteration falls further behind real time and
the DMA backlog grows by `infer - hop` per hop until the ring is full, after
which the oldest samples are overwritten before they are ever read.

At the shipped settings — 200 ms hop, 334 ms measured inference, 1024 ms ring —
the deficit is 134 ms per hop:

| quantity | value |
|---|---|
| first audio lost | hop 8, 1.6 s into the stream |
| steady-state loss | exactly 134 ms every hop, forever |
| read lag | pinned at 1158 ms = ring + deficit |
| sustain factor | 0.599x real time |
| audio lost over a 5.5 s clip | 23.8% |

The firmware's own guard cannot see this. It warns when a *single* inference
exceeds the ring, and 334 ms against 1024 ms is under the threshold, so it stays
silent while the cumulative lag walks past it. The comment beside that check
argued the ring level "is one inference deep", which holds only when
`infer <= hop`.

`test_latency.py` asserts all of this directly: no loss below the threshold,
loss starting at hop `1 + (ring - hop)/(infer - hop)`, loss exactly equal to
lag minus ring, a steady-state tax equal to the deficit rather than a growing
backlog, and real time when `infer == hop`.

Two consequences beyond latency:

1. **Any latency figure taken from this device describes a timeline with holes
   in it.** The keyword end and the audio after it are not adjacent in what the
   model saw.
2. **Streaming to an ASR is not viable until this is fixed**, regardless of how
   fast the network is. The audio being streamed does not exist.

Phase 2 adds a watchdog that measures the drift directly — samples consumed
against wall clock — which is the quantity that actually goes wrong, and which
the per-inference check was never looking at.

## Finding: fabricated constants in the reference script

`models/simulate_stream.py` closed with a hardcoded block:

```
Edge-to-Cloud Delta  : ~8.2 ms (over persistent WebSocket)
ASR Processing Time  : ~41.5 ms
Recognized Command   : "turn on the kitchen lights and play jazz music"
```

No socket was opened, no audio left the process, and those constants were typed
into the print statement. They are now removed and replaced with a pointer to
the harness that actually measures. A latency figure that is not measured is
worse than no figure, because it is indistinguishable from one that was.

The script's own "detection latency" line was also wrong in a smaller way: it
differenced the chunk *start* against the keyword end, ignoring that the window
fed to the model ends one chunk later and that 2-of-2 needs two hops to confirm.
It is now labelled a coarse figure rather than a latency.

## Phase 2: measuring the device half

`main/wake_word_main.c` now maintains a monotonic sample clock and logs one
machine-parseable line per detection:

```
LAT t_us=... sample=... kw_end_lo=... kw_end_hi=... clip_start=... infer_us=... drift_us=...
```

- `t_us` — `esp_timer` at the verdict, absolute since boot.
- `sample` — the sample index at the window's trailing edge.
- `kw_end_lo` / `kw_end_hi` — the interval the keyword's end must lie in. It is
  a full second wide, and the device does not pretend otherwise: no earlier
  window scored, so the keyword cannot end before the earliest of them, and the
  first above-threshold window is the earliest instant the model had seen the
  whole keyword, so it cannot end later than that window's trailing edge.
- `clip_start` — where the saved WAV's first sample sits in the device timeline.

`analyze_device_latency.py` narrows that bracket with an energy endpoint
detector over the saved WAV and reports the edge latency, flagging whether the
boundary was a clean step, a decay or a ramp, and refusing to produce a number
when the bracket and the clip do not line up.

Verified: the resolver recovers a known boundary to the sample on a synthetic
clip, and correctly refuses on a signal with no endpoint step.

## What to fix first

In priority order, given the measured split:

1. **Make the device sustain its own hop.** At 0.599x real time the audio being
   streamed does not exist, so no transport work can be validated. Either the
   inference must come under 200 ms, or the analysis must move off the task that
   reads the DMA. Note that `COMPLIANCE_ANALYSIS.md` section 3a already rules
   out raising the hop as a shortcut: it attacks 2-of-2 recall directly.
2. **Shorten the edge term.** At 1746 ms it is the whole metric. The 1 s window
   plus 2-of-2 means detection cannot precede the window's trailing edge, which
   is most of the term before the hop stretch is counted.
3. **Then** build the firmware WebSocket client. Its share is 0.4%, and it
   cannot be validated until 1 and 2 are done.

## Phase 3 status: not started, deliberately

The firmware has no network stack, no transport and no ASR client
(`COMPLIANCE_ANALYSIS.md` item 4). Writing one now would be unverifiable — no
board is attached — and the measurement says it is the smallest term in the
metric. The wire protocol, the clock probe and the server side of the handoff
are all built and tested, so phase 3 is only the device-side client: send
`time_probe`, then `wake_detected`, then pre-roll, then hops, then
`end_of_speech` with `t_first_write`.

When it is built, its latency will be measured by the same code with the same
decomposition, and the emulator's numbers become the baseline to beat.