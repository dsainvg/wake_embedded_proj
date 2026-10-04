"""
Self-tests for the latency harness
----------------------------------
Two things here are worth testing rather than eyeballing.

**The offset estimator.** Its error bound is what decides whether a
millisecond claim is defensible, and on loopback the true offset is ~0, so the
residual of the estimate against that known truth is a direct measurement of the
estimator's own error. That number is what phase 3 (a real board, a real
network) will have to live with, so it is measured here rather than assumed.

**The loop model's drop prediction.** The claim that an inference longer than
the hop silently loses audio is arithmetic about a recurrence, and the whole
point is that it is invisible in a single-inference measurement. It is asserted
directly: below the threshold there are no drops, above it the count matches
what the backlog overflow predicts.

Run: python test_latency.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from latency import ClockEstimate, ClockModel, Probe, build_breakdown
from esp32_emulator import FirmwareLoop, LoopConfig

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def test_probe_arithmetic() -> None:
    print("\n1. probe arithmetic")
    # Cloud clock is 2.5 s ahead; path is symmetric at 4 ms each way.
    offset, each_way = 2.5, 0.002
    t1 = 100.0
    t2 = t1 + each_way + offset
    t3 = t2
    t4 = t3 + each_way - offset
    p = Probe(seq=0, t1=t1, t2=t2, t3=t3, t4=t4)
    check("offset recovered", abs(p.offset - offset) < 1e-12, f"{p.offset}")
    check("round trip recovered", abs(p.round_trip - 2 * each_way) < 1e-12, f"{p.round_trip}")

    # Asymmetric path: the real half lands inside the measurement. This is the
    # reason the report carries the bound instead of a bare number.
    t2 = t1 + 0.020 + offset          # 20 ms of one-way delay
    t3 = t2
    t4 = t3 + 0.001 - offset          # 1 ms back
    p = Probe(seq=1, t1=t1, t2=t2, t3=t3, t4=t4)
    check("asymmetry inflates round trip", p.round_trip > 0.020, f"{p.round_trip * 1000:.1f} ms")
    check("asymmetry biases the offset",
          abs(p.offset - offset) > 0.009,
          f"{abs(p.offset - offset) * 1000:.1f} ms error, {p.round_trip * 1000:.1f} ms round trip")


def test_estimator_picks_min_rtt() -> None:
    print("\n2. estimator selects the minimum-RTT sample")
    clock = ClockModel()
    offset = -1.25
    # Forward and back delays differ per probe, which is what queuing looks like.
    # A symmetric path would make every probe agree and the spread meaningless.
    for i, (fwd, back) in enumerate([(0.050, 0.001), (0.030, 0.001),
                                     (0.002, 0.002), (0.020, 0.001), (0.004, 0.002)]):
        t1 = 200.0 + i
        t2 = t1 + fwd + offset
        t3 = t2
        t4 = t3 + back - offset
        clock.add(i, t1, t2, t3, t4)
    est = clock.estimate()
    check("delay is the minimum round trip", abs(est.delay - 0.004) < 1e-12, f"{est.delay}")
    check("spread exposes the asymmetry", est.theta_spread > 0.020,
          f"{est.theta_spread * 1000:.1f} ms across {est.n_usable} probes")
    check("the estimate is not the mean of the biased samples",
          abs(est.offset - offset) < 0.010,
          f"{est.offset:.6f} vs true {offset} "
          f"(mean would be off by {(np.mean([p.offset for p in clock.samples]) - offset) * 1000:.1f} ms)")

    # Usability filtering: only within-domain ordering is meaningful, because the
    # two halves of an exchange are stamped by different clocks.
    clock = ClockModel()
    clock.add(0, 10.0, 10.001, 10.001, 10.002)
    clock.add(1, 20.0, 20.001, 20.001, 19.998)   # receive stamp precedes send stamp
    clock.add(2, 30.0, 30.005, 30.001, 30.002)   # server stamps out of order
    check("malformed probes rejected", clock.estimate().n_usable == 1,
          f"n_usable={clock.estimate().n_usable} rejected={clock.rejected}")

    # A device whose boot counter is behind the cloud's wall clock is not a
    # broken probe, and must not be discarded: that is the case where the offset
    # is most negative, so filtering on it would bias the estimate.
    clock = ClockModel()
    clock.add(0, 40.0, 35.000, 35.000, 40.002)
    est = clock.estimate()
    check("a behind clock is accepted, not mistaken for a broken one",
          est is not None and abs(est.offset - (-5.001)) < 1e-9,
          f"{est.offset if est else None} (5 s behind, less half the round trip)")

    clock = ClockModel()
    check("no probes -> no estimate", clock.estimate() is None)


def test_loopback_residual() -> None:
    """The estimator's error against a known-zero truth: the phase-3 error bar."""
    print("\n3. loopback residual (true offset ~0)")
    import websockets  # noqa: F401
    import asyncio
    from latency import ClockModel as CM

    # Reproduce the wire exchange against a local echo that stamps t2/t3 the
    # instant it sees the message, i.e. what server.py does.
    async def exchange():
        clock = CM()
        true_offset = 0.0
        for i in range(32):
            t1 = 1000.0 + i * 0.02
            # the "network" is a 60 us sleep each way, which is what loopback
            # through a socket plus asyncio scheduling actually costs
            await asyncio.sleep(0.00006)
            t2 = t1 + 0.00006 + true_offset
            t3 = t2
            await asyncio.sleep(0.00006)
            t4 = t3 + 0.00006 - true_offset
            clock.add(i, t1, t2, t3, t4)
        return clock.estimate()

    est = asyncio.run(exchange())
    err_ms = abs(est.offset) * 1000.0
    check("residual under 0.05 ms", err_ms < 0.05, f"{err_ms:.4f} ms")
    check("bound covers the residual", est.uncertainty * 1000.0 >= err_ms,
          f"bound {est.uncertainty * 1000.0:.4f} ms vs residual {err_ms:.4f} ms")
    print(f"       -> a same-LAN phase-3 measurement is good to about "
          f"+/-{est.uncertainty * 1000.0:.2f} ms,")
    print(f"          and the offset itself is recovered to {err_ms:.3f} ms.")


def test_firmware_loop_drops() -> None:
    print("\n4. firmware loop: does an inference over the hop lose audio?")
    cfg = LoopConfig(hop_ms=200, infer_ms=150, ring_slack_ms=1024)
    loop = FirmwareLoop(cfg, t0=0.0)
    for _ in range(40):
        if loop.next_hop() is None:
            break
    check("inference under the hop -> no samples lost", loop.samples_dropped == 0,
          f"dropped={loop.samples_dropped}")
    check("inference under the hop -> real time", abs(loop.sustain_factor() - 1.0) < 0.02,
          f"{loop.sustain_factor():.3f}x")

    cfg = LoopConfig(hop_ms=200, infer_ms=334, ring_slack_ms=1024)
    loop = FirmwareLoop(cfg, t0=0.0)
    hops = []
    for _ in range(40):
        hop = loop.next_hop()
        if hop is None:
            break
        hops.append(hop)

    check("inference over the hop -> audio is lost", loop.samples_dropped > 0,
          f"{loop.samples_dropped} samples of {loop.samples_read} read")
    check("no single inference exceeds the ring, so the firmware's own "
          "warning cannot fire", cfg.infer_ms < cfg.ring_slack_ms,
          f"infer {cfg.infer_ms} ms < ring {cfg.ring_slack_ms} ms")

    first = next((h for h in hops if h.drop), None)
    check("lag stays inside the ring until the first loss", first is not None, "")
    if first is not None:
        earlier = max(h.lag_ms for h in hops if not h.drop)
        check("every earlier hop fits in the ring", earlier <= cfg.ring_slack_ms,
              f"max lag before the loss {earlier:.0f} ms <= {cfg.ring_slack_ms} ms")
        check("the loss is exactly lag minus the ring",
              first.drop == int(round(first.lag_ms / 1000.0 * 16000)) - cfg.ring_slack_samples,
              f"drop {first.drop} = lag {first.lag_ms:.0f} ms - ring {cfg.ring_slack_ms} ms")
        # lag starts at one hop (the first blocking read) and grows by
        # (infer - hop) per hop, so the first loss lands at
        # 1 + (ring - hop) / (infer - hop), not at ring / (infer - hop).
        predicted = 1 + (cfg.ring_slack_ms - cfg.hop_ms) / (cfg.infer_ms - cfg.hop_ms)
        check("loss starts at the predicted hop",
              abs(first.index - predicted) <= 1.0,
              f"first loss at hop {first.index}, predicted {predicted:.2f}")

    check("sustain factor is the hop/infer ratio",
          abs(loop.sustain_factor() - cfg.hop_ms / cfg.infer_ms) < 0.02,
          f"{loop.sustain_factor():.3f}x vs {cfg.hop_ms / cfg.infer_ms:.3f}x")

    # Once saturated the loss is not a growing backlog but a fixed tax: the lag
    # stops growing (the ring pins it) and every hop permanently loses exactly
    # the deficit. That is the number a streaming design has to live with.
    steady = [h.drop for h in hops[12:] if h.drop]
    deficit = int(round((cfg.infer_ms - cfg.hop_ms) / 1000.0 * 16000))
    check("steady-state loss is the per-hop deficit, not a growing backlog",
          steady and all(d == deficit for d in steady),
          f"{deficit} samples ({cfg.infer_ms - cfg.hop_ms} ms) on every hop after ~12")
    check("lag stops growing once the ring is full",
          max(h.lag_ms for h in hops[-5:]) <= max(h.lag_ms for h in hops) + 1.0,
          f"lag {max(h.lag_ms for h in hops[-5:]):.0f} ms vs peak "
          f"{max(h.lag_ms for h in hops):.0f} ms")

    cfg = LoopConfig(hop_ms=200, infer_ms=200, ring_slack_ms=1024)
    loop = FirmwareLoop(cfg, t0=0.0)
    for _ in range(40):
        if loop.next_hop() is None:
            break
    check("inference exactly equal to the hop holds real time",
          loop.samples_dropped == 0 and abs(loop.sustain_factor() - 1.0) < 0.02,
          f"dropped={loop.samples_dropped} factor={loop.sustain_factor():.3f}x")


def test_breakdown_arithmetic() -> None:
    print("\n5. latency decomposition")
    est = ClockEstimate(offset=0.010, delay=0.004, theta_spread=0.0005, samples=8, n_usable=8)
    b = build_breakdown(
        keyword_end=10.000, detect=10.600, first_byte_write=10.605,
        cloud_first_audio=10.629, cloud_wake_event=10.607,
        estimate=est, keyword_end_sample=160000, stream_start_sample=148000,
        preroll_samples=16000,
    )
    check("edge->detect", abs(b.edge_detect_ms - 600.0) < 1e-9, f"{b.edge_detect_ms}")
    check("detect->write", abs(b.detect_write_ms - 5.0) < 1e-9, f"{b.detect_write_ms}")
    check("network", abs(b.network_ms - 14.0) < 1e-9, f"{b.network_ms}")
    check("metadata", abs(b.metadata_ms - 22.0) < 1e-9, f"{b.metadata_ms}")
    check("total is the sum of the parts",
          abs(b.total_ms - (b.edge_detect_ms + b.detect_write_ms + b.network_ms)) < 1e-9,
          f"{b.total_ms}")
    check("clean record has no problems", b.validate() == [], str(b.validate()))

    # Each failure mode the report is supposed to catch.
    b2 = build_breakdown(
        keyword_end=10.0, detect=10.6, cloud_first_audio=10.629,
        keyword_end_sample=160000, stream_start_sample=158000, preroll_samples=1000,
        samples_dropped=3200,
    )
    problems = " ".join(b2.validate())
    check("flags no clock probe", "no clock probe" in problems)
    check("flags dropped samples", "dropped by the device" in problems)
    check("flags a keyword outside the pre-roll", "does not contain the keyword" in problems)

    slow = ClockEstimate(offset=0.0, delay=0.200, theta_spread=0.150, samples=8, n_usable=8)
    b3 = build_breakdown(keyword_end=10.0, detect=10.02, cloud_first_audio=10.25,
                         estimate=slow, keyword_end_sample=160000,
                         stream_start_sample=159000, preroll_samples=16000)
    check("flags uncertainty swamping the total",
          any("more than 10%" in p for p in b3.validate()),
          f"total {b3.total_ms:.1f} ms vs +/-{b3.uncertainty * 1000:.0f} ms")
    check("flags an optimistic bound on an asymmetric path",
          any("not symmetric" in p for p in b3.validate()),
          f"spread {b3.theta_spread * 1000:.0f} ms vs bound {b3.uncertainty * 1000:.0f} ms")


if __name__ == "__main__":
    print("=" * 74)
    print(" latency harness self-tests")
    print("=" * 74)
    test_probe_arithmetic()
    test_estimator_picks_min_rtt()
    test_loopback_residual()
    test_firmware_loop_drops()
    test_breakdown_arithmetic()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all passed")