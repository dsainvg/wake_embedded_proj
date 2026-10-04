"""
Edge/cloud clock reconciliation and latency decomposition
---------------------------------------------------------
The deployment metric is: *the time delta between the keyword ending and the
cloud ASR receiving the audio stream*. Two things have to be true before that
delta is a number rather than a guess.

**1. The two clocks have to be related.** The edge has no RTC and no NTP: its
only clock is a free-running microsecond counter since boot. The cloud's clock
is a different machine. Subtracting one from the other without measuring their
relationship produces a confident number with an unbounded error term, so the
offset is measured with an NTP-style probe exchange over the same socket the
audio will use. `ClockModel` holds the estimator and reports the error bound,
which is the round-trip delay -- that bound, not the offset, is what decides
whether a millisecond-scale claim is defensible.

**2. The delta has to be decomposed.** The total is dominated by the device,
not the network:

    L_total = (keyword end -> detection)      device, window + confirmation
            + (detection -> socket write)     device, task scheduling
            + (network + cloud ingest)        transport

The first term is where the time actually is: a 1 s analysis window plus a
2-of-2 peak-hold means detection cannot fire until the window's trailing edge
has passed the keyword, and an inference that outruns the hop stretches the
period on top of that. Reporting one number hides which of the three is worth
attacking, so every term is reported separately.

Units are float seconds throughout. The device reports integer microseconds and
is converted once, at the edge of this module.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional

# A probe exchange costs one round trip, so the window is kept short: the whole
# point is to track the current path, not the average path since boot. 32 is
# roughly a second of 30 Hz probing, long enough to have a usable minimum-RTT
# sample and short enough to follow a change in the network.
DEFAULT_PROBE_WINDOW = 32


@dataclass(frozen=True)
class Probe:
    """One completed NTP-style exchange. All four timestamps are seconds."""

    seq: int
    t1: float          # client sends
    t2: float          # server receives
    t3: float          # server sends
    t4: float          # client receives

    @property
    def offset(self) -> float:
        """Cristian's estimator: edge clock + offset = cloud clock."""
        return ((self.t2 - self.t1) + (self.t3 - self.t4)) / 2.0

    @property
    def round_trip(self) -> float:
        return (self.t4 - self.t1) - (self.t3 - self.t2)

    def is_usable(self) -> bool:
        """
        Reject the exchanges a clock can report but not mean.

        Only *within-domain* ordering is checked. The two halves of an exchange
        live in different clock domains -- one is a free-running counter since
        boot, the other is a wall clock -- so comparing a client stamp against a
        server stamp says nothing about correctness: a device whose counter is
        behind the cloud's wall clock produces t2 < t1 on a perfectly good
        exchange, and a naive check would discard exactly the samples where the
        offset is most negative.
        """
        return (
            self.t4 >= self.t1              # client clock monotonic across the exchange
            and self.t2 <= self.t3          # server clock monotonic across its own work
            and self.round_trip > 0.0
        )


@dataclass(frozen=True)
class ClockEstimate:
    """Result of reducing the probe window to one offset plus its error bound."""

    offset: float                     # add to an edge timestamp to get cloud time
    delay: float                      # best (minimum) round-trip observed
    theta_spread: float               # spread of offsets across the window
    samples: int
    n_usable: int

    @property
    def uncertainty(self) -> float:
        """
        Error on a latency derived from this offset.

        A one-way delay is only half the round trip if the path is symmetric;
        anything asymmetric lands in the measurement instead. This is the
        classical bound and it is optimistic -- it says nothing about path
        asymmetry, only about how much of the round trip is unaccounted for.
        """
        return self.delay / 2.0


class ClockModel:
    """
    Rolling window of probes, reduced to a single offset.

    The estimate is taken from the **minimum-RTT** probe rather than the mean.
    Queuing delay only ever adds time to a round trip, so the shortest exchange
    observed is the one least contaminated by it, and averaging would fold the
    queuing back in. The spread across the window is reported next to the
    estimate because a large spread is the signature of a path that is not
    symmetric, and a large spread means the bound below is optimistic.
    """

    def __init__(self, window: int = DEFAULT_PROBE_WINDOW) -> None:
        self._window: Deque[Probe] = deque(maxlen=window)
        self._rejected = 0

    def add(self, seq: int, t1: float, t2: float, t3: float, t4: float) -> Probe:
        probe = Probe(seq=seq, t1=t1, t2=t2, t3=t3, t4=t4)
        if probe.is_usable():
            self._window.append(probe)
        else:
            self._rejected += 1
        return probe

    @property
    def rejected(self) -> int:
        return self._rejected

    @property
    def samples(self) -> List[Probe]:
        return list(self._window)

    def estimate(self) -> Optional[ClockEstimate]:
        usable = [p for p in self._window if p.is_usable()]
        if not usable:
            return None
        best = min(usable, key=lambda p: p.round_trip)
        thetas = [p.offset for p in usable]
        return ClockEstimate(
            offset=best.offset,
            delay=best.round_trip,
            theta_spread=max(thetas) - min(thetas),
            samples=len(self._window),
            n_usable=len(usable),
        )

    def to_cloud(self, t_edge: float, est: ClockEstimate) -> float:
        return t_edge + est.offset


@dataclass
class LatencyBreakdown:
    """
    One turn's worth of latency, with every term kept separate.

    Device-side terms (detect vs keyword end, and write vs detect) are
    differences of two timestamps from the *same* clock, so the offset cancels
    and they need no correction. Only the terms that cross to the cloud are
    corrected, and only those carry the clock uncertainty.
    """

    # --- device clock, seconds ---
    keyword_end: float          # when the mic captured the keyword's last sample
    detect: float               # when the detector's decision was made
    cloud_first_audio: float    # when the server's receive() returned the first PCM frame

    first_byte_write: Optional[float] = None   # when the pre-roll was written to the socket
    cloud_wake_event: Optional[float] = None   # when the server saw the wake metadata

    # --- clock reconciliation ---
    offset: float = 0.0
    uncertainty: float = 0.0
    theta_spread: float = 0.0
    probe_samples: int = 0

    # --- sample domain (absolute sample indices on the device) ---
    keyword_end_sample: int = -1
    stream_start_sample: int = -1
    preroll_samples: int = 0
    samples_dropped: int = 0

    warnings: List[str] = field(default_factory=list)

    # -- derived terms, in milliseconds -----------------------------------

    @property
    def edge_detect_ms(self) -> float:
        """Keyword end -> detection decision. Same clock, so no correction."""
        return (self.detect - self.keyword_end) * 1000.0

    @property
    def detect_write_ms(self) -> Optional[float]:
        if self.first_byte_write is None:
            return None
        return (self.first_byte_write - self.detect) * 1000.0

    @property
    def network_ms(self) -> Optional[float]:
        if self.first_byte_write is None:
            return None
        corrected = self.first_byte_write + self.offset
        return (self.cloud_first_audio - corrected) * 1000.0

    @property
    def metadata_ms(self) -> Optional[float]:
        """Wake metadata -> first PCM frame, measured purely in the cloud clock."""
        if self.cloud_wake_event is None:
            return None
        return (self.cloud_first_audio - self.cloud_wake_event) * 1000.0

    @property
    def total_ms(self) -> float:
        """The metric: keyword end -> cloud ASR holding the audio that follows it."""
        corrected = self.keyword_end + self.offset
        return (self.cloud_first_audio - corrected) * 1000.0

    def validate(self) -> List[str]:
        """
        Check the measurement is coherent before anyone quotes it.

        Every entry here is a reason the number would be wrong rather than
        imprecise, so they are collected and reported rather than raised: one
        turn's warnings should not stop the rest of a run from being recorded.
        """
        problems = list(self.warnings)

        if self.first_byte_write is not None and self.first_byte_write < self.detect:
            problems.append("first byte written before the detection decision")

        if self.keyword_end_sample >= 0 and self.stream_start_sample >= 0:
            if self.stream_start_sample > self.keyword_end_sample:
                problems.append(
                    "stream starts after the keyword end: the ASR never receives "
                    f"the keyword itself ({self.stream_start_sample - self.keyword_end_sample} samples missed)")
            elif self.preroll_samples > 0:
                offset_in_clip = self.keyword_end_sample - self.stream_start_sample
                if offset_in_clip >= self.preroll_samples:
                    problems.append(
                        f"keyword end is {offset_in_clip} samples into the pre-roll but "
                        f"pre-roll is only {self.preroll_samples}: the ingested audio "
                        "does not contain the keyword")

        if self.samples_dropped > 0:
            problems.append(
                f"{self.samples_dropped} samples were dropped by the device: the stream "
                "has a discontinuity and its audio timeline is not the room's")

        if self.probe_samples == 0:
            problems.append(
                "no clock probe completed: edge and cloud clocks were never related, "
                "so total_ms is not a measurement")

        if self.probe_samples and self.uncertainty * 1000.0 > abs(self.total_ms) * 0.1:
            problems.append(
                f"clock uncertainty {self.uncertainty * 1000.0:.2f} ms is more than 10% "
                f"of total_ms ({self.total_ms:.2f} ms)")

        if self.theta_spread * 1000.0 > max(self.uncertainty * 1000.0, 1.0):
            problems.append(
                f"offset spread {self.theta_spread * 1000.0:.2f} ms across "
                f"{self.probe_samples} probes exceeds the "
                f"{self.uncertainty * 1000.0:.2f} ms bound: the path is not symmetric, "
                "so the round-trip/2 bound is optimistic")

        return problems

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.update(
            edge_detect_ms=round(self.edge_detect_ms, 3),
            detect_write_ms=None if self.detect_write_ms is None else round(self.detect_write_ms, 3),
            network_ms=None if self.network_ms is None else round(self.network_ms, 3),
            metadata_ms=None if self.metadata_ms is None else round(self.metadata_ms, 3),
            total_ms=round(self.total_ms, 3),
            uncertainty_ms=round(self.uncertainty * 1000.0, 3),
        )
        out["problems"] = self.validate()
        return out

    def format_table(self) -> str:
        """Human-readable decomposition, widest term first."""
        terms: List[tuple] = [
            ("keyword end -> detection", self.edge_detect_ms, ""),
        ]
        if self.detect_write_ms is not None:
            terms.append(("detection -> socket write", self.detect_write_ms, ""))
        if self.network_ms is not None:
            terms.append(("socket write -> cloud ingest", self.network_ms, f" +/-{self.uncertainty * 1000.0:.2f}"))
        else:
            terms.append(("socket write -> cloud ingest", float("nan"),
                          " (no device write timestamp: not reported)"))
        terms.sort(key=lambda t: (math.isnan(t[1]), -t[1]))

        lines = ["  latency decomposition", "  " + "-" * 58]
        for name, ms, note in terms:
            shown = "     n/a" if math.isnan(ms) else f"{ms:8.2f} ms"
            lines.append(f"  {name:<32} {shown}{note}")
        lines.append("  " + "-" * 58)
        lines.append(f"  {'TOTAL (keyword end -> cloud)':<32} {self.total_ms:8.2f} ms"
                     f" +/-{self.uncertainty * 1000.0:.2f}")
        return "\n".join(lines)


def build_breakdown(
    *,
    keyword_end: float,
    detect: float,
    cloud_first_audio: float,
    first_byte_write: Optional[float] = None,
    cloud_wake_event: Optional[float] = None,
    estimate: Optional[ClockEstimate] = None,
    keyword_end_sample: int = -1,
    stream_start_sample: int = -1,
    preroll_samples: int = 0,
    samples_dropped: int = 0,
    warnings: Optional[List[str]] = None,
) -> LatencyBreakdown:
    """Assemble a breakdown, filling the clock fields from an estimate if given."""
    est = estimate or ClockEstimate(offset=0.0, delay=0.0, theta_spread=0.0, samples=0, n_usable=0)
    return LatencyBreakdown(
        keyword_end=keyword_end,
        detect=detect,
        first_byte_write=first_byte_write,
        cloud_first_audio=cloud_first_audio,
        cloud_wake_event=cloud_wake_event,
        offset=est.offset,
        uncertainty=est.uncertainty,
        theta_spread=est.theta_spread,
        probe_samples=est.n_usable,
        keyword_end_sample=keyword_end_sample,
        stream_start_sample=stream_start_sample,
        preroll_samples=preroll_samples,
        samples_dropped=samples_dropped,
        warnings=list(warnings or []),
    )