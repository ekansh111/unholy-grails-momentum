"""Point-in-time index membership from a Clenow-style change-row file.

File format (one row per change date; each row is the FULL constituent list as
of that date):

    date,tickers
    1996-01-02,"AAL,AAMRQ,AAPL,ABI,..."

Membership on any date D = the tickers of the latest row at-or-before D
(forward-fill). This matches the Clenow project's `constituentsOn(D)` exactly,
giving survivorship-controlled point-in-time universes.
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import date, datetime


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


@dataclass
class Universe:
    dates: list[date]            # ascending change-row dates
    member_sets: list[frozenset] # member_sets[i] = members as of dates[i]

    @classmethod
    def from_file(cls, path: str, drop_sentinels: list[str] | None = None) -> "Universe":
        drop = set(drop_sentinels or [])
        dates: list[date] = []
        member_sets: list[frozenset] = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                d = _parse_date(row["date"])
                tickers = [t.strip() for t in row["tickers"].split(",") if t.strip()]
                tickers = [t for t in tickers if t not in drop]
                dates.append(d)
                member_sets.append(frozenset(tickers))
        # Ensure ascending (the files are, but be defensive).
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        dates = [dates[i] for i in order]
        member_sets = [member_sets[i] for i in order]
        return cls(dates, member_sets)

    def members_on(self, d: date) -> frozenset:
        """Members as of date d (latest change row with date <= d).

        Returns empty set for dates before the first change row.
        """
        idx = bisect.bisect_right(self.dates, d) - 1
        if idx < 0:
            return frozenset()
        return self.member_sets[idx]

    def all_tickers_ever(self) -> set[str]:
        out: set[str] = set()
        for s in self.member_sets:
            out |= s
        return out

    def first_date(self) -> date:
        return self.dates[0]
