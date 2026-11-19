"""Download chosen hours of An Open Book (Zenodo record 18184441) into data/raw/.

The record holds one archive per month and stream, tens of gigabytes each, so the
script takes single hourly files out of them without downloading the archives:

- book diffs and trades are plain tar: the script walks the 512-byte member
  headers with byte-range requests and fetches only the wanted members;
- order statuses are tar.xz, which cannot be entered in the middle: the archive is
  streamed from its start and decompressed until the wanted day has passed. Members
  are in chronological order, so an early day of the month costs little, a late one
  costs everything before it.

Files already present are skipped, so an interrupted run is simply restarted.
Downloads go through /api/records/<id>/files/<name>/content: the /records/... URL
starts returning a 92-byte error page after a few heavy downloads (journal, 23
August 2026).

    python scripts/fetch_data.py --date 20251201 --hours 0-23 --coins btc sol eth
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import lzma
import shutil
import ssl
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _report import hour_list

from hypermm import config
from hypermm.ingest import book_diffs, orders, trades

RECORD = "https://zenodo.org/api/records/18184441/files"
BLOCK = 512
TIMEOUT = 180
RETRIES = 8
RESUMES = 60  # Zenodo drops long transfers often; each resume costs nothing
# python.org builds on macOS ship without CA certificates; the system bundle works
CA_FILE = "/etc/ssl/cert.pem"
SSL = ssl.create_default_context(cafile=CA_FILE if Path(CA_FILE).exists() else None)


def retrying(task, *args):
    """Run `task` again on a network failure, whether at connecting or mid-transfer.

    Zenodo answers 500 now and then and sometimes drops a response half-way.
    """
    for attempt in range(RETRIES):
        try:
            return task(*args)
        except (
            OSError,
            EOFError,
            http.client.HTTPException,
            tarfile.TarError,
            lzma.LZMAError,
        ) as error:
            if attempt == RETRIES - 1:
                raise
            print(f"  retry {attempt + 1} of {task.__name__} after {error!r}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    raise AssertionError


def open_url(name: str, start: int | None = None, end: int | None = None):
    request = urllib.request.Request(
        f"{RECORD}/{name}/content", headers={"User-Agent": "hypermm-fetch"}
    )
    if start is not None:
        request.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    response = urllib.request.urlopen(request, timeout=TIMEOUT, context=SSL)
    if start is not None and response.status != 206:
        response.close()
        raise OSError(f"{name}: range request answered {response.status}")
    return response


def _read_range(name: str, start: int, length: int) -> bytes:
    with open_url(name, start, start + length - 1) as response:
        data = response.read()
    if len(data) != length:
        raise OSError(f"{name}: asked {length} bytes at {start}, got {len(data)}")
    return data


def read_range(name: str, start: int, length: int) -> bytes:
    return retrying(_read_range, name, start, length)


def save(stream, size: int, target: Path) -> None:
    """Write `size` bytes from `stream` to `target` through a .part file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    with open(part, "wb") as out:
        shutil.copyfileobj(stream, out, 1 << 20)
    if part.stat().st_size != size:
        raise OSError(f"{target}: expected {size} bytes, wrote {part.stat().st_size}")
    part.rename(target)
    print(f"  {target.relative_to(config.REPO_ROOT)}  {size / 1e6:.1f} MB", flush=True)


def header(block: bytes) -> tuple[str, int, bytes]:
    name = block[0:100].split(b"\0", 1)[0].decode()
    prefix = block[345:500].split(b"\0", 1)[0].decode()
    size = int(block[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
    return (f"{prefix}/{name}" if prefix else name), size, block[156:157]


def plain_tar_jobs(archive: str, date: str, wanted: dict[str, Path], index: Path) -> list[tuple]:
    """Locate members of a plain tar by walking its headers with range requests.

    Walking stops at the first member of another date once the wanted date has been
    seen: members of one date are stored together. Every header read is kept in
    `index`, so a later run resumes the walk where this one stopped instead of
    paying again for requests that take Zenodo seconds each. Returns one download
    job per missing member.
    """
    missing = {member: path for member, path in wanted.items() if not path.exists()}
    known = json.loads(index.read_text()) if index.exists() else {}
    walked = known.setdefault(archive, {"members": {}, "next": 0, "end": False})
    members = walked["members"]
    seen = any(name.split("/", 1)[0] == date for name in members)
    passed = seen and any(name.split("/", 1)[0] > date for name in members)
    offset, long_name = walked["next"], None
    while not walked["end"] and not passed and not set(missing) <= set(members):
        block = read_range(archive, offset, BLOCK)
        if block == bytes(BLOCK):
            walked["end"] = True
            break
        name, size, kind = header(block)
        data_start, offset = offset + BLOCK, offset + BLOCK + -(-size // BLOCK) * BLOCK
        if kind == b"L":  # GNU long name: the data is the next member's name
            long_name = read_range(archive, data_start, size).split(b"\0", 1)[0].decode()
            continue
        if kind in (b"x", b"g"):  # pax headers: nothing we need
            continue
        name, long_name = long_name or name, None
        members[name] = [data_start, size]
        walked["next"] = offset
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(known, indent=1))
        if name.split("/", 1)[0] == date:
            seen = True
        elif seen:
            passed = True
    jobs = [(archive, *members[m], path) for m, path in missing.items() if m in members]
    if len(jobs) < len(missing):
        raise SystemExit(f"{archive}: not found: {sorted(set(missing) - set(members))}")
    print(f"{archive}: {len(jobs)} files located", flush=True)
    return jobs


def _fetch_member(archive: str, start: int, size: int, target: Path) -> None:
    with open_url(archive, start, start + size - 1) as response:
        save(response, size, target)


def fetch_member(archive: str, start: int, size: int, target: Path) -> None:
    retrying(_fetch_member, archive, start, size, target)


def fetch_xz_tar(archive: str, date: str, wanted: dict[str, Path]) -> None:
    """Stream a chronologically ordered tar.xz from its start until `date` has passed.

    A dropped stream starts over from the archive's start, skipping what is saved.
    """
    retrying(_fetch_xz_tar, archive, date, wanted)


class Resuming(io.RawIOBase):
    """A download that, when the connection drops, reopens at the byte it had reached.

    Zenodo cuts long transfers off after a while; a tar.xz cannot be entered in the
    middle, so without this every drop meant starting the archive over.
    """

    def __init__(self, name: str):
        self.name, self.position, self.response = name, 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        for attempt in range(RESUMES):
            try:
                if self.response is None:
                    self.response = open_url(self.name, self.position or None)
                n = self.response.readinto(buffer)
                if n == 0 and len(buffer):
                    raise OSError("connection closed early")  # the archive is far longer
                self.position += n
                return n
            except (OSError, http.client.HTTPException) as error:
                if self.response is not None:
                    self.response.close()
                self.response = None
                print(
                    f"  {self.name}: resuming at {self.position:,} after {error!r}", file=sys.stderr
                )
                time.sleep(min(60, 5 * (attempt + 1)))
        raise OSError(f"{self.name}: gave up at byte {self.position:,}")

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
        super().close()


def _fetch_xz_tar(archive: str, date: str, wanted: dict[str, Path]) -> None:
    missing = {member: path for member, path in wanted.items() if not path.exists()}
    if not missing:
        return
    print(f"{archive}: {len(missing)} files, streaming from the start", flush=True)
    stream = io.BufferedReader(Resuming(archive), buffer_size=1 << 20)
    with stream, tarfile.open(fileobj=stream, mode="r|xz") as tar:
        for member in tar:
            member_date = member.name.split("/", 1)[0]
            if member_date > date or not missing:
                break
            if member.name in missing:
                save(tar.extractfile(member), member.size, missing.pop(member.name))
    if missing:
        raise SystemExit(f"{archive}: not found: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201", help="YYYYMMDD, December 2025")
    parser.add_argument(
        "--hours", type=hour_list, default=hour_list("0-23"), help="e.g. 0-23 or 10,12-14"
    )
    parser.add_argument("--coins", nargs="+", default=["btc", "sol"])
    parser.add_argument("--jobs", type=int, default=6, help="parallel downloads")
    parser.add_argument(
        "--rejected", action="store_true", help="also the rejected-order files (large)"
    )
    args = parser.parse_args()
    date, month = args.date, args.date[:6]
    index = args.raw / "zenodo_index.json"  # tar headers already walked

    def member(path: Path, root: Path) -> str:
        return path.relative_to(root).as_posix()

    with ThreadPoolExecutor(args.jobs) as pool:
        futures = []
        # the order streams need no index and are the longest: they start at once
        root = args.raw / "order_statuses"
        for coin in args.coins:
            for rejected in (False, True) if args.rejected else (False,):
                paths = [
                    orders.hour_path(root, date, h, coin, rejected=rejected) for h in args.hours
                ]
                kind = "rejected" if rejected else "orders"
                archive = f"{coin.lower()}_{kind}_{month}.tar.xz"
                wanted = {member(p, root): p for p in paths}
                futures.append(pool.submit(fetch_xz_tar, archive, date, wanted))

        # meanwhile the plain archives are indexed, each file queued once located
        for stream, reader, archive in (
            ("book_diffs", book_diffs, f"book_diffs_{month}.tar"),
            ("trades", trades, f"trades_{month[:4]}_{month[4:]}.tar"),
        ):
            root = args.raw / stream
            wanted = {
                member(p, root): p for p in (reader.hour_path(root, date, h) for h in args.hours)
            }
            for job in plain_tar_jobs(archive, date, wanted, index):
                futures.append(pool.submit(fetch_member, *job))

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
