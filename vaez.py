#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any, cast
from collections import Counter
import struct
import io
import zlib
import zipfile
import re
import time
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ---------------- small stable mixer (no hashlib) ----------------
MASK64 = 0xFFFFFFFFFFFFFFFF
PHI64 = 0x9E3779B97F4A7C15


def mix64(x: int) -> int:
    x = (x + PHI64) & MASK64
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= (x >> 31)
    return x & MASK64


def s_fingerprint(s: str) -> int:
    h = 0
    for b in s.encode('utf-8'):
        h = mix64((h + b) & MASK64)
    return h


def s_complement(s: str) -> int: return (~s_fingerprint(s)) & MASK64

# ---------------- color phases (quick diagnostic) ----------------


def counts_to_color(counts: Counter[str]) -> Tuple[float, float, float]:
    bad = counts.get('ℋ', 0)+counts.get('⊘', 0)+counts.get('∞', 0)
    good = counts.get('✓', 0)+counts.get('α', 0)
    tot = bad+good+sum(v for k, v in counts.items()
                       if k not in ('ℋ', '⊘', '∞', '✓', 'α'))
    if tot == 0:
        return (0.0, 0.0, 0.8)
    br, gr = bad/tot, good/tot
    hue = (0.33*gr - 0.33*br) % 1.0
    sat = min(1.0, br)
    val = 0.5 + 0.5*gr
    return float(hue), float(sat), float(val)


def color_phase(hsv: Tuple[float, float, float]) -> str:
    h, s, v = hsv
    if s < 0.15 and v > 0.85:
        return "calm"
    if s > 0.6 and v > 0.6:
        return "hot" if (h < 0.08 or h > 0.92) else "alert"
    if v < 0.3:
        return "dim"
    return "steady"

# ---------------- bit-LSB stats (literal side proxy) --------------


@dataclass(slots=True)
class BitLSB:
    ones: int = 0
    zeros: int = 0

    @property
    def bias(self) -> float:
        n = self.ones + self.zeros
        return (self.ones / n) if n else 0.5

    @property
    def entropy(self) -> float:
        p = self.bias
        q = 1.0 - p
        if p <= 0 or q <= 0:
            return 0.0
        if HAS_NUMPY:
            return float(-(p*np.log2(p) + q*np.log2(q)))
        else:
            import math
            return float(-(p*math.log2(p) + q*math.log2(q)))


def scan_bitlsb(b: bytes) -> BitLSB:
    if not b:
        return BitLSB()
    if HAS_NUMPY:
        arr = np.frombuffer(b, dtype=np.uint8)
        ones = int(np.bitwise_and(arr, 1).sum())
        zeros = int(arr.size - ones)
    else:
        # Fallback without numpy
        ones = sum(byte & 1 for byte in b)
        zeros = len(b) - ones
    return BitLSB(ones=ones, zeros=zeros)


# ---------------- phrase presets (bias emjkkkissions) -----------------
PHRASE_PRESETS: Dict[str, Tuple[str, ...]] = {
    "code": ("def", "class", "import", "return", "self", "for", "if", "else", "try", "except"),
    "text": ("the", "and", "of", "to", "in", "that", "is", "for", "with", "on"),
    "data": ("id", "name", "value", "type", "len", "count", "mean", "var", "sum"),
}

# ---------------- viral modifier gene w/ tanh-faded surprise ------


@dataclass(slots=True)
class VMGene:
    level: int = 0
    alpha: float = 0.5            # add-k smoothing
    fade: float = 0.2             # 0→keep history, 1→trust new surprise; used in EMA
    t_quantile: float = 0.95      # edge threshold (quantile over |grad|)
    emissions: Counter[str] = field(default_factory=Counter)
    error_counts: Counter[str] = field(default_factory=Counter)
    surprise_ema: float = 0.0     # EMA of normalized surprise
    last_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_phase: str = "steady"
    last_bitlsb: BitLSB = field(default_factory=BitLSB)

    def update_from_text(self, text: str, preset: Optional[str] = None) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9_]+", text)
        c = Counter(toks)
        # phrase preset bias (shared vocabulary)
        if preset and preset in PHRASE_PRESETS:
            for w in PHRASE_PRESETS[preset]:
                c[w] += 1  # tiny nudge
        self.emissions.update(c)
        self.last_hsv = counts_to_color(c)
        self.last_phase = color_phase(self.last_hsv)
        self.last_bitlsb = scan_bitlsb(text.encode('utf-8', errors='ignore'))
        return toks

    def _prob(self, sym: str) -> float:
        V = max(1, len(self.emissions))
        N = sum(self.emissions.values())
        return (self.emissions.get(sym, 0) + self.alpha) / (N + self.alpha * V)

    def surprise_series(self, tokens: Iterable[str]) -> np.ndarray:
        s = np.array([-np.log2(max(1e-12, self._prob(t)))
                     for t in tokens], dtype=np.float32)
        # normalize → tanh squash so "surp 1 → fade 0" is stable
        if s.size:
            mu, sigma = float(s.mean()), float(s.std() or 1.0)
            z = (s - mu) / sigma
            u = np.tanh(z)                       # [-1,1] scale
            # EMA: new = fade*u_mean + (1-fade)*old
            self.surprise_ema = float(
                self.fade * float(u.mean()) + (1.0 - self.fade) * self.surprise_ema)
            return u
        return s

    def edges(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if u.size == 0:
            return np.array([]), np.array([])
        # unit kernels
        K = np.array([-1, -2, 0, 2, 1], dtype=np.float32)
        K -= K.mean()
        K = K / (np.linalg.norm(K) + 1e-12)
        G = np.array([1, 2, 1], dtype=np.float32)
        G -= G.mean()
        G = G / (np.linalg.norm(G) + 1e-12)
        smooth = np.convolve(u, G, mode='same')
        grad = np.convolve(smooth, K, mode='same')
        g = np.abs(grad)
        hi = float(np.quantile(g, self.t_quantile))
        lo = 0.5 * hi
        mask = (g >= hi) | ((g >= lo) & np.concatenate(([False], g[1:] >= lo)))
        idx = np.nonzero(mask)[0]
        return idx, g

    # cosine/angle on "lengths" (any vectors) ----------------------
    @staticmethod
    def angle(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return float(np.pi/2)
        cos = float(np.clip(a.dot(b)/(na*nb), -1.0, 1.0))
        return float(np.arccos(cos))


# ---------------- reflex (double-ended solid) --------------------
MAGIC = b"DE1\0"
LE = "<"


@dataclass(slots=True)
class Entry:
    name: str
    off: int
    length: int
    crc32: int


@dataclass(slots=True)
class ReflexDE:
    entries: List[Entry]
    total_len: int
    f_index: List[Tuple[int, int]]
    r_index: List[Tuple[int, int]]
    payload_f: bytes
    payload_r: bytes

    def pack(self) -> bytes:
        flags = 1 if self.payload_r else 0
        n = len(self.entries)
        buf = io.BytesIO()
        buf.write(MAGIC)
        buf.write(struct.pack(LE+"HIQ", flags, n, self.total_len))
        for e in self.entries:
            nb = e.name.encode("utf-8")
            buf.write(struct.pack(LE+"H", len(nb)))
            buf.write(nb)
            buf.write(struct.pack(LE+"QQI", e.off, e.length, e.crc32))

        def _wix(ix: List[Tuple[int, int]]):
            buf.write(struct.pack(LE+"I", len(ix)))
            for u, c in ix:
                buf.write(struct.pack(LE+"QQ", u, c))
        _wix(self.f_index)
        buf.write(struct.pack(LE+"Q", len(self.payload_f)))
        buf.write(self.payload_f)
        _wix(self.r_index)
        buf.write(struct.pack(LE+"Q", len(self.payload_r)))
        buf.write(self.payload_r)
        return buf.getvalue()

    @staticmethod
    def unpack(blob: bytes) -> ReflexDE:
        br = io.BytesIO(blob)
        assert br.read(4) == MAGIC
        flags, n, total = struct.unpack(
            LE+"HIQ", br.read(struct.calcsize(LE+"HIQ")))
        ents = []
        for _ in range(n):
            (nl,) = struct.unpack(LE+"H", br.read(2))
            nm = br.read(nl).decode("utf-8")
            off, len_, c = struct.unpack(
                LE+"QQI", br.read(struct.calcsize(LE+"QQI")))
            ents.append(Entry(nm, off, len_, c))

        def _rix():
            (cnt,) = struct.unpack(LE+"I", br.read(4))
            out = []
            for _ in range(cnt):
                out.append(struct.unpack(LE+"QQ", br.read(16)))
            return out
        f_ix = _rix()
        (flen,) = struct.unpack(LE+"Q", br.read(8))
        fpay = br.read(flen)
        r_ix = _rix()
        (rlen,) = struct.unpack(LE+"Q", br.read(8))
        rpay = br.read(rlen)
        if not (flags & 1):
            r_ix, rpay = [], b""
        return ReflexDE(ents, total, f_ix, r_ix, fpay, rpay)


def _crc32(b: bytes) -> int: return zlib.crc32(b) & 0xFFFFFFFF


def _deflate_raw(b: bytes, level: int = 6) -> bytes:
    c = zlib.compressobj(level=level, wbits=-15)
    return c.compress(b)+c.flush()


def build_reflex_double_ended(members: Dict[str, bytes], *, chunk: int = 64*1024,
                              with_reverse: bool = True, level: int = 6) -> bytes:
    names = list(members.keys())
    pos = 0
    tape = io.BytesIO()
    ents = []
    for n in names:
        data = members[n]
        tape.write(data)
        ents.append(Entry(n, pos, len(data), _crc32(data)))
        pos += len(data)
    U = tape.getvalue()
    # forward index & payload
    f_ix = []
    comp = io.BytesIO()
    d = zlib.compressobj(level=level, wbits=-15)
    up = 0
    for w in range(0, len(U), chunk):
        if w > up:
            comp.write(d.compress(U[up:w]))
            up = w
        f_ix.append((w, comp.tell()))
    comp.write(d.compress(U[up:]))
    comp.write(d.flush())
    fpay = comp.getvalue()
    # reverse
    r_ix = []
    rpay = b""
    if with_reverse:
        R = U[::-1]
        comp2 = io.BytesIO()
        d2 = zlib.compressobj(level=level, wbits=-15)
        up2 = 0
        for w in range(0, len(R), chunk):
            if w > up2:
                comp2.write(d2.compress(R[up2:w]))
                up2 = w
            r_ix.append((w, comp2.tell()))
        comp2.write(d2.compress(R[up2:]))
        comp2.write(d2.flush())
        rpay = comp2.getvalue()
    return ReflexDE(ents, len(U), f_ix, r_ix, fpay, rpay).pack()

# ---------------- zip ingest → reflex + gene metadata ------------------------

def read_zip_members(zip_path: str) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            out[n] = zf.read(n)
    return out


def create_zip_from_directory(dir_path: str, zip_path: str, *, 
                            exclude_patterns: Optional[List[str]] = None,
                            compression: int = zipfile.ZIP_DEFLATED,
                            compresslevel: int = 6) -> None:
    """Create a zip file from directory contents with optional exclusion patterns"""
    if exclude_patterns is None:
        exclude_patterns = [r'\.pyc$', r'__pycache__', r'\.git', r'\.DS_Store']
    
    def should_exclude(path_str: str) -> bool:
        for pattern in exclude_patterns:
            if re.search(pattern, path_str):
                return True
        return False
    
    base_path = Path(dir_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    with zipfile.ZipFile(zip_path, "w", compression=compression, compresslevel=compresslevel) as zf:
        if base_path.is_file():
            # Single file
            if not should_exclude(base_path.name):
                zf.write(base_path, base_path.name)
        else:
            # Directory
            for file_path in base_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(base_path)
                    if not should_exclude(str(relative_path)):
                        zf.write(file_path, str(relative_path))


def generate_reflex_from_zip(zip_path: str, *, preset: Optional[str] = None,
                             chunk: int = 64*1024, with_reverse: bool = True,
                             level: int = 6) -> bytes:
    members = read_zip_members(zip_path)
    # build gene metadata per member (text-like files only)
    gene = VMGene(level=1)
    meta: Dict[str, bytes] = {}
    for name, data in members.items():
        try:
            text = data.decode("utf-8")
        except Exception:
            continue
        toks = gene.update_from_text(text, preset=preset)
        u = gene.surprise_series(toks)
        idx, g = gene.edges(u)
        hdr = {
            "version": 1,
            "phase": gene.last_phase,
            "hsv": [float(gene.last_hsv[0]), float(gene.last_hsv[1]), float(gene.last_hsv[2])],
            "bitlsb_bias": gene.last_bitlsb.bias,
            "bitlsb_entropy": gene.last_bitlsb.entropy,
            "surprise_ema": gene.surprise_ema,
            "edges_count": int(idx.size),
            "sig": f"{s_fingerprint(text) & 0xFFFFFFFF:08x}",
            "sig_comp": f"{s_complement(text) & 0xFFFFFFFF:08x}",
            "created": int(time.time())
        }
        try:
            import msgpack
            meta[f"registry/{name}.mpk"] = cast(bytes, msgpack.packb(
                hdr, use_bin_type=True, strict_types=True))
        except ImportError:
            # Fallback to json-like representation without msgpack
            import json
            meta[f"registry/{name}.json"] = json.dumps(hdr).encode('utf-8')
    members.update(meta)
    return build_reflex_double_ended(members, chunk=chunk, with_reverse=with_reverse, level=level)


# ---------------- tiny example (remove or guard) -----------------------------
if __name__ == "__main__":
    import sys

    import argparse
    parser = argparse.ArgumentParser("vaez: viral convolution vessel")
    parser.add_argument('-c', '--create', action='store_true', help='Create zip from directory')
    parser.add_argument('-p', '--preset', default=None, help='Phrase preset for gene analysis')
    parser.add_argument('-x', '--exclude', nargs='*', default=None, help='Exclusion patterns for zip creation')
    parser.add_argument('source', nargs='?', default=None, help='Source directory (for -c) or zip file')
    parser.add_argument("target", nargs='?', default=".vael.ce1", help='Target zip file (for -c) or reflex file')
    parser.add_argument("extras", nargs='*', default=None, help='Additional zip files to process')
    args = parser.parse_args()

    if args.create:
        # Create zip mode
        if not args.source:
            print("Error: source directory required for zip creation")
            sys.exit(1)
        source_path = Path(args.source)
        target_path = Path(args.target)
        
        try:
            create_zip_from_directory(
                str(source_path), 
                str(target_path),
                exclude_patterns=args.exclude
            )
            print(f"Created zip: {target_path.resolve()}")
        except Exception as e:
            print(f"Error creating zip: {e}")
            sys.exit(1)
    else:
        # Process zip mode (original functionality)
        if args.source and Path(args.source).exists():
            zips = [Path(args.source)]
        else:
            zips = list(Path(".").glob("*.ce1"))
            if args.extras:
                zips.extend(Path(c) for c in args.extras)
        
        dst = Path(args.target)
        
        for zp in zips:
            try:
                out = generate_reflex_from_zip(zp.as_posix(), preset=args.preset)
                dst.write_bytes(out)
                print(f"Processed: {zp} -> {dst.resolve()}")
            except Exception as e:
                print(f"Error processing {zp}: {e}")