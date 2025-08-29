#!/usr/bin/env python3
from __future__ import annotations
from typing import NamedTuple, Callable, Iterable, Optional, Any
from collections import Counter
from pathlib import Path
import struct, io, zlib, zipfile, re, time, json

try: import numpy as np; HAS_NUMPY = True
except ImportError: HAS_NUMPY = False

# Core Constants
PHI64 = 0x9E3779B97F4A7C15
MASK64 = 0xFFFFFFFFFFFFFFFF
MAGIC = b"DE1\0"

# Operator States
class Buffer(NamedTuple):
    data: bytes = b""
    mounts: tuple = ()  # (offset, mirror, target) tuples
    pos: int = 0

class Pen(NamedTuple):
    kernel: tuple = (-1, -2, 0, 2, 1)
    smooth: tuple = (1, 2, 1) 
    threshold: float = 0.95

class Time(NamedTuple):
    w: float = 1.0  # quaternion components
    x: float = 0.0
    y: float = 0.0  
    z: float = 0.0

class Scan(NamedTuple):
    ones: int = 0
    zeros: int = 0
    bias: float = 0.5
    entropy: float = 0.0

class Phase(NamedTuple):
    hue: float = 0.0
    sat: float = 0.0  
    val: float = 0.0
    name: str = "steady"

class Entry(NamedTuple):
    name: str
    off: int
    length: int
    crc32: int

# Mix Operator - hash operations  
def mix(x):
    x = (x + PHI64) & MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 27  
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return x & MASK64

fingerprint = lambda s: __import__('functools').reduce(lambda h, b: mix((h + b) & MASK64), s.encode('utf-8'), 0)
complement = lambda s: (~fingerprint(s)) & MASK64

# Buffer Operator - data mounting with mirrors/offsets  
def mount(buf: Buffer, offset: int = 0, mirror: bool = False, target: int = -1) -> Buffer:
    return buf._replace(mounts=buf.mounts + ((offset, mirror, target),))

def write_buf(buf: Buffer, data: bytes) -> Buffer:
    new_data = bytearray(buf.data)
    for offset, mirror, target in buf.mounts:
        pos = (target if target >= 0 else buf.pos) + offset
        chunk = data[::-1] if mirror else data
        if pos < len(new_data):
            new_data[pos:pos+len(chunk)] = chunk
        else:
            new_data.extend(b'\0' * (pos - len(new_data)) + chunk)
    return buf._replace(data=bytes(new_data), pos=buf.pos + len(data))

# Scan Operator - bit entropy analysis
def scan_bytes(data: bytes) -> Scan:
    if not data: return Scan()
    ones = sum(b & 1 for b in data) if not HAS_NUMPY else int(np.bitwise_and(np.frombuffer(data, np.uint8), 1).sum())
    zeros = len(data) - ones
    bias = ones / len(data) if data else 0.5
    if bias in (0, 1): entropy = 0.0
    else: 
        log2 = (lambda x: x.log2() if HAS_NUMPY else __import__('math').log2(x))
        entropy = float(-(bias * log2(bias) + (1-bias) * log2(1-bias)))
    return Scan(ones, zeros, bias, entropy)

# Phase Operator - color diagnostics from token counts
def phase_from_counts(counts: Counter) -> Phase:
    bad = sum(counts.get(s, 0) for s in ('ℋ', '⊘', '∞'))
    good = sum(counts.get(s, 0) for s in ('✓', 'α'))
    total = sum(counts.values())
    if not total: return Phase(0, 0, 0.8, "calm")
    
    br, gr = bad/total, good/total
    hue = (0.33*gr - 0.33*br) % 1.0
    sat, val = min(1.0, br), 0.5 + 0.5*gr
    
    if sat < 0.15 and val > 0.85: name = "calm"
    elif sat > 0.6 and val > 0.6: name = "hot" if (hue < 0.08 or hue > 0.92) else "alert"  
    elif val < 0.3: name = "dim"
    else: name = "steady"
    
    return Phase(hue, sat, val, name)

# Pen Operator - convolution drawing
def pen_convolve(pen: Pen, signal: tuple) -> tuple:
    if not signal: return ()
    if not HAS_NUMPY:
        # Simple convolution fallback
        k_norm = tuple(x - sum(pen.kernel)/len(pen.kernel) for x in pen.kernel)
        result = []
        for i in range(len(signal)):
            val = sum(signal[max(0, min(len(signal)-1, i+j-len(k_norm)//2))] * k 
                     for j, k in enumerate(k_norm))
            result.append(val)
        return tuple(result)
    
    k = np.array(pen.kernel, dtype=np.float32)
    k = (k - k.mean()) / (np.linalg.norm(k) + 1e-12)
    return tuple(np.convolve(signal, k, mode='same'))

def pen_edges(pen: Pen, signal: tuple) -> tuple:
    smooth = pen_convolve(pen._replace(kernel=pen.smooth), signal)
    grad = pen_convolve(pen, smooth)  
    g = tuple(abs(x) for x in grad)
    if not g: return ()
    hi = sorted(g)[int(len(g) * pen.threshold)]
    lo = hi * 0.5
    return tuple(i for i, val in enumerate(g) if val >= hi or (val >= lo and i > 0 and g[i-1] >= lo))

# Time Operator - quaternion temporal operations
def time_jump(t: Time, delta: float) -> Time:
    # Quaternion rotation for temporal jump
    cos_d, sin_d = __import__('math').cos(delta/2), __import__('math').sin(delta/2)
    return Time(
        t.w * cos_d - t.z * sin_d,
        t.x * cos_d + t.y * sin_d, 
        t.y * cos_d - t.x * sin_d,
        t.z * cos_d + t.w * sin_d
    )

def time_rewind(t: Time) -> Time:
    return Time(t.w, -t.x, -t.y, -t.z)  # Quaternion conjugate

def time_parallel(t: Time, branch: float) -> tuple[Time, Time]:
    t1 = time_jump(t, branch)
    t2 = time_jump(t, -branch) 
    return t1, t2

# Gene Operator - simplified viral analysis
def gene_analyze(text: str, preset: tuple = ()) -> dict:
    tokens = tuple(re.findall(r"[A-Za-z0-9_]+", text))
    counts = Counter(tokens + preset)  # General preset addition
    
    scan = scan_bytes(text.encode('utf-8', errors='ignore'))
    phase = phase_from_counts(counts)
    
    return {
        "tokens": len(tokens),
        "unique": len(set(tokens)),  
        "scan": scan._asdict(),
        "phase": phase._asdict(),
        "sig": f"{fingerprint(text) & 0xFFFFFFFF:08x}",
        "sig_comp": f"{complement(text) & 0xFFFFFFFF:08x}",
        "time": int(time.time())
    }

# Archive Operator - zip operations
def archive_read(path: str) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}

def archive_write(path: str, members: dict[str, bytes], 
                 exclude: tuple = (r'\.pyc$', r'__pycache__', r'\.git', r'\.DS_Store')) -> None:
    def excluded(name): return any(re.search(pat, name) for pat in exclude)
    
    base = Path(path) if isinstance(path, str) else path
    with zipfile.ZipFile(str(base) + '.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, data in members.items():
            if not excluded(name):
                zf.writestr(name, data)

def archive_from_dir(dir_path: str, zip_path: str, exclude: tuple = ()) -> None:
    base = Path(dir_path)
    if not base.exists(): raise FileNotFoundError(f"Not found: {dir_path}")
    
    def excluded(name): return any(re.search(pat, str(name)) for pat in (exclude or 
                        (r'\.pyc$', r'__pycache__', r'\.git', r'\.DS_Store')))
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if base.is_file():
            if not excluded(base.name): zf.write(base, base.name)
        else:
            for fp in base.rglob("*"):
                if fp.is_file() and not excluded(fp.relative_to(base)):
                    zf.write(fp, str(fp.relative_to(base)))

# Compression Operator - reflex double-ended
def compress_reflex(members: dict[str, bytes], chunk: int = 64*1024, reverse: bool = True) -> bytes:
    entries = []
    tape = io.BytesIO()
    
    for name, data in members.items():
        entries.append(Entry(name, tape.tell(), len(data), zlib.crc32(data) & 0xFFFFFFFF))
        tape.write(data)
    
    payload = tape.getvalue()
    
    # Forward compression with index
    f_index, f_data = [], io.BytesIO()
    comp = zlib.compressobj(6, wbits=-15)
    for i in range(0, len(payload), chunk):
        f_index.append((i, f_data.tell()))
        f_data.write(comp.compress(payload[i:i+chunk]))
    f_data.write(comp.flush())
    
    # Reverse compression  
    r_index, r_data = [], b""
    if reverse:
        rev_payload = payload[::-1]
        r_buf, comp2 = io.BytesIO(), zlib.compressobj(6, wbits=-15)
        for i in range(0, len(rev_payload), chunk):
            r_index.append((i, r_buf.tell()))
            r_buf.write(comp2.compress(rev_payload[i:i+chunk]))
        r_buf.write(comp2.flush())
        r_data = r_buf.getvalue()
    
    # Pack binary format
    buf = io.BytesIO()
    buf.write(MAGIC + struct.pack("<HIQ", 1 if reverse else 0, len(entries), len(payload)))
    
    for e in entries:
        name_bytes = e.name.encode('utf-8')
        buf.write(struct.pack("<H", len(name_bytes)) + name_bytes)
        buf.write(struct.pack("<QQI", e.off, e.length, e.crc32))
    
    def write_index(idx):
        buf.write(struct.pack("<I", len(idx)))
        for u, c in idx: buf.write(struct.pack("<QQ", u, c))
    
    write_index(f_index)
    buf.write(struct.pack("<Q", len(f_data.getvalue())) + f_data.getvalue())
    write_index(r_index) 
    buf.write(struct.pack("<Q", len(r_data)) + r_data)
    
    return buf.getvalue()

# Main processing pipeline
def process_zip(zip_path: str, preset: tuple = ()) -> bytes:
    members = archive_read(zip_path)
    meta = {}
    
    for name, data in members.items():
        try:
            text = data.decode('utf-8')
            analysis = gene_analyze(text, preset)
            meta[f"registry/{name}.json"] = json.dumps(analysis).encode('utf-8')
        except: pass
    
    return compress_reflex({**members, **meta})

if __name__ == "__main__":
    import sys, argparse
    
    p = argparse.ArgumentParser("vaez: viral convolution vessel")
    p.add_argument('-c', '--create', action='store_true', help='Create zip from directory')
    p.add_argument('-p', '--preset', default="", help='Preset tokens (comma-separated)')
    p.add_argument('-x', '--exclude', nargs='*', default=(), help='Exclusion patterns')
    p.add_argument('source', nargs='?', help='Source directory/zip')
    p.add_argument('target', nargs='?', default=".vael.ce1", help='Target file')
    args = p.parse_args()
    
    preset_toks = tuple(args.preset.split(',')) if args.preset else ()
    
    if args.create:
        if not args.source: print("Error: source required"); sys.exit(1)
        try:
            archive_from_dir(args.source, args.target, tuple(args.exclude))
            print(f"Created: {Path(args.target).resolve()}")
        except Exception as e: print(f"Error: {e}"); sys.exit(1)
    else:
        zips = [Path(args.source)] if args.source and Path(args.source).exists() else list(Path(".").glob("*.ce1"))
        for zp in zips:
            try:
                result = process_zip(str(zp), preset_toks)
                Path(args.target).write_bytes(result)
                print(f"Processed: {zp} -> {Path(args.target).resolve()}")
            except Exception as e: print(f"Error processing {zp}: {e}")