#!/usr/bin/env python3
from pathlib import Path
import zipfile, zlib, struct, io, re, json, time
from collections import Counter, namedtuple
from functools import reduce
import numpy as np

# Operators
Buf = namedtuple('Buf', 'data mounts')
Pen = namedtuple('Pen', 'k thresh')
Tick = namedtuple('Tick', 'w x y z')

PHI = 0x9E3779B97F4A7C15
MASK = 0xFFFFFFFFFFFFFFFF

# Mix 
def mix(x):
    x = (x + PHI) & MASK
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9; x &= MASK
    x ^= x >> 27; x *= 0x94D049BB133111EB; x &= MASK  
    return x ^ (x >> 31)

sig = lambda s: reduce(lambda h, b: mix(h + b), s.encode(), 0)

# Buffer
buf = lambda data=b"", mounts=(): Buf(data, mounts)
mount = lambda buf, off=0, mir=False: buf._replace(mounts=buf.mounts + ((off, mir),))
write = lambda buf, data: buf._replace(data=buf.data + data)

# Pen
pen = lambda k=(-1, -2, 0, 2, 1), thresh=0.95: Pen(k, thresh)
convolve = lambda pen, signal: tuple(np.convolve(signal, pen.k, 'same'))
edges = lambda pen, signal: tuple(np.where(np.abs(np.convolve(signal, pen.k, 'same')) >= np.quantile(np.abs(np.convolve(signal, pen.k, 'same')), pen.thresh))[0])

# Tick  
tick = lambda w=1.0, x=0.0, y=0.0, z=0.0: Tick(w, x, y, z)
jump = lambda t, d: Tick(t.w * d - t.z * d, t.x * d + t.y * d, t.y * d - t.x * d, t.z * d + t.w * d)
rewind = lambda t: Tick(t.w, -t.x, -t.y, -t.z)  
split = lambda t, d: (jump(t, d), jump(t, -d))

# Phase
def phase(counts):
    bad = sum(counts.get(s, 0) for s in 'ℋ⊘∞')
    good = sum(counts.get(s, 0) for s in '✓α') 
    total = sum(counts.values())
    if not total: return "calm"
    br, gr = bad/total, good/total
    hue, sat, val = (0.33*gr - 0.33*br) % 1, min(1, br), 0.5 + 0.5*gr
    if sat < 0.15 and val > 0.85: return "calm"
    if sat > 0.6 and val > 0.6: return "hot" if hue < 0.08 or hue > 0.92 else "alert"
    return "dim" if val < 0.3 else "steady"

# Gene
def analyze(text, preset=()):
    toks = tuple(re.findall(r"[A-Za-z0-9_]+", text))
    counts = Counter(toks + preset)
    bits = np.frombuffer(text.encode(), np.uint8)
    bias = float(np.bitwise_and(bits, 1).mean()) if len(bits) else 0.5
    return {
        "toks": len(toks), "uniq": len(set(toks)), "phase": phase(counts),
        "bias": bias, "sig": f"{sig(text):08x}", "t": int(time.time())
    }

# Archive
read_zip = lambda path: {n: zipfile.ZipFile(path).read(n) for n in zipfile.ZipFile(path).namelist()}

def write_zip(dir_path, zip_path, exclude=(r'\.git', r'__pycache__')):
    base = Path(dir_path)
    excl = lambda name: any(re.search(p, str(name)) for p in exclude)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if base.is_file():
            if not excl(base.name): zf.write(base, base.name)
        else:
            for fp in base.rglob("*"):
                if fp.is_file() and not excl(fp.relative_to(base)):
                    zf.write(fp, str(fp.relative_to(base)))

# Compress  
def compress(members):
    entries = []
    tape = io.BytesIO()
    for name, data in members.items():
        entries.append((name, tape.tell(), len(data), zlib.crc32(data)))
        tape.write(data)
    
    payload = tape.getvalue()
    comp = zlib.compressobj(6, wbits=-15)
    compressed = comp.compress(payload) + comp.flush()
    
    buf = io.BytesIO()
    buf.write(b"DE1\0" + struct.pack("<HI", 0, len(entries)))
    for name, off, length, crc in entries:
        nb = name.encode()
        buf.write(struct.pack("<H", len(nb)) + nb + struct.pack("<QQI", off, length, crc))
    buf.write(struct.pack("<Q", len(compressed)) + compressed)
    return buf.getvalue()

# Process
def process(zip_path, preset=()):
    members = read_zip(zip_path)
    meta = {}
    for name, data in members.items():
        try:
            text = data.decode()
            meta[f"reg/{name}.json"] = json.dumps(analyze(text, preset)).encode()
        except: pass
    return compress({**members, **meta})

if __name__ == "__main__":
    import sys, argparse
    p = argparse.ArgumentParser()
    p.add_argument('-c', action='store_true')
    p.add_argument('-p', default="")
    p.add_argument('src', nargs='?')
    p.add_argument('dst', nargs='?', default=".out")
    args = p.parse_args()
    
    preset = tuple(args.p.split(',')) if args.p else ()
    
    if args.c:
        write_zip(args.src, args.dst)
        print(f"Created: {Path(args.dst).resolve()}")
    else:
        result = process(args.src or next(Path(".").glob("*.zip"), ""), preset)
        Path(args.dst).write_bytes(result)
        print(f"Processed -> {Path(args.dst).resolve()}")