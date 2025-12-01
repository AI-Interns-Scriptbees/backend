# scripts/convert_npy_to_memmap.py
import numpy as np
from pathlib import Path
import sys

p = Path("content")
src = p / "embeddings.npy"
if not src.exists():
    print("ERROR: content\\embeddings.npy not found. Run this script from the project root where content/ is present.")
    sys.exit(2)

# backup (a simple copy)
bakdir = p.parent / "backup_embeddings"
bakdir.mkdir(parents=True, exist_ok=True)
bak = bakdir / "embeddings.npy"
print("Backing up", src, "to", bak)
try:
    with open(src, "rb") as fr, open(bak, "wb") as fw:
        fw.write(fr.read())
except Exception as e:
    print("Backup failed:", e)
    # continue anyway

# load the .npy (loads header + numbers, but returns numpy array)
print("Loading .npy ...")
arr = np.load(src, mmap_mode=None)
arr = arr.astype("float32")
if arr.ndim != 2:
    print("ERROR: expected a 2-D array. found shape:", arr.shape)
    sys.exit(3)
N, D = arr.shape
print("Loaded embeddings shape:", (N, D))

# write a raw memmap file (no numpy header) - file name: embeddings.memmap
memmap_path = p / "embeddings.memmap"
print("Writing raw memmap to", memmap_path)
m = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(N, D))
m[:] = arr[:]
m.flush()

# also write a raw binary (optional) name: embeddings.bin
bin_path = p / "embeddings.bin"
print("Writing raw binary to", bin_path)
with open(bin_path, "wb") as f:
    arr.tofile(f)

# report
total_floats = memmap_path.stat().st_size // 4
print("Total floats in memmap:", total_floats)
print("Floats per page (inferred dim):", total_floats // N)
print("Divisible?:", (total_floats % N == 0))
print("Done. If 'Divisible?: True' and inferred dim is 768, everything is good.")
