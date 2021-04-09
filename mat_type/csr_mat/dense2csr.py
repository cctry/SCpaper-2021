import numpy as np
import struct
import sys
import itertools


def pack_lists(buffer, *args, T='I'):
    for l in args:
        buffer += struct.pack(f"{len(l)}{T}", *l)
    return buffer


nrow = int(sys.argv[1])
ncol = int(sys.argv[2])

indptr = [ncol * i for i in range(0, nrow + 1)]
indices = []
for i in range(nrow):
    indices.extend(range(ncol))
print(len(indices))
vals = [1] * (nrow * ncol)

assert len(indptr) == nrow + 1
assert len(indices) == ncol * nrow
assert len(vals) == ncol * nrow

res = struct.pack("III", nrow, ncol, nrow * ncol)
res = pack_lists(res, indptr, indices, T='i')
res = pack_lists(res, vals, T='f')

with open(f"dense_{nrow}_{ncol}.csr", "wb+") as f:
    f.write(res)