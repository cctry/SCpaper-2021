import scipy.io
import scipy.sparse
import sys
import struct


def pack_lists(buffer, *args, T='I'):
    for l in args:
        buffer += struct.pack(f"{len(l)}{T}", *l)
    return buffer


path = sys.argv[1]
try:
    info = scipy.io.mminfo(path)
except Exception:
    print("Not a good file")
    exit(1)
mat = scipy.io.mmread(path)
assert isinstance(mat, scipy.sparse.coo_matrix)
mat = mat.tocsr()
res = struct.pack("III", mat.shape[0], mat.shape[1], mat.nnz)
res = pack_lists(res, mat.indptr, mat.indices, T='i')
res = pack_lists(res, mat.data, T='f')
name = path.split("/")[-1]
name = name.split(".")[-2]
with open(f"{name}.csr", "wb+") as f:
    f.write(res)
