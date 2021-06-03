import struct
import numpy as np
from sys import argv
from random import shuffle


def pack_list(buffer, lst):
    if isinstance(lst[0], float):
        buffer += struct.pack(f"{len(lst)}f", *lst)
    elif isinstance(lst[0], int):
        buffer += struct.pack(f"{len(lst)}I", *lst)
    else:
        raise TypeError
    return buffer


def compress(mat: np.ndarray, name: str):
    row_id = np.nonzero(mat.T[0][:])[0]
    # print(row_id)
    # print(mat)
    reduced = mat[~np.all(mat == 0, axis=1)]
    # print(reduced)
    reduced = reduced.astype(np.float32).tolist()
    buffer = struct.pack("III", nrow, ncol, len(row_id))
    buffer = pack_list(buffer, row_id.tolist())
    for r in reduced:
        buffer = pack_list(buffer, r)
    with open(f"{name}.row", "wb+") as f:
        f.write(buffer)


def gen_rand(nrow: int, ncol: int, sparsity: float) -> np.ndarray:
    mat = np.random.rand(nrow, ncol)
    nz_len = int(nrow*sparsity)
    idx = [1]*nz_len + [0]*(nrow-nz_len)
    shuffle(idx)
    assert len(idx) == mat.shape[0]
    for i in range(len(idx)):
        if idx[i] == 0:
            mat[i][:] = np.zeros((1, mat.shape[1]))
    return mat


if __name__ == '__main__':
    if argv[1] == "-g":  # ./script -g nrow ncol sparsity
        nrow = int(argv[2])
        ncol = int(argv[3])
        sparsity = float(argv[4])
        mat = gen_rand(nrow, ncol, sparsity)
        name = f"{nrow}_{ncol}_{sparsity}"
    else:  # ./script name.npy
        filename = argv[1]
        mat = np.load(filename)
        name = filename.split(".npy")[0]
    compress(mat, name)
