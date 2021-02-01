import multiprocessing
from multiprocessing import Pool
from itertools import tee

from batchedmoments import BatchedMoments


def test_multiprocessing_add():
    batchsize = 15
    samples = batchsize * batchsize * batchsize * batchsize
    gen1, gen2 = tee((list(range(n, n + batchsize)) for n in range(0, samples, batchsize)))
    data = iter(gen1)
    bm = BatchedMoments()(next(data))
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        for dbm in pool.imap_unordered(BatchedMoments(), data):
            bm += dbm
    data = iter(gen2)
    seq = BatchedMoments()(next(data))
    for batch in data:
        seq += BatchedMoments()(batch)
    assert seq == bm
