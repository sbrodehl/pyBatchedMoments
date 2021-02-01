import multiprocessing
from multiprocessing import Pool

from batchedmoments import BatchedMoments


def test_multiprocessing_add():
    batchsize = 15
    samples = batchsize * batchsize * batchsize * batchsize
    data = iter(list(range(n, n + batchsize)) for n in range(0, samples, batchsize))
    bm = BatchedMoments()(next(data))
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        for dbm in pool.imap_unordered(BatchedMoments(), data):
            bm += dbm
