from batchedmoments import BatchedMoments


def test_batches():
    data = list(range(100))
    batched = BatchedMoments()
    batchsize = 10
    for idx in range(len(data) // batchsize):
        st = idx * batchsize
        batched(data[st: st + batchsize])
        partial = BatchedMoments()(data[: st + batchsize])
        assert partial == batched
    full = BatchedMoments()(data)
    assert full == batched
