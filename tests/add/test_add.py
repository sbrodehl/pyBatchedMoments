from batchedmoments import BatchedMoments


def test_add():
    data = list(range(100))
    a = BatchedMoments()(data)
    b = BatchedMoments()(data[:len(data) // 2]) + BatchedMoments()(data[len(data) // 2:])
    assert a == b


def test_iadd():
    data = list(range(100))
    a = BatchedMoments()(data)
    b = BatchedMoments()(data[:len(data) // 2])
    b += BatchedMoments()(data[len(data) // 2:])
    assert a == b


def test_batched_add():
    data = list(range(100))
    batchsize = 10
    batched = BatchedMoments()(data[:batchsize])
    for idx in range(1, len(data) // batchsize):
        st = idx * batchsize
        batched += BatchedMoments()(data[st: st + batchsize])
        partial = BatchedMoments()(data[: st + batchsize])
        assert partial == batched
    full = BatchedMoments()(data)
    assert full == batched
