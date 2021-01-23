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
