from batchedmoments import BatchedMoments


def test_higher_order_disabled():
    data = list(range(100))
    batched = BatchedMoments(higher_order_moments=False)
    added = BatchedMoments(higher_order_moments=False)
    batchsize = 10
    for idx in range(len(data) // batchsize):
        st = idx * batchsize
        batched(data[st:st + batchsize])
        partial = BatchedMoments(higher_order_moments=False)(data[:st + batchsize])
        added += BatchedMoments(higher_order_moments=False)(data[st:st + batchsize])
        assert partial == batched
    full = BatchedMoments(higher_order_moments=False)(data)
    assert full == batched
