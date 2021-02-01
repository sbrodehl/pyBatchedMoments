# pyBatchedMoments

[![pypi-version](https://img.shields.io/pypi/v/batchedmoments)][pypi]
[![python-version](https://img.shields.io/pypi/pyversions/batchedmoments)][pypi]
[![Build, Test & Deploy](https://github.com/sbrodehl/pyBatchedMoments/workflows/Build,%20Test%20&%20Deploy%20to%20PyPI/badge.svg)](https://github.com/sbrodehl/PyBatchedMoments/actions?query=workflow%3A%22Build%2C+Test+%26+Deploy+to+PyPI%22)

[pyBatchedMoments][pyBM-gh] is a Python library for computing (batch-wise) sample statistics,
such as mean, variance, standard deviation, skewness and kurtosis.

In certain applications it is needed to compute simple statistics of a population, but with _textbook_ formulae
the calculation can suffer from loss of precision and can be numerically unstable.
Additionally, for large populations only a single pass over the values is feasible, therefore,
an incremental (_batch-wise_) approach is needed.

## Installation

To install the current release, run
```shell
pip install batchedmoments
```

### From Source

To install the latest development version (e.g. in [_editable mode_](https://pip.pypa.io/en/stable/reference/pip_install/#cmdoption-e)), run
```shell
git clone https://github.com/sbrodehl/pyBatchedMoments.git
pip install -e pyBatchedMoments
```

## Examples

We start with the simple use case of sample statistics of some (random) numbers.

```python
from batchedmoments import BatchedMoments

data = [2, 8, 0, 4, 1, 9, 9, 0]
bm = BatchedMoments()
bm(data)

# use computed values
# bm.mean, bm.std, ...
```
The result is equivalent to [numpy](https://numpy.org/doc/stable/reference/routines.statistics.html) (`mean`, `std` and `var`)
and [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) (`skew` and `kurtosis`).

### Batched Computation

Where [pyBatchedMoments][pyBM-gh] really shines is when the data is not available at once.
In this case, the data can be batched (split in _usable_ parts), and the statistics can be computed batch-wise.

```python
from batchedmoments import BatchedMoments

# a generator function which returns batches of data
data_iter = iter(list(range(n, n + 10)) for n in range(0, 1000, 10))

bm = BatchedMoments()
for batch in data_iter:
    bm(batch)

# use computed values
# bm.mean, bm.std, ...
```

### Distributed / Parallel Computation

The sample statistics of single batches can be computed independently and later be combined with the `add` operator.
The following example shows a multiprocessing use case, but the batches can be computed distributed among different
computers (nodes) as well.

```python
import multiprocessing
from multiprocessing import Pool
from batchedmoments import BatchedMoments

# a generator function which returns batches of data
data = iter(list(range(n, n + 10)) for n in range(0, 1000, 10))
# create object and initialize with first batch of data
bm = BatchedMoments()(next(data))
with Pool(processes=multiprocessing.cpu_count()) as pool:
    for dbm in pool.imap_unordered(BatchedMoments(), data):
        bm += dbm

# use computed values
# bm.mean, bm.std, ...
```

### Reduction of Axes

The `axis=...` keyword allows specifying axis or axes along which the sample statistics are computed.
The default (`None`) is to compute the sample statistics of the flattened array.

Working with data of shape `(1000, 3, 28, 28)` and specifying `axis=0` the computed statistics will have shape `(3, 28, 28)`.
If `axis=(0, 2, 3)` the computed statistics will have shape `(3,)`.

Using the `reduce` method the shape of the computed statistics can be further reduced at a later stage.
E.g. with data of shape `(1000, 3, 28, 28)` and `axis=(2, 3)` the computed statistics will have shape `(1000, 3)`.
By using `reduce(0)` the computed statistics will be reduced to shape `(3,)`.

## License

pyBatchedMoments uses a MIT-style license, as found in [LICENSE](LICENSE) file.


[pypi]: https://pypi.org/project/batchedmoments
[pyBM-gh]: https://github.com/sbrodehl/pyBatchedMoments
