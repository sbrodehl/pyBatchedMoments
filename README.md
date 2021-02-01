# pyBatchedMoments

[![pypi-version](https://img.shields.io/pypi/v/batchedmoments)][pypi]
[![python-version](https://img.shields.io/pypi/pyversions/batchedmoments)][pypi]
[![Build, Test & Deploy to PyPI](https://github.com/sbrodehl/pyBatchedMoments/workflows/Build,%20Test%20&%20Deploy%20to%20PyPI/badge.svg)](https://github.com/sbrodehl/PyBatchedMoments/actions?query=workflow%3A%22Build%2C+Test+%26+Deploy+to+PyPI%22)

[pyBatchedMoments][pyBM-gh] is a Python library for computing (batch-wise) sample statistics,
such as mean, variance, standard deviation, skewness and kurtosis.

In certain applications it is needed to compute simple statistics of a population, but with _textbook_ formulae
the calculation can suffer from loss of precision and can be numerically unstable.
Additionally, for large populations only a single pass over the values is feasible, therefore,
an incremental (_batch-wise_) approach is desirable.

## Install

To install the current release
```shell
pip install batchedmoments
```

## Example

```python
from batchedmoments import BatchedMoments

data = [2, 8, 0, 4, 1, 9, 9, 0]
bm = BatchedMoments()
bm(data)

# use computed values
bm.mean
bm.std
```


[pypi]: https://pypi.org/project/batchedmoments
[pyBM-gh]: https://github.com/sbrodehl/pyBatchedMoments
