# pylint: disable=unsubscriptable-object
from typing import Union, Iterable
import warnings
import numpy as np

__version__       = "1.0.0"
__title__         = "batchedmoments"
__description__   = "Computing (batch-wise) sample statistics."
__url__           = "https://github.com/sbrodehl/pyBatchedMoments"
__uri__           = __url__
__doc__           = __description__ + " <" + __uri__ + ">"
__documentation__ = __url__
__source__        = __url__
__tracker__       = __url__ + '/issues'
__author__        = "Sebastian Brodehl"
__license__       = "MIT License"
__copyright__     = "Copyright (c) 2021 " + __author__


class BatchedMoments:
    """Computes (batch-wise) sample statistics.

    Properties:
        mean        - returns the sample mean
        variance    - returns the sample variance
        std         - returns the sample standard deviation
        skewness    - returns the sample skewness
        kurtosis    - return the sample kurtosis

    """

    def __init__(self, axis: Union[tuple, int] = None, shape: tuple = None, ddof: int = 0):
        """
        Args:
            axis: Axis to be reduced. If None, a scalar value is computed (default)
            shape: Shape of the moments. If None, first update will initialize and set shape.
            ddof: "Delta Degrees of Freedom": the divisor used in the calculation is
                    ``N - ddof``, where ``N`` represents the number of elements. (default is zero)
        """
        self._n: int = 0
        self._ddof = ddof
        self.axis: Union[tuple, None] = None
        if axis is not None:
            self.axis = axis if isinstance(axis, tuple) else tuple([axis])
        self._m1: Union[np.ndarray, None] = None
        self._m2: Union[np.ndarray, None] = None
        self._m3: Union[np.ndarray, None] = None
        self._m4: Union[np.ndarray, None] = None
        self._initialized: bool = False
        self._moments_shape: Union[tuple, None] = None
        if shape is not None:  # initialization is possible
            axis = tuple([]) if self.axis is None else self.axis
            # data shape / dimensions must be
            data_shape = [0] * (len(axis) + len(shape))
            shape_idx = 0
            for ax, _ in enumerate(data_shape):
                # axis have fixed position
                if ax in axis:
                    data_shape[ax] = -1
                else:
                    # fill in the rest
                    data_shape[ax] = shape[shape_idx]
                    shape_idx += 1
            if shape_idx != len(shape):
                raise RuntimeError("There is something wrong with the shapes!")
            self._initialize(tuple(data_shape))

    def __len__(self):
        return self._n

    def reduce(self, axis: Union[tuple, int] = None) -> "BatchedMoments":
        """Reduce the moments along the given axis.

        Args:
            axis: Axis to be reduced. If None, a scalar value is computed (default)

        Returns:
            the reduced moments
        """
        if axis is not None:
            axis = axis if isinstance(axis, tuple) else tuple([axis])
        else:
            axis = self._moments_shape
        axis = list(axis)
        ax = axis.pop()
        _w = BatchedMoments(
            tuple(sorted(self.axis + tuple([ax]))),
            tuple([d for _i, d in enumerate(self.shape) if _i != ax])
        )
        for _i in range(self.shape[ax]):
            wi = BatchedMoments.from_(_w)
            wi._n = self._n
            wi._m1 = self._m1.take(_i, axis=ax)
            wi._m2 = self._m2.take(_i, axis=ax)
            wi._m3 = self._m3.take(_i, axis=ax)
            wi._m4 = self._m4.take(_i, axis=ax)
            _w += wi
        if len(axis) > 0:
            return _w.reduce(tuple(axis))
        return _w

    @staticmethod
    def _compute_ith_moment(
            t: np.ndarray,
            p: int,
            m_1: np.ndarray = None,
            axis: Union[tuple, int] = None
    ) -> np.ndarray:
        """Computes the p-th moment using the naive formulae `sum_{i=1}^n (x_i - mean)**p` over given axis."""
        t = t.copy()
        n = np.array(
            [t.shape[x] for x in axis]
            if axis is not None
            else list(t.shape)
        ).prod(dtype=np.float64)
        if m_1 is not None:
            t = t.astype(m_1.dtype)
            if axis is not None:
                m_1 = np.expand_dims(m_1, axis=axis)
            t = np.subtract(t, m_1, out=t)
        tt = t.copy()
        for _ in range(p - 1):
            t = np.multiply(t, tt, out=t)
        return np.divide(t, n, dtype=np.float64).sum(axis=axis)

    def update(self, t: np.ndarray) -> "BatchedMoments":
        n_b = int(np.array(
            [t.shape[x] for x in self.axis]
            if self.axis is not None
            else list(t.shape)
        ).prod(dtype=int).item())
        # the following is equivalent to `np.mean(t, axis=self.axis)`
        m1_b = self._compute_ith_moment(t, 1, axis=self.axis)
        # the following is equivalent to `np.var(t, axis=self.axis)`
        m2_b = n_b * self._compute_ith_moment(t, 2, m_1=m1_b, axis=self.axis)
        # higher order moments for custom shapes
        m3_b = n_b * self._compute_ith_moment(t, 3, m_1=m1_b, axis=self.axis)
        m4_b = n_b * self._compute_ith_moment(t, 4, m_1=m1_b, axis=self.axis)

        n_a = self._n
        n = n_a + n_b
        delta = m1_b - self._m1
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2

        # update M4
        self._m4 += m4_b
        self._m4 += 4.0 * delta * (n_a * m3_b - n_b * self._m3) / n
        self._m4 += 6.0 * delta2 * (n_a * n_a * m2_b + n_b * n_b * self._m2) / (n * n)
        self._m4 += delta4 * np.array(1.0 * (n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b)) / (n * n * n), dtype=np.float64)
        # update M3
        self._m3 += m3_b
        self._m3 += 3.0 * delta * (n_a * m2_b - n_b * self._m2) / n
        self._m3 += delta3 * n_a * n_b * (n_a - n_b) / (n * n)
        # update M2
        self._m2 += m2_b
        self._m2 += delta2 * n_a * n_b / n
        # update M1
        self._m1 += delta * n_b / n
        # increment seen samples
        self._n += n_b
        return self

    def _initialize(self, shape: Union[tuple, None]) -> bool:
        """Initialize buffers with the given shape of the data.
        The shape of the buffers is deduced from the data shape and the axis variable.
        """
        # reset stats
        self._n = 0
        self._m1 = self._m2 = self._m3 = self._m4 = None
        non_none_axis = self.axis if self.axis is not None else []
        self._moments_shape = tuple([
            x
            for _i, x in enumerate(shape)
            if _i not in non_none_axis
        ] if self.axis is not None else [])
        self._m1 = np.zeros(self._moments_shape)
        self._m2 = np.zeros(self._moments_shape)
        self._m3 = np.zeros(self._moments_shape)
        self._m4 = np.zeros(self._moments_shape)
        self._initialized = True
        return self._initialized

    def __eq__(self, other):
        if (  # axis is not compared, the shape of the moments is more important
            not isinstance(other, self.__class__)
            or self.shape != other.shape
            or self.ddof != other.ddof
        ):
            return False
        if (  # check values of moments
            not np.allclose(self.mean, other.mean, equal_nan=True)
            or not np.allclose(self.std, other.std, equal_nan=True)
            or not np.allclose(self.variance, other.variance, equal_nan=True)
            or not np.allclose(self.skewness, other.skewness, equal_nan=True)
            or not np.allclose(self.kurtosis, other.kurtosis, equal_nan=True)
        ):
            return False
        return True

    def __call__(self, x: Union[np.ndarray, Iterable, float, int]) -> "BatchedMoments":
        # check input
        if x is None:
            return self
        # convert to numpy if necessary
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        # check if initialized
        if not self._initialized:
            self._initialize(x.shape)
        # perform update
        return self.update(x)

    def _combine_moments(self, other: "BatchedMoments") -> tuple:
        """Computes combined moments of 'self' and 'other'.

        Args:
            other: BatchedMoments instance

        Returns: tuple of combined moments

        """
        n = self._n + other._n
        delta = other._m1 - self._m1
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2
        # m1
        m1 = self._m1 + other._n * delta / n
        # m2
        m2 = self._m2 + other._m2
        m2 += delta2 * self._n * other._n / n
        # m3
        m3 = self._m3 + other._m3
        m3 += 3.0 * delta * (self._n * other._m2 - other._n * self._m2) / n
        m3 += delta3 * self._n * other._n * (self._n - other._n) / (n * n)
        # m4
        m4 = self._m4 + other._m4
        m4 += 4.0 * delta * (self._n * other._m3 - other._n * self._m3) / n
        m4 += 6.0 * delta2 * (self._n * self._n * other._m2 + other._n * other._n * self._m2) / (n * n)
        m4 += delta4 * np.array((1.0 * self._n * other._n * (self._n * self._n - self._n * other._n + other._n * other._n)) / (n * n * n), dtype=np.float64)
        # check types
        if not isinstance(m1, np.ndarray):
            m1 = np.array(m1)
        if not isinstance(m2, np.ndarray):
            m2 = np.array(m2)
        if not isinstance(m3, np.ndarray):
            m3 = np.array(m3)
        if not isinstance(m4, np.ndarray):
            m4 = np.array(m4)
        return m1, m2, m3, m4

    @staticmethod
    def from_(other: "BatchedMoments") -> "BatchedMoments":
        """Create and initiate class from another instance."""
        if not other._initialized:
            raise RuntimeError("Can't initialize from non-initialized object.")
        return BatchedMoments(other.axis, other.shape, ddof=other.ddof)

    def __iadd__(self, other: "BatchedMoments") -> "BatchedMoments":
        # modify and return 'self'
        if not self._initialized:
            raise RuntimeError("Object not initialized!")
        if not other._initialized:
            raise RuntimeError("Object not initialized!")
        if self.shape != other.shape:
            raise RuntimeError("Won't broadcast shapes. You are on your own, sorry.")
        if self.axis != other.axis:
            warnings.warn("Axis in `add` method differ.", RuntimeWarning)

        self._m1, self._m2, self._m3, self._m4 = self._combine_moments(other)
        self._n = self._n + other._n
        return self

    def __add__(self, other: "BatchedMoments") -> "BatchedMoments":
        added = BatchedMoments.from_(self)
        added += self
        added += other
        return added

    @property
    def ddof(self) -> int:
        return self._ddof

    @property
    def shape(self) -> tuple:
        return self._moments_shape

    @property
    def mean(self) -> Union[np.ndarray, None]:
        return self._m1

    @property
    def variance(self) -> Union[np.ndarray, None]:
        try:
            return self._m2 / (self._n - self._ddof)
        except TypeError:
            return None

    @property
    def std(self) -> Union[np.ndarray, None]:
        try:
            return np.sqrt(self.variance)
        except TypeError:
            return None

    @property
    def skewness(self) -> Union[np.ndarray, None]:
        """Skewness is a measure of the asymmetry of a distribution (or data set).
         The skewness value can be positive, zero, negative, or undefined.

        References:
            Joanes, D. N., and C. A. Gill. "Comparing measures of sample skewness and kurtosis."
             Journal of the Royal Statistical Society: Series D (The Statistician) 47.1 (1998): 183-189.
             https://doi.org/10.1111/1467-9884.00122

        Returns:
            the sample skewness
        """
        try:
            return np.sqrt(1.0 * self._n) * self._m3 / pow(self._m2, 1.5)
        except TypeError:
            return None

    @property
    def kurtosis(self) -> Union[np.ndarray, None]:
        """Kurtosis is a measure of the "tailedness" of a distribution (or data set)
         relative to a normal distribution.

        References:
            Joanes, D. N., and C. A. Gill. "Comparing measures of sample skewness and kurtosis."
             Journal of the Royal Statistical Society: Series D (The Statistician) 47.1 (1998): 183-189.
             https://doi.org/10.1111/1467-9884.00122

        Returns:
            the sample kurtosis
        """
        try:
            return 1.0 * self._n * self._m4 / (self._m2 * self._m2) - 3.0
        except TypeError:
            return None

    def __repr__(self) -> str:
        return f"<BatchedMoments ({self._n}): {str(self.mean)} ± {str(self.std)}>"
