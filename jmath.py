# Python3
import numpy as np


def long_to_int64_array(val, ln):
    sz = ln / 64 + 1
    ar = np.zeros(sz, dtype=int)
    i64 = 2**64 - 1
    for ii in range(sz):
        ar[ii] = int(val & i64)
        val = val >> 64
    return ar


def int64_array_ro_long(ar):
    val = long(0)
    for ii in range(ar.shape[0]):
        val = val | ar[-ii - 1]
        print(val)
        if ii < ar.shape[0] - 1:
            val = val << 64
        print(val)
    return val


def count(a_l, a, inverse=False):
    """
    It returns the number of elements which are equal to
    the target value.
    In order to resolve when x is an array with more than
    one dimensions, converstion from array to list is used.
    """
    if inverse is False:
        x = np.where(np.array(a_l) == a)
    else:
        x = np.where(np.array(a_l) != a)

    return len(x[0].tolist())


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]


def autocorr_kmlee_k(y, k):
    ybar = np.mean(y)
    N = len(y)
    # print( N)
    # cross_sum = np.zeros((N-k,1))
    cross_sum = np.zeros(N - k)
    # print( cross_sum.shape, N, k)

    # Numerator, unscaled covariance
    # [Matlab] for i = (k+1):N
    for i in range(k, N):
        # [Matlab] cross_sum(i) = (y(i)-ybar)*(y(i-k)-ybar) ;
        # print( cross_sum.shape, i, k, N)
        # print( cross_sum[i-k])
        # print( (y[i]-ybar)*(y[i-k]-ybar))
        # cross_sum[i] = (y[i]-ybar)*(y[i-k]-ybar)
        cross_sum[i - k] = (y[i] - ybar) * (y[i - k] - ybar)

    # Denominator, unscaled variance
    yvar = np.dot(y - ybar, y - ybar)
    ta2 = np.sum(cross_sum) / yvar
    return ta2


def autocorr_kmlee(y, p=None):
    if p is None:
        p = len(y)
    # The results
    # ta = np.zeros((p,1))
    ta = np.zeros(p)
    # global N
    N = len(y)
    ybar = np.mean(y)

    # Generate ACFs at each lag i
    for i in range(p):
        ta[i] = autocorr_kmlee_k(y, i)

    return ta


def autocorrelate(x, method='numpy'):
    """
    Multiple approaches are considered.

    # kmlee method
    function ta2 = acf_k(y,k)
    % ACF_K - Autocorrelation at Lag k
    % acf(y,k)
    %
    % Inputs:
    % y - series to compute acf for
    % k - which lag to compute acf
    %
    global ybar
    global N
    cross_sum = zeros(N-k,1) ;

    % Numerator, unscaled covariance
    for i = (k+1):N
        cross_sum(i) = (y(i)-ybar)*(y(i-k)-ybar) ;
    end

    % Denominator, unscaled variance
    yvar = (y-ybar)'*(y-ybar) ;

    ta2 = sum(cross_sum) / yvar ;

    """
    if method == 'numpy':
        return autocorr(x)
    elif method == 'zeropadding':
        return np.correlate(x, x, mode='full')
    elif method == 'kmlee':
        return autocorr_kmlee(x)


def autocorrelate_m(X_org, method):
    """
    autocorrelate_m(X_org, method)
    Inputs
    ======
    method, string
    'numpy', 'zeropadding', 'kmlee'
    """
    X_l = []
    for i in range(X_org.shape[0]):
        x_org = X_org[i, :]
        x_ac = autocorrelate(x_org, method=method)
        X_l.append(x_ac)
    X = np.array(X_l)
    return X
