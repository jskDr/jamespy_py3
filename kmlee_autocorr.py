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