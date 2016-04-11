def long_to_int64_array( val, ln):
    sz = ln / 64 + 1
    ar = np.zeros( sz, dtype=int)
    i64 = 2**64 - 1
    for ii in range( sz):
        ar[ ii] = int(val & i64)
        val = val >> 64
    return ar

def int64_array_ro_long( ar):
    val = long(0)
    for ii in range( ar.shape[0]):
        val = val | ar[-ii-1]
        print val
        if ii < ar.shape[0] - 1:
            val = val << 64
        print val
    return val