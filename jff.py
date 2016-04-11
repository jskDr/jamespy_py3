"""
Finger print related codes are colected.
"""
def find_cluster( fa_list, thr = 0.5):
    """
    find similar pattern with 
    the first element: fa0
    """
    fa0 = fa_list[0]
    fa0_group = [fa0]
    fa_other = fa_list[1:]
    for fa_o in fa_list[1:]:
        tm_d = jchem.calc_tm_dist_int( fa0, fa_o)
        if tm_d > thr:
            fa0_group.append( fa_o)
            fa_other.remove( fa_o)
    
    return fa0_group, fa_other


def find_cluster_all( fa_list, thr = 0.5):
    """
    all cluster are founded based on threshold of 
    fingerprint similarity using greedy methods
    """
    fa_o = fa_list
    fa0_g_all = []
    
    while len( fa_o) > 0:
        fa0_g, fa_o = find_cluster( fa_o, thr)
        fa0_g_all.append( fa0_g)
    
    return fa0_g_all