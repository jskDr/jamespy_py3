# generate bead location

import numpy as np


def add_sort_center(center_l, center):
    center_l.append(center)
    return np.sort(center_l).tolist()


def calc_range(center_l, ang):
    center_l = np.sort(center_l).tolist()

    range_d = {'st': [], 'ed': []}
    st0 = ang / 2 + ang / 2
    range_d['st'].append(st0)

    for center in center_l[1:]:
        ed = center - ang / 2 - ang / 2
        st = center + ang / 2 + ang / 2
        range_d['ed'].append(ed)
        range_d['st'].append(st)

    ed0 = 360 - ang / 2 - ang / 2
    range_d['ed'].append(ed0)

    range_d['space'] = []
    for st, ed in zip(range_d['st'], range_d['ed']):
        range_d['space'].append(ed - st)

    return range_d


def calc_remained_range(range_d, ang):
    remained_range = 0
    for st, ed in zip(range_d['st'], range_d['ed']):
        if ed - st > ang:
            remained_range += ed - st

    # print("remained_range:", remained_range)
    return remained_range


def gen_pnt(remained_range):
    return np.random.rand() * remained_range


def remapping(range_d, pnt_in, ang):
    pnt_out = None
    for st, ed in zip(range_d['st'], range_d['ed']):
        if ed - st > ang:
            pnt_out = pnt_in + st
        else:
            continue
        if pnt_out < ed:
            break
        else:
            pnt_in -= (ed - st)
    return pnt_out


def shift_init(center_l):
    init = np.random.random() * 360
    return np.mod(np.add(center_l, init), 360).tolist()


class BEADS:
    def __init__(self, R=10, r=5 / 2, disp=False):
        self.disp = disp

        def get_ang(R, r):
            return np.arcsin(r / (R + r)) / np.pi * 180
        self.ang = get_ang(R, r)
        if disp:
            print("ang:", self.ang)

    def gen_bead_centers(self, beads=1):
        center_l = [0]

        for nb in range(1, beads):
            range_d = calc_range(center_l, self.ang)
            remained_range = calc_remained_range(range_d, self.ang)
            if remained_range == 0:
                break
            pnt = gen_pnt(remained_range)
            new_center = remapping(range_d, pnt, self.ang)
            center_l = add_sort_center(center_l, new_center)

            if self.disp:
                print('center_l:', center_l)
                print(range_d)
                print(remained_range)
                print(pnt)

        center_l = shift_init(center_l)
        if self.disp:
            print(center_l)

        return center_l
