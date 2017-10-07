# sjutils_sfile.py

from sjutils import sfile
import sjutils


def main():
    fname = sfile.unique_filename()
    print(fname)


def main2():
    fname = sjutils.sfile.unique_filename()
    print(fname)
