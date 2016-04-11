#James os library for utilizing python os library


import os

def _mfren_r0( src_startwith, dst_startwidh):
    
    if len( src_startwith) != len( dst_startwidh):
        raise Typeerror('The lengths of src_startwith and dst_startwith strings must be the same.')
    
    fnames = os.listdir(".")    
    fnames_Tang = [x for x in fnames if x.startswith( src_startwith)]    
    fnames_Tang2Tong = [ dst_startwidh+x[len(dst_startwidh):] for x in fnames if x.startswith( src_startwith)]

    for src, dst in zip( fnames_Tang, fnames_Tang2Tong):
        print src, '-->', dst
        os.rename( src, dst)

def mfren( src_startwith, dst_startwidh):
   
    fnames = os.listdir(".")    
    fnames_Tang = [x for x in fnames if x.startswith( src_startwith)]    
    fnames_Tang2Tong = [ dst_startwidh + x[len( src_startwith):] for x in fnames_Tang]

    for src, dst in zip( fnames_Tang, fnames_Tang2Tong):
        print src, '-->', dst
        os.rename( src, dst)