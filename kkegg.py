# KEGG related file
import json
import pandas as pd

def get_lr_string( KEGG_ID_list, sep_str):
    """
    Divide left and right kegg_id using seperation function in string.
    """
    kegg_id_l, kegg_id_r = [], []
    for kegg_id in KEGG_ID_list:
        kl, kr = kegg_id.split(sep_str)
        kegg_id_l.append( kl)
        kegg_id_r.append( kr)
    return kegg_id_l, kegg_id_r 
    #for kegg_id in df["KEGG_ID"]

def get_lr_kegg_id( KEGG_ID_list):
    return get_lr_string( KEGG_ID_list, " = ")

def get_lr_kegg_lr( kegg_id_l):
    return get_lr_string( kegg_id_l, " + ")

def get_l0r0_kegg( KEGG_ID_list):
    """
    Return the first id of each part in KEGG_ID.
    """
    l, r = get_lr_kegg_id( KEGG_ID_list)
    l0 = get_lr_kegg_lr( l)[0]
    r0 = get_lr_kegg_lr( r)[0]

    return l0, r0

def get_lr_smiles( kegg_d, KEGG_ID_list):
    """
    Translate SMILES
    """
    l0_l, r0_l = get_l0r0_kegg( KEGG_ID_list) 

    ls = [ kegg_d[l] for l in l0_l]
    rs = [ kegg_d[r] for r in r0_l]

    return ls, rs
    
class KEGG_ID_to_SMILES:
    """
    Translate KEGG_ID to Left and Right SMILES Strings.
    """
    def __init__(self, kegg_d_fname = 'sheet/Kegg_Dict.json'):
        """
        Usage
        -----
        ls, rs = KEGG_ID_to_SMILES().transform( df["KEGG_ID"].tolist())
        """
        with open( kegg_d_fname) as data_file:
            self.kegg_d = json.load(data_file)
    
    def transform( self, KEGG_ID_list):
        kegg_d = self.kegg_d
        return get_lr_smiles( kegg_d, KEGG_ID_list)
        
