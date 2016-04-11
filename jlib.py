import numpy as np 

def hello(name = 'no name'):
    """
    name is welcome by saying hello
    
    input: name - the welcome name
    """
    print('Hello {name}!'.format(**locals()))
    print('2015-3-2, 2:45pm')

def check_mol_similarity():
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    ms = [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
    fps = [FingerprintMols.FingerprintMol(x) for x in ms]
    print fps[0]
    print DataStructs.FingerprintSimilarity( fps[0], fps[1])
    print DataStructs.FingerprintSimilarity( fps[0], fps[2])
    print DataStructs.FingerprintSimilarity( fps[1], fps[2])
    print DataStructs.FingerprintSimilarity( fps[0], fps[0])

def mols_similarity( ms_smiles = ['CCOC', 'CCO', 'COC']):
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    ms = [Chem.MolFromSmiles( m_sm) for m_sm in ms_smiles]
    # [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
    fps = [FingerprintMols.FingerprintMol(x) for x in ms]
    print fps[0]
    print DataStructs.FingerprintSimilarity( fps[0], fps[1])
    print DataStructs.FingerprintSimilarity( fps[0], fps[2])
    print DataStructs.FingerprintSimilarity( fps[1], fps[2])
    print DataStructs.FingerprintSimilarity( fps[0], fps[0])

def _mols_similarity_base_r0( ms_smiles_mid, ms_smiles_base):
    """
    Input: dictionary type required such as {nick name: smiles code, ...}
    """
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols

    #processing for mid
    print( "Target: " + ms_smiles_mid.keys())
    ms_mid = [Chem.MolFromSmiles( m_sm) for m_sm in ms_smiles_mid.values()]
    # [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
    fps_mid = [FingerprintMols.FingerprintMol(x) for x in ms_mid]

    #processing for base
    print( "Base: " + ms_smiles_base.keys())
    ms_base = [Chem.MolFromSmiles( m_sm) for m_sm in ms_smiles_base.values()]
    # [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
    fps_base = [FingerprintMols.FingerprintMol(x) for x in ms_base]

    for (bx, f_b) in enumerate(fps_base):
        for (dx, f_d) in enumerate(fps_mid):
            print( "Base:{0}, Target:{1}".format( ms_smiles_base.keys()[bx], ms_smiles_mid.keys()[dx]))
            print( DataStructs.FingerprintSimilarity( f_b, f_d))


"""
core part is generated while addition is changed for both
"""
def mols_similarity_base_core( ms_smiles_mid, ms_smiles_base):
    """
    Input: dictionary type required such as {nick name: smiles code, ...}
    """
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols

    # Processing for mid
    print( "Target: ", ms_smiles_mid.keys())
    ms_mid = [Chem.MolFromSmiles( m_sm) for m_sm in ms_smiles_mid.values()]
    # [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
    fps_mid = [FingerprintMols.FingerprintMol(x) for x in ms_mid]

    #processing for base
    print( "Base: ", ms_smiles_base.keys())
    ms_base = [Chem.MolFromSmiles( m_sm) for m_sm in ms_smiles_base.values()]
    # [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
    fps_base = [FingerprintMols.FingerprintMol(x) for x in ms_base]

    return fps_base, fps_mid

def mols_similarity_base( ms_smiles_mid, ms_smiles_base):
    """
    Input: dictionary type required such as {nick name: smiles code, ...}
    """
    from rdkit import DataStructs

    [fps_base, fps_mid] = mols_similarity_base_core( ms_smiles_mid, ms_smiles_base)

    for (bx, f_b) in enumerate(fps_base):
        for (dx, f_d) in enumerate(fps_mid):
            print( "Base:{0}, Target:{1}".format( ms_smiles_base.keys()[bx], ms_smiles_mid.keys()[dx]))
            print( DataStructs.FingerprintSimilarity( f_b, f_d))

def mols_similarity_base_return( ms_smiles_mid, ms_smiles_base, property_of_base = None):
    """
    The results will be returned. 
        A * w = b, A and b will be returned.
        return A, b, w
    """
    from rdkit import DataStructs

    [fps_base, fps_mid] = mols_similarity_base_core( ms_smiles_mid, ms_smiles_base)

    Nb, Nm = len(fps_base), len(fps_mid)
    A = np.zeros( (Nm, Nb))
    b = np.zeros( Nb)

    for (bx, f_b) in enumerate(fps_base):
        for (mx, f_m) in enumerate(fps_mid):
            print( "Base:{0}, Target:{1}".format( ms_smiles_base.keys()[bx], ms_smiles_mid.keys()[mx]))
            A[mx, bx] =  DataStructs.FingerprintSimilarity( f_b, f_m)
            print( A[mx, bx])
        if property_of_base:
            b[ bx] = property_of_base[ bx]
            print( b[ bx])

    if property_of_base:
        print "b is obtained."
        return A, b
    else:
        return A

def mols_similarity_base_get_w( ms_smiles_mid, ms_smiles_base, property_of_base):
    """
    property_of_base, which is b, must be entered
    """
    [A, b] = mols_similarity_base_return( ms_smiles_mid, ms_smiles_base, property_of_base)

    w = np.dot( np.linalg.pinv(A), b)

    return w
