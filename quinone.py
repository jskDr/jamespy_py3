def gen_14BQ_OH():
    """
    return 1,4BQ species with OH functionals.
    """
    q_smiles_base = {}
    q_smiles_mid = {}
    q_smiles_base['1,4-BQ,2-OH'] = '[H]OC1=C([H])C(=O)C([H])=C([H])C1=O'
    q_smiles_base['1,4-BQ,Full-OH'] = 'OC1=C(O)C(=O)C(O)=C(O)C1=O'
    q_smiles_base['1,4-BQ'] = 'O=C1C=CC(=O)C=C1'

    q_smiles_mid['1,4-BQ'] = 'O=C1C=CC(=O)C=C1'
    q_smiles_mid['1,4-BQ,2-OH'] = 'OC1=CC(=O)C=CC1=O'
    q_smiles_mid['1,4-BQ,2,3-OH'] = 'OC1=C(O)C(=O)C=CC1=O'
    q_smiles_mid['1,4-BQ,2,3,5-OH'] = 'OC1=CC(=O)C(O)=C(O)C1=O'
    q_smiles_mid['1,4-BQ,Full-OH'] = 'OC1=C(O)C(=O)C(O)=C(O)C1=O'    

    return q_smiles_base, q_smiles_mid

def gen_910AQ_SO3H():
    """
    return 9,10AQ species with SO3H functionals.
    """
    q_smiles_base = {}
    q_smiles_mid = {}

    q_smiles_base['9,10AQ'] = 'O=C1C2C=CC=CC2C(=O)C2=C1C=CC=C2'
    q_smiles_base['9,10AQ,1-OH'] = 'OS(=O)(=O)C1=CC=CC2C1C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_base['9,10AQ,2-OH'] = 'OS(=O)(=O)C1=CC2C(C=C1)C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_base['9,10AQ,Full-OH'] = 'OS(=O)(=O)C1=C(C(=C(C2C1C(=O)C1=C(C2=O)C(=C(C(=C1S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O'

    q_smiles_mid['9,10AQ'] = 'O=C1C2C=CC=CC2C(=O)C2=C1C=CC=C2'
    q_smiles_mid['9,10AQ,1-OH'] = 'OS(=O)(=O)C1=CC=CC2C1C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_mid['9,10AQ,2-OH'] = 'OS(=O)(=O)C1=CC2C(C=C1)C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_mid['9,10AQ,1,2-OH'] = 'OS(=O)(=O)C1=C(C2C(C=C1)C(=O)C1=C(C=CC=C1)C2=O)S(O)(=O)=O'
    q_smiles_mid['9,10AQ,Full-OH'] = 'OS(=O)(=O)C1=C(C(=C(C2C1C(=O)C1=C(C2=O)C(=C(C(=C1S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O'

    return q_smiles_base, q_smiles_mid

def gen_smiles_quinone( quinone = '9,10AQ', r_group = 'SO3H'):
    if quinone == '1,4BQ' and r_group == 'OH':
        return gen_14BQ_OH()
    elif quinone == '9,10AQ' and r_group == 'SO3H':
        return gen_910AQ_SO3H()

