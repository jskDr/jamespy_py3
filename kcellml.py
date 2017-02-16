# kcell.py
# python3
import numpy as np
import pandas as pd

from sklearn import model_selection, svm, metrics, cluster, tree, ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import re

import kkeras

import klstm

param_d = {"n_cv_flt": 2, "n_cv_ln": 3, "cv_activation": "relu",
           'SVC:C': 100, 'SVC:gamma': 'auto',
           'RF:n_estimators': 100, 'RF:oob_score': False,
           'DNN:H1': 30, 'DNN:H2': 10}


def setparam(p_d):
    """
    setup global parameters

    Input
    ------
    p_d: dictionary
    This includes global parameters and values.

    Parameters
    -------------
    n_cv_flt, int: number of CNN filters
    n_cv_ln, int: length of a CNN filter
    """

    global param_d

    for k in p_d:
        param_d[k] = p_d[k]


def _clst_r0(X_train, y_train, X_test, y_test, nb_classes, disp=True):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    dt_score = model.score(X_test, y_test)

    model = svm.SVC(C=param_d['SVC:C'], gamma=param_d['SVC:gamma'])
    # print(model)
    model.fit(X_train, y_train)
    sv_score = model.score(X_test, y_test)

    model = kkeras.MLPC([X_train.shape[1],
                         param_d["DNN:H1"], param_d["DNN:H2"], nb_classes])
    model.fit(X_train, y_train, X_test, y_test, nb_classes)
    mlp_score = model.score(X_test, y_test)

    model = kkeras.CNNC(param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
                        l=[X_train.shape[1], param_d["DNN:H1"], param_d["DNN:H2"], nb_classes])
    model.fit(X_train, y_train, X_test, y_test, nb_classes)
    cnn_score = model.score(X_test, y_test)

    model = ensemble.RandomForestClassifier(n_estimators=param_d['RF:n_estimators'],
                                            oob_score=param_d['RF:oob_score'])
    # print(model)
    model.fit(X_train, y_train)
    rf_score = model.score(X_test, y_test)

    if disp:
        print("DT-C:", dt_score)
        print("SVC:", sv_score)
        print("DNN:", mlp_score)
        print("DCNN:", cnn_score)
        print("RF:", rf_score)

    return dt_score, sv_score, mlp_score, cnn_score, rf_score


def clst(X_train, y_train, X_test, y_test, nb_classes, disp=True,
         confusion_matric_return=False, matthews_corrcoef_return=False):
    """
    dt_score, sv_score, mlp_score, cnn_score, rf_score =
        clst(X_train, y_train, X_test, y_test, nb_classes, disp=True,
             confusion_matric_return=False):

    dt_score, sv_score, mlp_score, cnn_score, rf_score,
    dt_cf, sv_cf, mlp_cf, cnn_cf, rf_cf =
        clst(X_train, y_train, X_test, y_test, nb_classes, disp=True,
             confusion_matric_return=True):

    Claustering by multiple classification methods

    Inputs
    ======
    confusion_matric_return, True or False
    Whether it will return a confusion matric or not.
    """
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    dt_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    dt_cf = metrics.confusion_matrix(y_test, y_pred)
    if matthews_corrcoef_return:
        dt_mc = metrics.matthews_corrcoef(y_test, y_pred)

    model = svm.SVC(C=param_d['SVC:C'], gamma=param_d['SVC:gamma'])
    # print(model)
    model.fit(X_train, y_train)
    sv_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    sv_cf = metrics.confusion_matrix(y_test, y_pred)
    if matthews_corrcoef_return:
        sv_mc = metrics.matthews_corrcoef(y_test, y_pred)

    model = kkeras.MLPC([X_train.shape[1],
                         param_d["DNN:H1"], param_d["DNN:H2"], nb_classes])
    model.fit(X_train, y_train, X_test, y_test, nb_classes)
    mlp_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mlp_cf = metrics.confusion_matrix(y_test, y_pred)
    if matthews_corrcoef_return:
        mlp_mc = metrics.matthews_corrcoef(y_test, y_pred)

    model = kkeras.CNNC(param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
                        l=[X_train.shape[1], param_d["DNN:H1"], param_d["DNN:H2"], nb_classes])
    model.fit(X_train, y_train, X_test, y_test, nb_classes)
    cnn_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    cnn_cf = metrics.confusion_matrix(y_test, y_pred)
    if matthews_corrcoef_return:
        cnn_mc = metrics.matthews_corrcoef(y_test, y_pred)

    model = ensemble.RandomForestClassifier(n_estimators=param_d['RF:n_estimators'],
                                            oob_score=param_d['RF:oob_score'])
    # print(model)
    model.fit(X_train, y_train)
    rf_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    rf_cf = metrics.confusion_matrix(y_test, y_pred)
    if matthews_corrcoef_return:
        rf_mc = metrics.matthews_corrcoef(y_test, y_pred)

    if disp:
        print('Acrracuy, matthews corrcoef, confusion metrics')
        print("DT-C:", dt_score, dt_mc)
        print(dt_cf)
        print("SVC:", sv_score, sv_mc)
        print(sv_cf)
        print("DNN:", mlp_score, mlp_mc)
        print(mlp_cf)
        print("DCNN:", cnn_score, cnn_mc)
        print(cnn_cf)
        print("RF:", rf_score, rf_mc)
        print(rf_cf)

    return_l = [dt_score, sv_score, mlp_score, cnn_score, rf_score]
    if confusion_matric_return:
        return_l.extend([dt_cf, sv_cf, mlp_cf, cnn_cf, rf_cf])
    if matthews_corrcoef_return:
        return_l.extend([dt_mc, sv_mc, mlp_mc, cnn_mc, rf_mc])
    return return_l


def GET_clsf2_by_clst(nb_classes):
    def clsf2_by_clst(Xpart_cf, Xpart_ct):
        """
        Clustering is performed and then, classification performed by clustered indices.
        """
        cl_model = cluster.KMeans(n_clusters=nb_classes)
        cl_model.fit(Xpart_ct)
        yint = cl_model.predict(Xpart_ct)

        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(Xpart_cf, yint, test_size=0.2)

        return clst(X_train, y_train, X_test, y_test, nb_classes)

    return clsf2_by_clst


def GET_clsf2_by_yint(nb_classes, param_d=None,
                      confusion_matric_return=False,
                      matthews_corrcoef_return=False):
    """
    param_d, dict, defalut: None
    E.g., param_d = {"n_cv_flt": 2, "n_cv_ln": 3, "cv_activation": "relu"}
    """
    if param_d is not None:
        setparam(param_d)

    def _clsf2_by_yint_r0(X1part, yint, test_size=0.2, disp=True):
        """
        classification is performed by yint
        """
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X1part, yint, test_size=test_size)

        return clst(X_train, y_train, X_test, y_test, nb_classes, disp=disp)

    def clsf2_by_yint(X1part, yint, test_size=0.2, disp=True):
        """
        classification is performed by yint
        """
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X1part, yint, test_size=test_size)

        return clst(X_train, y_train, X_test, y_test, nb_classes,
                    disp=disp, confusion_matric_return=confusion_matric_return,
                    matthews_corrcoef_return=matthews_corrcoef_return)

    return clsf2_by_yint


def pd_df(ix, s_l, ms):
    VI = {1:"Velocity", 2:"Intensity", 12:"Combined"}
    ln = len( s_l)

    df_i = pd.DataFrame()
    df_i["Type"] = ["{}: ".format(ms) + str( ix)] * ln
    df_i["Clustering"] = [ VI[ix[0]]] * ln
    df_i["Classification"] = [ VI[ix[1]]] * ln
    df_i["Clustering method"] = [ "KMeans"] * ln
    df_i["Classification method"] = [ "DT", "SVC", "DNN", "DCNN", "RF"]
    df_i["Pc"] = s_l

    return df_i 

def pd_clsf2_by_clst( ix, Xpart_ct, Xpart_cf, nb_classes):
    print( "Type", ix, "- Clustering:", ix[1], "Classification:", ix[0])
    s_l = GET_clsf2_by_clst(nb_classes)(Xpart_cf, Xpart_ct)
    return pd_df( ix, s_l, "KMeans")

def pd_clsf2_by_yint( ix, yint, Xpart_cf, nb_classes):
    print( "Type", ix, "- Clustering:", ix[1], "Classification:", ix[0])
    s_l = GET_clsf2_by_yint(nb_classes)(Xpart_cf, yint)
    return pd_df( ix, s_l, "Science")

class Cell_Mat_Data:
    def __init__(self, ofname = None, time_range = None, norm_flag = False, 
                    dir_name = '../data/all-3x9', 
                    disp = False):
        """
        Input params
        ------------
        time_range = (s,e), 1D list
        Cut time of a range from s to e for every sequence 

        Usage
        -----
        Cell_Mat_Data( '../data/all-3x9-norm.csv', time_range=(199,251), norm_flag = True, disp = False)
        """
        self.mode_l = ['Velocity', 'Intensity', 'Distance']
        self.time_range = time_range
        self.norm_flag = norm_flag
        self.dir_name = dir_name
        self.disp = disp
        
        if ofname is not None:
            cell_df = self.get_df()
            cell_df.to_csv( ofname, index=False)

    def preprocessing(self, val_2d_l):
        time_range = self.time_range
        norm_flag = self.norm_flag
        
        if time_range is not None:
            val_2d_l = val_2d_l[:, time_range[0]:time_range[1]]
            
        if norm_flag:
            assert(time_range is not None)
            val_2d_l /= np.linalg.norm( val_2d_l, axis=1, keepdims=True)           
        return val_2d_l
    
    def get_df_i(self, pname, idx):
        mode_l = self.mode_l
        dir_name = self.dir_name
        disp = self.disp
        
        csv_fname = '{0}/{1}-{2}.csv'.format(dir_name,pname,idx+1)
        csv_df = pd.read_csv( csv_fname, header=None)
        val_2d_l = self.preprocessing( csv_df.values)
        val = val_2d_l.reshape( -1)

        if disp:
            print(pname, csv_fname, csv_df.shape)
            #print( val[:10])      

        # generate a new dataframe
        df_i = pd.DataFrame()
        df_i['Protein'] = [pname] * np.prod(val_2d_l.shape)
        df_i['Mode'] = [mode_l[ int(idx%3)]] * np.prod(val_2d_l.shape)
        df_i['Cluster'] = [int(idx/3)] * np.prod(val_2d_l.shape)
        df_i['sample'] = np.repeat( list(range( val_2d_l.shape[0])), val_2d_l.shape[1])
        df_i['time'] = list(range( val_2d_l.shape[1])) * val_2d_l.shape[0]
        df_i['value'] = val

        # print( df_i.shape, df_i.keys())
        
        return df_i
        
    def get_df(self):
        df_l = list()
        for pname in [ 'arp23', 'cofilin1', 'vasp']:
            for idx in range(9):
                df_i = self.get_df_i( pname, idx)
                df_l.append( df_i)
                
        df = pd.concat( df_l, ignore_index=True)
        return df

class Cell_Mat_Data4( Cell_Mat_Data):
    def __init__(self, ofname = None, time_range = None, norm_flag = False, 
                    dir_name = '../data/raw-4x9',
                    pname_l = [ 'arp23', 'cofilin1', 'VASP', 'Actin'],
                    disp = False):
        """
        Input params
        ------------
        time_range = (s,e), 1D list
        Cut time of a range from s to e for every sequence 

        pname_l, 1D string list
        list of candidate protein names
        e.g. pname_l = [ 'arp23', 'cofilin1', 'VASP', 'Actin']

        Usage
        -----
        Cell_Mat_Data( '../data/all-3x9-norm.csv', time_range=(199,251), norm_flag = True, disp = False)        
        """
        self.pname_l = [ 'arp23', 'cofilin1', 'VASP', 'Actin']
        super().__init__( ofname=ofname, time_range=time_range, norm_flag=norm_flag, 
                            dir_name=dir_name, disp=disp)

    def get_df(self):
        pname_l = self.pname_l

        df_l = list()
        for pname in pname_l:
            for idx in range(9):
                df_i = self.get_df_i( pname, idx)
                df_l.append( df_i)
                
        df = pd.concat( df_l, ignore_index=True)
        return df       

def _cell_get_X_r0( cell_df, Protein = 'VASP', Mode = 'Velocity', Cluster_l = [0]):
    """
    Input
    -----
    cell_df, pd.DataFrame
    key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
    e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

    """
    cell_df_vasp_velocity_c0 = cell_df[ 
        (cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.Cluster.isin( Cluster_l))]

    x_vec = cell_df_vasp_velocity_c0.value.values
    # x_vec.shape

    # search ln
    df_i = cell_df[ (cell_df.Protein == Protein) & (cell_df.Mode == Mode) 
                & (cell_df.Cluster == 0) & (cell_df['sample'] == 0)]
    l_time = df_i.shape[0]
    #print( l_time)

    X = x_vec.reshape( -1, l_time)
    #print( X.shape)
    
    return X

def cell_get_X( cell_df, Protein = 'VASP', Mode = 'Velocity', Cluster_l = [0]):
    """
    Input
    -----
    cell_df, pd.DataFrame
    key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
    e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

    """
    cell_df_vasp_velocity_c0 = cell_df[ 
        (cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.Cluster.isin( Cluster_l))]

    x_vec = cell_df_vasp_velocity_c0.value.values
    # x_vec.shape

    # search ln
    df_i0 = cell_df_vasp_velocity_c0[ cell_df_vasp_velocity_c0.Cluster == Cluster_l[0]]
    df_i = df_i0[ df_i0["sample"] == df_i0["sample"].values[0]]

    l_time = df_i.shape[0]
    #print( l_time)

    X = x_vec.reshape( -1, l_time)
    #print( X.shape)
    
    return X

def cell_get_cluster( cell_df, Protein = 'VASP', Mode = 'Velocity', cname = "Cluster"):
    """
    Input
    -----
    cell_df, pd.DataFrame
    key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
    e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

    """
    # To pick up cluster indices, consider only one time in a sequence
    # since all time samples in a sequence are associated to the same cluster.
    time_set = set( cell_df['time'])
    a_time = list(time_set)[0]

    cell_df_vasp_velocity_c0 = cell_df[ 
        (cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.time == a_time)]

    clusters_a = cell_df_vasp_velocity_c0[ cname].values
    
    return clusters_a

def _cell_get_cluster_r0( cell_df, Protein = 'VASP', Mode = 'Velocity'):
    """
    Input
    -----
    cell_df, pd.DataFrame
    key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
    e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

    """
    # To pick up cluster indices, consider only one time in a sequence
    # since all time samples in a sequence are associated to the same cluster.
    time_set = set( cell_df['time'])
    a_time = list(time_set)[0]

    cell_df_vasp_velocity_c0 = cell_df[ 
        (cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.time == a_time)]

    clusters_a = cell_df_vasp_velocity_c0["Cluster"].values
    
    return clusters_a

def cell_show_XcYc( cell_df, Protein = 'arp23'):
    X_VASP_Velocity = kcellml.cell_get_X(cell_df, Protein, 'Velocity')
    X_VASP_Intensity = kcellml.cell_get_X(cell_df, Protein, 'Intensity')

    X1 = X_VASP_Velocity
    X2 = X_VASP_Intensity
    #yint_org = np.loadtxt( 'sheet/vasp_y.gz').astype( int)
    #X1.shape

    #X1part = X1 
    X1part = X1 / np.linalg.norm( X1, axis = 1, keepdims=True)
    X2part = X2 / np.linalg.norm( X2, axis = 1, keepdims=True)

    cca = cross_decomposition.CCA(n_components=1)
    cca.fit(X1part.T, X2part.T)
    X_c, Y_c = cca.transform(X1part.T, X2part.T)

    line, = plt.plot( X_c, label = 'Velocity({})'.format(Protein))
    c = plt.getp( line, 'color')
    plt.plot( Y_c, '.-', color=c, label = 'Intensity({})'.format(Protein))
    plt.title('Canonical Correlation Analysis: {}'.format(Protein))
    plt.legend(loc = 0, fontsize='small')

def cell_show_avg_XY(cell_df, Protein = 'arp23'):
    X_VASP_Velocity = kcellml.cell_get_X(cell_df, Protein, 'Velocity')
    X_VASP_Intensity = kcellml.cell_get_X(cell_df, Protein, 'Intensity')

    X1 = X_VASP_Velocity
    X2 = X_VASP_Intensity
    #yint_org = np.loadtxt( 'sheet/vasp_y.gz').astype( int)
    #X1.shape

    #X1part = X1 
    X1part = X1 / np.linalg.norm( X1, axis = 1, keepdims=True)
    X2part = X2 / np.linalg.norm( X2, axis = 1, keepdims=True)

    X_c, Y_c = np.average( X1part, axis = 0), np.average( X2part, axis =0)    

    line, = plt.plot( X_c, label = 'Velocity({})'.format(Protein))
    c = plt.getp( line, 'color')
    plt.plot( Y_c, '.-', color=c, label = 'Intensity({})'.format(Protein))
    plt.title('Average: {}'.format(Protein))
    plt.legend(loc = 0, fontsize='small')

def get_nonan(x, disp = False):
    st_in = np.where( np.isnan(x)== False)[0][0]
    ed_ex = np.where( np.isnan(x[st_in:]) == True)[0][0] + st_in
    x_nonan = x[st_in: ed_ex]
    if disp: 
        print( st_in, ed_ex)
    return x_nonan

def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

def plot_nonan_xy( x, y):
    x = X_VASP_Intensity[k,:]
    y = X_VASP_Velocity[k,:]

    x_nonan = get_nonan(x)
    y_nonan = get_nonan(y)

    plot_xy( x_nonan, y_nonan)
    
def plot_xy( ac_x_nonan, ac_y_nonan):
    ac_x_nonan -= np.mean( ac_x_nonan)
    ac_y_nonan -= np.mean( ac_y_nonan)
    
    ac_x_nonan /= np.max( ac_x_nonan) / np.max( ac_y_nonan)

    plt.plot( ac_x_nonan)
    plt.plot( ac_y_nonan)

def plot_ac_xy( k):
    x = X_VASP_Intensity[k,:]
    y = X_VASP_Velocity[k,:]

    x_nonan = get_nonan(x)
    y_nonan = get_nonan(y)
    x_nonan.shape, y_nonan.shape

    ac_x_nonan = conv_circ( x_nonan, x_nonan)
    ac_y_nonan = conv_circ( y_nonan, y_nonan)

    ac_x_nonan -= np.mean( ac_x_nonan)
    ac_y_nonan -= np.mean( ac_y_nonan)
    
    ac_x_nonan /= np.max( ac_x_nonan) / np.max( ac_y_nonan)

    plt.plot( ac_x_nonan)
    plt.plot( ac_y_nonan)

def get_k( train_k, look_back):
    trainX, trainY = klstm.create_dataset(train_k.reshape(-1,1), look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    return trainX, trainY

def get_train_test( X, pca_order = 10):
    X = X.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.reshape(-1,1)).reshape( X.shape)

    if pca_order > 0:
        pca = PCA(3)
        X = pca.fit_transform(X)
        X = pca.inverse_transform(X)   
        
    n_samples = X.shape[0]
    train_size = int(n_samples * 0.67)
    test_size = n_samples - train_size
    train, test = X[0:train_size,:], X[train_size:n_samples,:]
    return train, test, scaler   

def get_VI( fname = '../data/raw-4x9-195to251.csv', time_range = (195, 251), clusters_l=[0,1,2]):
    cell_df = pd.read_csv( fname)

    pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']

    l_t = time_range[1] - time_range[0]

    V, I = {}, {}
    for pname in pname_l:
        print( "Reading:", pname)
        V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
        I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
        print( V[ pname].shape, I[pname].shape)
        
    return V, I

def get_VIC( fname = '../data/raw-4x9-195to251.csv', time_range = (195, 251), 
                clusters_l=[0,1,2],
                pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']):
    # cluster_l is fixed. Later, this information should be obtained automatically using set() and list()
    cell_df = pd.read_csv( fname)

    #l_t = time_range[1] - time_range[0]

    V, I, C = {}, {}, {}
    for pname in pname_l:
        print( "Reading:", pname)
        V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
        I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
        C[ pname] = cell_get_cluster(cell_df, pname)
        print( V[ pname].shape, I[pname].shape, C[pname].shape)
        
    return V, I, C

def get_VICC( fname = '../data/raw-4x9-195to251.csv', clusters_l=[0,1,2,4],
                pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']):
    # cluster_l is fixed. Later, this information should be obtained automatically using set() and list()
    cell_df = pd.read_csv( fname)

    #l_t = time_range[1] - time_range[0]

    V, I, C, Cell = {}, {}, {}, {}
    for pname in pname_l:
        print( "Reading:", pname)
        V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
        I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
        C[ pname] = cell_get_cluster(cell_df, pname)
        Cell[ pname] = cell_get_cluster(cell_df, pname, cname = "Cell")
        print( V[ pname].shape, I[pname].shape, C[pname].shape)
        
    return V, I, C, Cell

def get_VICCSS( fname = '../data/raw-4x9-195to251.csv', clusters_l=[0,1,2,4],
                pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']):
    """
    V, I, C, Cell, SV, SI = kcellml.get_VICCSS(...)
    """
    # cluster_l is fixed. Later, this information should be obtained automatically using set() and list()
    cell_df = pd.read_csv( fname)

    #l_t = time_range[1] - time_range[0]

    V, I, C, Cell, SV, SI = {}, {}, {}, {}, {}, {}
    for pname in pname_l:
        print( "Reading:", pname)
        V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
        I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
        SV[ pname] = cell_get_X(cell_df, pname, 'Velocity(Sax)', clusters_l)
        SI[ pname] = cell_get_X(cell_df, pname, 'Intensity(Sax)', clusters_l)	
        C[ pname] = cell_get_cluster(cell_df, pname)
        Cell[ pname] = cell_get_cluster(cell_df, pname, cname = "Cell")
        print( V[ pname].shape, I[pname].shape, C[pname].shape)
        
    return V, I, C, Cell, SV, SI

def get_VICCSSCN( fname = '../data/raw-4x9-195to251.csv', clusters_l=[0,1,2,3],
                pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']):
    """
    V, I, C, Cell, SV, SI, I_VASP_CN, I_VASP_CN_Sax = kcellml.get_VICCSSCN(...)
    """
    # cluster_l is fixed. Later, this information should be obtained automatically using set() and list()
    cell_df = pd.read_csv( fname)

    #l_t = time_range[1] - time_range[0]

    V, I, C, Cell, SV, SI = {}, {}, {}, {}, {}, {}
    for pname in pname_l:
        print( "Reading:", pname)
        V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
        I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
        SV[ pname] = cell_get_X(cell_df, pname, 'Velocity(Sax)', clusters_l)
        SI[ pname] = cell_get_X(cell_df, pname, 'Intensity(Sax)', clusters_l)	
        C[ pname] = cell_get_cluster(cell_df, pname)
        Cell[ pname] = cell_get_cluster(cell_df, pname, cname = "Cell")
        print( V[ pname].shape, I[pname].shape, C[pname].shape)

    assert len(pname_l) == 1
    assert ("VASP" in pname_l) or ("arp23" in pname_l) or ("Actin" in pname_l)
    prot = pname_l[0]
    I_VASP_CN = cell_get_X(cell_df, prot, "Intensity-CellNorm", clusters_l)
    I_VASP_CN_Sax = cell_get_X(cell_df, prot, "Intensity-CellNorm(Sax)", clusters_l)
    C_VASP_CN = cell_df[(cell_df["Protein"]==prot)&(cell_df["Mode"]=="Intensity-CellNorm")&
                (cell_df["time"]==0)]["Cluster"].values

    print("Perform sorting to align with Icn/Icnsax/y_cn.")
    for pname in pname_l:	
        arg_ysort = np.argsort( C[ pname])
        V[pname] = V[pname][arg_ysort,:]
        I[pname] = I[pname][arg_ysort,:]
        SV[pname] = SV[pname][arg_ysort,:]
        SI[pname] = SI[pname][arg_ysort,:]
        C[pname] = C[pname][arg_ysort]
        Cell[pname] = Cell[pname][arg_ysort]
    
    return V, I, C, Cell, SV, SI, I_VASP_CN, I_VASP_CN_Sax, C_VASP_CN	

def cell_gen_df_sample_cell( i_l, Mode, new_sample_idx, c_l, p_l, cell_l):
    """
    Mode is one of 'Intensity', 'Velocity'
    """
    sax_raw_df = pd.DataFrame()
    # intensity
    sax_raw_df['Protein'] = np.repeat(p_l, i_l.shape[1])
    sax_raw_df['Cell'] = np.repeat( cell_l, i_l.shape[1])
    sax_raw_df['Mode'] = Mode
    sax_raw_df['Cluster'] = np.repeat(c_l, i_l.shape[1])
    sax_raw_df['sample'] = np.repeat( new_sample_idx, i_l.shape[1])
    sax_raw_df['time'] = list(range(i_l.shape[1])) * i_l.shape[0]
    sax_raw_df['value'] = i_l.reshape(-1)

    return sax_raw_df

def RUN_cell_gen_df_sample_cell():
    # Protein list
    p_l = pd.read_csv( "../data/For_feature_classification_2/Matlab_f2/dminpool_more_56.dminpoolc_Protein.csv").values
    # Intensity list
    I_l = pd.read_csv("../data/For_feature_classification_2/Matlab_f2/dminpool_more_56.dminpool.csv", header = None).values
    # Velocity list
    V_l = pd.read_csv("../data/For_feature_classification_2/Matlab_f2/dminpool_more_56.dminpoolv.csv", header = None).values
    # Cluster assignment
    c_l = pd.read_csv("../data/For_feature_classification_2/Matlab_f2/cluster_assignment_ordered_T_0to3.csv").values
    cell_l = pd.read_csv( "../data/For_feature_classification_2/Matlab_f2/dminpool_more_56.dminpoolc_cell_index.csv").values
    # New Sample Index
    new_sample_idx = list(range(407)) + list(range(730-407)) + list(range(1753-730))

    # Sax Velocity
    sax_v_l = pd.read_csv("../data/For_feature_classification_2/Matlab_f2/dminpool_more_56_symbolic_data_symbolic_feat_data_ratio_4_alpha_size_4.csv", header = None).values

    new_sample_idx = list(range(407)) + list(range(730-407)) + list(range(1753-730))

    sax_raw_df_i = cell_gen_df_sample_cell(I_l, 'Intensity', new_sample_idx, c_l, p_l, cell_l)
    sax_raw_df_v = cell_gen_df_sample_cell(V_l, 'Velocity', new_sample_idx, c_l, p_l, cell_l)
    sax_raw_df_s = cell_gen_df_sample_cell(sax_v_l, 'Velocity(Sax)', new_sample_idx, c_l, p_l, cell_l)

    sax_raw_df = pd.concat( [sax_raw_df_v, sax_raw_df_i, sax_raw_df_s], ignore_index=True)
    sax_raw_df.shape

    print(sax_raw_df.shape[0] / 1753)

    sax_raw_df.to_csv("../data/For_feature_classification_2/Matlab_f2/sax_raw_df_cell.csv", index=False)

def cell_get_cell( ss_c, fold_fname):

    def search_cell_idx( ss_c):
        cidx_l = []
        for s in ss_c[1:]:
            rm = re.search('[0-9]+', s)
            cidx_l.append( int(rm.group()))
        return cidx_l

    cidx_l = search_cell_idx( ss_c)

    len(cidx_l)

    set( cidx_l)

    df = pd.DataFrame( cidx_l, columns=['Cell index'])

    print( df.shape)
    cell_fname = fold_fname[:-4] + "_cell_index.csv"
    df.to_csv( cell_fname, index=False)
    print( 'Cell information is saved to:')
    print( cell_fname)

    return df

def cell_get_protein( ss_c, protcell_fname):
    protein_l = [ss_c[0]]
    for protein in ss_c[1:-1]:
        if protein[1].isdigit():
            if protein[2:] == "Actin_dual":
                protein_l.append( "Actin")
            else:
                protein_l.append( protein[2:])
        else:
            if protein[1:] == "Actin_dual":
                protein_l.append( "Actin")
            else:
                protein_l.append( protein[1:])

    # Arp23 --> arp23 since I used that before
    p_l = []
    for p in protein_l:
        if p == "Arp23":
            p_l.append( "arp23")
        else:
            p_l.append( p)

    p_set = set(p_l)
    p_set_l = list( p_set)
    print(p_set)

    protein_df = pd.DataFrame()
    protein_df["Protein"] = p_l
    prot_fname = protcell_fname[:-4] +'_Protein.csv'
    protein_df.to_csv( prot_fname, index = False)
    print( 'Protein information is saved to:')
    print( prot_fname)
    
    return protein_df

def cell_get_protein_cell_from_matcsv( 
        fold = "../data/For_feature_classification_2/Matlab_f2/", 
        fname = "protcell_fname = dminpool_more_56.dminpoolc.csv"):
    """
    Return
    ======
    prot_df, pd.DataFrame

    cell_df, pd.DataFrame
    """

    # Read specific information such as protein and cell
    with open( fold + fname) as f:
        s = f.readline()

    # Remove comma
    s_l = s.split(",")
    ss = "".join(s_l)

    # split by _C since _C is included all strings
    ss_c = ss.split("_C")
    print( len(ss_c))

    prot_df = cell_get_protein( ss_c, fold + fname)
    cell_df = cell_get_cell( ss_c, fold + fname)

    return prot_df, cell_df

def cell_get_protein_from_matcsv( 
        fold = "../data/For_feature_classification_2/Matlab_f2/", 
        fname = "protcell_fname = dminpool_more_56.dminpoolc.csv"):

    protcell_fname = fold + fname
    
    # Read specific information such as protein and cell
    with open( protcell_fname) as f:
        s = f.readline()

    # Remove comma
    s_l = s.split(",")
    ss = "".join(s_l)

    # split by _C since _C is included all strings
    ss_c = ss.split("_C")
    print( len(ss_c))

    protein_l = [ss_c[0]]
    for protein in ss_c[1:-1]:
        if protein[1].isdigit():
            if protein[2:] == "Actin_dual":
                protein_l.append( "Actin")
            else:
                protein_l.append( protein[2:])
        else:
            if protein[1:] == "Actin_dual":
                protein_l.append( "Actin")
            else:
                protein_l.append( protein[1:])

    # Arp23 --> arp23 since I used that before
    p_l = []
    for p in protein_l:
        if p == "Arp23":
            p_l.append( "arp23")
        else:
            p_l.append( p)

    p_set = set(p_l)
    p_set_l = list( p_set)
    print(p_set)

    protein_df = pd.DataFrame()
    protein_df["Protein"] = p_l
    protein_df.to_csv( protcell_fname[-4] +'_Protein.csv', index = False)
    print( 'Protein information is saved to:')
    print( protcell_fname[-4] +'_Protein.csv')
    
    return protein_df

def get_new_sample_idx( 
        fold = '../data/For_feature_classification_2/Matlab_f2_3clusters/',
        fname = 'dminpool_more_56.dminpoolc_Protein.csv'): 
    
    protein_df = pd.read_csv( fold + fname)
    prot_a = protein_df["Protein"].values
    p_list, p_ln_l = [], []
    p_cur = ''
    p_cnt = 0
    for p in prot_a:
        if p != p_cur:
            if p_cur != '':
                print(p)
                p_ln_l.append(p_cnt)
                p_cnt = 0
            p_list.append( p)
            p_cur = p
        p_cnt += 1

    p_ln_l.append( prot_a.shape[0] - np.sum(p_ln_l))
    print(p_list, p_ln_l)

    for p in p_list:
        pos_2 = np.where( prot_a == p)
        pos = pos_2[0]
        l = pos[-1] - pos[0] + 1
        print( p, pos[0], pos[-1], pos[-1] - pos[0] + 1, 
              np.sum( pos - pos[0]), np.sum( pos - pos[0]) == l*(l-1)/2)
    
    new_sample_idx = []
    for pl in p_ln_l:
        new_sample_idx.extend( list( range( pl)))
        
    return new_sample_idx