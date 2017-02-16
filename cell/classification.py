# classification.py
import os
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import manifold, svm, model_selection, tree, metrics, cluster, ensemble
# from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
from time import time

from imblearn.under_sampling import RandomUnderSampler

# import kutil
# import kgrid
# import kkeras, kkeras_cv
import kcellml
# import jmath
import kmds
# from kcellml import pd_clsf2_by_clst, pd_clsf2_by_yint


def read_data(fname, Da, Db):
    """
    Read data for processing
    """
    Data = loadmat(fname)
    print(Data.keys())

    X2part = np.concatenate([Data[Da], Data[Db]], axis=0)
    print(X2part.shape)

    y = np.array([0] * Data[Da].shape[0] + [1] * Data[Db].shape[0])
    print(y.shape)

    return X2part, y


def get_X_y2_nb_classes(X2part, y):
    X = X2part
    y2 = y.copy()
    # plt.hist( y2)
    nb_classes = len(set(y2))

    return X, y2, nb_classes


def _do_classification_r0(X, y2, nb_classes, M=10, N=10):
    """
    Perform classification
    """
    defalut_param_d = {
        'SVC:C': 1.93, 'SVC:gamma': 0.037,
        'RF:n_estimators': 100, 'RF:oob_score': True
    }
    kcellml.setparam(defalut_param_d)

    clsf2_by_yint = kcellml.GET_clsf2_by_yint(nb_classes, confusion_matric_return=True,
                                              matthews_corrcoef_return=True)
    sc_ll = []
    cf_lll = []
    mc_ll = []
    dt_agg = 0
    for i_rs in range(N):
        print("i_rs", i_rs)
        Xeq, y2eq = RandomUnderSampler().fit_sample(X, y2)
        st = time()
        for i_sp in range(M):
            # print( "i_sp", i_sp)
            sc_cf_lx = clsf2_by_yint(Xeq, y2eq, test_size=0.3, disp=False)
            sc_ll.append(sc_cf_lx[:5])
            cf_lll.append(sc_cf_lx[5:10])
            mc_ll.append(sc_cf_lx[10:])
        ed = time()
        dt = ed - st
        dt_agg += dt
        print("Elapsed: {:.2f}s".format(dt),
              "Remaind time: {:.2f}m".format((N - i_rs - 1) * (dt_agg) / (i_rs + 1) / 60))
    sc_a2d = np.array(sc_ll)
    print(sc_a2d.shape)
    cf_a3d = np.array(cf_lll)
    print(cf_a3d.shape)
    mc_a2d = np.array(mc_ll)
    print(mc_a2d.shape)

    return sc_a2d, mc_a2d, cf_a3d


def do_classification(X, y2, nb_classes, M=10, N=10):
    """
    Perform classification
    """
    defalut_param_d = {
        'SVC:C': 1.93, 'SVC:gamma': 0.037,
        'RF:n_estimators': 100, 'RF:oob_score': True
    }
    kcellml.setparam(defalut_param_d)

    if nb_classes == 2:
        clsf2_by_yint = kcellml.GET_clsf2_by_yint(nb_classes,
                                                  confusion_matric_return=True,
                                                  matthews_corrcoef_return=True)
    elif nb_classes > 2:
        clsf2_by_yint = kcellml.GET_clsf2_by_yint(nb_classes,
                                                  confusion_matric_return=True,
                                                  matthews_corrcoef_return=False)
    else:
        raise ValueError('nb_classes should be equal to or larger than 2.')

    sc_ll = []
    cf_lll = []
    mc_ll = []
    dt_agg = 0
    for i_rs in range(N):
        print("i_rs", i_rs)
        Xeq, y2eq = RandomUnderSampler().fit_sample(X, y2)
        st = time()
        for i_sp in range(M):
            # print( "i_sp", i_sp)
            sc_cf_lx = clsf2_by_yint(Xeq, y2eq, test_size=0.3, disp=False)
            sc_ll.append(sc_cf_lx[:5])
            cf_lll.append(sc_cf_lx[5:10])
            if nb_classes == 2:
                mc_ll.append(sc_cf_lx[10:])
        ed = time()
        dt = ed - st
        dt_agg += dt
        print("Elapsed: {:.2f}s".format(dt),
              "Remaind time: {:.2f}m".format((N - i_rs - 1) * (dt_agg) / (i_rs + 1) / 60))
    sc_a2d = np.array(sc_ll)
    print(sc_a2d.shape)
    cf_a3d = np.array(cf_lll)
    print(cf_a3d.shape)
    
    if nb_classes == 2:
        mc_a2d = np.array(mc_ll)
        print(mc_a2d.shape)

        return sc_a2d, mc_a2d, cf_a3d
    else:
        return sc_a2d, cf_a3d


def _save_result_r0(fold, sc_a2d, mc_a2d, cf_a3d):
    if not os.path.exists(fold):
        os.mkdir(fold)
        # os.mkdir(fold+'/sheet')
    fold += '/'

    plt.boxplot(sc_a2d)
    plt.show()
    print('Accuracy', ["DT", "SVC", "DNN", "CDNN", "RF"])
    print(np.average(sc_a2d, axis=0))
    plt.boxplot(mc_a2d)
    plt.show()
    print('Matthews Corrcoef', ["DT", "SVC", "DNN", "CDNN", "RF"])
    print(np.average(mc_a2d, axis=0))

    np.save(fold + 'sc_a2d', sc_a2d)
    np.save(fold + 'cf_a3d', cf_a3d)
    np.save(fold + 'mc_a2d', mc_a2d)

    sc_df = pd.DataFrame(sc_a2d, columns=["DT", "SVC", "DNN", "CDNN", "RF"])
    # sc_df.plot(kind='bar')
    # plt.show()
    sc_df.to_csv(fold + 'sheet_sc_a2d.csv')
    sc_df.head()

    mc_df = pd.DataFrame(mc_a2d, columns=["DT", "SVC", "DNN", "CDNN", "RF"])
    # mc_df.plot(kind='bar')
    # plt.show()
    mc_df.to_csv(fold + 'sheet_mc_a2d.csv')
    mc_df.head()

    cf_a3d_avg = np.average(cf_a3d, axis=0)

    mode_l = ['DT', 'SVC', 'DNN', 'CDNN', 'RF']
    with open(fold + 'cf_a3d_avg.txt', 'w') as F:
        # dt_score, sv_score, mlp_score, cnn_score, rf_score
        for i, mode in enumerate(mode_l):
            F.write("{} Confusion Metrics\n".format(mode))
            F.write(", ".join([str(x) for x in cf_a3d_avg[i, 0, :]]))
            F.write('\n')
            F.write(", ".join([str(x) for x in cf_a3d_avg[i, 1, :]]))
            F.write('\n\n')

    print("Current working directory:")
    print(os.getcwd() + '/' + fold)
    print("Saved data")
    print(os.listdir(fold))


def save_result(fold, sc_a2d, mc_a2d, cf_a3d):
    if not os.path.exists(fold):
        os.mkdir(fold)
        # os.mkdir(fold+'/sheet')
    fold += '/'

    plt.boxplot(sc_a2d)
    plt.show()
    print('Accuracy', ["DT", "SVC", "DNN", "CDNN", "RF"])
    print(np.average(sc_a2d, axis=0))
    plt.boxplot(mc_a2d)
    plt.show()
    print('Matthews Corrcoef', ["DT", "SVC", "DNN", "CDNN", "RF"])
    print(np.average(mc_a2d, axis=0))

    np.save(fold + 'sc_a2d', sc_a2d)
    np.save(fold + 'cf_a3d', cf_a3d)
    np.save(fold + 'mc_a2d', mc_a2d)

    sc_df = pd.DataFrame(sc_a2d, columns=["DT", "SVC", "DNN", "CDNN", "RF"])
    # sc_df.plot(kind='bar')
    # plt.show()
    sc_df.to_csv(fold + 'sheet_sc_a2d.csv')
    sc_df.head()

    mc_df = pd.DataFrame(mc_a2d, columns=["DT", "SVC", "DNN", "CDNN", "RF"])
    # mc_df.plot(kind='bar')
    # plt.show()
    mc_df.to_csv(fold + 'sheet_mc_a2d.csv')
    mc_df.head()

    cf_a3d_avg = np.average(cf_a3d, axis=0)

    mode_l = ['DT', 'SVC', 'DNN', 'CDNN', 'RF']
    with open(fold + 'cf_a3d_avg.txt', 'w') as F:
        # dt_score, sv_score, mlp_score, cnn_score, rf_score
        for i, mode in enumerate(mode_l):
            F.write("{} Confusion Metrics\n".format(mode))
            for j in range(cf_a3d_avg.shape[1]):
                F.write(", ".join([str(x) for x in cf_a3d_avg[i, j, :]]))
                F.write('\n')
            # F.write(", ".join([str(x) for x in cf_a3d_avg[i, 1, :]]))
            F.write('\n')

    print("Current working directory:")
    print(os.getcwd() + '/' + fold)
    print("Saved data")
    print(os.listdir(fold))


def save_result_multiple_classes(fold, sc_a2d, cf_a3d):
    if not os.path.exists(fold):
        os.mkdir(fold)
        # os.mkdir(fold+'/sheet')
    fold += '/'

    plt.boxplot(sc_a2d)
    plt.xticks(range(1,6), ["DT", "SVC", "DNN", "CDNN", "RF"])
    plt.show()
    print('Accuracy', ["DT", "SVC", "DNN", "CDNN", "RF"])
    print(np.average(sc_a2d, axis=0))
    
    # plt.boxplot(mc_a2d)
    # plt.show()
    # print('Matthews Corrcoef', ["DT", "SVC", "DNN", "CDNN", "RF"])
    # print(np.average(mc_a2d, axis=0))

    np.save(fold + 'sc_a2d', sc_a2d)
    np.save(fold + 'cf_a3d', cf_a3d)
    # np.save(fold + 'mc_a2d', mc_a2d)

    sc_df = pd.DataFrame(sc_a2d, columns=["DT", "SVC", "DNN", "CDNN", "RF"])
    # sc_df.plot(kind='bar')
    # plt.show()
    sc_df.to_csv(fold + 'sheet_sc_a2d.csv')
    sc_df.head()

    # mc_df = pd.DataFrame(mc_a2d, columns=["DT", "SVC", "DNN", "CDNN", "RF"])
    # mc_df.plot(kind='bar')
    # plt.show()
    # mc_df.to_csv(fold + 'sheet_mc_a2d.csv')
    # mc_df.head()

    cf_a3d_avg = np.average(cf_a3d, axis=0)

    mode_l = ['DT', 'SVC', 'DNN', 'CDNN', 'RF']
    with open(fold + 'cf_a3d_avg.txt', 'w') as F:
        # dt_score, sv_score, mlp_score, cnn_score, rf_score
        for i, mode in enumerate(mode_l):
            F.write("{} Confusion Metrics\n".format(mode))
            for j in range(cf_a3d_avg.shape[1]):
                F.write(", ".join([str(x) for x in cf_a3d_avg[i, j, :]]))
                F.write('\n')
            # F.write(", ".join([str(x) for x in cf_a3d_avg[i, 1, :]]))
            F.write('\n')

    print("Current working directory:")
    print(os.getcwd() + '/' + fold)
    print("Saved data")
    print(os.listdir(fold))


def run(fname, Da, Db, save_fold,
        M=10, N=10):
    """
    Run this code.

    Input
    =====
    fname = 'sheet/classification/VASP_classfication_mapping_data.mat'
    Da, Db = 'Cluster1', 'Cluster23'
    save_fold = 'XX-VASP_1vs23_map'

    M, N = 10, 10
    Iteration counts for radomization and cross-validation
    """

    print("Reading data...")
    X2part, y = read_data(fname, Da, Db)

    X, y2, nb_classes = get_X_y2_nb_classes(X2part, y)

    print("Doing classificaiton...")
    sc_a2d, mc_a2d, cf_a3d = do_classification(X, y2, nb_classes, M=M, N=N)

    print("Saving the results...")
    save_result(save_fold, sc_a2d, mc_a2d, cf_a3d)


# ====================================
# Multiple Classes
# ====================================
def read_data_multple_classes(fname, D_l):
    """
    Read data for processing
    """
    Data = loadmat(fname)
    print(Data.keys())

    Data_con = [Data[Dx] for Dx in D_l]

    X2part = np.concatenate(Data_con, axis=0)
    print(X2part.shape)

    y_l = []
    for n, Dx in enumerate(D_l):
        y_l.extend([n] * Data[Dx].shape[0])

    y = np.array(y_l)
    print(y.shape)

    return X2part, y


def run_multiple_classes(fname, D_l, save_fold,
                         M=10, N=10):
    """
    Run this code.

    Input
    =====
    fname = 'sheet/classification/VASP_classfication_mapping_data.mat'
    Da, Db = 'Cluster1', 'Cluster23'
    save_fold = 'XX-VASP_1vs23_map'

    M, N = 10, 10
    Iteration counts for radomization and cross-validation
    """

    print("Reading data...")
    X2part, y = read_data_multple_classes(fname, D_l)

    X, y2, nb_classes = get_X_y2_nb_classes(X2part, y)
    print("...nb_classes ->", nb_classes)

    if nb_classes == 2:
        print("Doing classificaiton...")
        sc_a2d, mc_a2d, cf_a3d = do_classification(X, y2, nb_classes, M=M, N=N)

        print("Saving the results...")
        save_result(save_fold, sc_a2d, mc_a2d, cf_a3d)
    elif nb_classes > 2:
        print("Doing classificaiton...")
        sc_a2d, cf_a3d = do_classification(X, y2, nb_classes, M=M, N=N)

        print("Saving the results...")
        save_result_multiple_classes(save_fold, sc_a2d, cf_a3d)
    else:
        raise ValueError('nb_classes should be equal to or larger than 2.')


def show_tSNE(fname, D_l):
    print("Reading data...")
    X2part, y = read_data_multple_classes(fname, D_l)

    print('Showing tSNE...')
    kmds.plot_tSNE(X2part, y, digit=False)
    plt.show()
    return X2part, y
