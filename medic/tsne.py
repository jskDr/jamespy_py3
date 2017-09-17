import pandas as pd
import numpy as np

from medic import dl, kdl


def load_data(fname_gz):
    cell_df = pd.read_csv(fname_gz)

    Lx = cell_df['x'].max() + 1
    Ly = cell_df['y'].max() + 1
    Limg = cell_df['ID'].max() + 1
    print(Limg, Lx, Ly)

    return cell_df, Limg, Lx, Ly


def get_Xy(cell_df, Limg, Lx, Ly):
    cell_img_a = cell_df["image"].values.reshape(Limg, 1, Lx, Ly)

    cell_y = cell_df[(cell_df["x"] == 0) & (cell_df["y"] == 0)]["n_beads"].values
    cell_y.shape

    X = cell_img_a.reshape(cell_img_a.shape[0], -1)
    y = cell_y
    nb_classes = np.max(y) + 1

    print(X.shape, y.shape, nb_classes, set(y))

    return X, y, nb_classes


def main():
    # Generate and save data
    n_each = 10
    max_bd = 4
    N = n_each * max_bd
    fname_gz = "sheet.gz/cell_db{}_center_cell_3_nooverlap.cvs.gz".format(N)
    _ = kdl.gen_save_cell_db(N=N, fname_gz=fname_gz,
                             rand_pos_cell=False, max_bd=max_bd,
                             classification_mode="Center_Cell",
                             extra_bead_on=False,
                             flag_no_overlap_beads=True,
                             disp=2)

    # Load data
    cell_df, Limg, Lx, Ly = load_data(fname_gz)
    X, y, nb_classes = get_Xy(cell_df, Limg, Lx, Ly)

    machine = dl.Machine(X, y, Lx, Ly, nb_classes=nb_classes)
    machine.run(nb_epoch=300, verbose=0)


if __name__ == '__main__':
    main()
