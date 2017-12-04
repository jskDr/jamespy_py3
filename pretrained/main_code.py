import numpy as np
from keras.preprocessing import image

from amap.proc import HaS, get_rgb_mean
import pretrain


def get_rgb_mean_fold(fold='tiny-imagenet-200/train/'):
    fold = 'tiny-imagenet-200/train/'
    ig = image.ImageDataGenerator()  # rescale=1/255.0)
    it = ig.flow_from_directory(fold, target_size=(64, 64), batch_size=1000)
    mn_l = []
    for batch in range(it.samples // it.batch_size):
        X, _ = it.next()
        mn_l.append(get_rgb_mean(X))

    rgb_mean = np.mean(mn_l, axis=0)
    return rgb_mean


def main():
    # rgb_mean = get_rgb_mean_fold()
    # print(rgb_mean)
    rgb_mean = np.array([122.47496033,  114.24460602,
                         101.3369751], dtype=np.float32)

    print("HaS is included")
    pretrain.run(2, HaS, rgb_mean)

    print("HaS is not included")
    pretrain.run(2)


if __name__ == '__main__':
    main()
