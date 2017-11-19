
import numpy as np
from PIL import Image

def resizeX(X, basewidth=224):
    # basewidth = 224
    X_small = []
    for x in X:
        img = Image.fromarray(x)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        x_small = np.array(img)
        X_small.append(x_small)
    return np.array(X_small)  