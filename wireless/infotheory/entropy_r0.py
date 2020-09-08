from sage.all import *

class DRV:
    # descrete random variable
    def __init__(self):
        self.x = var('x')
        self.p = function('p')
        
def H(X):
    """
    Caculate entropy of descrete random variable X
    """
    x = X.x
    p = X.p
    return p(x)*log(x)

def test_010():
    X = DRV()
    print(H(X))
