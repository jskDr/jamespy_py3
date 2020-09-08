from sage.all import *

class DRV:
    # descrete random variable
    def __init__(self):
        self.x = var('x')
        self.p = function('p')
        self.dist = None # not define yet

class SUM:
    def __init__(self, f, x, X=None):
        self.f = f # function
        self.x = x # argument for sum
        self.X = X # all x
    
    def __repr__(self):
        if self.X is None:
            return f'Σ_{self.x} {self.f}'
        else:
            return f'Σ_{self.x} {self.f} in {self.X}'
        
    def __str__(self):
        return self.__repr__()
        
def H(X):
    """
    Caculate entropy of descrete random variable X
    """
    return SUM(p(X.x)*log(X.p(X.x)), X.x, X.dist)

def test_010():
    X = DRV()
    print(H(X))