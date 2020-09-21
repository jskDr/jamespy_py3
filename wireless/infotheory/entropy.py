from sage.all import *

class DRV:
    # descrete random variable
    def __init__(self, P=None):
        self.p = var('p')
        self.P = P # not define yet

class SUM:
    def __init__(self, f, p, P=None):
        self.f = f # function
        self.p = p # argument for sum
        self.P = P # all p
    
    def __repr__(self):
        if self.P is None:
            return f'Σ_{self.p} {self.f}'
        else:
            return f'Σ_{self.p} {self.f} in {self.P}'
        
    def __str__(self):
        return self.__repr__()
    
    def calc(self):
        if self.P is not None:
            H = 0
            for p in self.P:
                H += self.f(p=p)
            return H
        else:
            return self
        
def H(X):
    """
    Caculate entropy of descrete random variable X
    """
    f = -X.p*log(X.p,2) # use log2 for bits
    if X.P is None:
        return SUM(f, X.p, X.P)
    else:
        return SUM(f, X.p, X.P).calc().simplify()

def test_010():
    print('Not defined X')
    X = DRV()
    print(H(X))

def test_020():
    print('Defined pdf of X = [p, 1-p]')
    p = var('p')
    X = DRV([p, 1-p])
    Hx = H(X)

    print('Plot Entropy')
    pretty_print(Hx)
    p = plot(Hx,p,0,1)
    show(p)
    
    print('Maximum point')
    pretty_print(diff(Hx))
    p = plot(diff(Hx,p,0,1))
    show(p)
    
def test_030():
    print('Defined pdf of X = [1/2, 1/2]')
    X = DRV([Rational('1/2'), Rational('1/2')])
    Hx = H(X)

    print('Entropy: H(X)')
    pretty_print(Hx)

def entropy(p_list):
    h = 0
    for p in p_list:
        h += -p*log(p,2)
    return h

def entropy_x_and_y(Px_and_y_2d_list):
    h = 0
    for Px_row_list in Px_and_y_2d_list:
        for Px_and_y in Px_row_list:
            p = Px_and_y
            h += -p*log(p,2)
    return h    

def test_entropy_x_and_y():
    h = entropy_x_and_y([[Rational('1/8'),Rational('3/4')],[Rational('1/4'),Rational('1/4')]])
    pretty_print(h)

    print('Simple case: H(X,Y)')
    p = var('p')
    h = entropy_x_and_y([[(1-p)/2, p/2], [p/2, (1-p)/2]])
    img = plot(h, p, 0, 1)
    show(img)

def test_entropy():
    p = var('p')
    h = entropy([p, 1-p])
    img = plot(h, h, 0, 1, axes_labels=[r'$p$', r'$H(p)$'], 
               title='Entropy of $X$ ~ $(p, 1-p)$, i.e., $H(p) = −𝑝log_2(𝑝)+(𝑝−1)log_2(−𝑝+1)$')
    show(img)
    pretty_print(h)

class InformationEntropyX:
    def __init__(self, Px_list):
        self.Px_list = Px_list
    def Hx(self):
        h = 0
        for Px in self.Px_list:
            h += -Px*log(Px,2)
        return h

def test_InformationEntropyX():
    p = var('p')
    h = InformationEntropyX([p, 1-p]).Hx()
    img = plot(h, h, 0, 1, axes_labels=[r'$p$', r'$H(p)$'], 
               title='Entropy of $X$ ~ $(p, 1-p)$, i.e., $H(p) = −𝑝log_2(𝑝)+(𝑝−1)log_2(−𝑝+1)$')
    show(img)
    pretty_print(h)

class InformationEntropyXY:
    def __init__(self, Py_given_x_list):
        self.Px_list = Px_list
    def Hx(self):
        h = 0
        for Px in self.Px_list:
            h += -Px*log(Px,2)
        return h

