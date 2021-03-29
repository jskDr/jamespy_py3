"""
npolar를 이용해 polar Tx-Rx에서 decision boundary를 나타낸다.
- K=2, N=2인 encoding후 노이즈가 추가된 수신 신호를 (x,y) 순서로 좌표에 표시한다.
- x, y 입력은 사실 각각 0, 1 사이의 uniform한 입력을 넣어도 된다. 
  왜냐하면 디코딩시 바운드리를 구하기 위함이기 때문에 실제 입력이 어떤 값이였는지는 중요하지 않다.
- 표시할때 각 심볼의 색은 decoding 결과 (u0, u1)에 따른다.
- 수학적으로 u0, u1을 복호하는 f(), g() 수식이 있기 때문에 이를 이용해 바운드를 표시하는 것도 방법이다.
  이 때 g()는 f() 결과의 부호와 관련이 있기 때문에 그 부분을 고려해서 해석해야 함.
"""
import numpy as np
import matplotlib.pyplot as plt

from wireless.npolar import decode_frozen_array_n, frozen_decode_array_n

def coding(N_iter, frozen_flag_n=np.zeros(2, dtype=int)):   
    # N_code=2인 무작위 uninform(0,1) 값들로 이루어진 심볼을 생성함
    y_array = np.random.uniform(-2, 2, size=(N_iter, 2)) # -1 to 1
    ufd_array = decode_frozen_array_n(y_array, frozen_flag_n) # frozen을 고려한 함수로 변경되어야 함!
    ud_array = frozen_decode_array_n(ufd_array, frozen_flag_n)    
    return y_array, ud_array


def plot_boundary(N_iter=10):
    """
    Polar table
  s c:  u0 u1: x0 x1: y0 y1 (no noise 1-2x) 
  0 r:  0  0:  0  0:  1  1
  1 g:  0  1:  1  1: -1 -1
  2 b:  1  0:  1  0: -1  1
  3 k:  1  1:  0  1:  1 -1   
    """    
    y, ud = coding(N_iter)
    ud_s = ud[:,0]*2 + ud[:,1]
    ct = np.array(['r', 'g', 'b', 'k'])
    plt.scatter(y[:,0], y[:,1], color=ct[ud_s], marker='.')
    
def plot_boundary_K1(N_iter=10, frozen_flag_n=np.array([1,0])):
    """
  frozen_flag_n=np.array([1,0])
  Notice that Frozen force u0 = 0 (no 1)
  s c:  u0 u1: x0 x1: y0 y1 (no noise 1-2x) 
  0 r:  0  0:  0  0:  1  1
  1 b:  0  1:  1  1: -1 -1
   X :  1  0:  1  0: -1  1
   X :  1  1:  0  1:  1 -1   

 frozen_flag_n=np.array([0,1])
  Notice that Frozen force u0 = 0 (no 1)
  s c:  u0 u1: x0 x1: y0 y1 (no noise 1-2x) 
  0 r:  0  0:  0  0:  1  1
   X :  0  1:  1  1: -1 -1
  1 b:  1  0:  1  0: -1  1
   X :  1  1:  0  1:  1 -1   
    """        
    y, ud = coding(N_iter, frozen_flag_n)
    ud_s = ud[:,0] #[:,0] + ud[:,1]*2
    ct = np.array(['r', 'b'])
    plt.scatter(y[:,0], y[:,1], color=ct[ud_s], marker='.')

if __name__ == '__main__':
    plt.subplot(1,3,1)
    plot_boundary(10000)
    plt.subplot(1,3,2)
    plot_boundary_K1(10000, frozen_flag_n=np.array([1,0]))
    plt.subplot(1,3,3)
    plot_boundary_K1(10000, frozen_flag_n=np.array([0,1]))
    plt.show()