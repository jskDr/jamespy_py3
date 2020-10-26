def get_C_R():
    n, P, epsilon, t = var('n P epsilon t')

    C(P) = (1/2)*log(1+P,2)

    V(P) = (P/2) * (P+2) / (P+1)**2 * log(e,2)**2

    rho(n) = (1/2)*log(n,2)

    q(t) = (1/sqrt(2*pi))*e**((-t**2)/2)
    Q(x) = integral(q(t),(t,x,oo))
    ans = solve(Q(x) == epsilon, x)
    invQ(epsilon) = ans[0].rhs()

    M(n,P,epsilon) = n*C(P) - sqrt(n*V(P))*invQ(epsilon) + rho(n)
    R(n,P,epsilon) = M(n,P,epsilon)/n
    return C, R
    

def frc_plot(P=1, BLER=0.001, n_range=[1,2000]):
    C, R = get_C_R()
    p1 = plot(R(n,P,BLER),(n,n_range[0],n_range[1]))
    p2 = line([(n_range[0],C(P)),(n_range[1],C(P))], linestyle='--')
    p = p1 + p2
    p.show(ymin=0, ymax=C(P)+0.05, gridlines=True, axes_labels=['$n$', '$R(n)$'])    


def get_C(rho_0, z_range=[-10,10], fig_flag=True):
    rho, z = var('rho, z')
    c(rho,z) = 1/sqrt(2*pi) * e^(-z^2/2)*(1 - log(1+e^(-2*rho+2*z*sqrt(rho)),2))
    pretty_print('c:', c)
    if fig_flag:
        p1 = plot(c(rho_0,z),z,z_range[0],z_range[1])
        show(p1)
    C = integrate(c(rho=rho_0),(z,z_range[0],z_range[1]))
    return C.n()

def get_V(rho_0, z_range=[-10,10], fig_flag=True):
    rho, z = var('rho, z')
    C = get_C(rho_0,z_range=z_range)
    v(rho,z) = 1/sqrt(2*pi) * e^(-z^2/2)*(1 - log(1 + e^(-2*rho + 2*z*sqrt(rho)),2) - C)^2
    pretty_print('v:', v)
    if fig_flag:
        p1 = plot(v(rho_0,z),z,z_range[0],z_range[1])
        show(p1)
    V = integrate(v(rho=rho_0),(z,z_range[0],z_range[1]))
    return V.n()

def get_epsilon(EbNo_dB, R_0): 
    LN, t = var('LN t')
    EbNo = 10^(EbNo_dB/10)
    rho_0 = 2*R_0*EbNo
    q(t) = (1/sqrt(2*pi))*e^((-t^2)/2)
    Q(x) = integral(q(t),(t,x,oo))
    ep = (LN*(get_C(rho_0) - R_0) + (1/2)*log(LN,2))/sqrt(LN*get_V(rho_0))
    pretty_print('ep:', ep)
    EP = Q(ep)
    pretty_print('EP:', EP)
    return EP    