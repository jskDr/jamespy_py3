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
    
def frc_plot(P=1, BLER=0.001, n_range=[1,2000]):
    p1 = plot(R(n,P,BLER),(n,n_range[0],n_range[1]))
    p2 = line([(n_range[0],C(P)),(n_range[1],C(P))], linestyle='--')
    p = p1 + p2
    p.show(ymin=0, ymax=C(P)+0.05, gridlines=True, axes_labels=['$n$', '$R(n)$'])    