import scipy.misc as misc
from matplotlib.pyplot import *
from numpy import diag, dot
from numpy.linalg import svd, norm, matrix_rank

image = misc.ascent()
M = svd(image)
U = M[0]
S = M[1]
V_T = M[2]
s_full = diag(S)
r = matrix_rank(image)
n = len(image)
compression_ratio = []
frobenius_distance = []
for k in range(512):
    S[511-k] = 0
    s_full = diag(S)
    M_k = dot(dot(U,s_full),V_T)
    compression_ratio.append((2*k*n + k)/(2*n*r + r))
    frobenius_distance.append(norm(image - M_k))
    if k in {11,211,411,471,501}:
        imshow(M_k)
        title("k = "+str(511-k)+"\n"+"Compression ratio = "+str(compression_ratio[k])+"\n"+"Frobenius distance = "+str(frobenius_distance[k]))
        # gray()
        show()

plot(list(range(512)),frobenius_distance[::-1])
axis([0,512,0,25000])
ylabel("Frobineus distance from original image")
xlabel("k")
title("Frobenius distance")
show()
plot(list(range(512)),compression_ratio)
axis([0,512,0,1])
ylabel("Compression ratio")
xlabel("k")
title("Compression ratio")
show()