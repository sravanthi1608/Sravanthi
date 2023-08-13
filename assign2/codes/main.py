import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

A = np.array([1, -1])
B = np.array([-4, 6])
C = np.array([-3, -5])
BC = C - B

m_AD = np.array([-BC[1], BC[0]])  
  
equation_AD = np.array([A, m_AD])

def line_gen(A, B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(0, 1, len)
    for i in range(len):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    return x_AB


x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

AD_p = A
AD_m = m_AD
BC_p = np.array([-4, 6])
BC_m = np.array([-1, 11])

sp_p1 = BC_p - AD_p
det1 = np.cross(AD_m, BC_m)
t1 = np.cross(sp_p1, BC_m)/ det1
D = AD_p + t1*AD_m


# Plot the extended lines AC and AB
#plt.plot([B[0], C[0]],[B[1], C[1]], linestyle=':', color='b', label='Extended AC')
#plt.plot([A[0], F[0]],[A[1], F[1]], linestyle=':', color='b', label='Extended AB')

# Plot the shortened altitudes BE and CF
plt.plot([A[0], D[0]], [A[1], D[1]], linestyle='--', color='r', label='Altitude AD')
#plt.plot([C[0], F[0]], [C[1], F[1]], linestyle='--', color='g', label='Altitude CF')


A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
#F = F.reshape(-1,1)
tri_coords = np.block([[A,B,C,D]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


plt.show()
