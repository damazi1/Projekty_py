import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.integrate import dblquad
import sympy as sp


ma = np.loadtxt('136700.dat')
n = 21
m = 11

x = np.linspace(0, 2, n)
y = np.linspace(0, 1, m)
x, y = np.meshgrid(x, y)
z = np.zeros((m, n))


class Projekt:
    def __init__(self):
        self.wartosc = []

    def wyznacz_xyz(self, ma):
        n = ma.shape[0]
        x1, y1, z1 = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            x1[i] = ma[i][0]
            y1[i] = ma[i][1]
            z1[i] = ma[i][2]
        return x1, y1, z1

    def wyznacz(self, z, n, m, ma):
        k = 0
        for i in range(m):
            for j in range(n):
                z[i][j] = ma[k, 2]
                k += 1

    def wykres2D(self, x, y, z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none')
        plt.title('Mapa 2D', fontdict={
                  'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_box_aspect([2, 1, 0.0001])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def wykres3D(self, x, y, z):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none')
        plt.title('Mapa 3D', fontdict={
                  'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_box_aspect([2, 1, 1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def srednia(self, y, z):
        unique_y = np.unique(y)
        i = 0
        for y_val in unique_y:
            indices = np.where(y == y_val)
            print(unique_y[i], ' ', np.mean(z[indices]))
            i += 1

    def mediana(self, y, z):
        unique_y = np.unique(y)
        i = 0
        for y_val in unique_y:
            indices = np.where(y == y_val)
            print(unique_y[i], ' ', np.median(z[indices]))
            i += 1

    def odchylenie(self, y, z):
        unique_y = np.unique(y)
        i = 0
        for y_val in unique_y:
            indices = np.where(y == y_val)
            print(unique_y[i], ' ', np.std(z[indices]))
            i += 1

    def mianownik(self, x, k, n):
        m = 1
        for j in range(n):
            if j != k:
                m *= x[k]-x[j]
        return m

    def mianownik1(self, x, x1, k, n):
        m = 1
        for j in range(n):
            if j != k:
                m *= x1-x[j]
        return m

    def lan(self, x, y, iks, n):
        W = 0
        for j in range(n):
            W += (y[j]/(projekt.mianownik(x, j, n))) * \
                projekt.mianownik1(x, iks, j, n)
        return W

    def funl(self, X, Y):
        plt.plot(X, Y, 'bo')
        x1 = np.zeros(4)
        x2 = np.zeros(4)
        x3 = np.zeros(4)
        x4 = np.zeros(4)
        x5 = np.zeros(4)
        x6 = np.zeros(4)
        x7 = np.zeros(3)

        y1 = np.zeros(4)
        y2 = np.zeros(4)
        y3 = np.zeros(4)
        y4 = np.zeros(4)
        y5 = np.zeros(4)
        y6 = np.zeros(4)
        y7 = np.zeros(3)

        x1[0] = X[0]
        x1[1] = X[1]
        x1[2] = X[2]
        x1[3] = X[3]
        x2[0] = X[3]
        x2[1] = X[4]
        x2[2] = X[5]
        x2[3] = X[6]
        x3[0] = X[6]
        x3[1] = X[7]
        x3[2] = X[8]
        x3[3] = X[9]
        x4[0] = X[9]
        x4[1] = X[10]
        x4[2] = X[11]
        x4[3] = X[12]
        x5[0] = X[12]
        x5[1] = X[13]
        x5[2] = X[14]
        x5[3] = X[15]
        x6[0] = X[15]
        x6[1] = X[16]
        x6[2] = X[17]
        x6[3] = X[18]
        x7[0] = X[18]
        x7[1] = X[19]
        x7[2] = X[20]

        y1[0] = Y[0]
        y1[1] = Y[1]
        y1[2] = Y[2]
        y1[3] = Y[3]
        y2[0] = Y[3]
        y2[1] = Y[4]
        y2[2] = Y[5]
        y2[3] = Y[6]
        y3[0] = Y[6]
        y3[1] = Y[7]
        y3[2] = Y[8]
        y3[3] = Y[9]
        y4[0] = Y[9]
        y4[1] = Y[10]
        y4[2] = Y[11]
        y4[3] = Y[12]
        y5[0] = Y[12]
        y5[1] = Y[13]
        y5[2] = Y[14]
        y5[3] = Y[15]
        y6[0] = Y[15]
        y6[1] = Y[16]
        y6[2] = Y[17]
        y6[3] = Y[18]
        y7[0] = Y[18]
        y7[1] = Y[19]
        y7[2] = Y[20]

        iks = np.linspace(x1[0], x1[3], 100)
        plt.plot(iks, projekt.lan(x1, y1, iks, 4))
        iks = np.linspace(x2[0], x2[3], 100)
        plt.plot(iks, projekt.lan(x2, y2, iks, 4))
        iks = np.linspace(x3[0], x3[3], 100)
        plt.plot(iks, projekt.lan(x3, y3, iks, 4))
        iks = np.linspace(x4[0], x4[3], 100)
        plt.plot(iks, projekt.lan(x4, y4, iks, 4))
        iks = np.linspace(x5[0], x5[3], 100)
        plt.plot(iks, projekt.lan(x5, y5, iks, 4))
        iks = np.linspace(x6[0], x6[3], 100)
        plt.plot(iks, projekt.lan(x6, y6, iks, 4))
        iks = np.linspace(x7[0], x7[2], 100)
        plt.plot(iks, projekt.lan(x7, y7, iks, 3))

        plt.show()

    def aproksymacja3(self, x, y):
        M1 = np.zeros((3, 3))
        P1 = np.zeros(3)
        n = 3
        M1[0, 0] = n
        for i in range(n):
            M1[0, 1] = M1[0, 1]+x[i]
            M1[0, 2] = M1[0, 2]+x[i]**2
            M1[1, 0] = M1[1, 0]+x[i]
            M1[1, 1] = M1[1, 1]+x[i]**2
            M1[1, 2] = M1[1, 2]+x[i]**3
            M1[2, 0] = M1[2, 0]+x[i]**2
            M1[2, 1] = M1[2, 1]+x[i]**3
            M1[2, 2] = M1[2, 2]+x[i]**4

        for i in range(n):
            P1[0] += y[i]
            P1[1] += x[i]*y[i]
            P1[2] += (x[i]**2)*y[i]

        K1 = np.linalg.solve(M1, P1)

        return K1

    def f1(self, a0, a1, a2, x):
        return a0+a1*x+a2*x**2
    def fx1(self,a1,a2,x):
        return a1+a2*x
    def aproksymacja4(self, x, y):
        M1 = np.zeros((4, 4))
        P1 = np.zeros(4)
        n = 4
        M1[0, 0] = n
        for i in range(n):
            M1[0, 1] = M1[0, 1]+x[i]
            M1[0, 2] = M1[0, 2]+x[i]**2
            M1[0, 3] = M1[0, 3]+x[i]**3
            M1[1, 0] = M1[1, 0]+x[i]
            M1[1, 1] = M1[1, 1]+x[i]**2
            M1[1, 2] = M1[1, 2]+x[i]**3
            M1[1, 3] = M1[1, 3]+x[i]**4
            M1[2, 0] = M1[2, 0]+x[i]**2
            M1[2, 1] = M1[2, 1]+x[i]**3
            M1[2, 2] = M1[2, 2]+x[i]**4
            M1[2, 3] = M1[2, 3]+x[i]**5
            M1[3, 0] = M1[3, 0]+x[i]**3
            M1[3, 1] = M1[3, 1]+x[i]**4
            M1[3, 2] = M1[3, 2]+x[i]**5
            M1[3, 3] = M1[3, 3]+x[i]**6

        for i in range(n):
            P1[0] += y[i]
            P1[1] += x[i]*y[i]
            P1[2] += (x[i]**2)*y[i]
            P1[3] += (x[i]**3)*y[i]

        K1 = np.linalg.solve(M1, P1)
        return K1

    def f2(self, a0, a1, a2, a3, x):
        return a0+a1*x+a2*x**2+a3*x**3

    def funA(self, X, Y):
        x1 = np.zeros(4)
        x2 = np.zeros(4)
        x3 = np.zeros(4)
        x4 = np.zeros(4)
        x5 = np.zeros(4)
        x6 = np.zeros(4)
        x7 = np.zeros(3)

        y1 = np.zeros(4)
        y2 = np.zeros(4)
        y3 = np.zeros(4)
        y4 = np.zeros(4)
        y5 = np.zeros(4)
        y6 = np.zeros(4)
        y7 = np.zeros(3)

        x1[0] = X[0]
        x1[1] = X[1]
        x1[2] = X[2]
        x1[3] = X[3]
        x2[0] = X[3]
        x2[1] = X[4]
        x2[2] = X[5]
        x2[3] = X[6]
        x3[0] = X[6]
        x3[1] = X[7]
        x3[2] = X[8]
        x3[3] = X[9]
        x4[0] = X[9]
        x4[1] = X[10]
        x4[2] = X[11]
        x4[3] = X[12]
        x5[0] = X[12]
        x5[1] = X[13]
        x5[2] = X[14]
        x5[3] = X[15]
        x6[0] = X[15]
        x6[1] = X[16]
        x6[2] = X[17]
        x6[3] = X[18]
        x7[0] = X[18]
        x7[1] = X[19]
        x7[2] = X[20]

        y1[0] = Y[0]
        y1[1] = Y[1]
        y1[2] = Y[2]
        y1[3] = Y[3]
        y2[0] = Y[3]
        y2[1] = Y[4]
        y2[2] = Y[5]
        y2[3] = Y[6]
        y3[0] = Y[6]
        y3[1] = Y[7]
        y3[2] = Y[8]
        y3[3] = Y[9]
        y4[0] = Y[9]
        y4[1] = Y[10]
        y4[2] = Y[11]
        y4[3] = Y[12]
        y5[0] = Y[12]
        y5[1] = Y[13]
        y5[2] = Y[14]
        y5[3] = Y[15]
        y6[0] = Y[15]
        y6[1] = Y[16]
        y6[2] = Y[17]
        y6[3] = Y[18]
        y7[0] = Y[18]
        y7[1] = Y[19]
        y7[2] = Y[20]

        K1 = projekt.aproksymacja4(x1, y1)
        iks = np.linspace(x1[0], x1[3], 100)
        plt.plot(iks, projekt.f2(K1[0], K1[1], K1[2], K1[3], iks))
        K1 = projekt.aproksymacja4(x2, y2)
        iks = np.linspace(x2[0], x2[3], 100)
        plt.plot(iks, projekt.f2(K1[0], K1[1], K1[2], K1[3], iks))
        K1 = projekt.aproksymacja4(x3, y3)
        iks = np.linspace(x3[0], x3[3], 100)
        plt.plot(iks, projekt.f2(K1[0], K1[1], K1[2], K1[3], iks))
        K1 = projekt.aproksymacja4(x4, y4)
        iks = np.linspace(x4[0], x4[3], 100)
        plt.plot(iks, projekt.f2(K1[0], K1[1], K1[2], K1[3], iks))
        K1 = projekt.aproksymacja4(x5, y5)
        iks = np.linspace(x5[0], x5[3], 100)
        plt.plot(iks, projekt.f2(K1[0], K1[1], K1[2], K1[3], iks))
        K1 = projekt.aproksymacja4(x6, y6)
        iks = np.linspace(x6[0], x6[3], 100)
        plt.plot(iks, projekt.f2(K1[0], K1[1], K1[2], K1[3], iks))
        K1 = projekt.aproksymacja3(x7, y7)
        iks = np.linspace(x7[0], x7[2], 100)
        plt.plot(iks, projekt.f1(K1[0], K1[1], K1[2], iks))
        plt.plot(X, Y, 'bo')
        plt.show()

    def calkSa(self, K1):
        cp = 0
        xs = 0
        ys = 0
        n=10
        a = 0
        b = 2
        x=np.zeros(n+1)
        y=np.zeros(n+1)
        h=(b-a)/n
        for i in range(n+1):
            x[i]=a+i*h
            y[i]=(projekt.f2(K1[0], K1[1], K1[2],K1[3], x[i]))
            h = (b-a)/n
        for i in range(n):
            xs = (x[i]+x[i+1])/2
            ys = (projekt.f2(K1[0], K1[1], K1[2],K1[3], xs))
            cp += h*((y[i]+y[i+1]+4*ys)/6)
        return cp
    def calkSa1(self, x,y):
        cp = 0
        xs = 0
        ys = 0
        n=10
        a = 0
        b = 2
        x=np.zeros(n+1)
        y=np.zeros(n+1)
        h=(b-a)/n
        for i in range(n+1):
            x[i]=a+i*h
            y[i]=(projekt.lan(x3,y3,x[i],4))
            h = (b-a)/n
        for i in range(n):
            xs = (x[i]+x[i+1])/2
            ys = (projekt.lan(x3,y3,xs,4))
            cp += h*((y[i]+y[i+1]+4*ys)/6)
        return cp

    def wypz(self,x,z1,K1,zxA):
        z1=(projekt.f1(K1[0],K1[1],K1[2],x))
        zxA=projekt.fx1(K1[1],K1[2],x)
        return z1,zxA

    def poch(self,zx,pz,px):
        n=3
        for i in range (n):
            if(i==0):
                zx[i]=(pz[i+1]-pz[i])/(px[i+1]-px[i])
            elif(i==n-1):
                zx[i]=(pz[i]-pz[i-1])/(px[i]-px[i-1])
            else:
                zx[i]=(pz[i+1]-pz[i-1])/(px[i+1]-px[i-1])


projekt = Projekt()
x1, y1, z1 = projekt.wyznacz_xyz(ma)
projekt.wyznacz(z, n, m, ma)
# projekt.wykres2D(x1,y1,z1)
# projekt.wykres3D(x1, y1, z1)
# projekt.srednia(y1,z1)
# projekt.mediana(y1,z1)
# projekt.odchylenie(y1,z1)
# projekt.funl(x[2], z[2])
# projekt.funA(x[2], z[2])


x3 = np.zeros(4)
y3 = np.zeros(4)

x3[0] = x[2][6]
x3[1] = x[2][7]
x3[2] = x[2][8]
x3[3] = x[2][9]

y3[0] = z[2][6]
y3[1] = z[2][7]
y3[2] = z[2][8]
y3[3] = z[2][9]

X=sp.symbols('x')

K1=projekt.aproksymacja4(x3,y3)
a=0
b=2

cd = sp.integrate(projekt.f2(K1[0],K1[1],K1[2],K1[3],X),(X,a,b))
sa=projekt.calkSa(K1)
print ("CałkaA: ",cd," Całka SAA: ",sa)

iks=np.linspace(x3[0],x3[3],10)
K2=projekt.lan(x3,y3,iks,4)

cd1 = sp.integrate(projekt.lan(x3,y3,X,4),(X,a,b))
sa1=projekt.calkSa1(x3,y3)
print ("Całkal: ",cd1," Całka SAl: ",sa1)

z1=np.zeros(3)
zx1=np.zeros(3)

x7 = np.zeros(3)
y7 = np.zeros(3)

x7[0] = x[2][18]
x7[1] = x[2][19]
x7[2] = x[2][20]


y7[0] = z[2][18]
y7[1] = z[2][19]
y7[2] = z[2][20]

K3=projekt.aproksymacja3(x7,y7)

z1,zx1=projekt.wypz(x7,z1,K1,zx1)

zx=np.zeros(3)

projekt.poch(zx,y7,x7)

print (z1,"   ",zx1)

Rpx=zx-zx1

plt.plot(x7,zx,'b',label="pochodna funkcji")
plt.plot(x7,zx1,'--r',label="pochodna sprawdzenie ")
plt.plot(x7,Rpx,'orange',label='roznica')
plt.legend()
plt.xlabel("oś x")
plt.ylabel("oś y")
plt.show()