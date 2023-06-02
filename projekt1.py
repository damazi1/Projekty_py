import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.integrate import dblquad
from scipy.spatial import Delaunay
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
        surf = ax.plot_surface(x, y, z, cmap='plasma', edgecolor='none',vmin=-400, vmax=1050)
        plt.title('Mapa 2D', fontdict={
                  'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_box_aspect([2, 1, 0.0001])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.show()

    def wykres3D(self, x, y, z):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, z, cmap='plasma', edgecolor='none',vmin=-400, vmax=1050)
        plt.title('Mapa 3D', fontdict={
                  'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_box_aspect([2, 1, 1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def srednia(self,y,z):
        print("Średnia arytmetyczna: ")
        for i in range(11):
            print(round(y[i][i],1),"   ",np.mean(z[i]))
            

    def mediana(self, y, z):
        print("Mediana : ")
        for i in range(11):
            print(round(y[i][i],1),"   ",np.median(z[i]))

    def odchylenie(self, y, z):
        print("Odchylenie standardowe: ")
        for i in range(11):
            print(round(y[i][i],1),"   ",np.std(z[i]))

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
            W += (y[j]/(projekt.mianownik(x, j, n))) * projekt.mianownik1(x, iks, j, n)
        return W

    def funl(self, X, Y):
        plt.plot(X, Y, 'bo')
        X=np.round(X,2)
        x1 = [X[i:i+4] for i in range(0,len(X)-1,3)]
        y1 = [Y[i:i+4] for i in range(0,len(Y)-1,3)]
        for i in range(len(x1)):
            m=len(x1[i])-1
            iks = np.linspace(x1[i][0], x1[i][m], 101)
            plt.plot(iks, projekt.lan(x1[i], y1[i], iks, m+1), label=(f"{x1[i][0]}-{x1[i][m]}"))
        plt.legend()
        plt.title("Interpolacja Lagrange'a")
        plt.xlabel("oś X")
        plt.ylabel("oś Y")
        plt.show()
    def roznica(self,x, y):

        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y
        for j in range(1, n):
            for i in range(n-j):
                coef[i][j] = \
                    (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])

        return coef


    def newton(self,coef, x_data, x):

        n = len(x_data) - 1
        p = coef[n]
        for k in range(1, n+1):
            p = coef[n-k] + (x - x_data[n-k])*p
        return p


    def funN(self,X, Y):
        X=np.round(X,2)
        x1 = [X[i:i+4] for i in range(0,len(X)-1,3)]
        y1 = [Y[i:i+4] for i in range(0,len(Y)-1,3)]
        for i in range(len(x1)):
            m=len(x1[i])-1
            iks=np.linspace(x1[i][0],x1[i][m],101)
            y=projekt.newton(projekt.roznica(x1[i],y1[i])[0,:],x1[i],iks)
            plt.plot(iks, y,label=(f"{x1[i][0]}-{x1[i][m]}"))
        plt.plot(X, Y, 'bo')
        plt.legend()
        plt.title("Interpolacja Newtona")
        plt.xlabel("oś X")
        plt.ylabel("oś Y")
        plt.show()

    def aproksymacja2(self,x,y,n):
        M=np.zeros((2,2))
        P=np.zeros (2)
        M[0][0]=n
        for i in range (n):
            M[0][1]=M[1][0]+x[i]
            M[1][0]=M[1][0]+x[i]
            M[1][1]=M[1][1]+x[i]**2
        for i in range (n):
            P[0]=P[0]+y[i]
            P[1]=P[1]+x[i]*y[i]
        K=projekt.gauss(M,P)
        return K
    def f0(self,a0,a1,x):
        return a0+a1*x

    def aproksymacja3(self, x, y,n):
        M1 = np.zeros((3, 3))
        P1 = np.zeros(3)
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

        K1 = projekt.gauss(M1,P1)
        return K1

    def f1(self, a0, a1, a2, x):
        return a0+a1*x+a2*x**2
    def fx1(self,a1,a2,x):
        return a1+2*a2*x
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

        K1 = projekt.gauss(M1,P1)
        print(K1)
        return K1

    def f2(self, a0, a1, a2, a3, x):
        return a0+a1*x+a2*x**2+a3*x**3

    def funA(self, X, Y,n):
        K1 = projekt.aproksymacja3(X, Y,n)
        iks = np.linspace(X[0], X[20], 101)
        plt.plot(iks, projekt.f1(K1[0], K1[1], K1[2], iks),color="red",label=(f"{X[0]}-{X[20]}"))
        plt.plot(X, Y, 'bo')
        plt.legend()
        plt.title("Aproksymacja Średnio-kwadratowa")
        plt.xlabel("oś X")
        plt.ylabel("oś Y")
        plt.show()
    def funAl(self, X, Y,n):
        K1 = projekt.aproksymacja2(X, Y,n)
        iks = np.linspace(X[0],X[20],101)
        plt.plot(iks, projekt.f0(K1[0], K1[1],iks),color="red",label=(f"{X[0]}-{X[20]}"))
        plt.plot(X, Y, 'bo')
        plt.legend()
        plt.title("Aproksymacja Liniowa")
        plt.xlabel("oś X")
        plt.ylabel("oś Y")
        plt.show()

    def pole(sefl,ma):
        tri = Delaunay(ma[:, :2])
        triangle_areas = []
        for i in tri.simplices:
            p0, p1, p2 = ma[i]
            triangle_areas.append(
                0.5 * np.linalg.norm(
                    np.cross(p1 - p0, p2 - p0)
                )
            )
        surface_area = np.sum(triangle_areas)
        print("Pole powierzchni: ", surface_area)

    def calkSa(self, K1,x1):
        cp = 0
        xs = 0
        ys = 0
        n=10
        a=x1[0]
        b=x1[2]

        x=np.zeros(n+1)
        y=np.zeros(n+1)
        h=(b-a)/n
        for i in range(n+1):
            x[i]=a+i*h
            y[i]=(projekt.f1(K1[0], K1[1], K1[2],x[i]))
            h = (b-a)/n
        for i in range(n):
            xs = (x[i]+x[i+1])/2
            ys = (projekt.f1(K1[0], K1[1], K1[2],xs))
            cp += h*((y[i]+y[i+1]+4*ys)/6)
        return cp
    def calkSa1(self, x3,y3):
        cp = 0
        xs = 0
        ys = 0
        n=10
        a=x3[0]
        b=x3[2]

        x=np.zeros(n+1)
        y=np.zeros(n+1)
        h=(b-a)/n
        for i in range(n+1):
            x[i]=a+i*h
            y[i]=(projekt.lan(x3,y3,x[i],3))
            h = (b-a)/n
        for i in range(n):
            xs = (x[i]+x[i+1])/2
            ys = (projekt.lan(x3,y3,xs,3))
            cp += h*((y[i]+y[i+1]+4*ys)/6)
        return cp
    
    def calkiIiA(self,x,z,n):
        x1 = [x[2][i:i+3] for i in range(0,len(x[0])-1,2)]
        y1 = [z[2][i:i+3] for i in range(0,len(z[0])-1,2)]

        X=sp.symbols('x')

        K1=projekt.aproksymacja3(x1[2],y1[2],3)
        a=x1[2][0]
        b=x1[2][2]
        cd = sp.integrate(projekt.f1(K1[0],K1[1],K1[2],X),(X,a,b))
        sa=projekt.calkSa(K1,x1[2])
        print ("Całka Dokładna: ",cd," Całka z Aproksymacji średnio-kwadratowej: ",sa)

        cd1 = sp.integrate(projekt.lan(x1[2],y1[2],X,3),(X,a,b))
        sa1=projekt.calkSa1(x1[2],y1[2])
        print ("Całka Dokładna: ",cd1," Całka z interpolacij Lagrange'a: ",sa1)


    def wypz(self,x,K1,):
        z1=np.zeros(20)
        zx1=np.zeros(20)
        print("Wartosc funkcji:",projekt.f1(K1[0],K1[1],K1[2],x))
        z1=(projekt.f1(K1[0],K1[1],K1[2],x))
        zx1=projekt.fx1(K1[1],K1[2],x)
        return z1,zx1

    def poch(self,pz,px):
        n=20
        zx=np.zeros(n)
        for i in range (n):
            if(i==0):
                zx[i]=(pz[i+1]-pz[i])/(px[i+1]-px[i])
            elif(i==n-1):
                zx[i]=(pz[i]-pz[i-1])/(px[i]-px[i-1])
            else:
                zx[i]=(pz[i+1]-pz[i-1])/(px[i+1]-px[i-1])
        return zx
    def poch1(self,x,z):
        iks=np.linspace(0,2,20)
        zx=projekt.poch(z,x)

        plt.plot(iks,zx,'b',label="pochodna funkcji")
        plt.legend()
        plt.xlabel("oś x")
        plt.ylabel("oś y")
        plt.axhline(y=0,color='r')
        plt.show()
        return zx

    def iniciuj(self,X, Y):
        X=np.round(X,2)
        x1 = [X[i:i+4] for i in range(0,len(X)-1,3)]
        y1 = [Y[i:i+4] for i in range(0,len(Y)-1,3)]
        for i in range(len(x1)):
            m=len(x1[i])-1
            x_new=np.linspace(x1[i][0],x1[i][m],10)
            y_new=projekt.newton(projekt.roznica(x1[i],y1[i])[0,:],x1[i],x_new)
            print(projekt.lan(x1[i],y1[i],x_new,m+1)-y_new)

    def iniciuj1(self,X, Y,n):
        K1 = projekt.aproksymacja3(X, Y,n)
        iks = np.linspace(X[0],X[20], 20)
        W1=projekt.f1(K1[0], K1[1], K1[2], iks)
        K1 = projekt.aproksymacja2(X,Y,n)
        iks = np.linspace(X[0],X[20], 20)
        W2=projekt.f0(K1[0], K1[1],iks)
        print(W1,"\n",W2)
        print(W1-W2)
    
    def gauss(self,A,b):
        n=A.shape[0]
        C=np.zeros((n,n+1))
        C[:,0:n]=A
        C[:,n]=b
        x=np.zeros(n)
        for s in range(0, n-1):
            for i in range(s+1, n):
                # L = A[i,s] / A[s,s]
                for j in range(s+1, n+1):
                    C[i,j] = C[i,j] - (C[i,s] / C[s,s]) * C[s,j]
        x[n-1] = C[n-1,n]/C[n-1,n-1]
        for i in range(n-2,-1,-1):
            suma = 0.0
            for s in range(i+1, n):
                suma = suma + C[i,s] * x[s]
            x[i] = (C[i,n] - suma) / C[i,i]
        return x

projekt = Projekt()
projekt.wyznacz(z, n, m, ma)#Wyznacza Z
# projekt.wykres2D(x,y,z)     #Wykres 2D
# projekt.wykres3D(x,y,z)     #Wykres 3D
# projekt.srednia(y,z)        #Średnia Arytmetyczna
# projekt.mediana(y,z)        #Mediana
# projekt.odchylenie(y,z)     #Odchylenie Standardowe
# projekt.funN(x[2],z[2])     #Funkcja Interpolacja Newtona
# projekt.funl(x[2], z[2])    #Funkcja Interpolacja Lagrangea
# projekt.funAl(x[2],z[2],n)    #Funkcja Aproksymacja Liniowa
# projekt.funA(x[2], z[2],n)    #Funkcja Aproksymacja Średnio-Kwadratowa
# projekt.iniciuj(x[2],z[2]) #Funckcja odpowiedzialna za pokazywanie różnicy między funkcjami Interpolacyjnymi
# projekt.iniciuj1(x[2],z[2],n) #Funkcja odpowiedzialna za pokazywanie różnicy między funkcjami Aproksymacyjnymi
# projekt.pole(ma) #Liczy pole
# projekt.calkiIiA(x,z,n)       #Liczy całki
projekt.poch1(x[2],z[2])    #liczy pochodną