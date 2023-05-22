import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import sympy as sp

ma = np.loadtxt('136700.dat')
n = ma.shape[0]


class Projekt:
    def __init__(self):
        self.wartosc = []

    def wyznacz_xyz(self, ma, n):
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            x[i]= ma[i][0]
            y[i] =ma[i][1]
            z[i] = ma[i][2]
        return x, y, z

    def wykres2D(self, x, y, z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none')
        plt.title('Mapa 3D', fontdict={'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5,aspect=5)
        ax.set_box_aspect([2,1,0.0001])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def Wykres3D(self, x, y, z):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none')
        plt.title('Mapa 3D', fontdict={'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5,aspect=5)
        ax.set_box_aspect([2,1,1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


    def sredniamedianaodchylenie(self, x, y, z):
        unique_y = np.unique(y)
        mean_y = []
        median_y = []
        std_y = []
        for y_val in unique_y:
            indices = np.where(y == y_val)
            mean_y.append(np.mean(z[indices]))
            median_y.append(np.median(z[indices]))
            std_y.append(np.std(z[indices]))
        return mean_y, median_y, std_y
    
    def srednia (self,y,z):
        unique_y = np.unique(y)
        i=0
        for y_val in unique_y:
            indices=np.where(y==y_val)
            print(unique_y[i],' ',np.mean(z[indices]))
            i+=1
    def mediana(self,y,z):
        unique_y = np.unique(y)
        i=0
        for y_val in unique_y:
            indices=np.where(y==y_val)
            print(unique_y[i],' ',np.median(z[indices]))
            i+=1
    def odchylenie(self,y,z):
        unique_y = np.unique(y)
        i=0
        for y_val in unique_y:
            indices=np.where(y==y_val)
            print(unique_y[i],' ',np.std(z[indices]))
            i+=1
    
    def mianownik(self,x,k,n,z):
        m=1
        for j in range(n):
            if j!=k:
                m*=(x[k]-x[j])
        return z[k]/m
    
    def mianownik1(self,x,x1,k,n):
        m=1
        for i in range(n):
            if i!=k:
                m*=x1-x[i]
        return m

    def interpolacja_l1(self,ma,xp):
        n=21    
        x=np.zeros(n)
        y=np.zeros(n)
        a=np.zeros(n)
        yp=0
        for i in range(n):
            x[i]=ma[i+n,0]
            y[i]=ma[i+n,2]
        for i in range(n):
            p=1
            for j in range(n):
                if i!=j:
                    p=p*(xp-x[j])/(x[i]-x[j])
            yp = yp+p*y[i]
        # print('Interpolated value at %.3f is %.3f.' % (xp, yp))
        return yp
        # for i in range(n):
        #     a[i]=projekt.mianownik(x,i,n,y)    
        # for i in range (n):
        #     for j in range(len(iks)):
        #         x1+=projekt.mianownik1(x,iks[i],i,n)
        # for i in range (n): 
        #     W+=a[i]*x1
        # return W

    def interpolacja_l(self,ma,iks):
        n=21
        x=np.zeros(n)
        y=np.zeros(n)
        a=np.zeros(n)
        for i in range(n):
            x[i]=ma[i+n,0]
            y[i]=ma[i+n,1]
        for i in range(n):
            a[i]=y[i]/projekt.mianownik(x,i,n)
        x1=iks
        W=0
        for i in range(n):
            W+=a[i]*projekt.mianownik1(x,x1,i,n)
        return W
    
    def aproksymacja(self,ma):
        n=21
        M1=np.zeros((3,3))
        M2=np.zeros(3)
        x=np.zeros(n)
        y=np.zeros(n)
        z=np.zeros(n)
        for i in range(21):
            x[i]=ma[i+21,0]
            y[i]=ma[i+21,1]
            z[i]=ma[i+21,2]
        M1[0][0]=n
        for i in range(n):
            M1[0][1]+=x[i]
            M1[0][2]+=y[i]
            M1[1][0]+=x[i]
            M1[1][1]+=x[i]*x[i]
            M1[1][2]+=x[i]*y[i]
            M1[2][0]+=y[i]
            M1[2][1]+=x[i]*y[i]   
            M1[2][2]+=y[i]*y[i]
            M2[0]+=z[i]
            M2[1]+=x[i]*z[i]
            M2[2]+=y[i]*z[i]
        K=np.linalg.solve(M1,M2)
        print(K)

    def fun(x,y,z,a0,a1,a2):
        return 0
    def aproksymacja1(self,ma):
                
        M1=np.zeros((3,3))
        P1=np.zeros (3)
        x=np.zeros(n)
        y=np.zeros(n)
        z=np.zeros(n)
        for i in range(21):
            x[i]=ma[i+21,0]
            y[i]=ma[i+21,1]
            z[i]=ma[i+21,2]
        M1[0,0]=n
        for i in range (n):
            M1[0,1]=M1[0,1]+x[i]
            M1[0,2]=M1[0,2]+x[i]**2
            M1[1,0]=M1[1,0]+x[i]
            M1[1,1]=M1[1,1]+x[i]**2
            M1[1,2]=M1[1,2]+x[i]**3
            M1[2,0]=M1[2,0]+x[i]**2
            M1[2,1]=M1[2,1]+x[i]**3
            M1[2,2]=M1[2,2]+x[i]**4

        for i in range (n):
            P1[0]+=z[i]
            P1[1]+=x[i]*z[i]
            P1[2]+=(x[i]**2)*z[i]

        print (M1)
        print (P1)

        K1=np.linalg.solve(M1,P1)

        print (K1)
        return K1

    def fun1(self,a0,a1,a2,x):
        return a0+a1*x+a2*x**2

    def calkSa(self,K1):
        cp=0
        xs=0
        ys=0
        a=0
        b=2
        n=20
        x=np.zeros(n+1)
        y=np.zeros(n+1)
        h=(b-a)/n
        for i in range(n+1):
            x[i]=a+i*h
            y[i]=(projekt.fun1(K1[0],K1[1],K1[2],x[i]))
        for i in range (n):
            xs=(x[i]+x[i+1])/2
            ys=(projekt.fun1(K1[0],K1[1],K1[2],xs))
            cp+=h*((y[i]+y[i+1]+4*ys)/6)
        return cp
    def interl(self,ma):
        x=np.linspace(0,2,21)
        y=np.zeros(21)
        for i in range(21):
            y[i]=ma[i+21,2]
        
        xplt=np.linspace(x[0],x[-1])
        yplt=np.array([],float)

        for xp in xplt:
            yp =0
            for xi,yi in zip(x,y):
                yp+=yi*np.prod((xp-x[x!=xi])/(xi-x[x!=xi]))
            yplt=np.append(yplt,yp)
        plt.plot(x,y,'ro',xplt,yplt,'b-')
        plt.show()
projekt = Projekt()
x, y, z = projekt.wyznacz_xyz(ma, n)
# projekt.wykres2D(x, y, z)
# projekt.Wykres3D(x, y, z)
# print(projekt.sredniamedianaodchylenie(x,y,z))
# print ("Interpolacja lagrandża")
# projekt.interpolacja_l(ma,z)
# projekt.srednia(y,z)
# projekt.mediana(y,z)
# projekt.odchylenie(y,z)
a=0
b=2
K1=projekt.aproksymacja1(ma)
X=sp.symbols('x')
jd=projekt.interpolacja_l1(ma,0.2)
print("funkcja w punkcie 0.2",jd)
# cd1 = sp.integrate(projekt.interpolacja_l1(ma,X),(X,a,b))
# cd = sp.integrate(projekt.fun1(K1[0],K1[1],K1[2],X),(X,a,b))
# sa=projekt.calkSa(K1)
# print ("Całka: ",cd," Całka SA: ",cd1)

# projekt.interl(ma)

x1=np.zeros(21)
y1=np.zeros(21)
z1=np.zeros(21)
for i in range (21,42,1):
    x1[i-21]=x[i]
    y1[i-21]=z[i]
    z1[i-21]=z[i]
iks=np.arange(0.2,1.81,0.01)
w=np.zeros(len(iks))
for i in range(len(iks)):
    w[i]=projekt.interpolacja_l1(ma,iks[i])
plt.plot(x1,z1,'bo')
plt.plot(iks,w,'-b')
plt.show()
iks=np.linspace(0,2,100)

plt.plot(iks,projekt.fun1(K1[0],K1[1],K1[2],iks),'b-',label='Wykres aproksymacji kwadratowej')
plt.legend()
plt.show()

# y_f = interp1d(x1, y1, 'linear')
# x = np.linspace(0,2,1000)
# y = y_f(x)
# plt.scatter(x,y)
# plt.show()
# print("interpolacja liniowa: ",y_f(1.58))
# y_f1 = interp1d(x1, y1, 'cubic')
# print("interpolacja sześcienna: ",y_f1(1.58))
# h1=y_f(1.58)
# h2=y_f1(1.58)
# wynik=h1-h2
# print("roznica wynikow: ",wynik)
# x = np.linspace(0,2,1000)
# y = y_f1(x)
# plt.scatter(x,y)
# plt.show()