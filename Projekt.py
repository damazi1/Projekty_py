import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import quad
from scipy.integrate import solve_ivp

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
        colors = z
        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=colors, cmap='coolwarm')
        plt.title('Mapa 2D', fontdict={
                  'fontname': 'monospace', 'fontsize': 18})
        plt.legend(*scatter.legend_elements(), title="Wysokość")
        plt.show()

    def Wykres3D(self, x, y, z):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        plt.title('Mapa 3D', fontdict={
                  'fontname': 'monospace', 'fontsize': 18})
        fig.colorbar(surf, shrink=0.5, aspect=5)
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
    
    def interpolacja_l(self,ma,z):
        x=np.linspace(0,2,10)

        x_interp = np.linspace(0, 2, 21) 
        y_interp = np.zeros_like(x_interp)

        # Przeprowadzanie interpolacji dla każdej wartości x_interp
        for i in range(len(x_interp)):
            for j in range(len(x)):
                L = 1
                for k in range(len(x)):
                    if k != j:
                        L *= (x_interp[i] - x[k]) / (x[j] - x[k])
                y_interp[i] += z[j+21] * L

        # Wyświetlanie wyników
        for i in range(len(x_interp)):
            print(f'x={x_interp[i]:.2f}, y={y_interp[i]:.2f}')



projekt = Projekt()
x, y, z = projekt.wyznacz_xyz(ma, n)
# projekt.wykres2D(x, y, z)
# projekt.Wykres3D(x, y, z)
# print(projekt.sredniamedianaodchylenie(x,y,z))
# print ("Interpolacja lagrandża")
# projekt.interpolacja_l(ma,z)
projekt.srednia(y,z)
projekt.mediana(y,z)
projekt.odchylenie(y,z)
x1=np.zeros(21)
y1=np.zeros(21)
for i in range (21,42,1):
    x1[i-21]=x[i]
    y1[i-21]=z[i]
y_f = interp1d(x1, y1, 'linear')
x = np.linspace(0,2,1000)
y = y_f(x)
plt.scatter(x,y)
plt.show()
print(y_f(1.58))
y_f1 = interp1d(x1, y1, 'cubic')
print(y_f1(1.58))
x = np.linspace(0,2,1000)
y = y_f1(x)
plt.scatter(x,y)
plt.show()


