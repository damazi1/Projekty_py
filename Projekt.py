import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

ma = np.loadtxt('136700.DAT')
n = ma.shape[0]


class Projekt:
    def __init__(self):
        self.wartosc = []

    def wyznacz_xyz(self, ma, n):
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            x[i], y[i], z[i] = ma[i][0], ma[i][1], ma[i][2]
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

        # Wyznaczenie średniej, mediany i odchylenia standardowego dla każdej wartości y
        mean_y = []
        median_y = []
        std_y = []
        for y_val in unique_y:
            indices = np.where(y == y_val)
            mean_y.append(np.mean(z[indices]))
            median_y.append(np.median(z[indices]))
            std_y.append(np.std(z[indices]))
        return mean_y, median_y, std_y

    def interpolate_y(self,grid, y_coord, new_x):
        # Wybór współrzędnej y
        y_data = grid[:, y_coord]
        # Wyznaczenie nowych wartości osi x
        old_x = np.arange(len(y_data))
        # Interpolacja liniowa
        f = interp1d(old_x, y_data, kind='linear')
        # Wartości dla nowych osi x
        new_y = f(new_x)
        return new_y


projekt = Projekt()
x, y, z = projekt.wyznacz_xyz(ma, n)
# projekt.wykres2D(x, y, z)
# projekt.Wykres3D(x, y, z)

xx,yy =np.meshgrid(x, y)

projekt.interpolate_y(yy,1,x)