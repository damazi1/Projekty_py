import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2, 21)
Y = np.array([123, 543, 321, 412, 126, 754, 213, 543, 321, 654,
             342, 754, 213, 763, 12, 543, 231, 543, 654, 754, 123])


def mianownik(x, k, n):
    m = 1
    for j in range(n):
        if j != k:
            m *= x[k]-x[j]
    return m


def mianownik1(x, x1, k, n):
    m = 1
    for j in range(n):
        if j != k:
            m *= x1-x[j]
    return m


def lan(x, y, iks, n):
    W = 0
    for j in range(n):
        W += (y[j]/(mianownik(x, j, n)))*mianownik1(x, iks, j, n)
    print(W)
    return W


eks = np.linspace(0, 2, 101)


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

X=np.round (X,1)
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
plt.plot(iks, lan(x1, y1, iks, 4), label=(f"{x1[0]}-{x1[3]}"))
iks = np.linspace(x2[0], x2[3], 100,)
plt.plot(iks, lan(x2, y2, iks, 4), label=(f"{x2[0]}-{x2[3]}"))
iks = np.linspace(x3[0], x3[3], 100)
plt.plot(iks, lan(x3, y3, iks, 4), label=(f"{x3[0]}-{x3[3]}"))
iks = np.linspace(x4[0], x4[3], 100)
plt.plot(iks, lan(x4, y4, iks, 4), label=(f"{x4[0]}-{x4[3]}"))
iks = np.linspace(x5[0], x5[3], 100)
plt.plot(iks, lan(x5, y5, iks, 4), label=(f"{x5[0]}-{x5[3]}"))
iks = np.linspace(x6[0], x6[3], 100)
plt.plot(iks, lan(x6, y6, iks, 4), label=(f"{x6[0]}-{x6[3]}"))
iks = np.linspace(x7[0], x7[2], 100)
plt.plot(iks, lan(x7, y7, iks, 3), label=(f"{x7[0]}-{x7[2]}"))
plt.legend()

plt.show()
