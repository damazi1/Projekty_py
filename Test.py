import numpy as np

# Dane do interpolacji
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Stopień wielomianu
N = 3

# Dopasowanie wielomianu stopnia N do danych
coefficients = np.polyfit(x, y, N)

# Utworzenie wielomianu na podstawie uzyskanych współczynników
poly = np.poly1d(coefficients)

# Wygenerowanie wartości dla nowych punktów
new_x = np.linspace(min(x), max(x), 100)
new_y = poly(new_x)

# Wyświetlenie wyników
print("Współczynniki wielomianu:")
print(coefficients)
print("\nWartości dla nowych punktów:")
for i in range(len(new_x)):
    print("x =", new_x[i], "  y =", new_y[i])
