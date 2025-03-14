import numpy as np
import matplotlib.pyplot as plt

# Параметры
lambda_ = 2.14e-3  # Длина волны [м]
f = 50e-3         # Фокусное расстояние [м]
plate_size = 80e-3 # Размер пластины [м]
resolution = 128    # Разрешение сетки (можно увеличить до 256)
dx = plate_size / resolution  # Шаг сетки [м]

# Координаты в апертуре
xi = np.linspace(-plate_size / 2, plate_size / 2, resolution)
eta = np.linspace(-plate_size / 2, plate_size / 2, resolution)
XI, ETA = np.meshgrid(xi, eta)

# Входной пучок (например, гауссов)
w0 = 5000e-3  # Радиус пучка [м]
A_in = np.exp(-2 * (XI**2 + ETA**2) / w0**2)  # Амплитуда пучка

# Загрузка фазы из файла
phase_gs = np.load('phase_profile.npy')  # Фаза из алгоритма GS

# Поле в апертуре
E_aperture = A_in * np.exp(1j * phase_gs)

# Координаты в плоскости наблюдения
x = np.linspace(-plate_size / 2, plate_size / 2, resolution)
y = np.linspace(-plate_size / 2, plate_size / 2, resolution)
X, Y = np.meshgrid(x, y)

# Волновое число
k = 2 * np.pi / lambda_

# Функция для вычисления расстояния r
def compute_r(x, y, xi, eta, z):
    return np.sqrt((x - xi)**2 + (y - eta)**2 + z**2)

# Численное интегрирование интеграла Кирхгоффа для одной точки (x, y)
def kirchhoff_integral(x, y, z, E_aperture, XI, ETA, k, dx):
    r = compute_r(x, y, XI, ETA, z)
    integrand = E_aperture * (-1j * k / 2 / np.pi) * np.exp(1j * k * r) * z / r**2
    integral = np.sum(integrand) * dx**2  # Приближение методом прямоугольников
    return integral

# Проверка применимости приближения Френеля
def check_fresnel_approximation(z, L, lambda_):
    condition = z / (L**2 / (2 * lambda_))
    if condition > 10:
        return "Приближение Френеля применимо.", condition
    else:
        return "Приближение Френеля может быть неточным.", condition

# Вычисление поля в плоскости наблюдения
E_observation = np.zeros((resolution, resolution), dtype=complex)
for i in range(resolution):
    for j in range(resolution):
        E_observation[i, j] = kirchhoff_integral(X[i, j], Y[i, j], f, E_aperture, XI, ETA, k, dx)
        if (i * resolution + j) % 100 == 0:
            print(f"Прогресс: {i * resolution + j + 1}/{resolution**2}")

# Интенсивность
I_observation = np.abs(E_observation)**2

# Отображение параметров системы
print("Параметры системы:")
print(f"Длина волны: {lambda_ * 1e3:.2f} мм")
print(f"Фокусное расстояние: {f * 1e3:.2f} мм")
print(f"Размер пластины: {plate_size * 1e3:.2f} мм")
print(f"Разрешение: {resolution} x {resolution}")
print(f"Шаг сетки: {dx * 1e3:.2f} мм")

# Проверка применимости приближения Френеля
fresnel_check, condition = check_fresnel_approximation(f, plate_size, lambda_)
print(f"\nПроверка приближения Френеля: {fresnel_check}\nz/(L^2/2lambda) = {condition}")

# Визуализация
plt.figure(figsize=(15, 5))

# 1. Падающий пучок
plt.subplot(131)
plt.imshow(A_in, extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='gray')
plt.title('Падающий пучок')
plt.xlabel('x, мм')
plt.ylabel('y, мм')

# 2. Исходная фаза
plt.subplot(132)
plt.imshow(phase_gs, extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='viridis')
plt.title('Исходная фаза')
plt.xlabel('x, мм')
plt.ylabel('y, мм')

# 3. Интенсивность в плоскости наблюдения
plt.subplot(133)
plt.imshow(I_observation, extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='gray')
plt.title('Интенсивность (Киирхгофф)')
plt.xlabel('x, мм')
plt.ylabel('y, мм')

plt.tight_layout()
plt.show()

