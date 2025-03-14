import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_target_image(image_path, resolution):
    """
    Загружает изображение, преобразует в оттенки серого, изменяет размер до (resolution x resolution)
    и нормализует интенсивность так, чтобы максимум был равен 1.
    """
    img = Image.open(image_path).convert('L')  # Конвертируем в оттенки серого
    img = img.resize((resolution, resolution), Image.LANCZOS)  # Изменяем размер с использованием высококачественной фильтрации
    i_target = np.array(img, dtype=np.float64)
    # Нормировка: максимальное значение становится равным 1
    i_target = 1 - i_target / np.max(i_target)
    return i_target


def gerchberg_saxton(i_target, i_in, wavelength, focal_length, dx, iterations=100, visualize=False):
    # Нормировка целевой интенсивности
    power_in = np.sum(i_in)
    i_target_norm = i_target * (power_in / np.sum(i_target))

    # Инициализация случайной фазы
    phase = np.random.rand(*i_in.shape) * 2 * np.pi
    e_in = np.sqrt(i_in) * np.exp(1j * phase)

    # Фазовая маска линзы
    x = np.linspace(-i_in.shape[1] / 2 * dx, i_in.shape[1] / 2 * dx, i_in.shape[1])
    y = np.linspace(-i_in.shape[0] / 2 * dx, i_in.shape[0] / 2 * dx, i_in.shape[0])
    X, Y = np.meshgrid(x, y)
    #phase_lens = (np.pi * (X ** 2 + Y ** 2)) / (wavelength * focal_length)
    phase_lens = 0

    mse_history = []
    if visualize:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(iterations):
        """# В цикле итераций:
        if i < iterations * 1:  # Добавляем шум только в первой половине итераций
            phase_noise = 0 * np.pi * np.random.randn(*phase.shape) * (1 - i / iterations)
            phase = np.angle(e_in) + phase_noise
            e_in = np.sqrt(i_in) * np.exp(1j * phase)"""
        e_in = np.sqrt(i_in) * np.exp(1j * np.angle(e_in))
        e_out = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(1 * e_in)) * H_forward))
        e_out_prime = np.sqrt(i_target_norm) * np.exp(1j * np.angle(e_out))
        e_in = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(1 * e_out_prime)) * H_inverse))

        # Расчет ошибки
        mse = np.sum((np.abs(e_out) ** 2 - i_target_norm) ** 2) / np.sum(i_target_norm ** 2)
        mse_history.append(mse)

        if visualize and (i % 1 == 0 or i == iterations - 1):
            ax1.clear(), ax2.clear(), ax3.clear()
            ax1.semilogy(mse_history)
            ax1.set_title('MSE: %.2e' % mse)
            ax2.imshow(np.angle(e_in), cmap='viridis')
            ax3.imshow(np.abs(e_out) ** 2, cmap='gray')
            plt.pause(0.01)

    if visualize:
        plt.ioff()
    return np.angle(e_in), mse_history, e_in


# Параметры системы
lambda_ = 2.14e-3  # Длина волны [мм]
k = 2 * np.pi / lambda_
f = 200e-3  # Фокусное расстояние [мм]
resolution = 128  # Разрешение
plate_size = 50e-3  # Размер пластины [мм]
dx = plate_size / resolution  # Шаг сетки [мм]

# Входной пучок (гауссов)
x = np.linspace(-plate_size / 2, plate_size / 2, resolution)
y = np.linspace(-plate_size / 2, plate_size / 2, resolution)
X, Y = np.meshgrid(x, y)
w0 = 5000  # Радиус пучка
i_in = np.exp(-2 * (X ** 2 + Y ** 2) / w0 ** 2)

# Пропагаторы
ny, nx = i_in.shape
fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
fy = np.fft.fftshift(np.fft.fftfreq(ny, dx))
FX, FY = np.meshgrid(fx, fy)

test = np.exp(1j * k / f  * (X ** 2 + Y ** 2))

H_forward = np.exp(-1j * np.pi * lambda_ * f * (FX**2 + FY**2))
H_inverse = np.exp(1j * np.pi * lambda_ * f * (FX**2 + FY**2))
pre_part: complex = 1j * k / 2 / np.pi / f * np.exp(-1j * k * f)
pre_part_inv: complex = 1j * k / 2 / np.pi / f * np.exp(1j * k * f)

# Загрузка целевого изображения (укажите путь к вашему файлу)
image_path = 'C:/Users/Alexander/OneDrive/Desktop/Фокусаторы/SPR algorithm/cross_in_circle_1000x1000.jpg'
i_target = load_target_image(image_path, resolution)

# Запуск алгоритма
phase_gs, mse, e_in = gerchberg_saxton(i_target, i_in, lambda_, f, dx, iterations=100, visualize=True)

# Визуализация
plt.figure(figsize=(15, 5))
plt.subplot(221)
plt.imshow(i_target, cmap='gray')
plt.title('Целевое изображение')

plt.subplot(222)
plt.imshow(phase_gs, cmap='viridis')
plt.title('Фазовый профиль')
plt.colorbar()

plt.subplot(224)
plt.semilogy(mse)
plt.title('Эволюция ошибки')
plt.xlabel('Итерация')
plt.grid()

plt.subplot(223)
plt.imshow(np.abs(np.fft.ifft2(np.fft.fft2(pre_part * e_in) * H_forward)),cmap='gray')
plt.title('Фактическое изображение')
plt.show()

# Сохраняем фазовый профиль в файл 'phase_profile.npy'
np.save('phase_profile.npy', phase_gs)