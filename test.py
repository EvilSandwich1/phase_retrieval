import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка целевого изображения
def load_target_image(image_path, resolution):
    img = Image.open(image_path).convert('L')
    img = img.resize((resolution, resolution), Image.LANCZOS)
    i_target = np.array(img, dtype=np.float64)
    i_target = 1 - i_target / np.max(i_target)
    return i_target

# Параметры
lambda_ = 2.14e-3  # Длина волны [м]
f = 500e-3         # Фокусное расстояние [м]
resolution = 128
plate_size = 50e-3 # Размер пластины [м]
dx = plate_size / resolution

x = np.linspace(-plate_size / 2, plate_size / 2, resolution)
y = np.linspace(-plate_size / 2, plate_size / 2, resolution)
X, Y = np.meshgrid(x, y)
w0 = 10e-3  # Радиус пучка [м]
i_in = np.exp(-2 * (X ** 2 + Y ** 2) / w0 ** 2)

# Апертура
aperture = np.exp(-2 * (X ** 2 + Y ** 2) / (plate_size / 2) ** 2)
i_in = i_in * aperture

# Инициализация
ny, nx = i_in.shape
k = 2 * np.pi / lambda_
phase = np.random.rand(*i_in.shape) * 2 * np.pi
e_in = np.sqrt(i_in) * np.exp(1j * phase)

# Загрузка целевого изображения
image_path = 'C:/Users/Alexander/OneDrive/Desktop/Фокусаторы/SPR algorithm/cross2_300x300.png'
i_target = load_target_image(image_path, resolution)

# Пропагаторы
fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
fy = np.fft.fftshift(np.fft.fftfreq(ny, dx))
FX, FY = np.meshgrid(fx, fy)

H_forward = np.exp(1j * k / f / 2 / np.pi * ((X - FX) ** 2 + (Y - FY) ** 2))
H_inverse = np.exp(-1j * k / f / 2 / np.pi * ((X - FX) ** 2 + (Y - FY) ** 2))
prepart: complex = 1j * k / 2 / np.pi / f * np.exp(1j * k * f)
# Итерации
mse_history = []
iterations = 100
power_in = np.sum(i_in)
i_target_norm = i_target * (power_in / np.sum(i_target))

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for i in range(iterations):
    e_out = prepart * np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(e_in)) * np.fft.fft2(np.fft.ifftshift(H_forward))))
    e_out_prime = np.sqrt(i_target_norm) * np.exp(1j * np.angle(e_out))
    e_in = prepart * np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(e_out_prime)) * np.fft.fft2(np.fft.ifftshift(H_inverse))))


    mse = np.mean((np.abs(e_out) ** 2 / np.sum(np.abs(e_out) ** 2) - i_target_norm / power_in) ** 2)
    mse_history.append(mse)

    ax1.clear(), ax2.clear(), ax3.clear()
    ax1.semilogy(mse_history)
    ax1.set_title('MSE: %.2e' % mse)
    ax2.imshow(np.angle(e_in), cmap='viridis')
    ax3.imshow(np.abs(e_out) ** 2, cmap='gray')
    plt.pause(0.01)

# Визуализация
plt.figure(figsize=(15, 5))
plt.subplot(221)
plt.imshow(i_target, cmap='gray')
plt.title('Целевое изображение')

plt.subplot(222)
plt.imshow(np.angle(e_in), cmap='viridis')
plt.title('Фазовый профиль')
plt.colorbar()

plt.subplot(224)
plt.semilogy(mse)
plt.title('Эволюция ошибки')
plt.xlabel('Итерация')
plt.grid()

plt.subplot(223)
plt.imshow(np.abs(e_out) ** 2,cmap='gray')
plt.title('Фактическое изображение')
plt.show()
