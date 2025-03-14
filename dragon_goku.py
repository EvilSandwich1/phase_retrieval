import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Параметры системы
lambda_ = 2.14e-3  # Длина волны [м]
f = 50e-3  # Фокусное расстояние [м]
resolution = 128  # Разрешение
plate_size = 33.92e-3  # Размер пластины [м]
dx = plate_size / resolution  # Шаг сетки [м]

# Входной пучок (гауссов)
x = np.linspace(-plate_size / 2, plate_size / 2, resolution)
y = np.linspace(-plate_size / 2, plate_size / 2, resolution)
X, Y = np.meshgrid(x, y)
w0 = 5000e-3  # Радиус пучка
i_in = np.exp(-2 * (X ** 2 + Y ** 2) / w0 ** 2)


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

def angular_spectrum_propagation(field, wavelength, dx, dy, distance):
    """Распространение поля методом углового спектра."""
    ny, nx = field.shape
    k = 2 * np.pi / wavelength

    # Пространственные частоты
    fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, dy))
    FX, FY = np.meshgrid(fx, fy)

    # Вычисляем выражение под корнем и заменяем отрицательные значения на 0
    under_sqrt_raw = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    under_sqrt = np.maximum(under_sqrt_raw, 0)  # Гарантируем, что under_sqrt >= 0

    # Передаточная функция (учитываем только распространяющиеся волны)
    #H = -1j * k / 2 / np.pi / distance * np.exp(1j * k * distance * np.sqrt(under_sqrt))
    H = np.exp(1j * k * distance * np.sqrt(under_sqrt))
    H[under_sqrt_raw <= 0] = 0  # Обнуляем затухающие компоненты

    # Преобразование Фурье поля
    field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

    # Применение передаточной функции
    field_prop_fft = field_fft * H

    # Обратное преобразование
    field_prop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_prop_fft)))
    return field_prop


def gerchberg_saxton(i_target, i_in, wavelength, focal_length, dx, iterations=100, visualize=False, mse_threshold=0.01e-12):
    # Нормировка целевой интенсивности
    power_in = np.sum(i_in)
    i_target_norm = i_target * (power_in / np.sum(i_target))

    # Инициализация случайной фазы
    phase = np.random.rand(*i_in.shape) * 2 * np.pi
    phase_prev = np.zeros_like(phase)  # Для хранения предыдущей фазы
    tk_prev = 1 + np.zeros_like(phase) #
    e_in = np.abs(i_in) * np.exp(1j * phase)
    t = 0
    e_out_prev = 0

    # Фазовая маска линзы
    x = np.linspace(-i_in.shape[1] / 2 * dx, i_in.shape[1] / 2 * dx, i_in.shape[1])
    y = np.linspace(-i_in.shape[0] / 2 * dx, i_in.shape[0] / 2 * dx, i_in.shape[0])
    X, Y = np.meshgrid(x, y)

    mse_history = []
    if visualize:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(iterations):
        # Применение линзы и распространение
        e_out = angular_spectrum_propagation(e_in, wavelength, dx, dx, focal_length)

        # Наложение целевой амплитуды

        # Взвешенная стратегия в Фурье-пространстве (если ошибка перестала уменьшаться)
        if t == 1:
            # Применяем взвешенную стратегию
            beta = np.sum(np.abs(e_out)) / np.sum(i_target_norm)

            e_out_prime = 2 * beta * np.abs(i_target_norm) - np.abs(e_out)
            e_out_prime = e_out_prime * np.exp(1j * np.angle(e_out))
            # Обратное распространение с учетом взвешенной стратегии
            e_in_prime = angular_spectrum_propagation(e_out_prime, wavelength, dx, dx, -focal_length)

        else:
            e_out_prime = np.abs(i_target_norm) * np.exp(1j * np.angle(e_out))
            print(np.sum(abs(abs(e_out_prime) - abs(e_out_prev))))
            if i > 3 and abs(mse_history[-1] - mse_history[-2]) < mse_threshold: #np.sum(abs(abs(e_out_prime) - abs(e_out_prev))) <= mse_threshold:
                # Обратное распространение и коррекция фазы линзы
                t = 1
            e_in_prime = angular_spectrum_propagation(e_out_prime, wavelength, dx, dx, -focal_length)

        print(t)
        e_out_prev = e_out_prime
        # Градиентный спуск в реальном пространстве

        tk = np.angle(e_in_prime) - phase  # Фk - PHIk
        hk = np.angle(e_in_prime) - phase_prev  # Разница между текущей и предыдущей фазой
        alpha_k = np.sum(tk * tk_prev) / np.sum(tk_prev * tk_prev)
        tk_prev = tk
        phase_prev = np.angle(e_in_prime)  # Сохраняем текущую фазу для следующей итерации
        phase = np.angle(e_in_prime) + alpha_k * hk  # Обновляем фазу с учетом градиента

        # Наложение входной амплитуды
        e_in = np.abs(i_in) * np.exp(1j * phase)

        # Расчет ошибки
        mse = np.mean((np.abs(e_out) ** 2 / np.sum(np.abs(e_out) ** 2) - i_target_norm / power_in) ** 2)
        mse_history.append(mse)

        if visualize and (i % 10 == 0 or i == iterations - 1):
            ax1.clear(), ax2.clear(), ax3.clear()
            ax1.semilogy(mse_history)
            ax1.set_title('MSE: %.2e' % mse)
            ax2.imshow(np.angle(e_in),
                       extent=[-plate_size / 2 * 1e3, plate_size / 2 * 1e3, -plate_size / 2 * 1e3, plate_size / 2 * 1e3],
                       cmap='viridis')
            plt.xlabel('x, мм')
            plt.ylabel('y, мм')
            ax3.imshow(np.abs(e_out) ** 2,
                       extent=[-plate_size / 2 * 1e3, plate_size / 2 * 1e3, -plate_size / 2 * 1e3, plate_size / 2 * 1e3],
                       cmap='gray')
            plt.pause(0.01)

    if visualize:
        plt.ioff()
    return np.angle(e_in), mse_history, e_in


# Загрузка целевого изображения (укажите путь к вашему файлу)
image_path = 'D:/рабочий стол/Курганский ИД/фокусаторы/Фокусаторы/Фокусаторы/SPR algorithm/cross2_300x300.png'
i_target = load_target_image(image_path, resolution)

# Запуск алгоритма
phase_gs, mse, e_in = gerchberg_saxton(i_target, i_in, lambda_, f, dx, iterations=500, visualize=True)
# Сохраняем фазовый профиль в файл 'phase_profile.npy'
np.save('C:/Users/kurganskij/Desktop/Fireworks G-S/phase_profile.npy', phase_gs)

# Визуализация
plt.figure(figsize=(15, 5))
plt.subplot(221)
plt.imshow(i_target, extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='gray')
plt.title('Целевое изображение')

plt.subplot(222)
plt.imshow(phase_gs, extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='viridis')
plt.title('Фазовый профиль')
plt.colorbar()

plt.subplot(224)
plt.semilogy(mse)
plt.title('Эволюция ошибки')
plt.xlabel('Итерация')
plt.grid()

plt.subplot(223)
plt.imshow(np.abs(angular_spectrum_propagation(e_in, lambda_, dx, dx, f)), extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='gray')
plt.title('Фактическое изображение')
plt.show()


