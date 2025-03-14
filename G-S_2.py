import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import stl
from stl import mesh as mesh # pip install numpy-stl
from mpl_toolkits import mplot3d


# Параметры системы
lambda_ = 2.14e-3  # Длина волны [м]
f = 50e-3  # Фокусное расстояние [м]
resolution = 128  # Разрешение
plate_size = 80e-3  # Размер пластины [м]
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
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy)) * 2 * np.pi
    FX, FY = np.meshgrid(kx, ky)

    # Вычисляем выражение под корнем и заменяем отрицательные значения на 0
    under_sqrt = k ** 2 - FX ** 2 - FY ** 2
    under_sqrt = np.maximum(under_sqrt, 0)  # Гарантируем, что under_sqrt >= 0

    # Передаточная функция (учитываем только распространяющиеся волны)
    #H = -1j * k / 2 / np.pi / distance * np.exp(1j * k * distance * np.sqrt(under_sqrt))
    test = np.sqrt(under_sqrt)
    H = np.exp(1j * distance * np.sqrt(under_sqrt))
    H[under_sqrt <= 0] = 0  # Обнуляем затухающие компоненты

    # Преобразование Фурье поля
    field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

    # Применение передаточной функции
    field_prop_fft = field_fft * H

    # Обратное преобразование
    field_prop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_prop_fft)))
    return field_prop


def gerchberg_saxton(i_target, i_in, wavelength, focal_length, dx, iterations=100, visualize=False):
    # Нормировка целевой интенсивности
    power_in = np.sum(i_in)
    i_target_norm = i_target * (power_in / np.sum(i_target))

    # Инициализация случайной фазы
    phase = np.random.rand(*i_in.shape) * 2 * np.pi
    e_in = np.abs(i_in) * np.exp(1j * phase)

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
        # В цикле итераций:
        if i < iterations * 1:  # Добавляем шум только в первой половине итераций
            phase_noise = 0 * np.pi * np.random.randn(*phase.shape) * (1 - i / iterations)
            phase = np.angle(e_in) + phase_noise
            e_in = np.abs(i_in) * np.exp(1j * phase)

        # Применение линзы и распространение
        e_in_with_lens = e_in * np.exp(1j * phase_lens)
        e_out = angular_spectrum_propagation(e_in_with_lens, wavelength, dx, dx, focal_length)

        # Наложение целевой амплитуды
        i_current = np.abs(e_out) ** 2
        e_out_prime = np.abs(i_target_norm) * np.exp(1j * np.angle(e_out))

        # Обратное распространение и коррекция фазы линзы
        e_in_prime = angular_spectrum_propagation(e_out_prime, wavelength, dx, dx, -focal_length)
        #e_in_prime = e_in_prime * np.exp(-1j * phase_lens)

        # Наложение входной амплитуды
        e_in = np.abs(i_in) * np.exp(1j * np.angle(e_in_prime))

        # Расчет ошибки
        mse = np.mean((np.abs(e_out) ** 2 / np.sum(np.abs(e_out) ** 2) - i_target_norm / power_in) ** 2)
        mse_history.append(mse)

        if visualize and (i % 1 == 0 or i == iterations - 1):
            ax1.clear(), ax2.clear(), ax3.clear()
            ax1.semilogy(mse_history)
            ax1.set_title('MSE: %.2e' % mse)
            ax2.imshow(np.angle(e_in), extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='viridis')
            plt.xlabel('x, мм')
            plt.ylabel('y, мм')
            ax3.imshow(np.abs(e_out) ** 2, extent=[-plate_size/2*1e3, plate_size/2*1e3, -plate_size/2*1e3, plate_size/2*1e3], cmap='gray')
            plt.pause(0.01)

    if visualize:
        plt.ioff()
    return np.angle(e_in), mse_history, e_in

def shitpost_stl_gen(phase, x, y, wavelength, n=1.54, filename='phase_profile1.stl'):
    """
    Преобразует фазовый профиль в высотное поле по формуле:
      z = phase * wavelength / ((n-1)*2*pi)
    Затем генерирует 3D модель (треугольную сетку) и сохраняет её в STL‑файл.
    Добавлена тонкая пластинка с рельефом на поверхности.
    Параметр plate_thickness задаёт толщину пластинки (например, 2 мм).
    """


    # Преобразуем фазу в высоту (аналог MATLAB: z = matr*lambda/((n-1)*2*pi))
    z = phase * wavelength / ((n - 1) * 2 * np.pi)
    #z = z - np.min(z) + 1.0  # смещение: минимальное значение станет 1

    # Сетка: x и y — 1D массивы, z — 2D массив, размеры должны совпадать
    nx = len(x)
    ny = len(y)
    vertices = []
    # Создаём список вершин для каждой точки сетки
    for j in range(ny):
        for i in range(nx):
            vertices.append([x[i], y[j], z[j, i]])
    #vertices = np.array(vertices)

    # Генерация треугольников для каждого прямоугольного элемента сетки (2 треугольника на ячейку)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx0 = j * nx + i
            idx1 = j * nx + (i + 1)
            idx2 = (j + 1) * nx + i
            idx3 = (j + 1) * nx + (i + 1)
            faces.append([idx0, idx2, idx1])
            faces.append([idx1, idx2, idx3])

    base_thickness = np.min(z) - 2e-3

    """
    idx0 = [0, 0, base_thickness]
    idx1 = [0, y[ny-1], base_thickness]
    idx2 = [x[nx-1], 0, base_thickness]
    idx3 = [0, y[ny-1], base_thickness]
    faces.append([idx2, idx1, idx0])  # Первый треугольник (обратная ориентация)
    faces.append([idx3, idx1, idx2])  # Второй треугольник (обратная ориентация)
    """

    # Создаём боковые грани только по краям модели

    # Количество вершин верхней поверхности
    num_upper_vertices = nx * ny

    # Создаём вершины для нижней поверхности (все точки по краям)
    for j in range(ny):
        vertices.append([x[0], y[j], base_thickness])  # Левая граница
    for j in range(ny):
        vertices.append([x[-1], y[j], base_thickness])  # Правая граница
    for i in range(1, nx - 1):  # Углы уже добавлены, пропускаем их
        vertices.append([x[i], y[0], base_thickness])  # Передняя граница
    for i in range(1, nx - 1):  # Углы уже добавлены, пропускаем их
        vertices.append([x[i], y[-1], base_thickness])  # Задняя граница

    # Создаём грани для боковых стенок
    # Левая граница (300 точек)
    for j in range(ny - 1):
        v_top0 = j * nx
        v_bottom0 = num_upper_vertices + j
        v_top1 = (j + 1) * nx
        v_bottom1 = num_upper_vertices + (j + 1)
        faces.append([v_top0, v_bottom0, v_top1])
        faces.append([v_bottom0, v_bottom1, v_top1])

    # Правая граница (300 точек)
    for j in range(ny - 1):
        v_top0 = j * nx + (nx - 1)
        v_bottom0 = num_upper_vertices + ny + j
        v_top1 = (j + 1) * nx + (nx - 1)
        v_bottom1 = num_upper_vertices + ny + (j + 1)
        faces.append([v_top0, v_bottom0, v_top1])
        faces.append([v_bottom0, v_bottom1, v_top1])

    # Передняя граница (298 точек)
    v_top0 = 0
    v_bottom0 = num_upper_vertices
    v_top1 = 1
    v_bottom1 = num_upper_vertices + 2 * ny #90600
    faces.append([v_top0, v_bottom0, v_top1])
    faces.append([v_bottom0, v_bottom1, v_top1])

    for i in range(1, nx - 2):  # Учитываем пропуск углов
        v_top0 = i
        v_bottom0 = num_upper_vertices + 2 * ny + i - 1
        v_top1 = i + 1
        v_bottom1 = num_upper_vertices + 2 * ny + i
        faces.append([v_top0, v_bottom0, v_top1])
        faces.append([v_bottom0, v_bottom1, v_top1])

    v_top0 = nx - 2
    v_bottom0 = num_upper_vertices + 3 * ny - 3 #90897
    v_top1 = nx - 1
    v_bottom1 = num_upper_vertices + ny #90300
    faces.append([v_top0, v_bottom0, v_top1])
    faces.append([v_bottom0, v_bottom1, v_top1])

    # Задняя граница (298 точек)
    v_top0 = num_upper_vertices - nx
    v_bottom0 = num_upper_vertices + nx - 1 #90299
    v_top1 = num_upper_vertices - nx + 1
    v_bottom1 = num_upper_vertices + 3 * nx - 2 #90898
    faces.append([v_top0, v_bottom0, v_top1])
    faces.append([v_bottom0, v_bottom1, v_top1])

    for i in range(1, nx - 2):  # Учитываем пропуск углов
        v_top0 = num_upper_vertices - nx + i
        v_bottom0 = num_upper_vertices + 3 * nx - 3 + i
        v_top1 = num_upper_vertices - nx + i + 1
        v_bottom1 = num_upper_vertices + 3 * nx - 2 + i
        faces.append([v_top0, v_bottom0, v_top1])
        faces.append([v_bottom0, v_bottom1, v_top1])

    v_top0 = num_upper_vertices - 2
    v_bottom0 = num_upper_vertices + 4 * nx - 5 #91195
    v_top1 = num_upper_vertices - 1
    v_bottom1 = num_upper_vertices + 2 * nx - 1 #90599
    faces.append([v_top0, v_bottom0, v_top1])
    faces.append([v_bottom0, v_bottom1, v_top1])

    faces.append([num_upper_vertices, num_upper_vertices + 2 * ny - 1, num_upper_vertices + ny - 1])
    faces.append([num_upper_vertices + 2 * ny - 1, num_upper_vertices, num_upper_vertices + ny])

    vertices = np.array(vertices)
    faces = np.array(faces)

    # Создаём объект mesh для STL
    phase_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            phase_mesh.vectors[i][j] = vertices[f[j], :]

    phase_mesh.save(filename, mode = stl.Mode.ASCII)
    print(f"STL файл сохранён как {filename}")

    # Создаём 3D-ось
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Добавляем модель на ось
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(phase_mesh.vectors))

    # Автоматическое масштабирование
    scale = phase_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Показываем модель
    plt.show()


# Загрузка целевого изображения (укажите путь к вашему файлу)
image_path = 'D:/рабочий стол/Курганский ИД/фокусаторы/G-S/G-S/images/krest_dlya_kamery.png'
i_target = load_target_image(image_path, resolution)

# Запуск алгоритма
phase_gs, mse, e_in = gerchberg_saxton(i_target, i_in, lambda_, f, dx, iterations=2000, visualize=True)
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

shitpost_stl_gen(phase_gs, x, y, wavelength=lambda_, n=1.65, filename='phase_profile1.stl')


