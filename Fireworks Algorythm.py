import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from PIL import Image

# Параметры
lambda_ = 520e-9  # длина волны (м), 520 нм
w = 400e-6  # радиус талии гауссова пучка (м), 400 мкм
N_x = 256  # количество точек по x
N_y = 256  # количество точек по y
dx = 20e-6  # интервал выборки (м), 20 мкм
dy = 20e-6
f = 0.2  # фокусное расстояние линзы (м), 200 мм

# Координаты входной плоскости
x = (np.arange(N_x) - N_x / 2) * dx
y = (np.arange(N_y) - N_y / 2) * dy
X, Y = np.meshgrid(x, y)

# Амплитуда гаусс |ова пучка
A_input = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
A_input /= np.sqrt(np.sum(A_input ** 2))  # Нормализация

# Координаты выходной плоскости
du = lambda_ * f / (N_x * dx)
dv = lambda_ * f / (N_y * dy)
u = np.fft.fftfreq(N_x, d=1) * du * N_x
v = np.fft.fftfreq(N_y, d=1) * dv * N_y
U, V = np.meshgrid(u, v)

# Желаемая интенсивность: квадратный пучок 760 мкм × 760 мкм
side = 760e-6
s = (np.abs(U) < side / 2) & (np.abs(V) < side / 2)
I_desired = np.zeros((N_y, N_x))
I_desired[s] = 1 / np.sum(s)  # Нормализация
A_desired = np.sqrt(I_desired)

# Параметры алгоритма Fireworks
N = 20  # количество фейерверков
m = 30  # параметр для ограничения числа искр
a = 0.1  # минимальная доля искр
b = 0.9  # максимальная доля искр
A_max = np.pi  # максимальная амплитуда взрыва
n_gaussian = 5  # количество гауссовских искр
sigma_gaussian = 0.1  # стандартное отклонение для гауссовских искр
max_generations = 10  # максимальное количество поколений
M_gs = 200  # количество итераций GS на оценку
epsilon = 2.2204e-16  # малое значение для избежанияDделения на ноль


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

# Алгоритм Герчберга-Сакстона (GS)
def gs_algorithm(A_input, A_desired, phi_init, num_iterations):
    phi = phi_init.copy()
    for _ in range(num_iterations):
        field_input = A_input * np.exp(1j * phi)
        field_output = fftshift(fft2(field_input))
        psi = np.angle(field_output)
        field_output_desired = A_desired * np.exp(1j * psi)
        field_input_back = ifft2(ifftshift(field_output_desired))
        phi = np.angle(field_input_back)
    field_input_final = A_input * np.exp(1j * phi)
    field_output_final = fftshift(fft2(field_input_final))
    I_output = np.abs(field_output_final) ** 2
    return phi, I_output


# Вычисление фитнес-функций (RMSE и дифракционная эффективность)
def compute_fitness(I_output, I_desired, s):
    I_s = I_output[s]
    I_desired_s = I_desired[s]
    RMSE = np.sqrt(np.sum((I_s - I_desired_s) ** 2) / np.sum(I_desired_s ** 2))
    eta = np.sum(I_s) / np.sum(I_output)
    return RMSE, eta


# Нерегулярная сортировка для многокритериальной оптимизации
def non_dominated_sorting(fitness_values):
    fronts = []
    remaining = list(range(len(fitness_values)))
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            rmse_i, eta_i = fitness_values[i]
            for j in remaining:
                rmse_j, eta_j = fitness_values[j]
                if rmse_j < rmse_i and eta_j > eta_i:
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining = [idx for idx in remaining if idx not in front]
    return fronts


# Инициализация N случайных фаз
population = [np.random.rand(N_y, N_x) * 2 * np.pi for _ in range(N)]

# Основной цикл FW-GS
for t in range(max_generations):
    # Оценка всех решений в популяции
    fitness_list = []
    for phi in population:
        phi_opt, I_output = gs_algorithm(A_input, A_desired, phi, M_gs)
        RMSE, eta = compute_fitness(I_output, I_desired, s)
        fitness_list.append((RMSE, eta))

    # Нерегулярная сортировка
    fronts = non_dominated_sorting(fitness_list)

    # Назначение рангов
    ranks = np.zeros(N)
    for rank, front in enumerate(fronts, start=1):
        for idx in front:
            ranks[idx] = rank

    # Вычисление интенсивности взрыва S_i'
    Rank_max = max(ranks)
    sum_denominator = sum(Rank_max - rank + epsilon for rank in ranks)
    S_prime = [(Rank_max - rank + epsilon) / sum_denominator for rank in ranks]

    # Вычисление количества искр S_i
    S = [max(min(round(m * s), round(b * m)), round(a * m)) for s in S_prime]

    # Вычисление амплитуды взрыва A_i
    Rank_min = min(ranks)
    sum_denominator_A = sum(rank - Rank_min + epsilon for rank in ranks)
    A = [A_max * (rank - Rank_min + epsilon) / sum_denominator_A for rank in ranks]

    # Генерация искр
    sparks = []
    for i in range(N):
        num_sparks = S[i]
        for _ in range(num_sparks):
            phi_spark = population[i] + np.random.uniform(-A[i], A[i], size=(N_y, N_x))
            phi_spark = np.mod(phi_spark, 2 * np.pi)
            sparks.append(phi_spark)

    # Генерация гауссовских искр
    for _ in range(n_gaussian):
        i = np.random.randint(N)
        phi_gauss = population[i] * np.random.normal(1, sigma_gaussian, size=(N_y, N_x))
        phi_gauss = np.mod(phi_gauss, 2 * np.pi)
        sparks.append(phi_gauss)

    # Оценка всех фейерверков и искр
    all_solutions = population + sparks
    all_fitness = []
    for phi in all_solutions:
        phi_opt, I_output = gs_algorithm(A_input, A_desired, phi, M_gs)
        RMSE, eta = compute_fitness(I_output, I_desired, s)
        all_fitness.append((RMSE, eta))

    # Выбор следующего поколения
    fronts_all = non_dominated_sorting(all_fitness)
    selected_indices = []
    for front in fronts_all:
        if len(selected_indices) + len(front) <= N:
            selected_indices.extend(front)
        else:
            remaining = N - len(selected_indices)
            selected_indices.extend(np.random.choice(front, remaining, replace=False))
            break
    population = [all_solutions[idx] for idx in selected_indices]

# Выбор лучшего решения из Pareto front
best_idx = np.argmin([all_fitness[idx][0] for idx in fronts_all[0]])
best_phi = all_solutions[best_idx]

# Финальная оптимизация GS
phi_final, I_final = gs_algorithm(A_input, A_desired, best_phi, 400)
RMSE_final, eta_final = compute_fitness(I_final, I_desired, s)

print(f"Final RMSE: {RMSE_final:.4f}, Diffractive efficiency: {eta_final:.4f}")


# Визуализация
def plot_intensity(I, title, extent):
    plt.imshow(I, extent=extent, origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('u (м)')
    plt.ylabel('v (м)')


extent_u = [-u.max(), u.max(), -v.max(), v.max()]

plt.figure(figsize=(15, 5))

# 1. Входная интенсивность (Гаусс)
plt.subplot(1, 4, 1)
plot_intensity(A_input ** 2, 'Входная интенсивность (Гаусс)', extent_u)

# 2. Желаемая интенсивность (Квадрат)
plt.subplot(1, 4, 2)
plot_intensity(I_desired, 'Желаемая интенсивность (Квадрат)', extent_u)

# 3. Выходная интенсивность (FW-GS)
plt.subplot(1, 4, 3)
plot_intensity(I_final, 'Выходная интенсивность (FW-GS)', extent_u)

# 4. Фазовое распределение
plt.subplot(1, 4, 4)
plt.imshow(phi_final, extent=extent_u, cmap='hsv')
plt.colorbar()
plt.title('Фазовое распределение')
plt.xlabel('x (м)')
plt.ylabel('y (м)')

plt.tight_layout()
plt.show()