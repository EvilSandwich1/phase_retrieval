import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# -------------------------------
# Функция загрузки целевого изображения
# -------------------------------
def load_target_image(image_path, resolution):
    """
    Загружает изображение, преобразует в оттенки серого, изменяет размер до (resolution x resolution)
    и нормализует интенсивность так, чтобы максимум был равен 1.
    """
    img = Image.open(image_path).convert('L')  # Конвертируем в оттенки серого
    img = img.resize((resolution, resolution), Image.LANCZOS)  # Высококачественное изменение размера
    i_target = np.array(img, dtype=np.float64)
    # Нормировка: максимальное значение становится равным 1, инвертируется для получения бинарного паттерна
    i_target = 1 - i_target / np.max(i_target)
    return i_target


# -------------------------------
# Метод углового спектра для распространения поля
# -------------------------------
def angular_spectrum_propagation(field, wavelength, dx, dy, distance):
    ny, nx = field.shape
    k = 2 * np.pi / wavelength

    # Пространственные частоты
    fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, dy))
    FX, FY = np.meshgrid(fx, fy)

    under_sqrt = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    under_sqrt = np.maximum(under_sqrt, 0)

    H = -1j * k / (2 * np.pi * distance) * np.exp(1j * k * distance * np.sqrt(under_sqrt))
    H[under_sqrt <= 0] = 0

    field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    field_prop_fft = field_fft * H
    field_prop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_prop_fft)))
    return field_prop


# -------------------------------
# Алгоритм Gerchberg–Saxton с возможностью задания начальной фазы
# -------------------------------
def gerchberg_saxton(i_target, i_in, wavelength, focal_length, dx, iterations=100, visualize=False, init_phase=None):
    power_in = np.sum(i_in)
    i_target_norm = i_target * (power_in / np.sum(i_target))

    # Инициализация фазы: если не передана, используем случайную фазу
    if init_phase is None:
        phase = np.random.rand(*i_in.shape) * 2 * np.pi
    else:
        phase = init_phase.copy()
    e_in = np.abs(i_in) * np.exp(1j * phase)

    # Если требуется, можно добавить фазовую маску линзы (здесь нулевая)
    phase_lens = 0

    mse_history = []
    if visualize:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(iterations):
        # Добавляем затухающий шум в фазу
        phase_noise = 0 * np.pi * np.random.randn(*phase.shape) * (1 - i / iterations)
        phase = np.angle(e_in) + phase_noise
        e_in = np.abs(i_in) * np.exp(1j * phase)

        e_in_with_lens = e_in * np.exp(1j * phase_lens)
        e_out = angular_spectrum_propagation(e_in_with_lens, wavelength, dx, dx, focal_length)

        # Наложение целевой амплитуды
        i_current = np.abs(e_out) ** 2
        e_out_prime = np.abs(i_target_norm) * np.exp(1j * np.angle(e_out))

        e_in_prime = angular_spectrum_propagation(e_out_prime, wavelength, dx, dx, -focal_length)
        e_in = np.abs(i_in) * np.exp(1j * np.angle(e_in_prime))

        mse = np.mean((np.abs(e_out) ** 2 / np.sum(np.abs(e_out) ** 2) - i_target_norm / power_in) ** 2)
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


# -------------------------------
# Функция оценки кандидата (начальной фазы)
# Выполняется заданное число итераций GS и вычисляются критерии:
#   f1 – ошибка (MSE), f2 – отрицательная дифракционная эффективность
# -------------------------------
def evaluate_candidate(candidate, i_target, i_in, wavelength, focal_length, dx, gs_iter):
    phase, mse_history, e_in = gerchberg_saxton(i_target, i_in, wavelength, focal_length, dx, iterations=gs_iter,
                                                visualize=False, init_phase=candidate)
    mse_final = mse_history[-1]
    e_out = angular_spectrum_propagation(e_in, wavelength, dx, dx, focal_length)
    I_out = np.abs(e_out) ** 2
    # Определяем целевую область как те пиксели, где i_target > 0.5 (при бинарном целевом изображении)
    mask = (i_target > 0.5)
    if np.sum(I_out) == 0:
        efficiency = 0
    else:
        efficiency = np.sum(I_out * mask) / np.sum(I_out)
    f1 = mse_final
    f2 = -efficiency  # поскольку оптимизация идёт по минимуму (чем ниже f2, тем выше эффективность)
    return f1, f2, phase


# -------------------------------
# Недоминированная сортировка для многокритериальной оптимизации (по Парето)
# -------------------------------
def non_dominated_sort(fitness_list):
    n = len(fitness_list)
    ranks = [None] * n
    dominated_count = [0] * n
    dominates = [[] for _ in range(n)]
    fronts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            f1_i, f2_i = fitness_list[i]
            f1_j, f2_j = fitness_list[j]
            # Кандидат i доминирует j, если оба критерия не хуже и хотя бы один строго лучше
            if (f1_i <= f1_j and f2_i <= f2_j) and (f1_i < f1_j or f2_i < f2_j):
                dominates[i].append(j)
            elif (f1_j <= f1_i and f2_j <= f2_i) and (f1_j < f1_i or f2_j < f2_i):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            ranks[i] = 1
    current_front = [i for i in range(n) if dominated_count[i] == 0]
    fronts.append(current_front)
    rank = 1
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    ranks[j] = rank + 1
                    next_front.append(j)
        rank += 1
        current_front = next_front
        if current_front:
            fronts.append(current_front)
    return ranks, fronts


# -------------------------------
# Вычисление числа искр (интенсивности взрыва) для каждого кандидата
# согласно формуле (4) и ограничение по формуле (5)
# -------------------------------
def compute_explosion_intensity(ranks, m, a, b, eps):
    rank_max = max(ranks)
    intensities = []
    denominator = sum(rank_max - r + eps for r in ranks)
    for r in ranks:
        S = m * (rank_max - r + eps) / denominator if denominator != 0 else m
        S_min = round(a * m)
        S_max = round(b * m)
        S = round(S)
        if S < S_min:
            S = S_min
        if S > S_max:
            S = S_max
        intensities.append(S)
    return intensities


# -------------------------------
# Вычисление амплитуды взрыва для каждого кандидата по формуле (6)
# -------------------------------
def compute_explosion_amplitude(ranks, A_max, eps):
    rank_min = min(ranks)
    denominator = sum(r - rank_min + eps for r in ranks)
    amplitudes = []
    for r in ranks:
        A = A_max * (r - rank_min + eps) / denominator if denominator != 0 else A_max
        amplitudes.append(A)
    return amplitudes


# -------------------------------
# Генерация искровых решений (sparks) на основе кандидата.
# Для каждого кандидата генерируются S равномерно распределённых искр и n гауссовских искр.
# Применяется отображение, чтобы значения фазы оставались в пределах [xmin, xmax].
# -------------------------------
def generate_sparks(candidate, S, A, n, xmin, xmax):
    sparks = []
    shape = candidate.shape
    # Равномерный шум
    for _ in range(S):
        noise = np.random.uniform(-A, A, size=shape)
        spark = candidate + noise
        spark = xmin + np.abs(spark - xmin) % (xmax - xmin)
        sparks.append(spark)
    # Гауссовский шум
    for _ in range(n):
        noise = np.random.normal(0, A, size=shape)
        spark = candidate + noise
        spark = xmin + np.abs(spark - xmin) % (xmax - xmin)
        sparks.append(spark)
    return sparks


# -------------------------------
# Основная функция FW-GS алгоритма
# (сочетание поиска оптимальной начальной фазы с помощью fireworks algorithm и последующего полного GS)
# Согласно статье :contentReference[oaicite:1]{index=1}.
# -------------------------------
def fw_gs(i_target, i_in, wavelength, focal_length, dx, gs_iter_eval=200, fw_max_eval=2000, visualize=False):
    # Параметры fireworks алгоритма (см. Таблицу 1 статьи)
    N = 20  # Размер популяции
    m = 30  # Константа для вычисления числа искр
    n = 10  # Число гауссовских искр на кандидата
    a = 0.1
    b = 0.9
    eps = 2.2204e-16
    A_max = np.pi  # Максимальная амплитуда взрыва
    xmin = -np.pi
    xmax = np.pi

    # Инициализация популяции: N кандидатов с фазовыми масками, равномерно распределёнными в [xmin, xmax]
    population = [np.random.uniform(xmin, xmax, size=i_in.shape) for _ in range(N)]
    evaluations = 0
    pop_data = []  # Список словарей: {candidate, f1, f2, phase}
    for cand in population:
        f1, f2, final_phase = evaluate_candidate(cand, i_target, i_in, wavelength, focal_length, dx, gs_iter_eval)
        pop_data.append({'candidate': cand, 'f1': f1, 'f2': f2, 'phase': final_phase})
        evaluations += 1

    # Основной цикл оптимизации
    while evaluations < fw_max_eval:
        fitness_list = [(d['f1'], d['f2']) for d in pop_data]
        ranks, fronts = non_dominated_sort(fitness_list)
        S_list = compute_explosion_intensity(ranks, m, a, b, eps)
        A_list = compute_explosion_amplitude(ranks, A_max, eps)

        new_candidates = []
        for d, S, A in zip(pop_data, S_list, A_list):
            sparks = generate_sparks(d['candidate'], S, A, n, xmin, xmax)
            new_candidates.extend(sparks)

        new_pop_data = []
        for cand in new_candidates:
            f1, f2, final_phase = evaluate_candidate(cand, i_target, i_in, wavelength, focal_length, dx, gs_iter_eval)
            new_pop_data.append({'candidate': cand, 'f1': f1, 'f2': f2, 'phase': final_phase})
            evaluations += 1
            if evaluations >= fw_max_eval:
                break

        pop_data.extend(new_pop_data)
        fitness_list = [(d['f1'], d['f2']) for d in pop_data]
        ranks, fronts = non_dominated_sort(fitness_list)
        for i, d in enumerate(pop_data):
            d['rank'] = ranks[i]
        # Отбираем N лучших кандидатов по рангу, а при равенстве – по сумме f1+f2
        pop_data.sort(key=lambda d: (d['rank'], d['f1'] + d['f2']))
        pop_data = pop_data[:N]
        print(f"Evaluations: {evaluations}, Best candidate: f1={pop_data[0]['f1']:.2e}, f2={pop_data[0]['f2']:.2e}")

    best_candidate = pop_data[0]
    best_phase = best_candidate['phase']
    # Финальный запуск полного GS алгоритма с оптимизированной начальной фазой
    final_phase, mse_history, e_in = gerchberg_saxton(i_target, i_in, wavelength, focal_length, dx, iterations=100,
                                                      visualize=visualize, init_phase=best_phase)
    return final_phase, mse_history, e_in


# -------------------------------
# Основной блок: настройка параметров системы и запуск алгоритма
# -------------------------------
# Параметры системы
lambda_ = 2.14e-3  # Длина волны [м]
f = 50e-3  # Фокусное расстояние [м]
resolution = 300  # Разрешение
plate_size = 50e-3  # Размер пластины [м]
dx = plate_size / resolution  # Шаг сетки [м]

# Входной пучок (гауссов)
x = np.linspace(-plate_size / 2, plate_size / 2, resolution)
y = np.linspace(-plate_size / 2, plate_size / 2, resolution)
X, Y = np.meshgrid(x, y)
w0 = 5000e-3  # Радиус пучка
i_in = np.exp(-2 * (X ** 2 + Y ** 2) / w0 ** 2)

# Загрузка целевого изображения
image_path = 'C:/Users/Alexander/OneDrive/Desktop/Фокусаторы/SPR algorithm/cross2_300x300.png'
i_target = load_target_image(image_path, resolution)

# Выбор режима работы: стандартный GS или FW-GS
use_fw_gs = True  # установите False для стандартного GS

if use_fw_gs:
    phase_result, mse_history, e_in_final = fw_gs(i_target, i_in, lambda_, f, dx, gs_iter_eval=200, fw_max_eval=2000,
                                                  visualize=True)
else:
    phase_result, mse_history, e_in_final = gerchberg_saxton(i_target, i_in, lambda_, f, dx, iterations=100,
                                                             visualize=True)

# Визуализация результатов
plt.figure(figsize=(15, 5))
plt.subplot(221)
plt.imshow(i_target, cmap='gray')
plt.title('Целевое изображение')

plt.subplot(222)
plt.imshow(phase_result, cmap='viridis')
plt.title('Фазовый профиль')
plt.colorbar()

plt.subplot(224)
plt.semilogy(mse_history)
plt.title('Эволюция ошибки')
plt.xlabel('Итерация')
plt.grid()

plt.subplot(223)
plt.imshow(np.abs(angular_spectrum_propagation(e_in_final, lambda_, dx, dx, f)) ** 2, cmap='gray')
plt.title('Фактическое изображение')
plt.show()

# Сохранение фазового профиля
np.save('phase_profile.npy', phase_result)