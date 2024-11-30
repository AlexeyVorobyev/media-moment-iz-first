"""
Модуль с дискретным преобразованием Фурье
"""
import numpy as np


def pad_to_power_of_two(matrix):
    """
    Дополняет 2D массив нулями до ближайших размеров, являющихся степенями двойки.

    Parameters:
    matrix : numpy.ndarray
        Исходный 2D массив.

    Returns:
    numpy.ndarray, tuple
        Дополненный массив и исходные размеры.
    """
    rows, cols = matrix.shape
    new_rows = 2 ** int(np.ceil(np.log2(rows)))
    new_cols = 2 ** int(np.ceil(np.log2(cols)))
    padded_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)
    for i in range(rows):
        for j in range(cols):
            padded_matrix[i, j] = matrix[i, j]
    return padded_matrix, (rows, cols)


def crop_to_original_size(padded_matrix, original_shape):
    """
    Обрезает дополненный 2D массив до исходных размеров.

    Parameters:
    padded_matrix : numpy.ndarray
        Дополненный 2D массив.
    original_shape : tuple
        Исходные размеры (строки, столбцы).

    Returns:
    numpy.ndarray
        Массив обрезанных размеров.
    """
    rows, cols = original_shape
    return padded_matrix[:rows, :cols]

def fft_2d(matrix: np.ndarray) -> np.ndarray:
    """
     Проводит быстрое преобразование Фурье (FFT) для матрицы

    Parameters:
    matrix : numpy.ndarray
        Входящая матрица

    Returns:
    numpy.ndarray
    """

    rows, cols = matrix.shape
    if np.log2(rows) % 1 > 0 or np.log2(cols) % 1 > 0:
        fft_rows = np.array([dft_slow(row) for row in matrix])
        fft_result = np.array([dft_slow(col) for col in fft_rows.T]).T
    else:
        # Применить fft для каждой строки
        fft_rows = np.array([fft(row) for row in matrix])

        # Применить fft для каждого столбца
        fft_result = np.array([fft(col) for col in fft_rows.T]).T

    return fft_result


def dft_slow(x: np.ndarray) -> np.ndarray:
    """
    Произвести дискретное преобразование Фурье одномерного массива
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x: np.ndarray) -> np.ndarray:
    """
    Проводит быстрое преобразование Фурье (FFT) для одномерного массива, используя алгоритм Кули-Тьюки.

    Parameters:
    x : numpy.ndarray
        Входящий массив

    Returns:
    numpy.ndarray
    """
    N = len(x)
    if N <= 1:
        return x
    if np.log2(N) % 1 > 0:
        raise ValueError("Size of input must be a power of 2")

    even = fft(x[0::2])
    odd = fft(x[1::2])

    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return np.array([even[k] + T[k] for k in range(N // 2)] +
                    [even[k] - T[k] for k in range(N // 2)])


if __name__ == "__main__":
    # Тест, что результат нашего преобразования совпадает с реализацией NumPy
    x = np.random.random((512, 256))
    my_fft = fft_2d(x)
    num = np.fft.fft2(x)
    print(num)
    print(np.allclose(my_fft, num))
    print(my_fft)
