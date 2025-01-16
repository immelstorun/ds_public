import numpy as np

low_range = 1
high_range = 101

def predictor_01_random_search(number:int=1) -> int:
    """
    Запускает алгоритм случайного поиска для угадывания числа.

    Args:
        number (int, optional): Загаданное число. По умолчанию равно 1.

    Returns:
        int: Количество попыток, необходимых для угадывания числа.
    """

    count = 0

    while True:
        count += 1
        predict_number = np.random.randint(low_range, high_range) # Предполагаемое число
        if number == predict_number:
            break # Выход из цикла, если угадали
    return count


def predictor_02_more_less(number: int = 1) -> int:
    """
    Использует алгоритм бинарного поиска для угадывания числа.

    Args:
        number (int, optional): Загаданное число. По умолчанию равно 1.

    Returns:
        int: Количество попыток, необходимых для угадывания числа.
    """
    count = 0
    predict = np.random.randint(low_range, high_range)

    while number != predict:
        count += 1
        if number > predict:
            predict += 1
        elif number < predict:
            predict -= 1

    return count

def predictor_03_binary_search(guess):
    """
    Запускает алгоритм бинарного поиска для угадывания числа.

    Args:
        guess: Загаданное число.

    Returns:
        int: Количество попыток, необходимых для угадывания числа, или -1, если число не входит в диапазон от 1 до 100.
    """
    low = low_range
    high = high_range
    attempts = 0

    while low < high:
        mid = (low + high) // 2
        attempts += 1

        if mid < guess:
            low = mid + 1
        else:
            high = mid

    if low == guess:
        return attempts
    else:
        return -1

def scorer(predictor) -> int:
    """
    Вычисляет среднее количество попыток, необходимых для угадывания числа с использованием заданной функции предсказания.

    Args:
        predictor: Функция предсказания.

    Returns:
        tuple: Кортеж, содержащий имя функции предсказания и среднее количество попыток.
    """

    count_ls = []
    np.random.seed(42)  # Устанавливаем сид для воспроизводимости
    random_array = np.random.randint(low_range, high_range, size=(1000))  # Генерируем список из 1000 случайных чисел

    for number in random_array:
        count_ls.append(predictor(number))

    score = int(np.mean(count_ls))
        
    return (predictor.__name__, score)


    
#     # RUN
# if __name__ == '__main__':
#     scorer(predictor_01_random_search)