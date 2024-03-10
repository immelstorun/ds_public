import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats # библиотека для расчетов
import statsmodels.api as sm # для qq plot

def df_info(df):
    # import pandas as pd
    """
    Функция для получения информации о датафрейме.

    Args:
        df (pandas.DataFrame): Входной датафрейм.

    Returns:
        None
    """
    len_df = len(df)
    count = 0
    na_col_name = []
    na_col_size = []
    unique_list = []

    for col in df.columns:
        notnull = df[col].notna().sum()
        if notnull < len_df:
            count += 1
        if df[col].isna().sum() > 0:
            na_col_name.append(col)
            col_size = df[col].isna().sum() / len(df) * 100
            na_col_size.append(col_size)

        unique_list.append(len(df[col].unique()))

    pivot = pd.DataFrame(data=unique_list, index=df.columns, columns=['Уникальные значения'])

    print(f'Количество записей: \t {len_df}')
    print(f'Количество столбцов: \t {len(df.columns)}')
    print(f'Явных дубликатов: \t {df.duplicated().sum()}')
    print(f'Пропуски присутствуют в {count} столбцах из {len(df.columns)}:')
    display(pd.DataFrame(na_col_size, index=na_col_name, columns=['Пропущено %']).sort_values(by='Пропущено %', ascending=False))
    print('Обобщенная информация:')
    display(df.info(verbose=False))
    print('Первые 3 строки:')
    display(df.head(3))

def low_information_features(df):
    # import pandas as pd
    """
    Функция для определения неинформативных признаков в датафрейме.
    
    Признаки считаются неинформативными, если одно значение встречается более чем в 95%
    случаев или если более 95% значений являются уникальными.
    
    Args:
        df (pandas.DataFrame): Входной датафрейм.
        
    Returns:
        Union[list, str]: Список неинформативных признаков или сообщение об их отсутствии.
    """
    low_information_cols = []  # инициализация списка для неинформативных признаков

    for col in df.columns:  # цикл по всем столбцам
        top_freq = df[col].value_counts(normalize=True).max()  # наибольшая относительная частота в признаке
        nunique_ratio = df[col].nunique() / df[col].count()  # доля уникальных значений от размера признака

        # сравниваем наибольшую частоту с порогом 95%
        if top_freq > 0.95:
            low_information_cols.append(col)  # добавляем столбец в список
            # выводим информацию о признаке
            print(f'{col}: {round(top_freq*100, 2)}% одинаковых значений')

        # сравниваем долю уникальных значений с порогом 95%
        if nunique_ratio > 0.95:
            low_information_cols.append(col)  # добавляем столбец в список
            # выводим информацию о признаке
            print(f'{col}: {round(nunique_ratio*100, 2)}% уникальных значений')

    if not low_information_cols:
        return "Нет неинформативных признаков"
    else:
        return low_information_cols  # возвращаем список неинформативных признаков
    
def plot_missing_values(df):
    # import pandas as pd
    """
    Функция для построения столбчатой диаграммы, отображающей процент пропущенных значений в каждом столбце датафрейма.

    Args:
        df (pandas.DataFrame): Входной датафрейм.

    Returns:
        None
    """
    # Вычисление процента пропущенных значений
    nans = 100 * df.isnull().mean().sort_values(ascending=False)[df.isnull().mean().sort_values(ascending=False) > 0]

    # Построение столбчатой диаграммы
    nans.plot(
        kind='bar', 
        logy=False,
        figsize=(10,5),
        title='Соотношение пропусков, %'
    )

def plot_sample_data(sample_data_a, sample_data_b, title_a='a', title_b='b'):
    # import pandas as pd
    # import seaborn as sns
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy import stats # библиотека для расчетов
    # import statsmodels.api as sm # для qq plot
    """
    Функция для создания графиков QQ, гистограмм и boxplot для двух выборок.

    Args:
    sample_data_a: Первая выборка данных
    sample_data_b: Вторая выборка данных
    title_a: Название для первой выборки
    title_b: Название для второй выборки
    """
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.5)

    # Создание графика QQ-графика для sample_data_a
    # sm.qqplot(sample_data_a, line='45', ax=axs[0, 0])
    stats.probplot(sample_data_a, plot=axs[0, 0]) # qq plot
    axs[0, 0].set_title(f'График QQ для {title_a}')

    # Создание графика QQ-графика для sample_data_b
    # sm.qqplot(sample_data_b, line='45', ax=axs[0, 1])
    stats.probplot(sample_data_b, plot=axs[0, 1]) # qq plot
    axs[0, 1].set_title(f'График QQ для {title_b}')

    # Создание гистограммы для sample_data_a
    sns.histplot(sample_data_a, kde=True, ax=axs[1, 0])
    axs[1, 0].set_title(f'Гистограмма для {title_a}')

    # Создание гистограммы для sample_data_b
    sns.histplot(sample_data_b, kde=True, ax=axs[1, 1])
    axs[1, 1].set_title(f'Гистограмма для {title_b}')
    
    # Создание графика boxplot для sample_data_a с отрисовкой среднего и медианы
    sns.boxplot(sample_data_a, ax=axs[2, 0], showmeans=True, meanline=True, showfliers=False)
    axs[2, 0].set_title(f'Boxplot для {title_a} \n mean: {np.mean(sample_data_a):.2f}, median: {np.median(sample_data_a):.2f}')
    
    # Создание графика boxplot для sample_data_b с отрисовкой среднего и медианы
    sns.boxplot(sample_data_b, ax=axs[2, 1], showmeans=True, meanline=True, showfliers=False)
    axs[2, 1].set_title(f'Boxplot для {title_b} \n mean: {np.mean(sample_data_b):.2f}, median: {np.median(sample_data_b):.2f}')

    plt.show()

def remove_outliers(df, group_column, value_column, threshold=3.5):
    """
    Удаляет выбросы из датафрейма для заданных групп.

    Аргументы:
    df : pandas.DataFrame
        Датафрейм, из которого удаляются выбросы.
    group_column : str
        Название столбца для группировки данных.
    value_column : str
        Название столбца, в котором проверяются выбросы.
    threshold : float, optional
        Пороговое значение Z-оценки для определения выбросов. По умолчанию 3.5.

    Возвращает:
    pandas.DataFrame
        Датафрейм без выбросов, с сохранением оригинальной структуры.
    """
    
    # Получаем уникальные значения для группировки
    groups = df[group_column].unique()
    # Создаем пустой DataFrame для хранения очищенных данных
    clean_data = pd.DataFrame()
    
    # Обходим все группы
    for group in groups:
        # Выбираем данные по текущей группе
        group_data = df[df[group_column] == group]
        # Вычисляем медиану значений в группе
        median = group_data[value_column].median()
        # Вычисляем медиану абсолютных отклонений от медианы
        mad = np.median(np.abs(group_data[value_column] - median))
        # Вычисляем модифицированную Z-оценку для каждого значения в группе
        modified_z_score = 0.6745 * (group_data[value_column] - median) / mad
        # Фильтруем данные, исключая выбросы
        group_clean_data = group_data[abs(modified_z_score) < threshold]
        # Объединяем очищенные данные с данными предыдущих групп
        clean_data = pd.concat([clean_data, group_clean_data])
    
    # Возвращаем очищенный датафрейм
    return clean_data