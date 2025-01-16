import matplotlib.pyplot as plt

def dict_plotter(results_dict):
    labels = list(results_dict.keys())
    values = list(results_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Установка размеров фигуры (ширина, высота)

    plt.bar(labels, values)
    plt.xlabel('Функции предсказания')
    plt.ylabel('Количество попыток')
    plt.title('Количество попыток для каждой функции предсказания')
    plt.xticks(rotation=45)

    # Добавление подписей над каждым столбцом
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha='center', va='bottom', fontsize=12)
        
    # Добавление сетки на график
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.tight_layout()
    return plt.show()