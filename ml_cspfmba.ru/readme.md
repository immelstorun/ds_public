![TG profile](https://img.shields.io/badge/seikin__alexey-blue?logo=telegram&logoColor=yellow)

# Проект "Качество воздуха"
Подробная сводка ответов по заданию приведена в readme.

## Оглавление
[1. Описание проекта](#описание-проекта)

[2. Описание данных](#описание-данных)

[3. План работы](#план-работы)

[4. Результаты](#результаты)

[5. Выводы](#выводы)

## Задачи от заказчика

| Номер | Критерий  | Результат |
|:------------- |:---------------| :-------------|
| 1 |Интерпретируйте названия переменных. Какие вредные вещества и примеси охватывал проводившийся мониторинг?|[Перейти](#01-описание-состава-атмосферы-который-мониторился-станциями)|
| 2 |Существуют ли в представленных данных выбросы? Если таковые есть, каким образом будете от них избавляться?|[Перейти](#02-выбросы-в-данных)|
| 3 |Существуют ли корреляционные связи между представленными переменными. Если такие связи есть, насколько они сильны?| [Перейти](#03-мультиколлинеарность) |
|--|--|--|
| 4 |Создайте классификационную модель, которая по составу атмосферного воздуха сможет предсказать, в каком районе вы находитесь. В ходе создания попробуйте несколько методов классификации.|[Перейти](#04-базовая-модель)|
| 4* |Вкратце расскажите о математической базе, стоящей за используемым алгоритмом классификации. Каким образом происходит расчет модели, т. е. какой используется алгоритм и каким образом осуществляется предсказание на основе вновь поступивших данных.|[Перейти](#04-описание-использованного-алгоритма)|
| 4** |Использовали ли подбор гиперпараметров, если использовали, то что это были за гиперпараметры и каким образом осуществлялся подбор?|[Перейти](#04-описание-техники-подбора-гиперпараметров)|
|--|--|--|
| 5 |Проверьте работоспособность созданной классификационной модели. Расскажите какие метрики качества вы использовали и почему. Если вы создали несколько моделей проверьте работоспособность каждой из них. Интерпретируйте полученные результаты, а именно: существуют ли районы, которые весьма отличаются от всех прочих по характеристикам загрязнения атмосферы.| [Перейти](#05-описание-метрик-производительности) |
| 5* |Возможно ли интерпретировать модель, то есть объяснить, что она делает и почему, каким образом осуществляется предсказание? Возможно ли на основе этой интерпретации сделать выводы о наблюдаемых явлениях?| [Перейти](#05-интерпретация-коэффициентов) |
| 5** |Попробуйте применить метамоделирование. Каким образом вы его реализовали и смогли ли вы улучшить показатели качества классификации?| [Перейти](#05-метамоделирование) |
|--|--|--|
| 6 |Проверьте предположение о том, что если станции наблюдения расположены относительно недалеко друг от друга, то они регистрируют в целом схожие характеристики загрязнения атмосферы| [Перейти](#06-проверка-гипотезы) |
| 7 |Изменяется ли степень загрязнения атмосферы в зависимости от времени года, а если изменяется, то насколько существенно и в каких районах? Может ли качество модели быть улучшено, если ко входящим переменным добавить день или месяц наблюдения? Каким образом вы реализуете введение новой переменной в модель?| [Перейти](#07-сезонность-и-введение-новых-признаков) |

| -- |--| ✔ |

## Результаты

### 01. Признаки - описание состава атмосферы, который мониторился станциями  
  - $SO_2$ — диоксид серы. Главным источником диоксида серы является сжигание ископаемого топлива, такого как уголь, нефть и газ. SO2 может приводить к образованию кислотных дождей, когда соединяется с водными каплями в атмосфере. 
  - $NO_2$ -  диоксид азота. Главным источником диоксида азота является сжигание ископаемого топлива, особенно в автотранспорте и энергетических установках. Является важным компонентом смога и агрессивного дождя. Он может способствовать образованию озона на низких уровнях атмосферы. NO2 также может привести к кислотификации почв и водных ресурсов
  - $NO_2$ — озон.  Озон образуется в атмосфере в результате химических реакций между азотными оксидами (NOx), углеводородами и солнечным светом. На низких уровнях атмосферы озон является загрязнителем и может иметь негативное воздействие на растения.
  - $CO$ — угарный газ. Результат неполного сгорания органики. Метаболический яд. Является главным источником антропогенного выброса углерода в атмосферу. Он также является одним из главных вкладчиков в формирование парникового эффекта и изменение климата
  - $PM2.5$ - частицы, взвешенные в воздухе, которые имеют диаметр меньше 2.5 микрометра, могут взвешиваться в воздухе в течение длительного времени. PM2.5 может включать в себя такие вещества, как пыль, сажу, летучие органические соединения и химические вещества. 
  - $PM10$ - частицы, взвешенные в воздухе, которые имеют диаметр меньше 10 микрометров, могут взвешиваться в воздухе в течение длительного времени. Содержат в т.ч. пыльцу, сажу, споры грибов

Прочие имена переменных понятны интуитивно и в дополнительном описании не нуждаются.

### 02. Выбросы в данных
В данных обнаружены выбросы во всех действительных признаках. Их распределение и значения хорошо видны на гистограммах. Методов борьбы с выбросами несколько. В данном случаем применены:
* Фильтрация на основе статистических показателей: Использование межквартильного размаха (IQR), где данные за пределами 1.5 * IQR от первого и третьего квартилей считаются выбросами. Использованы квартили 0,05 / 0,95
* Так же использован метод ограничения данных: применение верхних и нижних порогов для данных вручную или исходя из доменных знаний. Принудительно удалены отрицательные значения концентраций и размеров, тк физически они не имеют смысла.

### 03. Мультиколлинеарность
Поскольку признаками являются как действительные так и категориальные признаки - применим метод phik, тк это статистический показатель, разработанный как расширение коэффициента корреляции Пирсона, но он предназначен для эффективной работы с категориальными, порядковыми и интервальными данными.

Если ранг матрицы соответствует количеству признаков, это означает, что все признаки являются линейно независимыми.
Формально линейной зависимости между факторами нет, и матрица факторов имеет максимальный ранг, но обнаружена мультиколлинеарность - матрица корреляции практически вырождена, несмотря на то что имеет максимальный ранг.

Обнаружены связи между признаками разной степени силы. К удалению выбраны признаки с силой связи между сильной и умеренной.

В нашем случае список признаков к удалению, где корреляция больше THR = 0.7: ['co', 'pm2_5'] 
После удаления столбцов были расчитаны матрица корреляции, ее детерминант (который стал чуть больше: от 0.00241 к 0.1), а так-же визуализированы итоговые парные корреляции признаков.

### 04. Базовая модель
Построены базовые модели без подбора гиперпараметров. Для каждой модели выведена матрица правильных попаданий и ошибок по классам (confusion matrix). В список моделей добавлена Dummy модель классификатора - как индикатор адекватности моделей. Перечень использованных моделей на этапе построения базовой модели:

* DummyClassifier(strategy='most_frequent'),
* LogisticRegression(solver='lbfgs', max_iter=100),   
* GaussianNB(),
* KNeighborsClassifier(n_neighbors=5),
* GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
* AdaBoostClassifier(n_estimators=50, learning_rate=1.0),
* DecisionTreeClassifier(max_depth=None, criterion='gini'),
* RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
* SVC(kernel='rbf', C=1.0, gamma='scale'),
* CatBoostClassifier(verbose=0, random_seed=RANDOM_STATE),
* LGBMClassifier(random_state=RANDOM_STATE)

### 04*. Описание использованного алгоритма
Описание работы алгоритма классификации
Логистическая регрессия — это статистическая модель, используемая для прогнозирования вероятности наступления события путем применения логистической функции к линейной комбинации входных переменных.

Математическая формулировка:
Линейная комбинация входных переменных (предикторов): $z = w0 + w1*x1 + w2*x2 + ... + wn*xn$ где $w0$ — интерсепт (свободный член), $wi$ — веса модели, $xi$ — предикторы.

Логистическая функция (сигмоид): $\sigma(z) = \frac{1}{1 + e^{-z}}$ Эта функция преобразует линейную комбинацию в вероятность (от 0 до 1).

Алгоритм расчета модели:
Модель логистической регрессии обычно обучается методом максимального правдоподобия. Алгоритмы, такие как градиентный спуск или более продвинутые варианты (например, L-BFGS, Ньютона-Рафсона), используются для оптимизации логистической функции стоимости, которая измеряет разницу между предсказанными вероятностями и фактическими метками классов.

Предсказание:
После того как модель обучена (т.е., определены веса wi), предсказание вероятности того, что новый пример данных принадлежит положительному классу, происходит следующим образом:

Вычисляется линейная комбинация входных переменных для нового примера данных.
Полученное значение подается на вход логистической функции.
Результат логистической функции интерпретируется как вероятность наступления события (принадлежность к классу 1). Если вероятность больше заданного порога (например, 0.5), пример относят к классу 1, иначе — к классу 0.
Таким образом, логистическая регрессия предоставляет вероятностную оценку того, что пример данных принадлежит определенному классу, что делает ее мощным инструментом для классификационных задач.

### 04** Описание техники подбора гиперпараметров

В коде используется `RandomizedSearchCV` для подбора гиперпараметров. Этот метод осуществляет выборку заданного числа комбинаций параметров из указанных распределений (distribs) и выполняет кросс-валидацию для каждого набора. В качестве метрики для оценки лучших параметров используется '`accuracy`' (точность).

Стандартизация данных с помощью `StandardScaler` в каждом пайплайне важна, поскольку она обеспечивает равный вклад каждого признака в вычисления расстояний в моделях, чувствительных к масштабу данных, например, логистическая регрессия и метод ближайших соседей. Это шаг предварительной обработки, который преобразует данные так, чтобы их среднее значение было равно нулю, а стандартное отклонение — единице.

Для защиты от утечки данных в процессе обучения код использует кросс-валидацию внутри цикла `RandomizedSearchCV` (с параметром cv=5). Это обеспечивает, что модель никогда не тестируется на данных, которые она видела во время обучения, что могло бы привести к завышенной оценке эффективности.

Применение простой базовой модели (dummy model) полезно в качестве отправной точки для сравнения. Она использует простые правила, такие как 'most_frequent', 'stratified' или 'uniform', для прогнозирования. Это позволяет убедиться, что более сложные модели действительно извлекают закономерности из данных, а не просто работают так же или немного лучше, чем тривиальная модель.

#### Описание гиперпараметров

рассмотрим каждый гиперпараметр модели LogisticRegression в контексте мультиклассовой классификации:

* `C`:

Этот гиперпараметр контролирует обратную величину силы регуляризации. Меньшие значения соответствуют более сильной регуляризации. Регуляризация помогает предотвратить переобучение, штрафуя модель за слишком большие веса. В мультиклассовой классификации регуляризация может влиять на способность модели обобщать информацию по различным классам.

* `solver`:

Этот гиперпараметр указывает, какой алгоритм использовать для решения оптимизационной задачи. Для мультиклассовой классификации такие решатели, как 'newton-cg', 'lbfgs' и 'sag', могут обрабатывать мультиномиальную потерю, в то время как 'liblinear' ограничен схемой один-против-всех, если только не использовать 'auto', который автоматически выбирает решатель на основе данных.

* `penalty`:

Этот параметр указывает, какой нормы следует придерживаться при регуляризации. Чаще всего используется 'l2'-штраф, который подходит ко всем решателям, кроме 'liblinear', при использовании в мультиклассовой классификации. Он способствует созданию модели с лучшей обобщающей способностью, так как стимулирует меньшие, более разбросанные веса.

* `multi_class`:

Этот гиперпараметр определяет подход, используемый для решения задач мультиклассовой классификации. 'ovr' обозначает схему "один против всех", когда для каждого класса обучается отдельный бинарный классификатор. Другой вариант - 'multinomial', который рассматривает проблему как единую мультиномиальную задачу, что часто приводит к более точным оценкам вероятностей.

* `max_iter`:

Этот гиперпараметр указывает максимальное количество итераций, которые решатель выполняет для сходимости. В мультиклассовых задачах, где оптимизация проводится по большему числу классов и, возможно, по большему набору данных, важно иметь достаточное количество итераций, чтобы модель могла сойтись к хорошему решению.

Каждый из этих гиперпараметров можно настроить для оптимизации производительности модели логистической регрессии в задаче мультиклассовой классификации. Обычно используется поиск по сетке (grid search) или случайный поиск (randomized search) для экспериментирования с различными комбинациями этих гиперпараметров для нахождения наилучшей конфигурации модели.

### 05. Описание метрик производительности

Проведено тестирование Логистической регресии с следующими результатами:
* `Accuracy`: 0.5115
* `F1 Score`: 0.5052
* `ROC AUC`: 0.8185
* `Confusion Matrix`:

[[27 20  6  7  9  7]<br>
 [11 46  1  4  9  6]<br>
 [ 1  3 46 13  4  1]<br>
 [ 5  1 11 51  1  0]<br>
 [ 1 21  5  3 33 12]<br>
 [ 8 18  2  5 18 20]]

 Можно отметить, что часть районов предсказывается с намного меньшим количеством ошибок. К таким выводом можно 
 прийти в том числе и после интерпретации коэфициентов модели.


`Точность (Accuracy)` - это простейшая метрика, которая показывает долю правильно классифицированных примеров среди всех примеров. Она рассчитывается как отношение числа правильных предсказаний к общему числу предсказаний.

`F1-мера` - это гармоническое среднее между точностью (precision) и полнотой (recall). F1-мера более устойчива к несбалансированным классам, чем простая точность. В многоклассовой классификации F1-мера может быть рассчитана несколькими способами:

`Macro-average F1(применена в модели)` - рассчитывается отдельно для каждого класса и затем усредняется. Это дает равный вес каждому классу.

`Weighted-average F1` - учитывает количество примеров в каждом классе при усреднении, что делает его полезным для несбалансированных наборов данных.

`Micro-average F1` - считает общее количество TP (истинно положительных результатов), FP (ложноположительных результатов), и FN (ложноотрицательных результатов) по всем классам, а затем вычисляет F1-меру.

`ROC AUC` - площадь под кривой ошибок (ROC, Receiver Operating Characteristic), это мера способности модели различать классы. В многоклассовой классификации ROC AUC может быть рассчитана для каждого класса против всех остальных (One-vs-Rest) или используя подход One-vs-One, где рассчитывается AUC для каждой пары классов.

`Матрица ошибок (Confusion Matrix)` - это таблица, показывающая количество правильных и неправильных предсказаний, разбитых по классам. Это позволяет более детально увидеть, какие классы модель предсказывает точно, а с какими возникают проблемы.

### 05*. Интерпретация коэффициентов

Коэффициенты модели логистической регрессии отражают влияние единицы изменения признака на логарифм шансов принадлежности к определенному классу, при условии, что все остальные признаки остаются неизменными. Положительный коэффициент указывает на то, что с увеличением значения признака увеличивается вероятность принадлежности к классу, для которого этот коэффициент был вычислен. Отрицательный коэффициент говорит об обратном — с увеличением признака вероятность принадлежности к классу уменьшается. Величина коэффициента показывает силу влияния: чем больше абсолютное значение коэффициента, тем сильнее признак влияет на принадлежность к классу. Термин Intercept (свободный член) можно интерпретировать как логарифм шансов принадлежности к классу, когда значения всех признаков равны нулю.

В нашем случае можно сделать вывод, что значения признаков не позволяют добиться хорошего качества.

### 05**. Метамоделирование:

Создана метамодель, используя стекинг с тремя базовыми моделями (KNN, случайный лес и SVM) и мета-классификатором (логистическая регрессия). После обучения модели на обучающем наборе данных, код выводит точность метамодели на тестовом наборе, детальный отчет по классификации для каждого класса и матрицу ошибок, которая показывает количество правильно и неправильно классифицированных экземпляров.
Стекинг немного помог повысить точность предсказания, но этот метод требует отдельной настройки и подготовки базовых моделей. В нашем случае в предобработку для каждой базовой модели добавлены различные варианты. Например метод главных компонент или создание полиномиальных признаков разных степеней.


### 06. Проверка гипотезы 
`Нулевая гипотеза (H0)`: Не существует статистически значимой линейной связи между расстоянием между станциями наблюдения и схожестью характеристик загрязнения атмосферы. То есть любая обнаруженная корреляция является результатом случайности.

`Альтернативная гипотеза (H1)`: Существует статистически значимая линейная связь между расстоянием между станциями наблюдения и схожестью характеристик загрязнения атмосферы.

`Результаты`:

`Коэффициент корреляции (r)`: 0.4836552976855788
`P-value`: 0.06775864774806167
`Интерпретация`:

Коэффициент корреляции r = 0.48 указывает на наличие умеренной положительной связи между расстоянием между станциями и схожестью характеристик загрязнения. Однако полученное p-value превышает общепринятый порог статистической значимости α = 0.05, что не позволяет нам отклонить нулевую гипотезу.

Таким образом, на уровне значимости 5% мы не можем утверждать, что наблюдаемая корреляция между расположением станций и схожестью данных о загрязнении атмосферы является статистически значимой. Это не исключает возможности наличия связи, но для подтверждения такой связи требуются дополнительные данные или более мощное исследование.


### 07. Сезонность и введение новых признаков
Построение графика процентного изменения концентрации поллютантов по временам года позволяет отметить несколько закономерностей:
* не всегда изменения концентраций отдельных поллютантов коррелируют по времени года
* в целом видна выраженная сезонность по всем газам, кроме диоксида серы и районам
* изменения концентраций диоксида серы тоже подвержено сезонности, но паттерны для районов отличаются по сезонам
* наибольшим вариациям подвержен угарный газ и озон, при этом просматривается их обратная корреляция 
На метамодели с введениеми данных о сезоне и месяце удалось достигнуть повышения точности на тестовой выборке до 0,63 (было 0,6)

[к списку результатов](#задачи-от-заказчика)

## Описание проекта
Задание включает анализ данных из двух таблиц: 
* AerialEcoData_Seoul.csv
* Station_coordinates.csv

Данные имеют весьма косвенное отношение к биологии, тем не менее они весьма интересны в аспекте построения классификационных моделей.

На территории Сеула есть несколько станций наблюдения за экологической обстановкой. В течение 2017-го года на каждой станции выполнялся мониторинг состояния атмосферного воздуха – оценивалась степень загрязнения вредными примесями. 

## Описание данных
В таблице AerialEcoData_Seoul.csv представлены суточные медианные значения показателей загрязнения атмосферы. 

В таблицу Station_coordinates.csv сведены географические координаты станций наблюдения, а также указано в каких районах Сеула они расположены. 

## План работы
- [x]  Шаг 1: Подготовка данных
    - [x]  Загрузка данных
    - [x]  Предобработка датасетов
    - [x]  Расширение признакового пространства


- [X]  Шаг 2: Исследовательский анализ данных
    - [x]  Карта района
    - [x]  Пропуски и явные дубликаты
    - [x]  Выбросы
    - [x]  Предобработка данных
    - [x]  Мультиколлинеарность
    - [x]  Баланс классов таргета

  
- [ ]  Шаг 3: Построение базовой модели для задачи классификации
    - [x]  Проверка модели на адекватность
    - [x]  Построение базовой модели классификации

- [ ]  Шаг 4: Перебор моделей с подбором гиперпараметров
    - [x] Описание гиперпараметров и алгоритма
- [ ]  Шаг 5: Тест лучшей модели
    - [x]  Проведение тестирования
- [ ]  Шаг 6: Метамоделирование
    - [x]  Построение метамодели
- [ ]  Шаг 7: Проверка гипотезы о связи расстояния между районами и уровнем загрязнений
    - [x]  Расчет статистической значимости результатов
- [ ]  Шаг 8: Проверка гипотезы о временах года
    - [x]
- [ ]  Шаг 9: Общий вывод
    - [x]

## Выводы

В ходе работы над заданем были опробованы различные техники для создания моделей машинного обучения для предсказания района города по составу атмосферы по содержанию SO2, NO2, O3, CO, PM10 и PM2.5. 

Категориальные признаки кодируются методом one-hot encoding, в то время как числовые признаки нормализуются и возможно трансформируются с использованием полиномиальных признаков.

В качестве базовых моделей используются три различных алгоритма: метод ближайших соседей, случайный лес и машина опорных векторов. Эти базовые модели объединяются с помощью мета-классификатора, в данном случае это логистическая регрессия с мультиномиальным распределением. Эффективность ансамблевой модели оценивается с использованием кросс-валидации, и её производительность отражается в виде точности на обучающей и валидационной выборках.

Невозможность достижения высокой точности предсказаний может быть связана с рядом проблем:

Качество данных: В целом в анализе рассмотрен не большой датасет. Из него были убраны выбросы. Но в целом их репрезентативность нужно увеличивать за счет получения данных из доп источников.

Выбор признаков: Не смотря на коррелирующие признаки они не удалялись при обучении моделей. Более того их удаление это снижало точность.

Сложность модели: Для предсказаний строились быстрые и простые модели для ускорения процесса разработки.

Несбалансированность классов: В нашем случае классы таргета сбалансированы

Настройка гиперпараметров: Как было указано выше для увеличения точности можно отдельно и долго работать с перебором как самих моделей так и их гиперпараметров

Метод ансамблирования: Выбор базовых моделей и мета-классификатора может не быть идеальным для конкретной задачи. Это отдельное поле для исследований.


1 Выполнено отлично. Дано подробное описание с превосходным оформлением, по сути – краткая лекция по видам экологических загрязнений в атмосфере. Алексей также позаботился о том, чтобы скрипт запускался по нажатию на любой машине, где есть доступ к интернету, так как он загрузил исследуемые массивы данных на внешний источник и ввел несколько строчек кода, которые осуществляют автоматическую загрузку этих массивов, если их нет в папке.

2 Выполнено хорошо. Как вариант, можно было рассмотреть определение выбросов с применением МГК.

3, 4, 4*, 4** Выполнено отлично, подробные разъяснения по применению логистической регрессии  – отдельный плюс.

5, 5*, 5**  Выполнено на высоте (5+).

6 Выполнено хорошо, засчитано применение математического аппарата для решения.

7 +/- Часть кода выдает ошибки.

Плюс также в представлении проекта на Github’е.

Итог – отлично справился с заданием.
Это тестовое задание на должность Data Scientist для ФГБУ ЦСП ФМБА России отдел АиПМБРЗ. deadline (1 декабря 2023 в 11:51 + 7 дней)
