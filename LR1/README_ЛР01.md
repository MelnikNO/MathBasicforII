# Лабораторная работа №01: SimpleRNN Many‑to‑One

## Название работы
**SimpleRNN для бинарной классификации последовательностей (many‑to‑one)**

## Задание
Реализовать и обучить рекуррентную нейронную сеть на синтетических данных для задачи бинарной классификации последовательностей.

### Цель работы
Научиться:
- строить бинарные метки на основе суммы элементов последовательности;
- создавать модель с использованием слоя SimpleRNN;
- компилировать и обучать модель;
- оценивать качество модели на тестовых данных.

## Что было сделано

### 1. Создание датасета
Реализована функция `make_dataset()`, которая:
- генерирует случайные последовательности из +1.0 и -1.0;
- вычисляет сумму по временной оси;
- формирует бинарные метки по правилу `(sum > 0)`.

### 2. Разделение данных
Данные разделены на обучающую и тестовую выборки в соотношении 80/20 с сохранением баланса классов.

| Выборка | Форма X | Форма y |
|---------|---------|---------|
| Train   | (4800, 12, 1) | (4800,) |
| Test    | (1200, 12, 1) | (1200,) |

### 3. Построение модели
Создана Sequential-модель со следующей архитектурой:
- Входной слой: `Input(shape=(12, 1))`
- Слой SimpleRNN: 16 нейронов, активация `tanh`
- Выходной слой: `Dense(1, activation='sigmoid')`

### 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `binary_crossentropy`
- Метрика: `accuracy`

### 5. Обучение
Модель обучена с параметрами:
- Эпохи: 10
- Размер батча: 64
- Валидационная выборка: 20% от обучающей

### 6. Оценка модели
На тестовой выборке получены следующие метрики:

| Метрика | Значение |
|---------|----------|
| Loss | 0.287 |
| Accuracy (evaluate) | 0.875 |
| Accuracy (ручной расчёт) | 0.875 |

### 7. Визуализация
Построены графики динамики accuracy и loss в процессе обучения.

## Вывод
Модель SimpleRNN успешно справилась с задачей бинарной классификации последовательностей.
Точность на тестовых данных составила **87.5%**, что подтверждает корректность реализации и способность сети обучаться на синтетических данных.

---

# Скриншоты выполнения

<img width="1508" height="800" alt="image" src="https://github.com/user-attachments/assets/47704aae-9f4e-4cde-be0e-59fb89feed6a" />

<img width="1116" height="523" alt="image" src="https://github.com/user-attachments/assets/335d77a6-9701-4a9c-804b-acc50230a38b" />


<img width="1195" height="642" alt="image" src="https://github.com/user-attachments/assets/be46a53f-036a-4c68-988a-d46e36bcdf99" />

<img width="1048" height="441" alt="image" src="https://github.com/user-attachments/assets/dd328552-1a47-4ad3-94c6-5d1802a40547" />


<img width="1016" height="549" alt="image" src="https://github.com/user-attachments/assets/902d00bb-89bb-4d8c-9db4-ffcdf63af174" />



<img width="1015" height="413" alt="image" src="https://github.com/user-attachments/assets/e2f8e807-e7e1-4e7b-acfe-38575acc0590" />

<img width="1347" height="579" alt="image" src="https://github.com/user-attachments/assets/93924242-b605-4e18-9e87-4dd0c5a80393" />

<img width="964" height="273" alt="image" src="https://github.com/user-attachments/assets/c1523947-1b98-4781-a38f-358b7cdf02d9" />


<img width="1375" height="594" alt="image" src="https://github.com/user-attachments/assets/d95b8345-9b20-44db-8daa-edbc0e8d78a6" />

<img width="1190" height="326" alt="image" src="https://github.com/user-attachments/assets/0afc6af5-74dd-4047-9eb5-43a5a7657ed0" />



