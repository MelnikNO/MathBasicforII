# Лабораторная работа №02: LSTM Many‑to‑Many

## Название работы
**LSTM для бинарной классификации последовательностей (many‑to‑many)**

## Задание
Реализовать и обучить LSTM-модель, которая выдает бинарную метку на **каждом шаге** временной последовательности.

### Цель работы
Научиться:
- строить покадровые метки на основе кумулятивной суммы элементов последовательности;
- создавать модель LSTM с параметром `return_sequences=True`;
- вычислять две метрики качества: `token_accuracy` (точность по шагам) и `sequence_accuracy` (точность по целым последовательностям);
- понимать разницу между локальной и строгой точностью.

## Что было сделано

### 1. Создание датасета
Реализована функция `make_dataset()`, которая:
- генерирует случайные последовательности из +1.0 и -1.0;
- вычисляет кумулятивную сумму по временной оси;
- формирует покадровые бинарные метки по правилу `(cumsum > 0)`.

```python
def make_dataset(n_samples: int = 6500, seq_len: int = 16):
    x = np.random.choice([-1.0, 1.0], size=(n_samples, seq_len, 1)).astype(np.float32)
    cumsum = np.cumsum(x[:, :, 0], axis=1)
    y = (cumsum > 0).astype(np.float32)[..., None]
    return x, y, cumsum
```

## 2. Разделение данных
Данные разделены на обучающую и тестовую выборки в соотношении 80/20.

| Выборка | Форма X | Форма y |
|---------|---------|---------|
| Train   | (5200, 16, 1) | (5200, 16, 1) |
| Test    | (1300, 16, 1) | (1300, 16, 1) |

**Важное отличие от ЛР01:** теперь `y` имеет форму `(N, T, 1)` — метка на каждом шаге.

## 3. Построение модели
Создана Sequential-модель с LSTM слоем:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16, 1)),
    tf.keras.layers.LSTM(units=32, return_sequences=True, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Ключевой параметр: `return_sequences=True` — без него модель вернула бы только один вектор на последовательность.
```

## 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `binary_crossentropy`
- Метрика: `accuracy` (здесь это `token_accuracy`)

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 12
- Размер батча: 64
- Валидационная выборка: 20% от обучающей

## 6. Оценка модели
На тестовой выборке получены две метрики:

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Loss | 0.152 | Функция потерь |
| Token Accuracy | 0.942 | Доля правильных отдельных шагов |
| Sequence Accuracy | 0.815 | Доля полностью правильных последовательностей |

## 7. Визуализация
Построены графики динамики функции потерь и token_accuracy в процессе обучения.

## Ключевые понятия

### Token Accuracy vs Sequence Accuracy

| Метрика | Формула | Смысл |
|---------|---------|-------|
| Token Accuracy | (правильные шаги) / (все шаги) | Локальная точность |
| Sequence Accuracy | (правильные последовательности) / (все последовательности) | Строгая точность |

**Пример:** Если модель предсказала 3 шага из 4 правильно:
- Token Accuracy = 3/4 = 0.75
- Sequence Accuracy = 0/1 = 0.0 (последовательность целиком неверна)

## Вывод
Модель LSTM успешно справилась с задачей many‑to‑many классификации:
- **Token Accuracy = 94.2%** — модель правильно предсказывает отдельные шаги
- **Sequence Accuracy = 81.5%** — более строгая метрика

Ключевой вывод: для many‑to‑many задач критически важно использовать `return_sequences=True` и понимать разницу между локальной и строгой точностью.



---

# Скриншоты выполнения

<img width="966" height="529" alt="image" src="https://github.com/user-attachments/assets/dacef63b-c47e-4554-a408-6b7e0a9fb2cb" />

<img width="1119" height="632" alt="image" src="https://github.com/user-attachments/assets/a152b353-3e56-43f6-9602-2577be095f00" />

<img width="1267" height="801" alt="image" src="https://github.com/user-attachments/assets/f88c2a77-be54-4087-a72f-2ddeb72dd9b4" />


<img width="1449" height="576" alt="image" src="https://github.com/user-attachments/assets/5466bcea-91dc-412a-a3ca-ba5bbf38904b" />


<img width="1136" height="627" alt="image" src="https://github.com/user-attachments/assets/a769fc30-4314-4242-8359-3b30666e4e60" />

<img width="999" height="516" alt="image" src="https://github.com/user-attachments/assets/c2574cac-33cd-4b8c-bab1-391cac4870fd" />

<img width="1288" height="198" alt="image" src="https://github.com/user-attachments/assets/b909d510-13b7-483c-b89b-5e776058aff2" />

<img width="1267" height="530" alt="image" src="https://github.com/user-attachments/assets/a4494d2a-c50c-4b9e-bd96-18bb72e597bb" />

<img width="1381" height="770" alt="image" src="https://github.com/user-attachments/assets/2a512e94-0935-4e0a-aa91-6cb54c3426cd" />

<img width="1115" height="258" alt="image" src="https://github.com/user-attachments/assets/3cb77295-ac93-4495-86de-ee8c28fa67f9" />

<img width="1421" height="704" alt="image" src="https://github.com/user-attachments/assets/7197b81a-0a09-4ea8-a46a-073486438bb7" />

<img width="1433" height="554" alt="image" src="https://github.com/user-attachments/assets/fe6dce68-172f-4fa3-bf56-5a87bfdd1a60" />

<img width="1569" height="562" alt="image" src="https://github.com/user-attachments/assets/be89de0f-4d25-4237-9237-7c0d977c15f7" />

