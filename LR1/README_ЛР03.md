# Лабораторная работа №03: GRU Encoder-Decoder для seq2seq

## Название работы
**GRU Encoder-Decoder для обратного порядка последовательности (seq2seq)**

## Задание
Реализовать модель seq2seq на базе GRU, которая разворачивает входную последовательность целочисленных токенов в обратном порядке.

### Цель работы
Научиться:
- работать с дискретными токенами и специальными токенами (PAD, SOS, EOS);
- строить encoder-decoder архитектуру с двумя входами;
- реализовывать teacher forcing через сдвиг decoder_input и decoder_target;
- вычислять метрики token_accuracy и exact_match.

## Формат данных
- Вход (encoder_input): последовательность из 6 целочисленных токенов (1-9)
- Выход (decoder_target): та же последовательность в обратном порядке
- Специальные токены:
  - `PAD = 0` — заполнение до фиксированной длины
  - `SOS = 10` — начало последовательности декодера
  - `EOS = 11` — конец последовательности
- Размер словаря: 12 (1-9 + PAD + SOS + EOS)

## Что было сделано

### 1. Создание датасета
Реализованы функции для генерации данных со сдвигом decoder_input и decoder_target:

```python
def make_one_sample(min_len: int = 3, max_len: int = ENC_LEN):
    length = np.random.randint(min_len, max_len + 1)
    seq = np.random.randint(1, 10, size=length, dtype=np.int32)
    rev = seq[::-1]  # обратный порядок
    return seq, rev

# Сдвиг: decoder_input = [SOS] + rev, decoder_target = rev + [EOS]
dec_in = [SOS_ID] + rev.tolist()
dec_out = rev.tolist() + [EOS_ID]


## 2. Разделение данных
Данные разделены на обучающую и тестовую выборки в соотношении 80/20.

| Выборка | Форма encoder_input | Форма decoder_input | Форма decoder_target |
|---------|---------------------|---------------------|----------------------|
| Train   | (5600, 6) | (5600, 7) | (5600, 7, 1) |
| Test    | (1400, 6) | (1400, 7) | (1400, 7, 1) |

## 3. Построение модели
Создана Encoder-Decoder модель с двумя входами:

```python
def build_model(vocab_size=12, emb_dim=24, latent_dim=48):
    # Энкодер
    encoder_inputs = tf.keras.layers.Input(shape=(ENC_LEN,))
    encoder_seq = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True),
        tf.keras.layers.GRU(latent_dim)
    ])
    context = encoder_seq(encoder_inputs)
    
    # Декодер
    decoder_inputs = tf.keras.layers.Input(shape=(DEC_LEN,))
    decoder_seq = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
    ])
    dec_emb = decoder_seq(decoder_inputs)
    
    dec_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True)
    dec_out = dec_gru(dec_emb, initial_state=context)
    
    # Выходной слой
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(dec_out)
    
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

## 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `sparse_categorical_crossentropy` (для целочисленных меток)
- Метрика: `accuracy` (token_accuracy)

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 16
- Размер батча: 64
- Валидационная выборка: 20% от обучающей

## 6. Оценка модели
На тестовой выборке получены две метрики:

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Loss | 0.052 | Функция потерь |
| Token Accuracy | 0.983 | Доля правильных отдельных токенов |
| Exact Match | 0.912 | Доля полностью правильных последовательностей |

## 7. Визуализация
Построены графики динамики функции потерь и token_accuracy в процессе обучения.

## Сравнение с ЛР02

| Параметр | ЛР02 (LSTM many-to-many) | ЛР03 (GRU seq2seq) |
|----------|--------------------------|---------------------|
| Тип задачи | many‑to‑many | seq2seq |
| Входы модели | Один | Два (encoder, decoder) |
| Тип данных | Вещественные числа | Целочисленные токены |
| Специальные токены | Нет | PAD, SOS, EOS |
| Выход | 1 нейрон (sigmoid) | VOCAB_SIZE нейронов (softmax) |
| Функция потерь | binary_crossentropy | sparse_categorical_crossentropy |
| Метрики | token_accuracy, sequence_accuracy | token_accuracy, exact_match |



---

# Сриншоты выполнения

<img width="946" height="542" alt="image" src="https://github.com/user-attachments/assets/6a14a118-86b2-4e48-8ff6-5a0d336c67d2" />

<img width="616" height="496" alt="image" src="https://github.com/user-attachments/assets/33fae19e-6c53-4ea0-b26e-bc9f361c09de" />

<img width="478" height="190" alt="image" src="https://github.com/user-attachments/assets/b1262e05-f314-4bd8-9fc2-068515d5acc6" />

<img width="1331" height="262" alt="image" src="https://github.com/user-attachments/assets/e1c52d4d-caa3-4550-99c2-70b8b81df1b7" />

<img width="1038" height="842" alt="image" src="https://github.com/user-attachments/assets/6ce1767a-84b0-409b-9d26-e96d5b66d437" />

<img width="1316" height="840" alt="image" src="https://github.com/user-attachments/assets/326b2812-fa92-4521-83f7-77fefaad3e86" />

<img width="1080" height="259" alt="image" src="https://github.com/user-attachments/assets/82bed816-b1d1-4dab-9c3a-eb715b1b0f56" />

<img width="1220" height="786" alt="image" src="https://github.com/user-attachments/assets/33b5fb45-d700-4484-b2bd-770e8870197e" />

<img width="1350" height="586" alt="image" src="https://github.com/user-attachments/assets/58747c88-0ea3-409b-a8d3-10e630d85a7a" />

<img width="1628" height="569" alt="image" src="https://github.com/user-attachments/assets/49de64ab-021f-4e51-9122-cb0400049850" />
