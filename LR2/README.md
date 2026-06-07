# Лабораторная работа 2: GRU seq2seq с Luong attention для reverse-sequence

## Задание
Реализовать `GRU` encoder-decoder с `Luong attention`, который разворачивает входную последовательность в обратном порядке.

### Цель работы
Научиться:
- понимать bottleneck plain seq2seq и зачем нужен attention;
- реализовывать cross-attention между encoder и decoder;
- работать с тензорами `query`, `key`, `value`, `context`, `attention_scores`;
- интерпретировать attention_scores через тепловую карту (heatmap);
- сравнивать качество attention-модели с plain seq2seq.

## Формат данных
- Вход (encoder_input): последовательность из 10 целочисленных токенов (1-9)
- Выход (decoder_target): та же последовательность в обратном порядке
- Специальные токены:
  - `PAD = 0` — заполнение до фиксированной длины
  - `SOS = 10` — начало последовательности декодера
  - `EOS = 11` — конец последовательности
- Размер словаря: 12 (1-9 + PAD + SOS + EOS)
- Реальная длина последовательности: случайная в диапазоне 4..10

## Что было сделано

### 1. Создание датасета
Реализованы функции для генерации данных со сдвигом decoder_input и decoder_target:

```python
def make_one_sample(min_len: int = 4, max_len: int = ENC_LEN):
    length = np.random.randint(min_len, max_len + 1)
    seq = np.random.randint(1, 10, size=length, dtype=np.int32)
    rev = seq[::-1]  # обратный порядок
    return seq, rev

## 2. Разделение данных
Данные разделены на обучающую и тестовую выборки в соотношении 80/20.

| Выборка | Форма encoder_input | Форма decoder_input | Форма decoder_target |
|---------|---------------------|---------------------|----------------------|
| Train   | (6400, 10) | (6400, 11) | (6400, 11, 1) |
| Test    | (1600, 10) | (1600, 11) | (1600, 11, 1) |

## 3. Построение модели
Создана Encoder-Decoder модель с Luong attention:

```python
def build_model(vocab_size=12, emb_dim=32, latent_dim=64):
    # Энкодер
    encoder_inputs = tf.keras.layers.Input(shape=(ENC_LEN,))
    enc_emb = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, encoder_state = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)(enc_emb)
    
    # Декодер
    decoder_inputs = tf.keras.layers.Input(shape=(DEC_LEN,))
    dec_emb = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)(decoder_inputs)
    decoder_outputs, _ = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=encoder_state)
    
    # Attention
    context, attention_scores = tf.keras.layers.Attention(score_mode='dot', return_attention_scores=True)([decoder_outputs, encoder_outputs])
    
    # Объединение и выход
    merged = tf.keras.layers.Concatenate()([decoder_outputs, context])
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(merged)
    
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

## 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `sparse_categorical_crossentropy`
- Метрика: `accuracy` (token_accuracy)

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 18
- Размер батча: 64
- Валидационная выборка: 20% от обучающей

## 6. Оценка модели
На тестовой выборке получены метрики:

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Loss | 0.038 | Функция потерь |
| Token Accuracy | 0.989 | Доля правильных отдельных токенов |
| Exact Match | 0.932 | Доля полностью правильных последовательностей |

## 7. Визуализация
Построены графики динамики функции потерь и token_accuracy, а также heatmap внимания для одного примера.

## Вывод
Модель GRU Encoder-Decoder с Luong attention успешно справилась с задачей seq2seq:
- **Token Accuracy = 98.9%** — модель правильно предсказывает отдельные токены
- **Exact Match = 93.2%** — более строгая метрика

Ключевой вывод: attention улучшает качество на длинных последовательностях за счёт возможности смотреть на все позиции энкодера, а не только на одно финальное состояние. Heatmap подтверждает осмысленное поведение: фокус внимания движется по антидиагонали.

---

# Скриншоты

<img width="1023" height="546" alt="image" src="https://github.com/user-attachments/assets/4542e8c5-7dea-4df4-83cd-adaa30235930" />

<img width="773" height="280" alt="image" src="https://github.com/user-attachments/assets/571be7ec-e7f3-4c21-b8f1-2cbd2c407287" />


<img width="821" height="504" alt="image" src="https://github.com/user-attachments/assets/e73c922e-d45f-4ce7-ae04-3dbd94caa5de" />


<img width="1123" height="821" alt="image" src="https://github.com/user-attachments/assets/b6aaa225-b977-4ae9-8a7d-c0bed1a558a6" />

<img width="1039" height="850" alt="image" src="https://github.com/user-attachments/assets/50109151-6478-4ba8-822a-57b41c86843e" />

<img width="1287" height="700" alt="image" src="https://github.com/user-attachments/assets/9aeac177-0e9f-413a-a6aa-7fb21865e265" />

<img width="1322" height="838" alt="image" src="https://github.com/user-attachments/assets/7f2f7a58-681f-4e7e-9f5c-4afa48255fdd" />

<img width="1391" height="785" alt="image" src="https://github.com/user-attachments/assets/bde786ed-2e7f-4bc0-8b1b-92035fe7676a" />

<img width="1208" height="360" alt="image" src="https://github.com/user-attachments/assets/11f3cd6d-57e5-4de6-bafb-33b46a10b2c1" />

<img width="1123" height="118" alt="image" src="https://github.com/user-attachments/assets/844890b3-4d16-4293-ab99-d50f5bcef65f" />

<img width="1298" height="382" alt="image" src="https://github.com/user-attachments/assets/fe2f9f25-a2a7-4bc9-bd2e-d3fd3c650b51" />

<img width="920" height="886" alt="image" src="https://github.com/user-attachments/assets/45a3ec67-fc37-4bbc-97db-dacff2ff43a3" />

<img width="1537" height="532" alt="image" src="https://github.com/user-attachments/assets/5b31ee91-9558-40f0-9da6-ab30bd8ce7f5" />











