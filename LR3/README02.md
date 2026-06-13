# Лабораторная работа 02: Transformer Encoder на IMDB

## Название работы
**Transformer Encoder для бинарной классификации тональности отзывов IMDB**

## Задание
Реализовать Transformer encoder модель для классификации тональности отзывов IMDB (positive/negative) на реальных текстовых данных.

### Цель работы
Научиться:
- применять Transformer encoder к реальным текстовым данным;
- переиспользовать готовые компоненты (TokenAndPositionEmbedding, TransformerEncoderBlock) из предыдущей лабораторной;
- работать с реальным датасетом IMDB (подготовка данных, паддинг, декодирование);
- визуализировать attention на реальных отзывах.

## Формат данных
- Вход: последовательности целочисленных токенов (отзывы IMDB)
- Словарь: 10,000 наиболее частотных слов
- Длина последовательности: 200 токенов (после паддинга)
- `PAD = 0` — заполнение до фиксированной длины
- Метка:
  - `1` — положительный отзыв
  - `0` — отрицательный отзыв

## Что было сделано

### 1. Загрузка и подготовка данных
Загружен датасет IMDB через `keras.datasets.imdb.load_data()`:

```python
(x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.imdb.load_data(num_words=10000)

# Взяты подмножества для CPU-friendly обучения
x_train = x_train_full[:5000]
y_train = y_train_full[:5000]
x_test = x_test_full[:2000]
y_test = y_test_full[:2000]

# Паддинг до MAXLEN=200
x_train = pad_sequences(x_train, maxlen=200, padding='post', value=0)
x_test = pad_sequences(x_test, maxlen=200, padding='post', value=0)
```

## 2. Разделение данных

| Выборка | Форма X | Количество примеров |
|---------|---------|---------------------|
| Train   | (5000, 200) | 5000 |
| Test    | (2000, 200) | 2000 |

## 3. Построение модели
Переиспользованы компоненты из `03-Transformer / ЛР01`:

```python
class TokenAndPositionEmbedding(layers.Layer):
    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

class TransformerEncoderBlock(layers.Layer):
    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        # Self-attention с маской
        attention_output, attention_scores = self.att(
            query=inputs, value=inputs, key=inputs,
            attention_mask=attention_mask,
            return_attention_scores=True
        )
        # Residual + LayerNorm + FFN + Residual + LayerNorm
        out1 = self.layernorm1(inputs + self.dropout1(attention_output))
        proj_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(proj_output))
        return out2 if not return_attention_scores else (out2, attention_scores)
```

## 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `binary_crossentropy`
- Метрика: `accuracy`

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 3 (для быстрого прототипирования)
- Размер батча: 64
- Валидационная выборка: 20% от обучающей

## 6. Оценка модели
На тестовой выборке получены метрики:

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Loss | 0.582 | Функция потерь |
| Accuracy | 0.762 | Доля правильных предсказаний |

## 7. Декодирование отзыва
Для проверки работы модели выполнен обратный перевод токенов в слова:

```python
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = '[PAD]'
reverse_word_index[1] = '[START]'
reverse_word_index[2] = '[OOV]'

def decode_review(token_ids):
    return [reverse_word_index.get(int(token), f'#{int(token)}') for token in token_ids]
```

## 8. Визуализация attention
Построена heatmap self-attention для одного тестового отзыва, обрезанная до первых 30 содержательных токенов (без PAD-хвоста).

## Вывод
Модель Transformer encoder успешно справилась с задачей классификации тональности отзывов IMDB:
- **Test Accuracy = 76.2%** (на подмножестве 5000/2000 примеров)
- Модель демонстрирует осмысленное поведение на реальных текстовых данных
- Attention визуализация позволяет интерпретировать, на какие токены модель обращает внимание

Ключевой вывод: архитектура Transformer encoder, разработанная на синтетических данных, успешно переносится на реальную задачу NLP без изменений, что подтверждает универсальность подхода.

## Сравнение с предыдущими лабораторными

| Параметр | Transformer на toy-задаче | Transformer на IMDB |
|----------|--------------------------|---------------------|
| Тип данных | Синтетические токены | Реальные тексты |
| Длина последовательности | 12 | 200 |
| Размер словаря | 16 | 10,000 |
| Параметры модели | ~14k | ~333k |
| Целевая метрика | >95% | >75% |


---

# Скриншот

<img width="1045" height="540" alt="image" src="https://github.com/user-attachments/assets/b38d1453-92e5-46c5-8fa5-810304cbe1ba" />

<img width="763" height="580" alt="image" src="https://github.com/user-attachments/assets/454ffb3b-0ad4-4ef1-a5f4-17382e7837dd" />

<img width="1472" height="853" alt="image" src="https://github.com/user-attachments/assets/731616a4-d375-4368-94ea-2154b2c1569c" />

<img width="1215" height="547" alt="image" src="https://github.com/user-attachments/assets/3027d399-b290-49b7-8489-ef29b6daa40f" />

<img width="787" height="396" alt="image" src="https://github.com/user-attachments/assets/42b760ac-59c0-4781-b85c-735668f3aa0b" />

<img width="957" height="822" alt="image" src="https://github.com/user-attachments/assets/c2f43f1e-5a66-4ef4-803b-8b7562963fa2" />

<img width="1458" height="271" alt="image" src="https://github.com/user-attachments/assets/a545e757-f83f-446c-bc68-99f034b50cd9" />

<img width="1300" height="560" alt="image" src="https://github.com/user-attachments/assets/df615224-13ac-4402-bf2a-4bada6e400a6" />

<img width="975" height="254" alt="image" src="https://github.com/user-attachments/assets/5def944d-d014-470e-8585-82c9fd8fde34" />

<img width="297" height="431" alt="image" src="https://github.com/user-attachments/assets/b9752be2-7316-4b37-855b-7242b5516096" />

<img width="1040" height="773" alt="image" src="https://github.com/user-attachments/assets/7d03d482-a1d8-44ba-ba0e-770fbe1c1a8d" />














