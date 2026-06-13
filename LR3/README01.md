# Лабораторная работа 01: Transformer Encoder для order-sensitive задачи

## Название работы
**Transformer Encoder для бинарной классификации последовательностей (order‑sensitive toy task)**

## Задание
Реализовать Transformer encoder, который определяет порядок следования двух специальных токенов (7 и 3) в последовательности.

### Цель работы
Научиться:
- понимать разницу между self-attention и cross-attention;
- реализовывать позиционное кодирование (positional embedding);
- работать с padding mask в Transformer encoder;
- визуализировать attention scores для интерпретации работы модели;
- решать задачу, чувствительную к порядку токенов (без позиционного кодирования задача не решается).

## Формат данных
- Вход: последовательность из 12 целочисленных токенов (1-15)
- Специальные токены:
  - `7` и `3` — ключевые токены, порядок которых определяет метку
  - `PAD = 0` — заполнение до фиксированной длины
- Реальная длина последовательности: случайная в диапазоне 4..12
- Метка:
  - `1` — если токен `7` встречается раньше токена `3`
  - `0` — если токен `3` встречается раньше токена `7`

## Что было сделано

### 1. Создание датасета
Реализована функция `generate_order_dataset()`, которая:
- генерирует последовательности со случайной длиной;
- гарантирует присутствие токенов 7 и 3 в каждой последовательности;
- формирует бинарные метки на основе порядка этих токенов.

```python
def generate_order_dataset(num_samples, seq_len=SEQ_LEN, min_len=MIN_LEN):
    for i in range(num_samples):
        length = rng.integers(min_len, seq_len + 1)
        remaining = length - 2
        filler = rng.choice(filler_tokens, size=remaining, replace=True)
        
        if rng.random() < 0.5:
            seq_list = [KEY_TOKEN, VALUE_TOKEN] + filler.tolist()
            y[i] = 1  # 7 раньше 3
        else:
            seq_list = [VALUE_TOKEN, KEY_TOKEN] + filler.tolist()
            y[i] = 0  # 3 раньше 7
        
        X[i, :length] = seq_list[:length]


## 2. Разделение данных
Данные разделены на обучающую и тестовую выборки в соотношении 80/20 с сохранением баланса классов.

| Выборка | Форма X | Количество примеров |
|---------|---------|---------------------|
| Train   | (4000, 12) | 4000 |
| Test    | (1000, 12) | 1000 |

## 3. Построение модели
Создана Transformer encoder модель со следующими компонентами:

```python
class TokenAndPositionEmbedding(layers.Layer):
    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        token_embeddings = self.token_emb(inputs)
        position_embeddings = self.pos_emb(positions)
        return token_embeddings + position_embeddings

class TransformerEncoderBlock(layers.Layer):
    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        # Self-attention с маской
        attn_output, attn_scores = self.att(
            query=inputs, value=inputs, key=inputs,
            attention_mask=attention_mask,
            return_attention_scores=True
        )
        # Residual + LayerNorm + FFN + Residual + LayerNorm
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2 if not return_attention_scores else (out2, attn_scores)

## 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `binary_crossentropy`
- Метрика: `accuracy`

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 6
- Размер батча: 64
- Валидационная выборка: 20% от обучающей

## 6. Оценка модели
На тестовой выборке получены метрики:

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Loss | 0.162 | Функция потерь |
| Accuracy | 0.952 | Доля правильных предсказаний |

## 7. Проверка на ручных примерах

| Последовательность | Вероятность | Предсказание | Ожидание |
|-------------------|-------------|--------------|----------|
| [7, 5, 2, 3] | 0.983 | 1 (7 раньше 3) | ✓ |
| [3, 5, 2, 7] | 0.021 | 0 (3 раньше 7) | ✓ |

## 8. Визуализация attention
Построена heatmap self-attention для одного тестового примера, обрезанная до non-PAD части последовательности.

## Вывод
Модель Transformer encoder успешно справилась с задачей:
- **Test Accuracy = 95.2%**
- Модель правильно различает порядок токенов 7 и 3
- Attention heatmap показывает осмысленное взаимодействие между ключевыми токенами

Ключевой вывод: без позиционного кодирования задача не решается, так как self-attention сам по себе не чувствителен к порядку. Это демонстрирует важность positional embedding в Transformer архитектуре.

## Сравнение с предыдущими лабораторными

| Параметр | RNN + Attention | Transformer Encoder |
|----------|-----------------|---------------------|
| Тип внимания | Cross-attention | Self-attention |
| Рекуррентность | Есть (GRU) | Нет |
| Позиционная информация | Передаётся через состояние | Positional embedding |
| Маскировка | Только decoder | Padding mask |
| Параметры | ~40k | ~14k |



---


# Скриншооты выполнения

<img width="1122" height="528" alt="image" src="https://github.com/user-attachments/assets/42cdb506-ead9-4d80-809c-b22f762359b9" />

<img width="701" height="737" alt="image" src="https://github.com/user-attachments/assets/5ecdb68a-c744-46d2-9d5a-16f336c583f8" />

<img width="1213" height="647" alt="image" src="https://github.com/user-attachments/assets/d7d145d9-6fe6-42ba-8419-49967674232e" />

<img width="1134" height="729" alt="image" src="https://github.com/user-attachments/assets/1406911f-7668-426f-aaf1-53323c4fdb18" />

<img width="1533" height="221" alt="image" src="https://github.com/user-attachments/assets/01a52a24-4476-4075-91dc-26a138754231" />

<img width="1030" height="851" alt="image" src="https://github.com/user-attachments/assets/4679f27f-55f8-483e-9bff-e50e233ebad1" />

<img width="1339" height="382" alt="image" src="https://github.com/user-attachments/assets/1342e886-30ba-4f69-9d35-8acc2308e767" />

<img width="1332" height="577" alt="image" src="https://github.com/user-attachments/assets/0dd897c5-a3fb-4ebe-8ecc-fd77c7e6d4f8" />

<img width="895" height="659" alt="image" src="https://github.com/user-attachments/assets/f17ccbc5-9e02-47fd-b652-a6580e87d365" />

<img width="1371" height="826" alt="image" src="https://github.com/user-attachments/assets/3db07dce-7d2f-46cc-ae80-b05f00f5fcad" />

<img width="901" height="730" alt="image" src="https://github.com/user-attachments/assets/3bcd426e-5636-478e-b936-08749377a8a5" />









