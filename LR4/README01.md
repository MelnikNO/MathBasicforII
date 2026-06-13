# Лабораторная работа 01: Decoder-only Transformer для предсказания следующего токена

## Название работы
**Decoder-only Transformer с причинной маской для авторегрессионного предсказания следующего токена**

## Задание
Реализовать декодерный трансформер (decoder-only Transformer) с причинной маской (causal mask) для предсказания следующего токена в детерминированной последовательности.

### Цель работы
Научиться:
- реализовывать причинную маску (causal mask) для запрета доступа к будущим токенам;
- строить декодерный блок с self-attention и позиционным кодированием;
- обучать модель на детерминированных данных с правилом второго порядка;
- генерировать новые токены в авторегрессионном режиме (greedy decoding);
- диагностировать корректность причинной маски через attention scores.

## Формат данных
- Вход `X`: последовательность токенов без последнего шага, форма `(N, T)`
- Цель `Y`: последовательность со сдвигом на 1 шаг влево, форма `(N, T)`
- Токены:
  - `PAD = 0` — заполнение
  - `BOS = 1` — начало последовательности
  - `EOS = 2` — конец последовательности
  - `A = 3, B = 4, C = 5, D = 6` — полезные токены
- Длина последовательности: 17 токенов (включая BOS и EOS)
- Длина полезной нагрузки: 9 токенов
- Правило генерации: следующий токен зависит от ДВУХ предыдущих (правило второго порядка)

## Что было сделано

### 1. Создание датасета
Реализована функция `build_synthetic_corpus()` с детерминированным правилом второго порядка:

```python
def next_payload_token(prev_prev, prev):
    # Правило: зависит от двух предыдущих токенов
    map_if_ac = {A: B, B: C, C: A, D: B}
    map_if_bd = {A: C, B: A, C: D, D: A}
    return map_if_ac[prev] if prev_prev in (A, C) else map_if_bd[prev]
```

## 2. Разделение данных
Данные разделены на обучающую, валидационную и тестовую выборки (без перемешивания).

| Выборка | Форма X | Количество примеров |
|---------|---------|---------------------|
| Train   | (4096, 17) | 4096 |
| Val     | (512, 17)  | 512  |
| Test    | (512, 17)  | 512  |

## 3. Построение модели
Создана декодерная трансформер-модель с причинной маской:

```python
class TokenAndPositionEmbedding(layers.Layer):
    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        token_emb = self.token_embedding(inputs)
        pos_emb = self.position_embedding(positions)
        return token_emb + pos_emb

class CausalDecoderBlock(layers.Layer):
    def call(self, inputs, padding_mask=None, return_attention_scores=False):
        # Причинная маска (нижнетреугольная)
        causal_mask = build_causal_mask(seq_len)
        # Комбинируем с padding_mask
        combined_mask = causal_mask & padding_mask_3d
        
        # Self-attention с маской
        attn_out = self.self_attention(query=inputs, value=inputs, key=inputs, attention_mask=combined_mask)
        
        # Residual + LayerNorm + FFN + Residual + LayerNorm
        out1 = self.norm_1(inputs + self.dropout_1(attn_out))
        ffn_out = self.ffn(out1)
        out2 = self.norm_2(out1 + self.dropout_2(ffn_out))
        return out2
```

## 4. Компиляция
- Оптимизатор: `adam`
- Функция потерь: `masked_sparse_crossentropy` (учитывает только не-PAD позиции)
- Метрика: `masked_token_accuracy`

```python
def masked_sparse_crossentropy(y_true, y_pred):
    per_token = sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    return tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)
```

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 25
- Размер батча: 64
- Валидационная выборка: (512, 17)

## 6. Оценка модели
На тестовой выборке получены метрики:

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Loss | 0.152 | Функция потерь |
| Token Accuracy | 0.984 | Доля правильных предсказаний |
| Perplexity | 1.164 | exp(loss) — неопределённость модели |

**Целевые пороги достигнуты:**
- token_accuracy >= 0.97
- perplexity <= 1.30

## 7. Детерминированная генерация
Реализована жадная генерация (greedy decoding) для авторегрессионного предсказания:

```python
def generate_greedy(model, prompt_ids, max_new_tokens):
    current = list(prompt_ids)
    for _ in range(max_new_tokens):
        input_padded = pad_sequence(current)
        probs = model.predict(input_padded, verbose=0)
        next_token = np.argmax(probs[0, len(current)-1, :])
        current.append(next_token)
        if next_token == EOS_ID: break
    return current
```


## 8. Диагностика внимания
Построена heatmap self-attention и проверена корректность причинной маски:

- Сумма весов внимания в будущем: < 1e-4 (утечки нет)
- Диагональ содержит ненулевые значения (токен видит себя)
- Матрица имеет нижнетреугольную структуру

## Вывод
Модель decoder-only Transformer успешно справилась с задачей авторегрессионного предсказания следующего токена:
- **Test Token Accuracy = 98.4%**
- **Test Perplexity = 1.164** (очень низкая неопределённость)
- **Генерация:** 18/20 правильных продолжений

Ключевой вывод: причинная маска эффективно запрещает доступ к будущим токенам, что подтверждается диагностикой внимания (сумма весов в будущем < 1e-4). Архитектура decoder-only Transformer является основой для современных LLM (GPT и подобных).

## Сравнение с предыдущими лабораторными

| Параметр | Encoder-only Transformer | Decoder-only Transformer |
|----------|-------------------------|--------------------------|
| Тип внимания | Self-attention (без маски) | Self-attention + causal mask |
| Основная задача | Классификация | Генерация |
| Режим работы | Однопроходный | Авторегрессионный |
| Доступ к будущему | Есть (полный контекст) | Нет (только прошлое) |
| Маскировка | Padding mask | Padding mask + causal mask |

























