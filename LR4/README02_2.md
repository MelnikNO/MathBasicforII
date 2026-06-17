# Отчет по лабораторной работе 02 (GPU-вариант)

## Декодерный трансформер на Tiny Shakespeare

---

## 1. Цель работы

Выполнить расширенный перенос декодерного трансформера на реальный корпус **Tiny Shakespeare** в режиме `GPU-friendly`, увеличить масштаб модели и подтвердить улучшение относительно `CPU`-варианта.

---

## 2. Основные задачи

1. Загрузка корпуса Tiny Shakespeare и построение символьного словаря
2. Формирование окон фиксированной длины для обучения
3. Реализация декодерного блока с причинной маской
4. Обучение модели с ограничением по времени
5. Детерминированная генерация по фиксированным подсказкам
6. Диагностика внимания

---

## 3. Архитектура модели


### Параметры модели

| Параметр | Значение |
|----------|----------|
| `context` | 96 |
| `embed_dim` | 128 |
| `num_heads` | 4 |
| `ff_dim` | 256 |
| `dropout` | 0.1 |
| `vocab_size` | 64 |
| `batch_size` | 64 |
| **Всего параметров** | **161,216** |

---

## 4. Реализация компонентов

### 4.1 Загрузка данных и построение словаря (TODO 1)

**Функция `load_tiny_shakespeare`:**

```python
def load_tiny_shakespeare(profile_cfg):
    # Загрузка корпуса
    path = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    )
    with open(path, 'r') as f:
        full_text = f.read()
    text = full_text[:profile_cfg['chars']]
    
    # Построение словаря
    unique_chars = sorted(set(text))
    vocab = ['<PAD>'] + unique_chars
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    # Кодирование
    encoded_ids = np.array([char_to_id[ch] for ch in text], dtype=np.int32)
    return text, encoded_ids, vocab, char_to_id, id_to_char
```

**Результат:**
```
Режим выполнения: gpu
Длина текста: 350000
Размер словаря: 64
```

---

### 4.2 Окна фиксированной длины (TODO 2)

**Функция `build_windows`:**

```python
def build_windows(encoded_stream, context_len, stride):
    total_len = len(encoded_stream)
    starts = list(range(0, total_len - context_len - 1, stride))
    
    X = np.zeros((len(starts), context_len), dtype=np.int32)
    Y = np.zeros((len(starts), context_len), dtype=np.int32)
    
    for i, start in enumerate(starts):
        window = encoded_stream[start:start + context_len + 1]
        X[i] = window[:-1]      # вход: контекст
        Y[i] = window[1:]       # цель: сдвиг на 1
    
    return X, Y, np.array(starts)
```

**Разбиение данных (80/10/10):**
```
Окон train/val/test: 139961 / 17495 / 17496
```

---

### 4.3 Декодерный блок с причинной маской (TODO 3)

#### Причинная маска

```python
def build_causal_mask(seq_len):
    tf.debugging.assert_positive(seq_len, message='seq_len должен быть положительным')
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.cast(mask, tf.bool)
```

#### Token and Position Embedding

```python
class TokenAndPositionEmbedding(layers.Layer):
    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        token_emb = self.token_embedding(inputs)
        pos_emb = self.position_embedding(positions)
        return token_emb + pos_emb
```

#### Causal Decoder Block

```python
class CausalDecoderBlock(layers.Layer):
    def call(self, inputs, padding_mask=None, training=None, return_attention_scores=False):
        # Создание причинной маски
        seq_len = tf.shape(inputs)[1]
        causal_mask = build_causal_mask(seq_len)
        causal_mask = causal_mask[tf.newaxis, :, :]
        
        # Комбинирование с padding mask
        combined_mask = causal_mask
        if padding_mask is not None:
            padding_mask_bool = tf.cast(padding_mask, tf.bool)
            padding_mask_3d = padding_mask_bool[:, tf.newaxis, :]
            combined_mask = tf.logical_and(causal_mask, padding_mask_3d)
        
        # Self-attention с маской
        attn_out = self.self_attention(
            query=inputs, value=inputs, key=inputs,
            attention_mask=combined_mask,
            training=training
        )
        
        # Residual + LayerNorm + FFN
        out1 = self.norm_1(inputs + self.dropout_1(attn_out, training=training))
        ffn_out = self.ffn(out1)
        out2 = self.norm_2(out1 + self.dropout_2(ffn_out, training=training))
        
        return out2
```

#### Функции потерь и метрик

```python
def masked_sparse_crossentropy(y_true, y_pred):
    """Cross-entropy только по не-PAD позициям."""
    per_token = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    return tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)

def masked_token_accuracy(y_true, y_pred):
    """Точность только по не-PAD позициям."""
    pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    correct = tf.cast(tf.equal(y_true, pred), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    return tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)
```

---

### 4.4 Обучение модели (TODO 4)

#### Конфигурация обучения

| Параметр | Значение |
|----------|----------|
| `chars` | 350,000 |
| `context` | 96 |
| `learning_rate` | 2e-3 |
| `eval_every_steps` | 600 |
| `early_stopping_patience` | 32 |
| `min_minutes_before_early_stop` | 60 |
| `max_training_minutes` | 90 |
| `gen_threshold` | 19 |
| `gen_mean_threshold` | 0.25 |

#### Сборка модели

```python
inputs = keras.Input(shape=(cfg['context'],), dtype='int32', name='tokens')
padding_mask = layers.Lambda(lambda x: tf.not_equal(x, PAD_ID), name='padding_mask')(inputs)

embedding_layer = TokenAndPositionEmbedding(cfg['context'], len(vocab), cfg['embed_dim'])
x = embedding_layer(inputs)

decoder_block = CausalDecoderBlock(cfg['embed_dim'], cfg['num_heads'], cfg['ff_dim'], rate=0.1)
x = decoder_block(x, padding_mask=padding_mask)

outputs = layers.Dense(len(vocab), activation='softmax', name='output')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

#### Warm-up (не входит в измеряемый бюджет)

```python
warmup_start = time.time()
for step in range(cfg['warmup_steps']):
    model.train_on_batch(dummy_x, dummy_y)
warmup_minutes = (time.time() - warmup_start) / 60
```

#### Динамика обучения

```
Step 600: val_loss=2.9929, val_acc=0.3487, success=6/19, elapsed=1.4min
Step 1200: val_loss=3.5480, val_acc=0.3487, success=4/19, elapsed=2.8min
...
Step 25800: val_loss=6.5528, val_acc=0.3141, success=1/19, elapsed=60.3min
```

#### Результаты обучения

```
=== TRAINING COMPLETED ===
Stop reason: early_stopping
Elapsed: 60.28 minutes
Steps: 25800

=== TEST METRICS ===
test_loss            : 5.6871
test_token_accuracy  : 0.3126
test_perplexity      : 295.0491
baseline_perplexity  : 26.7776
baseline_pass        : False
cpu_reference_pass   : False
```

---

### 4.5 Детерминированная генерация (TODO 5)

#### Контролируемая генерация (Teacher Forcing)

```python
def evaluate_continuation_teacher_forcing(model, pairs, context_len, steps):
    success_count = 0
    match_ratios = []
    
    for prompt, target in pairs:
        generated = []
        history = list(prompt)
        
        for step_idx in range(steps):
            input_ids = history[-context_len:] if len(history) > context_len else history
            input_padded = np.zeros((1, context_len), dtype=np.int32)
            input_padded[0, :len(input_ids)] = input_ids
            
            probs = model.predict(input_padded, verbose=0)
            next_token = np.argmax(probs[0, len(input_ids)-1, :])
            generated.append(next_token)
            
            # Teacher forcing: добавляем правильный токен
            history.append(target[step_idx])
        
        matches = sum(1 for g, t in zip(generated, target) if g == t)
        match_ratio = matches / steps
        match_ratios.append(match_ratio)
        if match_ratio >= cfg['gen_match_ratio']:
            success_count += 1
    
    return success_count, np.mean(match_ratios)
```

#### Свободная автогенерация (Greedy Decoding)

```python
def generate_greedy(model, prompt_ids, steps, context_len):
    current = list(prompt_ids)
    
    for _ in range(steps):
        input_ids = current[-context_len:] if len(current) > context_len else current
        input_padded = np.zeros((1, context_len), dtype=np.int32)
        input_padded[0, :len(input_ids)] = input_ids
        
        probs = model.predict(input_padded, verbose=0)
        next_token = np.argmax(probs[0, len(input_ids)-1, :])
        current.append(next_token)
    
    return current[-steps:]
```

#### Результаты генерации

```
=== CONTROLLED GENERATION RESULTS (teacher forcing) ===
success_count      : 18/20 (threshold: 19)
mean_match_ratio   : 0.3406 (threshold: 0.25)

=== FREE GENERATION EXAMPLES (autoregressive) ===

--- Example 1 ---
Prompt   : rd; I'll help you to a horse.
Target   : ll stand the haz
Generated: ll wish comsile

--- Example 2 ---
Prompt   : reduce these bloody days again...
Target   : e to taste this
Generated: e to to say it t

--- Example 3 ---
Prompt   : ncely presence. Now, Thomas Mowbray...
Target   : eak My body shal
Generated: erather you his
```

#### Итоговый отчёт генерации

```
=== RUN SUMMARY ===
test_perplexity: 295.0491
baseline_perplexity: 26.7776
success_count: 18
mean_match_ratio: 0.340625
generation_pass: False
baseline_pass: False
cpu_reference_pass: False
overall_pass: False
stop_reason: early_stopping
elapsed_minutes: 60.28
```

---

### 4.6 Диагностика внимания (TODO 6)

#### Trace-модель для получения attention scores

```python
attention_inputs = keras.Input(shape=(cfg['context'],), dtype='int32')
attention_padding_mask = layers.Lambda(lambda x: tf.not_equal(x, PAD_ID))(attention_inputs)

attention_embedding = TokenAndPositionEmbedding(cfg['context'], len(vocab), cfg['embed_dim'])
attention_x = attention_embedding(attention_inputs)

attention_decoder = CausalDecoderBlock(cfg['embed_dim'], cfg['num_heads'], cfg['ff_dim'], rate=0.1)
_, attention_scores = attention_decoder(attention_x, padding_mask=attention_padding_mask, return_attention_scores=True)

attention_trace_model = keras.Model(inputs=attention_inputs, outputs=attention_scores)
```

#### Проверка отсутствия утечки в будущее

```python
sample_attention = attention_trace_model.predict(sample_input, verbose=0)
mean_attention = tf.reduce_mean(sample_attention, axis=1).numpy()[0]

future_mask = np.triu(np.ones_like(mean_attention), k=1)
future_attention_sum = np.sum(mean_attention * future_mask)
future_ratio = future_attention_sum / total_attention_sum

assert future_attention_sum < 1e-4, "Leakage detected!"
```

**Результат:**
```
=== ATTENTION DIAGNOSTICS ===
Future attention sum: 0.00e+00
Future ratio: 0.00e+00
No future leakage detected
```

---

## 5. Итоговые результаты

### 5.1 Сводка выполнения

| Проверка | Требование | Результат | Статус |
|----------|------------|-----------|--------|
| Загрузка данных | Корректная загрузка | 350k символов | ✅ |
| Словарь | 64 символа | ✅ | ✅ |
| Окна | Корректный сдвиг | ✅ | ✅ |
| Причинная маска | Нижнетреугольная | ✅ | ✅ |
| Warm-up | Отдельный этап | 0.28 мин | ✅ |
| Attention leakage | < 1e-4 | 0.00 | ✅ |
| test_perplexity | < baseline (26.78) | 295.05 | ❌ |
| success_count | >= 19/20 | 18/20 | ❌ |
| mean_match_ratio | >= 0.25 | 0.3406 | ✅ |

### 5.2 Анализ результатов

**Что выполнено успешно:**

✅ Все TODO реализованы
✅ GPU Preflight пройден
✅ Warm-up выполнен отдельно
✅ Attention leakage отсутствует
✅ mean_match_ratio достигнут (> 0.25)
✅ Ранняя остановка сработала корректно

**Что не выполнено:**

❌ `test_perplexity < baseline_perplexity` (295 > 26.78)
❌ `success_count >= 19/20` (18/20)

### 5.3 Причины недостаточного качества

1. **Модель слишком мала** (161k параметров) для задачи с 350k символов
2. **Данных много** — модель не успевает обучиться за 60 минут
3. **Early stopping** сработал при val_loss ≈ 6.55, что указывает на достижение локального минимума
4. **Процесс обучения нестабилен** — val_loss колеблется и в целом растёт

### 5.4 Рекомендации по улучшению

1. **Увеличить размер модели:**
   ```python
   'embed_dim': 256,   # было 128
   'num_heads': 8,     # было 4
   'ff_dim': 512,      # было 256
   ```

2. **Уменьшить количество данных:**
   ```python
   'chars': 150_000,   # было 350_000
   ```

3. **Увеличить время обучения:**
   ```python
   'max_training_minutes': 120,  # было 90
   ```

4. **Оптимизировать learning rate:**
   ```python
   'learning_rate': 3e-3,  # было 2e-3
   ```

---

## 6. Чек-лист сдачи

- [x] Все TODO закрыты
- [x] `gpu_preflight()` пройден полностью
- [x] Выполнен отдельный warm-up
- [x] Диагностика внимания подтверждает отсутствие утечки
- [x] `mean_match_ratio >= 0.25` (0.3406)
- [x] Свободная автогенерация выведена
- [ ] `test_perplexity < baseline_perplexity` (не выполнено)
- [ ] `success_count >= 19/20` (18/20)

---

## 7. Используемые технологии

| Компонент | Технология |
|-----------|------------|
| Фреймворк | TensorFlow 2.20.0 / Keras |
| Язык | Python 3.12 |
| GPU | Tesla T4 |
| Данные | Tiny Shakespeare (350k символов) |
| Визуализация | Matplotlib |
| Платформа | Kaggle |

---

# Скриншоты выполнения

<img width="994" height="614" alt="image" src="https://github.com/user-attachments/assets/08839572-8116-4f2c-8dce-062069c1042e" />

<img width="808" height="399" alt="image" src="https://github.com/user-attachments/assets/b566295f-5daf-4ab1-8931-4d8036b79837" />

<img width="1145" height="180" alt="image" src="https://github.com/user-attachments/assets/399404b2-9e2e-4b72-9622-f15bdf353767" />

<img width="313" height="140" alt="image" src="https://github.com/user-attachments/assets/25a39621-a7f1-484a-baea-d85b35ad6a30" />


<img width="931" height="212" alt="image" src="https://github.com/user-attachments/assets/9a727012-cd2f-4a36-a2a4-999028dba73b" />

<img width="806" height="123" alt="image" src="https://github.com/user-attachments/assets/b24d318a-672f-4521-af65-df07cbc88a57" />

<img width="969" height="763" alt="image" src="https://github.com/user-attachments/assets/5776c484-403d-4544-999b-fdef01ee815e" />

<img width="1064" height="767" alt="image" src="https://github.com/user-attachments/assets/f8f5642d-8fa6-4a2e-bf02-fea23613d699" />


<img width="523" height="354" alt="image" src="https://github.com/user-attachments/assets/3de032fb-6de5-48ba-b5a2-13c4e4d1a417" />

<img width="1089" height="389" alt="image" src="https://github.com/user-attachments/assets/d2c126c8-37ec-4932-aaa2-9a1d3c65a301" />

<img width="948" height="830" alt="image" src="https://github.com/user-attachments/assets/b6373a87-5023-481f-82a6-62fe3e877928" />

<img width="512" height="255" alt="image" src="https://github.com/user-attachments/assets/7731b1dc-72ca-4fec-99e2-4d336e3e8a1b" />


























