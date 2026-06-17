# Отчет по лабораторному практикуму 05-Full-Transformer

## Полный трансформер encoder-decoder на Tiny Shakespeare

---

## 1. Цель работы

Реализовать полный контур **encoder-decoder** трансформера для задачи предсказания следующего токена на корпусе **Tiny Shakespeare**. Работа включает:

- Реализацию контракта данных с символьной токенизацией
- Построение архитектуры полного трансформера с encoder и decoder
- Обучение модели с teacher forcing
- Детерминированную генерацию продолжений
- Диагностику механизма внимания

---

## 2. Архитектура модели


### Параметры модели

| Параметр | Значение |
|----------|----------|
| `d_model` | 192 |
| `num_heads` | 6 |
| `d_ff` | 512 |
| `num_layers` | 3 |
| `dropout` | 0.1 |
| `src_len` | 80 |
| `tgt_len` | 80 |
| `vocab_size` | 64 |
| **Всего параметров** | **2,560,576** |

---

## 3. Реализация компонентов

### 3.1 Контракт данных (TODO 1)

**Задача:** Преобразовать непрерывный текст в обучающие пары «контекст → продолжение».

**Реализованные функции:**

```python
def load_tiny_shakespeare_text(profile_cfg) -> str:
    """Загрузка корпуса Tiny Shakespeare."""
    
def build_char_vocabulary(text) -> tuple[list[str], dict, dict]:
    """Построение символьного словаря с токенами <PAD> и <SOS>."""
    
def encode_text(text, char_to_id) -> np.ndarray:
    """Кодирование текста в индексы."""
    
def build_seq2seq_windows(token_ids, src_len, tgt_len, stride, sos_id):
    """Формирование окон для encoder/decoder."""
```

**Формат данных:**

```
encoder_input = ids[i : i + SRC_LEN]
target = ids[i + SRC_LEN : i + SRC_LEN + TGT_LEN]
decoder_input = [SOS] + target[:-1]
decoder_target = target
```

**Проверка:** Формы тензоров корректны, сдвиг через SOS работает

---

### 3.2 Причинная маска (TODO 2)

**Задача:** Реализовать нижнетреугольную маску, запрещающую доступ к будущим позициям.

```python
def build_causal_mask(seq_len: int) -> np.ndarray:
    """Строит нижнетреугольную причинную маску."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))
```

**Проверка:**
```
Маска формы: (5, 5)
Причинная маска работает корректно
```
Диагональ разрешена, верхний треугольник запрещен

---

### 3.3 Модель encoder-decoder (TODO 3)

#### Positional Encoding

```python
def sinusoidal_position_encoding(length: int, depth: int) -> tf.Tensor:
    """Синусоидальное позиционное кодирование."""
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### Encoder Block

```python
class TransformerEncoderBlock(keras.layers.Layer):
    """Блок кодировщика: Self-Attention + FFN + Residual + LayerNorm."""
    def call(self, x, attention_mask, training):
        attn_out = self.self_attention(x, x, attention_mask=attention_mask)
        x = self.norm_1(x + self.dropout_1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm_2(x + self.dropout_2(ffn_out))
        return x
```

#### Decoder Block

```python
class TransformerDecoderBlock(keras.layers.Layer):
    """Блок декодера: Masked Self-Attention + Cross-Attention + FFN."""
    def call(self, x, encoder_output, self_attention_mask, cross_attention_mask, training):
        # Masked Self-Attention (causal)
        self_out = self.self_attention(x, x, attention_mask=self_attention_mask, use_causal_mask=True)
        x = self.norm_1(x + self.dropout_1(self_out))
        
        # Cross-Attention (query from decoder, key/value from encoder)
        cross_out = self.cross_attention(x, encoder_output, attention_mask=cross_attention_mask)
        x = self.norm_2(x + self.dropout_2(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm_3(x + self.dropout_3(ffn_out))
        return x
```

#### Полная модель

```python
class FullTransformerModel(keras.Model):
    """Полный encoder-decoder трансформер."""
    def call(self, inputs, training=False, return_attention=False):
        encoder_tokens, decoder_tokens = inputs
        enc_mask, dec_mask, cross_mask = self._masks(encoder_tokens, decoder_tokens)
        
        # Encoder
        x = self.encoder_embedding(encoder_tokens)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=enc_mask, training=training)
        encoder_output = x
        
        # Decoder
        x = self.decoder_embedding(decoder_tokens)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, dec_mask, cross_mask, training=training)
        
        # Output projection
        logits = self.output_layer(x)
        return logits
```


### 3.4 Обучение и перплексия (TODO 4)

#### Функции потерь и метрик

```python
def masked_loss(y_true, y_pred):
    """Sparse Categorical Crossentropy только по не-PAD позициям."""
    losses = loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), losses.dtype)
    return tf.reduce_sum(losses * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

def masked_accuracy(y_true, y_pred):
    """Точность только по не-PAD позициям."""
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    correct = tf.cast(tf.equal(y_true, pred), tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)
```

#### Baseline Perplexity

```python
def calculate_unigram_baseline_perplexity(train_targets, eval_targets, vocab_size):
    """Частотная baseline-перплексия (униграммная модель)."""
    counts = np.bincount(train_flat, minlength=vocab_size)
    probs = counts / counts.sum()
    nll = -np.mean(np.log(probs[eval_flat]))
    return np.exp(nll)
```


### 3.5 Детерминированная генерация (TODO 5)

#### Авторегрессионная генерация

```python
def evaluate_single_probe(model, prompt, expected, char_to_id, id_to_char, cfg, sos_id, pad_id):
    """Авторегрессионная генерация с top-k метрикой."""
    for step in range(gen_len):
        # Подготовка входа декодера с уже сгенерированным префиксом
        decoder_tokens[0, 0] = sos_id
        decoder_tokens[0, 1:1+upto] = generated_ids[:upto]
        
        # Предсказание следующего токена
        logits = model((encoder_tokens, decoder_tokens), training=False)
        next_id = np.argmax(logits[0, step])
        generated_ids.append(next_id)
        
        # Проверка попадания в top-k
        topk_ids = np.argpartition(logits[0, step], -k)[-k:]
        topk_hits.append(int(expected_id in topk_ids))
```


### 3.6 Диагностика внимания (TODO 6)

**Задача:** Проверить, что причинная маска действительно запрещает доступ к будущим позициям.

```python
# Получение attention scores первого decoder блока
logits, attention_scores = model((sample_encoder, sample_decoder), return_attention=True)
mean_attention = tf.reduce_mean(attention_scores[0], axis=1).numpy()[0]

# Проверка верхнего треугольника (будущие позиции)
future_mask = np.triu(np.ones_like(mean_attention), k=1)
future_attention_sum = np.sum(mean_attention * future_mask)

assert future_attention_sum < 1e-4, "Обнаружена утечка в будущие позиции!"
```

## 4. Итоговый отчет

### 4.1 Сводка результатов

| Проверка | Требование | Результат | Статус |
|----------|------------|-----------|--------|
| Контракт данных | Корректные формы и сдвиг | ✓ | ✅ |
| Причинная маска | Нижнетреугольная | ✓ | ✅ |
| Модель | Forward pass успешен | ✓ | ✅ |
| Test Perplexity | < Baseline (27.68) | **5.56** | ✅ |
| Generation success | >= 18/20 | **20/20** | ✅ |
| Mean match ratio | >= 0.70 | **0.8080** | ✅ |
| Attention leakage | < 1e-4 | **0.00** | ✅ |

### 4.2 Финальный вердикт

```json
{
  "profile": "GPU-friendly",
  "test_perplexity": 5.5567,
  "baseline_perplexity": 27.6842,
  "baseline_pass": true,
  "generation_success_count": 20,
  "generation_mean_match_ratio": 0.8080,
  "generation_pass": true,
  "attention_future_sum": 0.0,
  "attention_pass": true,
  "overall_pass": true
}
```


---

## 5. Выводы

### 5.1 Достигнутые результаты

1. **Архитектура**: Реализован полный encoder-decoder трансформер с:
   - Синусоидальным позиционным кодированием
   - Многоголовым self-attention и cross-attention
   - Причинной маской в декодере
   - Residual связями и Layer Normalization

2. **Обучение**: 
   - Модель успешно обучилась за 18 эпох
   - Достигнута test perplexity **5.56** (значительно ниже baseline 27.68)
   - Test accuracy: **50.65%**

3. **Генерация**:
   - Достигнут порог **20/20** успешных продолжений
   - Среднее соответствие (top-k): **0.8080**

4. **Диагностика**:
   - Подтверждено отсутствие утечки в будущие позиции
   - Сумма весов в будущем: **0.00**

### 5.2 Ключевые особенности реализации

- **Teacher forcing**: Использован для эффективного обучения
- **Causal masking**: Обеспечивает авторегрессионность
- **Top-k метрика**: Более гибкая оценка качества генерации
- **Masked loss**: Игнорирование PAD-токенов

### 5.3 Практическая значимость

Разработанная модель может быть использована для:
- Генерации текста в стиле Шекспира
- Изучения механизмов внимания в трансформерах
- Базового бенчмарка для NLP задач на символьном уровне

---

## 6. Сравнение с предыдущими лабораторными

| Параметр | Decoder-only (ЛР02) | Full Transformer (ЛР05) |
|----------|---------------------|-------------------------|
| Архитектура | Только декодер | Encoder + Decoder |
| Вход | Один поток токенов | Encoder: контекст, Decoder: целевой |
| Attention | Self-attention + causal | Self-attention + Cross-attention + causal |
| Основная задача | Генерация следующего токена | Предсказание на основе контекста |
| Маскировка | Causal mask | Encoder mask + Causal mask + Cross mask |
| Данные | Синтетические (7 токенов) | Tiny Shakespeare (64 символа) |

---

## 7. Чек-лист сдачи

- [x] Реализованы все TODO 1..6
- [x] `test_perplexity < baseline_perplexity` (5.56 < 27.68)
- [x] `success_count >= 18/20` (20/20)
- [x] `mean_match_ratio >= 0.70` (0.8080)
- [x] Диагностика подтверждает отсутствие утечки в будущее
- [x] Все мини-проверки пройдены
- [x] Итоговый отчёт сформирован

---

## 8. Используемые технологии

| Компонент | Технология |
|-----------|------------|
| Фреймворк | TensorFlow 2.20.0 / Keras |
| Язык | Python 3.12 |
| GPU | Tesla T4 |
| Данные | Tiny Shakespeare |
| Визуализация | Matplotlib |
| Формат | Jupyter Notebook |

---

---

# Скриншоты

<img width="821" height="837" alt="image" src="https://github.com/user-attachments/assets/ef8fb8d8-1521-4f94-a4fb-a6767b351385" />

<img width="1136" height="649" alt="image" src="https://github.com/user-attachments/assets/ffb34c56-593d-43af-84fa-a7b41a9236b4" />

<img width="651" height="225" alt="image" src="https://github.com/user-attachments/assets/84a8246d-3764-416f-bcd5-bed9ce709c2b" />

<img width="670" height="195" alt="image" src="https://github.com/user-attachments/assets/ac368dd7-4946-4867-b0c2-688472181616" />

<img width="881" height="809" alt="image" src="https://github.com/user-attachments/assets/b2549e28-3455-4551-a1af-6fb5d37f98fa" />

<img width="1121" height="723" alt="image" src="https://github.com/user-attachments/assets/ccf5010a-df7a-4007-846e-043a51babb4f" />

<img width="635" height="177" alt="image" src="https://github.com/user-attachments/assets/15d0232b-b00f-4053-ab33-8f08a7967251" />

<img width="775" height="823" alt="image" src="https://github.com/user-attachments/assets/f40c194c-0465-4e91-8a87-3530f3e12307" />

<img width="764" height="825" alt="image" src="https://github.com/user-attachments/assets/5b097f0f-483f-4be9-b5df-7415c4823806" />
















