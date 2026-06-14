# Лабораторная работа: Decoder-only Transformer на Tiny Shakespeare (CPU-вариант)

## Название работы
**Decoder-only Transformer с причинной маской для генерации текста на корпусе Tiny Shakespeare**

## Задание
Перенести декодерный трансформер с причинной маской с синтетических данных на реальный текстовый корпус Tiny Shakespeare для авторегрессионного предсказания следующего символа.

### Цель работы
Научиться:
- обрабатывать реальные текстовые данные (построение словаря символов, кодирование);
- строить обучающие окна фиксированной длины с заданным шагом;
- применять декодерный трансформер к реальной задаче генерации текста;
- сравнивать качество модели с частотным базовым ориентиром (baseline perplexity);
- генерировать текст в авторегрессионном режиме.

## Формат данных
- Корпус: Tiny Shakespeare (первые 250,000 символов)
- Словарь: уникальные символы + `<PAD>` (размер ~65 символов)
- Формат входа `X`: последовательность из 64 символов (контекст)
- Формат цели `Y`: сдвиг на 1 символ вправо (next-token prediction)
- Шаг между окнами: 3
- `PAD = 0` — заполнение до фиксированной длины

## Что было сделано

### 1. Загрузка корпуса и построение словаря
Загружен корпус Tiny Shakespeare через `tf.keras.utils.get_file()` и построено символьное кодирование:

```python
def load_tiny_shakespeare(profile_cfg):
    path = tf.keras.utils.get_file('shakespeare.txt', url)
    with open(path, 'r') as f:
        full_text = f.read()
    text = full_text[:profile_cfg['chars']]
    unique_chars = sorted(set(text))
    vocab = ['<PAD>'] + unique_chars
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for i, ch in enumerate(vocab)}
    encoded_ids = np.array([char_to_id[ch] for ch in text], dtype=np.int32)
    return text, encoded_ids, vocab, char_to_id, id_to_char
```

## 2. Построение окон и разбиение
Созданы обучающие окна фиксированной длины 64 с шагом 3, выполнено детерминированное индексное разбиение (80/10/10):

| Выборка | Форма X | Количество примеров |
|---------|---------|---------------------|
| Train   | (32331, 64) | 32,331 |
| Val     | (4041, 64)  | 4,041 |
| Test    | (4042, 64)  | 4,042 |

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
        causal_mask = build_causal_mask(seq_len)
        if padding_mask is not None:
            padding_mask_bool = tf.cast(padding_mask, tf.bool)
            padding_mask_3d = padding_mask_bool[:, tf.newaxis, :]
            combined_mask = tf.logical_and(causal_mask, padding_mask_3d)
        attn_out = self.self_attention(query=inputs, value=inputs, key=inputs, attention_mask=combined_mask)
        out1 = self.norm_1(inputs + self.dropout_1(attn_out))
        ffn_out = self.ffn(out1)
        out2 = self.norm_2(out1 + self.dropout_2(ffn_out))
        return out2
```

## 4. Компиляция
- Оптимизатор: `Adam` (learning_rate = 2e-3)
- Функция потерь: `masked_sparse_crossentropy` (учитывает только не-PAD позиции)
- Метрика: `masked_token_accuracy`

## 5. Обучение
Модель обучена с параметрами:
- Эпохи: 3
- Размер батча: 64
- Валидационная выборка: 4,041 примеров

## 6. Оценка модели

| Метрика | Значение | Что измеряет |
|---------|----------|--------------|
| Test Loss | 1.452 | Функция потерь |
| Test Token Accuracy | 0.523 | Доля правильных предсказаний |
| Test Perplexity | 4.271 | exp(loss) — неопределённость модели |
| Baseline Perplexity | 8.234 | Частотный ориентир (предсказание по частоте) |

**Целевые пороги достигнуты:**
- `test_perplexity < baseline_perplexity` (4.27 < 8.23)

## 7. Детерминированная генерация
Реализована жадная генерация для 20 фиксированных подсказок из тестовой выборки:

```python
def generate_greedy(model, prompt_ids, steps, context_len):
    current = list(prompt_ids)
    for _ in range(steps):
        input_padded = np.zeros((1, context_len), dtype=np.int32)
        input_padded[0, :len(input_ids)] = input_ids
        probs = model.predict(input_padded, verbose=0)
        next_token = np.argmax(probs[0, len(input_ids)-1, :])
        current.append(next_token)
    return current[-steps:]
```

## 8. Диагностика внимания
Построена heatmap self-attention и проверена корректность причинной маски:

- Сумма весов внимания в будущем: < 1e-4 (утечки нет)
- Матрица имеет нижнетреугольную структуру

## Вывод
Модель decoder-only Transformer успешно справилась с задачей авторегрессионного предсказания следующего символа на реальном корпусе Tiny Shakespeare:
- **Test Perplexity = 4.27** (значительно лучше baseline 8.23)
- **Генерация:** успешно преодолены оба порога (success_count и mean_match_ratio)

Ключевой вывод: архитектура decoder-only Transformer, отработанная на синтетических данных, успешно переносится на реальную задачу генерации текста. Причинная маска корректно запрещает доступ к будущим позициям, что подтверждается диагностикой внимания.

## Сравнение с предыдущими лабораторными

| Параметр | ЛР01 (синтетика) | ЛР02 CPU (Tiny Shakespeare) |
|----------|------------------|------------------------------|
| Тип данных | Детерминированные токены A/B/C/D | Реальные символы |
| Размер словаря | 7 | ~65 |
| Длина контекста | 17 | 64 |
| Параметры модели | ~38k | ~96k |
| Эпохи | 25 | 3 |
| Perplexity | 1.16 | 4.27 |
| Baseline сравнение | Нет | Есть (8.23) |


---

# Скриншот выполнения

<img width="1173" height="552" alt="image" src="https://github.com/user-attachments/assets/5cf61ac5-fbb7-4ecf-b63e-860c5f2f2899" />

<img width="715" height="249" alt="image" src="https://github.com/user-attachments/assets/67b5b8a9-703e-4ee0-a75a-d2f9dd89200c" />

<img width="1010" height="686" alt="image" src="https://github.com/user-attachments/assets/14d712cb-f58a-4cd1-8a68-079421a41d89" />

<img width="1311" height="499" alt="image" src="https://github.com/user-attachments/assets/645f8d9f-48d9-4bb5-bbf6-1bb8d9d1e0b8" />

<img width="1106" height="824" alt="image" src="https://github.com/user-attachments/assets/eda2452a-accd-44c2-b31c-aaa7579b9655" />

<img width="1022" height="499" alt="image" src="https://github.com/user-attachments/assets/f6825921-483d-4803-b649-3cc4566badd5" />

<img width="1544" height="803" alt="image" src="https://github.com/user-attachments/assets/653ce177-c5f7-4cee-9649-564d3b28c94d" />

<img width="1520" height="387" alt="image" src="https://github.com/user-attachments/assets/0ddd55e7-7d82-46f1-96ec-2c28cb65e2fb" />

<img width="1452" height="717" alt="image" src="https://github.com/user-attachments/assets/0196702f-b65d-4096-9109-192181db2397" />

<img width="720" height="220" alt="image" src="https://github.com/user-attachments/assets/1165f5fd-9865-459c-9bcb-aa8c71eba79f" />

<img width="1061" height="600" alt="image" src="https://github.com/user-attachments/assets/e862c09e-2abd-44b7-b8a4-3a7bea4b97eb" />

<img width="1116" height="990" alt="image" src="https://github.com/user-attachments/assets/46c23d50-c91f-4f5d-9aa5-1cac0317d576" />

<img width="853" height="239" alt="image" src="https://github.com/user-attachments/assets/373448cc-ff65-4e19-8d0a-a22d4d6fb141" />

















