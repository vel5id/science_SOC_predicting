# pre-ml: S prediction with enriched features + SSL pretraining

## Цель
Предсказание серы (S, мг/кг) с использованием:
- **Кросс-контаминированных OOF-признаков** от лучших моделей (pH/GBDT, SOC/ET, NO3/RF)
- **Временны́х рядов спутниковой динамики** (100 признаков: L8/S2 × 25 индексов × {mean, std, slope, cv})
- **Self-supervised pretraining** (MAE) для улучшения CNN/MLP

## Структура
```
pre-ml/
├── build_sulfur_dataset.py  # Сборка датасета
├── ssl_pretrain.py          # SSL pretraining (Masked Autoencoder)
├── train_sulfur.py          # Сравнение моделей (RF, XGB, MLP, SSL)
├── checkpoints/             # Веса энкодера после pretraining
└── results/                 # Метрики и графики
```

## Запуск
```bash
# 1. Собрать датасет
python pre-ml/build_sulfur_dataset.py

# 2. Self-supervised pretraining (опционально, ~2-3 мин на GPU)
python pre-ml/ssl_pretrain.py

# 3. Обучение и оценка всех моделей
python pre-ml/train_sulfur.py
```

## Self-supervised pretraining

### Почему это работает?
- У нас 1085 образцов — мало для DL, но SSL pretraining не требует лейблов
- Автоэнкодер учит **структуру многоспектральных данных**: временну́ю динамику,
  пространственную вариабельность, корреляции между индексами
- Fine-tuning малого head поверх замороженного encoder — гораздо устойчивее,
  чем обучение всей сети с нуля

### Архитектура (Masked Autoencoder — SCARF подход)
```
Pretraining:
  Input → [маскируем 35%] → Encoder → z(64) → Decoder → ̂X
  Loss: MSE только по маскированным признакам

Fine-tuning для S:
  Input → Encoder (frozen/trainable) → z(64) → Head(32→1) → S_pred
```

### Почему MAE лучше обычного автоэнкодера?
- Принудительная реконструкция пропущенных признаков учит **контекстуальные зависимости**
- Аналогично BERT для NLP: маскируем часть входа → предсказываем
- В результате encoder кодирует богатое представление, не привязанное к одной задаче

## Ожидаемые результаты
| Метод | Field-LOFO ρ | Farm-LOFO ρ |
|-------|-------------|-------------|
| RF baseline | ~0.37-0.42 | ~0.06-0.10 |
| XGB baseline | ~0.40-0.45 | ~0.05-0.09 |
| MLP vanilla | ~0.30-0.38 | ~0.04-0.08 |
| MLP SSL frozen | ~0.35-0.42 | ~0.06-0.11 |
| MLP SSL finetune | ~0.38-0.45 | ~0.06-0.12 |

**Важно:** при Farm-LOFO ρ < 0.15 для серы — это подтверждает фундаментальную
проблему: серá определяется историей хозяйства, не наблюдаемой из космоса.
