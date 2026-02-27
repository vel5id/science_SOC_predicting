"""
agronomic_metrics.py
====================
Метрики для аграриев:
  1) RPD (Ratio of Performance to Deviation)
  2) RPIQ (Ratio of Performance to InterQuartile distance)
  3) CCC (Lin's Concordance Correlation Coefficient)
  4) Классификационные метрики для дефицита серы

Агро-стандарты серы (mg/kg):
  < 6   — дефицит (нужно удобрять)
  6-12  — среднее содержание (норма)
  > 12  — высокое содержание

Для фермера:
  - Precision (дефицит) =TP / (TP + FP)
    "Если модель говорит дефицит, насколько часто это правда?"
  - Recall (дефицит) = TP / (TP + FN)
    "Модель находит какой % реальных дефицитов?"
  - F1 = 2 * P*R / (P+R)
  - Matthews Corr Coeff (MCC) для класса дефицит
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix


def compute_standard_metrics(y_true, y_pred):
    """Стандартные регрессионные метрики."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr, pearsonr

    rho, _ = spearmanr(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)

    return {
        "rho": float(rho),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": float(pearson_r),
    }


def compute_rpd(y_true, y_pred):
    """Ratio of Performance to Deviation.

    RPD = SD(y_true) / RMSE(y_true, y_pred)

    Интерпретация:
      RPD < 1.0  — непригодна для любого использования
      RPD 1.0-1.4 — только для очень грубых прогнозов
      RPD 1.4-1.8 — подходит для грубых оценок
      RPD 1.8-2.5 — подходит для точных оценок
      RPD > 2.5  — подходит для высокоточных прогнозов
    """
    from sklearn.metrics import mean_squared_error

    sd = np.std(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if rmse == 0:
        return np.inf
    return float(sd / rmse)


def compute_rpiq(y_true, y_pred):
    """Ratio of Performance to InterQuartile distance.

    RPIQ = IQR(y_true) / RMSE(y_true, y_pred)

    Похож на RPD, но более устойчив к выбросам.
    """
    from sklearn.metrics import mean_squared_error

    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    iqr = q3 - q1
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if rmse == 0 or iqr == 0:
        return np.inf
    return float(iqr / rmse)


def compute_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient.

    CCC = (2 * ρ * σ_true * σ_pred) / (σ_true² + σ_pred² + (μ_true - μ_pred)²)

    Комбинирует корреляцию (ρ) и точность (близость линии идеального согласия к y=x).
    CCC ∈ [-1, 1], где 1 = идеальное согласие.

    Интерпретация:
      CCC > 0.99  — отличное согласие
      CCC 0.95-0.99 — очень хорошее
      CCC 0.90-0.95 — хорошее
      CCC 0.80-0.90 — среднее
      CCC < 0.80  — слабое
    """
    mu_true = np.mean(y_true)
    mu_pred = np.mean(y_pred)
    sigma_true = np.std(y_true)
    sigma_pred = np.std(y_pred)

    if sigma_true == 0 or sigma_pred == 0:
        return np.nan

    rho = np.corrcoef(y_true, y_pred)[0, 1]

    numerator = 2 * rho * sigma_true * sigma_pred
    denominator = sigma_true**2 + sigma_pred**2 + (mu_true - mu_pred)**2

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def compute_agronomic_classification(y_true, y_pred):
    """Классификационные метрики для дефицита серы.

    Пороги (mg/kg):
      Класс 0 — дефицит:    < 6   (нужны удобрения)
      Класс 1 — среднее:    6-12  (норма, нет вмешательства)
      Класс 2 — высокое:    > 12  (переизбыток)

    Для фермера критично:
      - Recall для класса 0 (находи ВСЕ дефициты → не пропусти поле с дефицитом)
      - Precision для класса 0 (когда говоришь дефицит, это должно быть правда → не удобри зря)
    """

    # Пороги для классификации
    DEFICIT_THRESHOLD = 6
    NORMAL_THRESHOLD = 12

    # Обе переменные классифицируем
    y_true_class = np.zeros_like(y_true, dtype=int)
    y_true_class[(y_true >= DEFICIT_THRESHOLD) & (y_true < NORMAL_THRESHOLD)] = 1
    y_true_class[y_true >= NORMAL_THRESHOLD] = 2

    y_pred_class = np.zeros_like(y_pred, dtype=int)
    y_pred_class[(y_pred >= DEFICIT_THRESHOLD) & (y_pred < NORMAL_THRESHOLD)] = 1
    y_pred_class[y_pred >= NORMAL_THRESHOLD] = 2

    # Метрики для каждого класса
    results = {}

    # Общие метрики
    cm = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2])
    results["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_class, y_pred_class, labels=[0, 1, 2], zero_division=0
    )

    for class_idx, class_name in enumerate(["deficit", "normal", "high"]):
        results[f"{class_name}_precision"] = float(precision[class_idx])
        results[f"{class_name}_recall"] = float(recall[class_idx])
        results[f"{class_name}_f1"] = float(f1[class_idx])
        results[f"{class_name}_support"] = int(support[class_idx])

    # Matthews Correlation Coefficient для бинарной классификации (дефицит vs остальное)
    y_true_binary = (y_true_class == 0).astype(int)
    y_pred_binary = (y_pred_class == 0).astype(int)
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    results["mcc_deficit_vs_rest"] = float(mcc)

    # Accuracy для каждого класса
    results["overall_accuracy"] = float(np.mean(y_true_class == y_pred_class))

    return results


def compute_all_agronomic_metrics(y_true, y_pred):
    """Полный набор метрик для аграриев."""
    metrics = {}

    # Стандартные
    metrics.update(compute_standard_metrics(y_true, y_pred))

    # RPD и RPIQ
    metrics["rpd"] = compute_rpd(y_true, y_pred)
    metrics["rpiq"] = compute_rpiq(y_true, y_pred)

    # CCC
    metrics["ccc"] = compute_ccc(y_true, y_pred)

    # Классификация
    agro = compute_agronomic_classification(y_true, y_pred)
    metrics.update(agro)

    return metrics


def format_agronomic_report(metrics):
    """Красивый отчёт для фермера."""
    report = []
    report.append("=" * 70)
    report.append("ОТЧЁТ ДЛЯ АГРАРИЕВ: Прогноз содержания серы (S)")
    report.append("=" * 70)

    # Точность модели
    report.append("\n[1] ТОЧНОСТЬ РЕГРЕССИОННОГО ПРОГНОЗА")
    report.append(f"  R² = {metrics.get('r2', np.nan):.3f}  (доля объяснённой дисперсии)")
    report.append(f"  RMSE = {metrics.get('rmse', np.nan):.2f} mg/kg  (средняя ошибка)")
    report.append(f"  MAE = {metrics.get('mae', np.nan):.2f} mg/kg  (средняя абсолютная ошибка)")

    # RPD/RPIQ
    report.append("\n[2] ОЦЕНКА ПРИГОДНОСТИ МОДЕЛИ (RPD/RPIQ)")
    rpd = metrics.get('rpd', np.nan)
    rpiq = metrics.get('rpiq', np.nan)
    rpd_status = ""
    if rpd < 1.4:
        rpd_status = "❌ непригодна"
    elif rpd < 1.8:
        rpd_status = "⚠️  только грубые оценки"
    elif rpd < 2.5:
        rpd_status = "✅ подходит для точных оценок"
    else:
        rpd_status = "✅✅ высокоточная модель"
    report.append(f"  RPD = {rpd:.2f}  {rpd_status}")
    report.append(f"  RPIQ = {rpiq:.2f}")

    # CCC
    report.append("\n[3] СОГЛАСОВАННОСТЬ ПРОГНОЗА (Lin's CCC)")
    ccc = metrics.get('ccc', np.nan)
    ccc_status = ""
    if ccc > 0.99:
        ccc_status = "✅✅ отличное согласие"
    elif ccc > 0.95:
        ccc_status = "✅ очень хорошее"
    elif ccc > 0.90:
        ccc_status = "✅ хорошее"
    elif ccc > 0.80:
        ccc_status = "⚠️  среднее"
    else:
        ccc_status = "❌ слабое"
    report.append(f"  CCC = {ccc:.3f}  {ccc_status}")

    # Классификация дефицита
    report.append("\n[4] ОБНАРУЖЕНИЕ ДЕФИЦИТА СЕРЫ (< 6 mg/kg)")
    report.append(f"  Точность (Precision) = {metrics.get('deficit_precision', 0):.1%}")
    report.append("    → Если модель говорит 'дефицит', это правда в {:.0%} случаев".format(
        metrics.get('deficit_precision', 0)))

    report.append(f"  Полнота (Recall) = {metrics.get('deficit_recall', 0):.1%}")
    report.append("    → Модель находит {:.0%} реальных дефицитов".format(
        metrics.get('deficit_recall', 0)))

    report.append(f"  F1-score = {metrics.get('deficit_f1', 0):.3f}")

    # Практический вывод
    report.append("\n[5] ПРАКТИЧЕСКАЯ РЕКОМЕНДАЦИЯ ДЛЯ ФЕРМЕРА")
    precision = metrics.get('deficit_precision', 0)
    recall = metrics.get('deficit_recall', 0)

    if recall > 0.85 and precision > 0.75:
        report.append("✅ Модель ПРИГОДНА для поддержки решений об удобрении")
        report.append(f"   Модель будет находить {recall:.0%} проблемных полей,")
        report.append(f"   и {precision:.0%} её рекомендаций будут верны.")
    elif recall > 0.70:
        report.append("⚠️  Модель ЧАСТИЧНО пригодна. Используй с осторожностью.")
        report.append(f"   Риск пропустить дефицит: {(1-recall):.0%}")
    elif recall > 0.30:
        report.append("⚠️⚠️ Модель имеет ограниченную полезность.")
        report.append(f"   Находит только {recall:.0%} дефицитов (риск {(1-recall):.0%})")
        report.append(f"   Точность прогноза при дефиците: {precision:.0%}")
    else:
        report.append("❌ Модель НЕ ПРИГОДНА для этой задачи.")

    # Статистика
    report.append("\n[6] СТАТИСТИКА ДЕФИЦИТА")
    report.append(f"  Реальный дефицит в данных: {metrics.get('deficit_support', 0)} образцов")
    report.append(f"  Модель предсказала дефицит: {metrics.get('deficit_support', 0) + metrics.get('deficit_support', 0)} образцов")

    return "\n".join(report)


if __name__ == "__main__":
    # Пример
    np.random.seed(42)
    y_true = np.array([3, 5, 8, 10, 15, 18, 4, 6, 12])
    y_pred = np.array([4, 6, 7, 11, 14, 17, 5, 7, 11])

    metrics = compute_all_agronomic_metrics(y_true, y_pred)

    print("Метрики:")
    for k, v in metrics.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    print("\n" + format_agronomic_report(metrics))
