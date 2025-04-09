# Rohlik Sales Forecasting Challenge
![image](https://github.com/user-attachments/assets/021d82d3-96c1-4f07-9802-17e96d7d4d92)

## Overview

Rohlik Group, a leading European e-grocery innovator, is revolutionising the food retail industry. We operate across 11 warehouses in Czech Republic, Germany, Austria, Hungary, and Romania.
We are now transitioning from the Rohlik Orders Forecasting Challenge to the Rohlik Sales Forecasting Challenge, as we continue with our set of challenges. This challenge focuses on predicting the sales of each selected warehouse inventory for next 14 days using historical sales data.

___

Группа компаний "Рохлик", ведущий европейский новатор в области электронных продуктов питания, совершает революцию в сфере розничной торговли продуктами питания. Мы работаем на 11 складах в Чехии, Германии, Австрии, Венгрии и Румынии.
Сейчас мы переходим от задачи "Прогнозирование заказов Rohlik" к задаче "Прогнозирование продаж Rohlik", продолжая наш набор задач. Эта задача направлена на прогнозирование продаж каждого выбранного складского запаса на следующие 14 дней с использованием исторических данных о продажах.

___

## Используемые данные

- `sales_train.csv` — Исторические продажи товаров
- `sales_test.csv` — Примеры для прогнозирования
- `calendar.csv` — Календарные данные (включая праздники)
- `inventory.csv` — Остатки на складе
- `test_weights.csv` — Веса для метрики оценки

## EDA (Анализ данных)

- Анализ сезонности: отчётливо видна **недельная и месячная сезонность**
- Признаки дня недели и дня месяца **не показали значимых отклонений**
- Построены графики: временные ряды, гистограммы, агрегаты по времени

## Feature Engineering

Созданы **временные и лаговые признаки**, которые помогают модели захватывать:
- Год, месяц, квартал, день недели, выходные
- Периодические признаки через `sin`/`cos`
- **Лаги** (1, 7, 14 дней) и **скользящее среднее**
- Признаки тренда: линейный, полиномиальный, логарифмический и экспоненциальный

```python
def generate_time_features(...):
    ...
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    ...
```

## Модель

Используется **XGBoost Regressor** внутри `sklearn.Pipeline`, который объединяет:
- Преобразование признаков (масштабирование + OneHot)
- Предсказатель
- Специальная метрика `WMAE` (взвешенная MAE) с учетом важности магазинов

## Подбор гиперпараметров

С применением **Optuna** и `TimeSeriesSplit`:

```python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

Лучшие параметры сохраняются и используются в финальной модели.

## Итеративный прогноз

Прогноз на тесте строится по принципу **"авторегрессии"** — каждый следующий день предсказывается, используя фактические и предсказанные значения предыдущих дней:

```python
for uid in unique_ids:
    ...
    for row in test_rows.iterrows():
        ...
        temp_df = generate_time_features(...)
        temp_df = generate_lag_features(...)
        y_pred = model.predict(...)
```
