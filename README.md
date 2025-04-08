# Rohlik Sales Forecasting Challenge
![image](https://github.com/user-attachments/assets/021d82d3-96c1-4f07-9802-17e96d7d4d92)

## Overview

Rohlik Group, a leading European e-grocery innovator, is revolutionising the food retail industry. We operate across 11 warehouses in Czech Republic, Germany, Austria, Hungary, and Romania.
We are now transitioning from the Rohlik Orders Forecasting Challenge to the Rohlik Sales Forecasting Challenge, as we continue with our set of challenges. This challenge focuses on predicting the sales of each selected warehouse inventory for next 14 days using historical sales data.

___

Группа компаний "Рохлик", ведущий европейский новатор в области электронных продуктов питания, совершает революцию в сфере розничной торговли продуктами питания. Мы работаем на 11 складах в Чехии, Германии, Австрии, Венгрии и Румынии.
Сейчас мы переходим от задачи "Прогнозирование заказов Rohlik" к задаче "Прогнозирование продаж Rohlik", продолжая наш набор задач. Эта задача направлена на прогнозирование продаж каждого выбранного складского запаса на следующие 14 дней с использованием исторических данных о продажах.

___

### Prepare Data
<pre>
  ```
def generate_time_features(df, calendar, date_column, add_trend_seasonality=False):
    """
    Генерация временных признаков из временной метки.

    Параметры:
    - df: DataFrame с данными.
    - date_column: название столбца с временной меткой (должен быть типа datetime).
    - add_trend_seasonality: bool, добавить ли признаки тренда и сезонности.

    Возвращает:
    - DataFrame с добавленными признаками.
    """
    df = df.copy()

    df = df.merge(calendar, on = ['date', 'warehouse'], how = 'left')
    
    # Убедимся, что столбец с датами имеет тип datetime
    df[date_column] = pd.to_datetime(df[date_column])

    df = df.sort_values(by=['date', 'warehouse'])

    # Создаем признак макс скидки
    discount_cols = df.filter(regex=r'^type_\d+_discount$').columns
    df['max_discount'] = df[discount_cols].max(axis=1)
    
    # Основные временные признаки
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek  # Понедельник = 0, Воскресенье = 6
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['day_of_year'] = df[date_column].dt.dayofyear  # День года
    df['week_of_year'] = df[date_column].dt.isocalendar().week  # Номер недели

    # Периодические признаки (синус/косинус для циклической природы времени)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Признаки тренда и сезонности
    if add_trend_seasonality:
        df['linear_trend'] = (df[date_column] - df[date_column].min()).dt.days  # Линейный тренд
        df['seasonality_month'] = np.sin(2 * np.pi * df['month'] / 12) * np.cos(2 * np.pi * df['month'] / 12)
        df['seasonality_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)

	# Добавление полиномиального тренда
    df["squared_trend"] = np.square(np.arange(len(df)))
    df["log_trend"] = np.log1p(np.arange(len(df)))  # Логарифмический
    df["exp_trend"] = np.exp(np.arange(len(df)) / len(df))  # Экспоненциальный

    df['id'] = df['unique_id'].astype(str) + "_" + df['date'].astype(str)
    
    df = df.drop(['holiday_name'] + list(discount_cols), axis=1) 
    df = df.dropna()

    return df

  
  def generate_lag_features(df):
    
    # Лаги
    lag_days = [1, 7, 14]
    for lag in lag_days:
        df[f'lag_{lag}'] = df.groupby('unique_id')['sales'].shift(lag)

    # Добавляем скользящее среднее
    window_sizes = [7, 14]
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df.groupby('unique_id')['sales'].shift(1).rolling(window=window).mean()


    return df
  ```
</pre>


### Optuna
<pre>
  ```
def objective(trial):
    
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
        "objective" : "reg:squarederror"
    }


    # Создаем препроцессор для числовых и категориальных признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feat),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)
        ]
    )

    # Инициализируем модель с подобранными гиперпараметрами
    model = XGBRegressor(**params)
    
    # Собираем Pipeline: сначала препроцессинг, затем регрессор
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    

    # Определяем TimeSeriesSplit для кросс-валидации (например, 3 разбиений)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Используем отрицательную среднюю абсолютную ошибку как метрику (чем больше — тем лучше)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=custom_wmae_score)
    
    # Возвращаем среднее значение метрики по кросс-валидации
    return np.mean(scores)


# Создаем исследование (study) с направлением максимизации (так как метрика отрицательная, и её максимум соответствует минимальной MAE)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)


# Вывод лучших гиперпараметров
print("Лучший trial:")
best_trial = study.best_trial
print("  Значение метрики:", best_trial.value)
print("  Гиперпараметры:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
  ```
</pre>


### Model
<pre>
  ```
# Значение метрики Optuna: -0.33016935935972924
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

best_params = {
    "objective": "reg:squarederror",
    "n_estimators": 101,
    "max_depth": 7,
    "learning_rate": 0.05335853605410118,
    "subsample": 0.876364265110239,
    "colsample_bytree": 0.8253151666142489,
    "reg_alpha": 0.2930905580203602,
    "reg_lambda": 0.18076715508639507
}

# Создаем препроцессор для числовых и категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_feat),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)
    ]
)

# Инициализируем модель с подобранными гиперпараметрами
model = XGBRegressor(**best_params)

# Собираем Pipeline: сначала препроцессинг, затем регрессор
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Обучаем модель
pipeline.fit(X_train, y_train)

# Предсказываем
y_pred = pipeline.predict(X_test)

# Считаем метрики
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

print(f'MSE : {MSE}')
print(f'MAE : {MAE}')
  
MSE : 0.2141229538335658
MAE : 0.3098679927794184
  ```
</pre>
