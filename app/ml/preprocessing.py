from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    'base_price',
    'demand_index',
    'competitor_price',
    'inventory_level',
    'customer_behavior_score',
    'seasonal_index',
    'demand_trend',
    'price_elasticity',
    'competitor_price_diff',
    'seasonal_factor',
]

CATEGORICAL_FEATURES = ['category']


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    rolling_window = min(5, len(data))
    data['demand_trend'] = (
        data['demand_index'].rolling(window=rolling_window, min_periods=1).mean()
    )
    safe_demand = np.clip(data['demand_index'], 1e-3, None)
    data['price_elasticity'] = (data['competitor_price'] - data['base_price']) / safe_demand
    data['competitor_price_diff'] = data['competitor_price'] - data['base_price']
    data['seasonal_factor'] = data['seasonal_index'] * (1 + (data['demand_index'] / 100.0))
    return data
