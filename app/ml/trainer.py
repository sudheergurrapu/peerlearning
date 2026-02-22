from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import add_engineered_features, build_preprocessor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


class PricingModelService:
    def __init__(self) -> None:
        self.best_model_name = ''
        self.best_pipeline: Pipeline | None = None
        self.cv_results: dict[str, dict] = {}
        self._fit()

    def _generate_training_data(self, n: int = 1800) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        categories = np.array(['electronics', 'fashion', 'grocery', 'general'])
        base_price = rng.uniform(10, 500, n)
        demand_index = rng.uniform(5, 100, n)
        competitor_price = base_price * rng.uniform(0.8, 1.25, n)
        inventory_level = rng.integers(5, 200, n)
        customer_behavior_score = rng.uniform(0.1, 1.0, n)
        seasonal_index = rng.uniform(0.8, 1.35, n)
        category = rng.choice(categories, n)

        conversion_rate = np.clip(
            0.45
            + 0.003 * demand_index
            - 0.0015 * ((competitor_price - base_price) / np.maximum(base_price, 1)) * 100
            + 0.08 * customer_behavior_score,
            0.05,
            0.98,
        )
        revenue = base_price * conversion_rate * (1 + 0.002 * demand_index) * seasonal_index

        df = pd.DataFrame(
            {
                'base_price': base_price,
                'demand_index': demand_index,
                'competitor_price': competitor_price,
                'inventory_level': inventory_level,
                'customer_behavior_score': customer_behavior_score,
                'seasonal_index': seasonal_index,
                'category': category,
                'target_revenue': revenue,
            }
        )
        return add_engineered_features(df)

    def _fit(self) -> None:
        df = self._generate_training_data()
        X = df.drop(columns=['target_revenue'])
        y = df['target_revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        models = {
            'random_forest': (
                RandomForestRegressor(random_state=7),
                {'model__n_estimators': [120, 180], 'model__max_depth': [6, 10]},
            ),
            'gradient_boosting': (
                GradientBoostingRegressor(random_state=7),
                {'model__n_estimators': [120, 180], 'model__learning_rate': [0.03, 0.07]},
            ),
        }

        if XGBRegressor is not None:
            models['xgboost'] = (
                XGBRegressor(
                    objective='reg:squarederror',
                    random_state=7,
                    n_estimators=150,
                    eval_metric='rmse',
                ),
                {'model__max_depth': [4, 6], 'model__learning_rate': [0.03, 0.1]},
            )

        best_r2 = -np.inf
        for name, (model, grid) in models.items():
            pipeline = Pipeline([('preprocessor', build_preprocessor()), ('model', model)])
            search = GridSearchCV(
                pipeline,
                grid,
                scoring='r2',
                cv=3,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            preds = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            cv_metrics = cross_validate(
                best_model,
                X_train,
                y_train,
                cv=3,
                scoring=('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'),
            )
            self.cv_results[name] = {
                'mae': float(mae),
                'mse': float(mse),
                'r2': float(r2),
                'cv_mae': float(-cv_metrics['test_neg_mean_absolute_error'].mean()),
                'cv_mse': float(-cv_metrics['test_neg_mean_squared_error'].mean()),
                'cv_r2': float(cv_metrics['test_r2'].mean()),
                'best_params': search.best_params_,
            }

            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_pipeline = best_model

    def predict_revenue(self, sample_df: pd.DataFrame) -> tuple[float, float]:
        if self.best_pipeline is None:
            raise RuntimeError('Model not trained')
        engineered = add_engineered_features(sample_df)
        prediction = float(self.best_pipeline.predict(engineered)[0])
        confidence = float(min(max(self.cv_results[self.best_model_name]['cv_r2'], 0.0), 1.0))
        return prediction, confidence
