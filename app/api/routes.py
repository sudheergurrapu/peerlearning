from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from app.core.config import CONFIG
from app.ml.reinforcement import QLearningPriceAdjuster
from app.ml.trainer import PricingModelService
from app.schemas.pricing import PredictRequest, PredictResponse

router = APIRouter()
model_service = PricingModelService()
rl_adjuster = QLearningPriceAdjuster()


for _ in range(400):
    rl_adjuster.train_episode(
        base_price=120,
        demand_index=55,
        inventory_level=45,
    )


@router.get('/health', tags=['system'])
def healthcheck():
    return {'status': 'ok', 'best_model': model_service.best_model_name}


@router.get('/metrics', tags=['system'])
def metrics():
    return model_service.cv_results


@router.post('/predict', response_model=PredictResponse, tags=['pricing'])
def predict(payload: PredictRequest):
    sample = pd.DataFrame(
        [
            {
                'base_price': payload.base_price,
                'demand_index': payload.demand_index,
                'competitor_price': payload.competitor_price,
                'inventory_level': payload.inventory_level,
                'customer_behavior_score': payload.customer_behavior_score,
                'seasonal_index': payload.seasonal_index,
                'category': payload.category,
            }
        ]
    )

    predicted_revenue, confidence = model_service.predict_revenue(sample)

    ml_target_price = payload.base_price * (1 + (payload.demand_index - 50) / 250)
    competitor_anchor = payload.competitor_price * 0.98
    blended_price = (0.55 * ml_target_price) + (0.45 * competitor_anchor)

    rl_delta = rl_adjuster.suggest_adjustment(payload.demand_index, payload.inventory_level)
    adjusted = blended_price * (1 + rl_delta)

    min_price = payload.base_price * CONFIG.min_threshold
    max_price = payload.base_price * CONFIG.max_threshold
    bounded = max(min(adjusted, max_price), min_price)

    max_step_up = payload.base_price * (1 + CONFIG.max_step_change)
    max_step_down = payload.base_price * (1 - CONFIG.max_step_change)
    optimized_price = max(min(bounded, max_step_up), max_step_down)

    static_revenue = payload.base_price * (payload.demand_index / 100) * payload.seasonal_index
    uplift_pct = ((predicted_revenue - static_revenue) / max(static_revenue, 1e-3)) * 100

    baseline_conversion = max(min(payload.demand_index / 100, 0.95), 0.05)
    dynamic_conversion = max(
        min(
            baseline_conversion
            + ((payload.competitor_price - optimized_price) / max(payload.base_price, 1)) * 0.08,
            0.98,
        ),
        0.05,
    )
    conversion_improvement_pct = ((dynamic_conversion - baseline_conversion) / baseline_conversion) * 100

    kpis = {
        'revenue_growth': round(uplift_pct, 2),
        'profit_margin': round(((optimized_price - (0.58 * payload.base_price)) / optimized_price) * 100, 2),
        'conversion_rate': round(dynamic_conversion * 100, 2),
        'inventory_turnover': round((payload.demand_index / max(payload.inventory_level, 1)) * 12, 2),
    }

    return PredictResponse(
        optimized_price=round(optimized_price, 2),
        revenue_prediction=round(predicted_revenue, 2),
        confidence_score=round(confidence, 3),
        expected_revenue_uplift_pct=round(uplift_pct, 2),
        conversion_rate_improvement_pct=round(conversion_improvement_pct, 2),
        kpis=kpis,
        model_used=model_service.best_model_name,
        constraints={
            'min_price': round(min_price, 2),
            'max_price': round(max_price, 2),
            'volatility_cap_pct': CONFIG.max_step_change * 100,
        },
    )
