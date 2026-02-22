from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    product_id: str = Field(..., description='Unique product identifier')
    category: str = Field('general', description='Product category')
    base_price: float = Field(..., gt=0)
    demand_index: float = Field(..., ge=0)
    competitor_price: float = Field(..., gt=0)
    inventory_level: int = Field(..., ge=0)
    customer_behavior_score: float = Field(0.5, ge=0, le=1)
    seasonal_index: float = Field(1.0, ge=0)


class PredictResponse(BaseModel):
    optimized_price: float
    revenue_prediction: float
    confidence_score: float
    expected_revenue_uplift_pct: float
    conversion_rate_improvement_pct: float
    kpis: dict
    model_used: str
    constraints: dict
