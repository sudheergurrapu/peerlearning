const form = document.getElementById('pricingForm');
const chartCtx = document.getElementById('pricingChart');
let pricingChart;

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

async function fetchPrediction(payload) {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

function renderChart(staticPrice, dynamicPrice, staticRevenue, dynamicRevenue) {
  if (pricingChart) pricingChart.destroy();
  pricingChart = new Chart(chartCtx, {
    type: 'bar',
    data: {
      labels: ['Static', 'Dynamic'],
      datasets: [
        { label: 'Price', data: [staticPrice, dynamicPrice], backgroundColor: ['#9ca3af', '#2563eb'] },
        { label: 'Revenue', data: [staticRevenue, dynamicRevenue], backgroundColor: ['#d1d5db', '#10b981'] },
      ],
    },
    options: { responsive: true, maintainAspectRatio: true },
  });
}

async function runPrediction() {
  const payload = {
    product_id: document.getElementById('productId').value,
    category: document.getElementById('category').value,
    base_price: Number(document.getElementById('basePrice').value),
    demand_index: Number(document.getElementById('demandIndex').value),
    competitor_price: Number(document.getElementById('competitorPrice').value),
    inventory_level: Number(document.getElementById('inventoryLevel').value),
    customer_behavior_score: 0.65,
    seasonal_index: 1.1,
  };

  const result = await fetchPrediction(payload);
  setText('optimizedPrice', `$${result.optimized_price}`);
  setText('revenuePrediction', `$${result.revenue_prediction}`);
  setText('confidenceScore', result.confidence_score);
  setText('upliftPct', `${result.expected_revenue_uplift_pct}%`);
  setText('conversionPct', `${result.conversion_rate_improvement_pct}%`);
  setText('modelUsed', result.model_used);

  setText('kpiRevenue', `${result.kpis.revenue_growth}%`);
  setText('kpiMargin', `${result.kpis.profit_margin}%`);
  setText('kpiConversion', `${result.kpis.conversion_rate}%`);
  setText('kpiTurnover', result.kpis.inventory_turnover);

  const staticRevenue = payload.base_price * (payload.demand_index / 100) * payload.seasonal_index;
  renderChart(payload.base_price, result.optimized_price, staticRevenue, result.revenue_prediction);
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  runPrediction();
});

setInterval(() => {
  const demand = Number(document.getElementById('demandIndex').value);
  const jitter = (Math.random() * 4) - 2;
  document.getElementById('demandIndex').value = Math.max(1, Math.round((demand + jitter) * 10) / 10);
  runPrediction();
}, 12000);

runPrediction();
