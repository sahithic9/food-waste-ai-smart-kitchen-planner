"""Shared constants for the Food Waste AI project."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "food_waste_data.csv"
MODEL_DIR = BASE_DIR / "models"
WASTE_MODEL_PATH = MODEL_DIR / "waste_model.joblib"
DEMAND_Q_PATH = MODEL_DIR / "demand_quantile_models.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
IMPORTANCE_PATH = MODEL_DIR / "importance.csv"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"

# Features known BEFORE cooking starts (no data leakage).
CATEGORICAL = ["location","venue_type", "meal_type", "day_of_week", "menu_type", "event_type"]
NUMERIC = [
    "expected_customers", "temperature_c", "rainfall_mm", "is_holiday",
    "is_exam_week", "menu_popularity", "month", "last_week_waste_kg",
    "prev_day_waste_kg",
]
ENGINEERED = ["prep_per_customer"]  # food_prepared_kg / expected_customers

# Waste model also knows how much was planned/cooked.
WASTE_FEATURES = CATEGORICAL + NUMERIC + ["food_prepared_kg"] + ENGINEERED
# Demand model must NOT see food_prepared_kg.
DEMAND_FEATURES = CATEGORICAL + NUMERIC

LOCATIONS = ["Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Delhi"]
VENUES = ["Restaurant", "Hostel Mess", "College Cafeteria", "Event Hall"]
MEALS = ["Breakfast", "Lunch", "Dinner"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MENUS = ["Veg", "Non-Veg", "Mixed"]
EVENTS = ["None", "Wedding", "Corporate", "Festival", "Sports Day"]

# Business assumptions (editable in the app sidebar)
DEFAULT_COST_PER_KG = 120.0            # Rs. cost of cooked food per kg
DEFAULT_SHORTAGE_COST_PER_KG = 200.0   # Rs. penalty per kg of unmet demand
CO2E_PER_KG_WASTE = 2.5                # kg CO2e per kg food wasted (approx.)