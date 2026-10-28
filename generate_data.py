"""Create a realistic synthetic food-waste dataset (no target leakage)."""
import numpy as np
import pandas as pd
from config import DATA_PATH, DAYS, VENUES, MEALS, MENUS, EVENTS, LOCATIONS

RNG = np.random.default_rng(42)
VENUE_SIZE = {"Restaurant": 120, "Hostel Mess": 350, "College Cafeteria": 500, "Event Hall": 300}
MEAL_KG_PP = {"Breakfast": 0.30, "Lunch": 0.50, "Dinner": 0.45}  # kg per person
# Location effects: temperature offset (C), footfall multiplier, monsoon months
LOC_TEMP = {"Hyderabad": 0.0, "Bengaluru": -4.0, "Chennai": 2.0, "Mumbai": 1.0, "Delhi": 0.5}
LOC_FOOTFALL = {"Hyderabad": 1.00, "Bengaluru": 1.05, "Chennai": 0.95, "Mumbai": 1.15, "Delhi": 1.10}
LOC_RAIN_MONTHS = {"Hyderabad": (6, 7, 8, 9), "Bengaluru": (5, 6, 7, 8, 9, 10), "Chennai": (10, 11, 12),
                   "Mumbai": (6, 7, 8, 9), "Delhi": (7, 8)}
LOC_RAIN_SCALE = {"Hyderabad": 1.0, "Bengaluru": 0.9, "Chennai": 1.1, "Mumbai": 1.8, "Delhi": 0.8}
EVENT_BOOST = {"None": 1.0, "Wedding": 1.6, "Corporate": 0.9, "Festival": 1.4, "Sports Day": 1.2}


def generate(n_days: int = 365) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for location, venue in [(l, v) for l in LOCATIONS for v in VENUES]:
        history = {m: [30.0] * 7 for m in MEALS}
        for d in dates:
            dow = DAYS[d.dayofweek]
            weekend = d.dayofweek >= 5
            holiday = int(RNG.random() < 0.04)
            exam = int(d.month in (4, 5, 11, 12) and RNG.random() < 0.25)
            month_temp = 26 + 8 * np.sin((d.month - 3) / 12 * 2 * np.pi)
            temp = month_temp + LOC_TEMP[location] + RNG.normal(0, 2.5)
            rain = (max(0.0, RNG.gamma(1.2, 6 * LOC_RAIN_SCALE[location]) - 5) if d.month in LOC_RAIN_MONTHS[location]
                    else max(0.0, RNG.gamma(0.6, 3) - 3))
            event = RNG.choice(EVENTS, p=[0.9, 0.03, 0.03, 0.02, 0.02]) if venue in ("Event Hall", "Restaurant") else "None"
            for meal in MEALS:
                if venue == "Event Hall" and event == "None" and RNG.random() < 0.7:
                    continue
                menu = RNG.choice(MENUS, p=[0.4, 0.3, 0.3])
                popularity = float(np.clip(RNG.normal(3.4, 0.9), 1, 5))
                base = VENUE_SIZE[venue] * LOC_FOOTFALL[location]
                if venue in ("Hostel Mess", "College Cafeteria"):
                    base *= 0.55 if weekend else 1.0
                    base *= 0.85 if holiday else 1.0
                else:
                    base *= 1.25 if weekend else 1.0
                expected = base * EVENT_BOOST[event] * (0.6 if meal == "Breakfast" else 1.0)
                expected = max(15, int(RNG.normal(expected, expected * 0.06)))
                # --- hidden real behaviour ---
                show = 1.0 + RNG.normal(0, 0.09)
                show -= 0.12 if rain > 10 else 0
                show -= 0.10 if exam and venue == "College Cafeteria" else 0
                show += 0.05 if (temp > 34 and meal != "Breakfast") else 0
                actual = max(5, expected * show)
                appetite = MEAL_KG_PP[meal] * (0.75 + 0.1 * popularity) * (1.05 if menu == "Non-Veg" else 1.0)
                demand_kg = actual * appetite * (1 + RNG.normal(0, 0.04))
                prepared = expected * MEAL_KG_PP[meal] * RNG.uniform(1.00, 1.35)
                consumed = min(prepared, demand_kg)
                plate_waste = consumed * RNG.uniform(0.02, 0.07)
                waste = (prepared - consumed) + plate_waste
                prev_day = history[meal][-1]
                week_ago = history[meal][-7]
                history[meal].append(waste)
                rows.append(dict(
                    date=d, location=location, venue_type=venue, meal_type=meal, day_of_week=dow, menu_type=menu,
                    event_type=event, expected_customers=expected, temperature_c=round(temp, 1),
                    rainfall_mm=round(rain, 1), is_holiday=holiday, is_exam_week=exam,
                    menu_popularity=round(popularity, 1), month=d.month,
                    last_week_waste_kg=round(week_ago, 2), prev_day_waste_kg=round(prev_day, 2),
                    food_prepared_kg=round(prepared, 2), food_demand_kg=round(demand_kg, 2),
                    food_waste_kg=round(waste, 2),
                ))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows -> {DATA_PATH}")
    print(df[["food_prepared_kg", "food_demand_kg", "food_waste_kg"]].describe().round(2))