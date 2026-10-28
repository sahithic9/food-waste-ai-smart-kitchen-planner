"""Interactive dashboard tab for the Food Waste AI app."""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config as C
import backtest

GREEN, LIGHT, ORANGE, RED = "#2D6A4F", "#95D5B2", "#F4A261", "#E76F51"


@st.cache_data
def load_data():
    return pd.read_csv(C.DATA_PATH, parse_dates=["date"])


@st.cache_data
def load_ai_rows(cost, shortage):
    return backtest.per_row(cost, shortage)


def _pct(num, den):
    return 100 * num / den if den else 0.0


def _share(d, col):
    """Waste share (%) of cooked food, grouped by a column."""
    g = d.groupby(col, observed=True)[["food_waste_kg", "food_prepared_kg"]].sum()
    return 100 * g["food_waste_kg"] / g["food_prepared_kg"]


def _delta(now, before):
    return None if not before else f"{100 * (now - before) / before:+.1f}% vs previous period"


def render(cost: float, short_cost: float):
    df = load_data()

    # ---------------- Filters ----------------
    st.subheader("Filters")
    f1, f2, f3, f4 = st.columns(4)
    cities = f1.multiselect("📍 City", C.LOCATIONS, default=C.LOCATIONS, key="dash_city")
    venues = f2.multiselect("🏢 Venue", C.VENUES, default=C.VENUES, key="dash_venue")
    meals = f3.multiselect("🍽️ Meal", C.MEALS, default=C.MEALS, key="dash_meal")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    rng = f4.date_input("📅 Dates", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="dash_dates")
    start, end = (rng[0], rng[1]) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax)

    base = df[df["location"].isin(cities) & df["venue_type"].isin(venues) & df["meal_type"].isin(meals)]
    d = base[(base["date"] >= pd.Timestamp(start)) & (base["date"] <= pd.Timestamp(end))]
    if d.empty:
        st.warning("No records match these filters. Widen your selection.")
        return

    span = pd.Timestamp(end) - pd.Timestamp(start)
    prev = base[(base["date"] >= pd.Timestamp(start) - span - pd.Timedelta(days=1)) & (base["date"] < pd.Timestamp(start))]

    waste, prepared = d["food_waste_kg"].sum(), d["food_prepared_kg"].sum()
    p_waste, p_prepared = prev["food_waste_kg"].sum(), prev["food_prepared_kg"].sum()

    # ---------------- KPI cards ----------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total waste", f"{waste:,.0f} kg", _delta(waste, p_waste), delta_color="inverse")
    k2.metric("Waste share of cooked food", f"{_pct(waste, prepared):.1f}%",
              None if not p_prepared else f"{_pct(waste, prepared) - _pct(p_waste, p_prepared):+.1f} pts", delta_color="inverse")
    k3.metric("Cost of waste", f"₹{waste * cost:,.0f}")
    k4.metric("CO₂e from waste", f"{waste * C.CO2E_PER_KG_WASTE / 1000:,.1f} t")
    k5.metric("Avg waste per meal", f"{waste / len(d):.1f} kg", f"{len(d):,} meals", delta_color="off")

    # ---------------- Auto insights ----------------
    by_city = _share(d, "location")
    by_day = _share(d, "day_of_week")
    by_meal = _share(d, "meal_type")
    st.info(f"💡 Highest waste share: **{by_city.idxmax()}** ({by_city.max():.1f}%) · worst day: **{by_day.idxmax()}** "
            f"({by_day.max():.1f}%) · worst meal: **{by_meal.idxmax()}** ({by_meal.max():.1f}%)")

    # ---------------- Trend + city ----------------
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Weekly waste trend (kg)**")
        weekly = d.set_index("date")["food_waste_kg"].resample("W").sum()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(weekly.index, weekly.values, color=LIGHT, lw=1.5, label="Weekly")
        ax.plot(weekly.index, weekly.rolling(4, min_periods=1).mean(), color=GREEN, lw=2.5, label="4-week average")
        ax.set_ylabel("kg"); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with c2:
        st.markdown("**Waste share by city (%)**")
        s = by_city.sort_values()
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.barh(s.index, s.values, color=[ORANGE if v == s.max() else GREEN for v in s.values])
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    # ---------------- Heatmap + venue ----------------
    c3, c4 = st.columns([3, 2])
    with c3:
        st.markdown("**When does waste happen? Waste share by day × meal (%)**")
        piv = d.pivot_table(index="day_of_week", columns="meal_type", values=["food_waste_kg", "food_prepared_kg"], aggfunc="sum")
        heat = (100 * piv["food_waste_kg"] / piv["food_prepared_kg"]).reindex(index=C.DAYS, columns=[m for m in C.MEALS if m in piv["food_waste_kg"].columns])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        im = ax.imshow(heat.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(heat.shape[1]), heat.columns); ax.set_yticks(range(heat.shape[0]), heat.index)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                if pd.notna(heat.values[i, j]):
                    ax.text(j, i, f"{heat.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
        st.pyplot(fig)
    with c4:
        st.markdown("**Total waste by venue type (kg)**")
        st.bar_chart(d.groupby("venue_type")["food_waste_kg"].sum().sort_values(ascending=False), color=GREEN)

    # ---------------- Weather ----------------
    c5, c6 = st.columns(2)
    dd = d.copy()
    dd["Rain"] = pd.cut(dd["rainfall_mm"], [-0.1, 0.5, 5, 15, 1000], labels=["Dry", "Light", "Moderate", "Heavy"])
    dd["Temp"] = pd.cut(dd["temperature_c"], [-50, 25, 30, 35, 80], labels=["<25°C", "25-30°C", "30-35°C", ">35°C"])
    for col, holder, title in [("Rain", c5, "Waste share by rainfall (%)"), ("Temp", c6, "Waste share by temperature (%)")]:
        g = _share(dd, col)
        holder.markdown(f"**{title}**")
        holder.bar_chart(g, color=ORANGE)

    # ---------------- Worst offenders ----------------
    st.markdown("**🚨 Top 10 waste hotspots (city · venue · meal)**")
    hot = (d.groupby(["location", "venue_type", "meal_type"])
             .agg(meals=("food_waste_kg", "size"), waste_kg=("food_waste_kg", "sum"), prepared_kg=("food_prepared_kg", "sum"))
             .reset_index())
    hot["waste_%"] = (100 * hot["waste_kg"] / hot["prepared_kg"]).round(1)
    hot["cost_₹"] = (hot["waste_kg"] * cost).round(0)
    hot = hot.sort_values("waste_kg", ascending=False).head(10).drop(columns="prepared_kg")
    hot["waste_kg"] = hot["waste_kg"].round(0)
    st.dataframe(hot, use_container_width=True, hide_index=True)

    # ---------------- AI impact ----------------
    st.markdown("---")
    st.subheader("🤖 AI impact (held-out future period)")
    st.caption("What would have happened if kitchens had cooked the AI-recommended quantity? Uses the city, venue and meal filters above.")
    ai = load_ai_rows(cost, short_cost)
    ai = ai[ai["location"].isin(cities) & ai["venue_type"].isin(venues) & ai["meal_type"].isin(meals)]
    if ai.empty:
        st.warning("No held-out rows for these filters.")
    else:
        w0, w1 = ai["waste_current"].sum(), ai["waste_ai"].sum()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Waste today", f"{w0:,.0f} kg")
        a2.metric("Waste with AI plan", f"{w1:,.0f} kg", f"-{_pct(w0 - w1, w0):.1f}%", delta_color="inverse")
        a3.metric("Money saved", f"₹{(w0 - w1) * cost:,.0f}", f"{(w0 - w1) * C.CO2E_PER_KG_WASTE / 1000:,.1f} t CO₂e avoided", delta_color="off")
        s0, s1 = 100 * (ai["short_current"] > 0.5).mean(), 100 * (ai["short_ai"] > 0.5).mean()
        a4.metric("Meals running short", f"{s1:.0f}%", f"{s1 - s0:+.1f} pts vs today", delta_color="inverse")
        comp = ai.groupby("location")[["waste_current", "waste_ai"]].sum().rename(
            columns={"waste_current": "Current practice", "waste_ai": "With AI plan"})
        st.bar_chart(comp, color=[RED, GREEN])

    with st.expander("Download filtered data"):
        st.download_button("⬇️ CSV", d.to_csv(index=False), "filtered_food_waste.csv")