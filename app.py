"""Food Waste AI - smart kitchen planner.   Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config as C
import engine
import dashboard

_icon = C.BASE_DIR  / "logo.png"
st.set_page_config(page_title="Food Waste AI", page_icon=str(_icon) if _icon.exists() else "🍽️", layout="wide")

if not C.WASTE_MODEL_PATH.exists():
    st.error("Models not found. Run `python generate_data.py` then `python train_model.py` first.")
    st.stop()

M = engine.metrics()

# ---------------- Sidebar: business assumptions ----------------
logo = C.BASE_DIR  / "logo.png"
if logo.exists():
    st.sidebar.image(str(logo),use_container_width=True)
st.sidebar.header("⚙️ Cost assumptions")
cost = st.sidebar.number_input("Cost of cooked food (₹/kg)", 10.0, 2000.0, C.DEFAULT_COST_PER_KG, 10.0)
short_cost = st.sidebar.number_input("Penalty for running short (₹/kg)", 10.0, 5000.0, C.DEFAULT_SHORTAGE_COST_PER_KG, 10.0,
                                     help="Lost sales / emergency cooking / unhappy customers per kg of unmet demand.")
st.sidebar.caption(f"Target service level = {short_cost / (short_cost + cost):.0%} "
                   "(higher penalty → cook more to avoid shortages).")
st.sidebar.markdown("---")
st.sidebar.write(f"**Best model:** {M['best_model']}")
st.sidebar.write(f"**Test MAE:** {min(r['test_mae'] for r in M['comparison']):.2f} kg")

banner = C.BASE_DIR  / "banner.jpg"
if banner.exists():
    st.image(str(banner),use_container_width=True)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Dashboard", "🎯 Predict & Recommend", "📅 Weekly Planner", "📂 Batch CSV", "📊 Model Insights", "🔎 Data Explorer"])

with tab0:
    dashboard.render(cost, short_cost)

# ---------------- Tab 1 ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        location = st.selectbox("📍 Location (city)", C.LOCATIONS)
        venue = st.selectbox("Venue type", C.VENUES)
        meal = st.selectbox("Meal", C.MEALS, index=1)
        day = st.selectbox("Day of week", C.DAYS)
        menu = st.selectbox("Menu type", C.MENUS)
        event = st.selectbox("Special event", C.EVENTS)
    with c2:
        expected = st.number_input("Expected customers", 5, 5000, 300, 10)
        popularity = st.slider("Menu popularity (1 = unpopular, 5 = favourite)", 1.0, 5.0, 3.5, 0.1)
        temp = st.slider("Temperature (°C)", 5.0, 48.0, 30.0, 0.5)
        rain = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0, 1.0)
        month = st.selectbox("Month", list(range(1, 13)), index=pd.Timestamp.today().month - 1)
    with c3:
        holiday = st.checkbox("Public holiday")
        exam = st.checkbox("Exam week")
        last_week = st.number_input("Waste same day last week (kg)", 0.0, 1000.0, 15.0, 1.0)
        prev_day = st.number_input("Waste yesterday (kg)", 0.0, 1000.0, 15.0, 1.0)
        default_prep = float(expected * {"Breakfast": 0.30, "Lunch": 0.50, "Dinner": 0.45}[meal] * 1.2)
        prepared = st.number_input("Food you plan to cook (kg)", 1.0, 5000.0, round(default_prep, 1), 5.0)

    row = dict(location=location,venue_type=venue, meal_type=meal, day_of_week=day, menu_type=menu, event_type=event,
               expected_customers=expected, temperature_c=temp, rainfall_mm=rain, is_holiday=int(holiday),
               is_exam_week=int(exam), menu_popularity=popularity, month=month, last_week_waste_kg=last_week,
               prev_day_waste_kg=prev_day, food_prepared_kg=prepared)

    if st.button("🔮 Predict", type="primary"):
        res = engine.recommend(row, cost, short_cost)
        lo_r, hi_r = M["residual_interval_80"]
        w = res["waste_current"]
        lo, hi = max(0.0, w + lo_r), max(0.0, w + hi_r)
        risk = engine.risk_level(w, prepared)
        icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[risk]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted waste", f"{w:.1f} kg", help=f"80% range: {lo:.1f} – {hi:.1f} kg")
        k2.metric("Waste risk", f"{icon} {risk}", f"{w / prepared:.0%} of cooked food", delta_color="off")
        k3.metric("Recommended to cook", f"{res['recommended_kg']:.1f} kg", f"{res['recommended_kg'] - prepared:+.1f} kg vs plan")
        k4.metric("Waste if you follow AI", f"{res['waste_recommended']:.1f} kg", f"-{res['waste_saved_kg']:.1f} kg", delta_color="inverse")
        st.caption(f"80% prediction range for current plan: **{lo:.1f} – {hi:.1f} kg**")

        s1, s2, s3 = st.columns(3)
        s1.success(f"💰 Money saved per meal: **₹{res['money_saved']:,.0f}**")
        s2.success(f"🌍 CO₂e avoided: **{res['co2_saved_kg']:.0f} kg**")
        s3.info(f"📈 Your current plan covers demand with ~**{res['service_level_current']:.0%}** probability")

        q = res["demand_quantiles"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(["Demand forecast"], [q[0.9] - q[0.1]], left=q[0.1], color="#cfe8ff", label="80% demand range")
        ax.plot([q[0.5]] * 2, [-0.3, 0.3], color="#1f77b4", lw=3, label=f"Median demand {q[0.5]:.0f} kg")
        ax.axvline(prepared, color="#d62728", ls="--", label=f"Your plan {prepared:.0f} kg")
        ax.axvline(res["recommended_kg"], color="#2ca02c", lw=2, label=f"AI recommendation {res['recommended_kg']:.0f} kg")
        ax.set_xlabel("kg of food"); ax.legend(fontsize=8, loc="upper right"); ax.set_yticks([])
        st.pyplot(fig)

        with st.expander("Why this recommendation?"):
            st.write(f"A kg wasted costs ₹{cost:.0f}; a kg short costs ₹{short_cost:.0f}. The cost-optimal choice is to cook "
                     f"the **{res['critical_ratio']:.0%} quantile** of forecast demand (newsvendor rule), "
                     "estimated with quantile Gradient Boosting.")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Plan the whole week")
    st.caption("Edit the table, the AI fills in waste forecast and recommended quantity for each day.")
    base = pd.DataFrame({
        "day_of_week": C.DAYS, "expected_customers": [300, 300, 300, 300, 320, 180, 160],
        "rainfall_mm": [0.0] * 7, "is_holiday": [0] * 7, "event_type": ["None"] * 7})
    wk_location = st.selectbox("📍 Location (city)", C.LOCATIONS, key="wkl")
    wk_venue = st.selectbox("Venue", C.VENUES, key="wkv")
    wk_meal = st.selectbox("Meal", C.MEALS, index=1, key="wkm")
    edited = st.data_editor(base, num_rows="fixed", use_container_width=True,
                            column_config={"event_type": st.column_config.SelectboxColumn(options=C.EVENTS),
                                           "day_of_week": st.column_config.TextColumn(disabled=True)})
    if st.button("Forecast week"):
        out = []
        kgpp = {"Breakfast": 0.30, "Lunch": 0.50, "Dinner": 0.45}[wk_meal]
        for _, r in edited.iterrows():
            rw = dict(location=wk_location,venue_type=wk_venue, meal_type=wk_meal, day_of_week=r["day_of_week"], menu_type="Mixed",
                      event_type=r["event_type"], expected_customers=int(r["expected_customers"]), temperature_c=30.0,
                      rainfall_mm=float(r["rainfall_mm"]), is_holiday=int(r["is_holiday"]), is_exam_week=0,
                      menu_popularity=3.4, month=pd.Timestamp.today().month, last_week_waste_kg=15.0, prev_day_waste_kg=15.0,
                      food_prepared_kg=int(r["expected_customers"]) * kgpp * 1.2)
            rec = engine.recommend(rw, cost, short_cost)
            out.append(dict(day=r["day_of_week"], usual_plan_kg=round(rw["food_prepared_kg"], 1),
                            ai_recommended_kg=round(rec["recommended_kg"], 1),
                            predicted_waste_usual=round(rec["waste_current"], 1),
                            predicted_waste_ai=round(rec["waste_recommended"], 1),
                            money_saved=round(rec["money_saved"])))
        res_df = pd.DataFrame(out)
        st.dataframe(res_df, use_container_width=True)
        st.bar_chart(res_df.set_index("day")[["predicted_waste_usual", "predicted_waste_ai"]])
        st.success(f"Weekly saving: ₹{res_df['money_saved'].sum():,.0f}  |  "
                   f"{(res_df['predicted_waste_usual'] - res_df['predicted_waste_ai']).sum():.0f} kg food")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Upload many meals at once")
    need = [c for c in C.WASTE_FEATURES if c not in C.ENGINEERED]
    template = pd.read_csv(C.DATA_PATH, nrows=5)[need]
    st.download_button("⬇️ Download CSV template", template.to_csv(index=False), "template.csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        d = pd.read_csv(up)
        missing = [c for c in need if c not in d.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            d["predicted_waste_kg"] = engine.predict_waste_df(d).round(2)
            d["waste_risk"] = [engine.risk_level(a, b) for a, b in zip(d["predicted_waste_kg"], d["food_prepared_kg"])]
            st.dataframe(d, use_container_width=True)
            st.metric("Total predicted waste", f"{d['predicted_waste_kg'].sum():,.0f} kg",
                      f"₹{d['predicted_waste_kg'].sum() * cost:,.0f} lost")
            st.download_button("⬇️ Download results", d.to_csv(index=False), "predictions.csv")

# ---------------- Tab 4 ----------------
with tab4:
    st.subheader("Model comparison (trained on past, tested on future dates)")
    comp = pd.DataFrame(M["comparison"]).set_index("model")
    st.dataframe(comp.style.highlight_min(subset=["cv_mae", "test_mae", "test_rmse"], color="#c8f7c5")
                 .highlight_max(subset=["test_r2"], color="#c8f7c5"), use_container_width=True)
    st.caption(f"Winner: **{M['best_model']}**. Training rows: {M['n_train']}, test rows: {M['n_test']}. "
               f"Demand P10–P90 interval covers {M['demand_p10_p90_coverage']:.0%} of real outcomes (ideal 80%).")

    a, b = st.columns(2)
    with a:
        imp = pd.read_csv(C.IMPORTANCE_PATH).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(imp["feature"], imp["importance"], color="#2a9d8f")
        ax.set_title("What drives food waste?"); ax.set_xlabel("Increase in MAE when shuffled (kg)")
        st.pyplot(fig)
    with b:
        p = pd.read_csv(C.PREDICTIONS_PATH)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(p["actual"], p["predicted"], s=6, alpha=0.4)
        m = max(p["actual"].max(), p["predicted"].max())
        ax.plot([0, m], [0, m], "r--"); ax.set_xlabel("Actual waste (kg)"); ax.set_ylabel("Predicted (kg)")
        ax.set_title("Actual vs Predicted (unseen future data)")
        st.pyplot(fig)

    st.subheader("💼 Business backtest")
    import backtest
    bt = backtest.run(cost, short_cost)
    x1, x2, x3 = st.columns(3)
    x1.metric("Waste reduction", f"{bt['waste_reduction_pct']:.1f}%", f"{bt['waste_current_kg'] - bt['waste_ai_kg']:,.0f} kg")
    x2.metric("Total cost reduction", f"{bt['cost_saving_pct']:.1f}%", f"₹{bt['total_cost_current'] - bt['total_cost_ai']:,.0f}")
    x3.metric("Shortage days (AI vs current)", f"{bt['shortage_days_ai_pct']:.0f}%",
              f"{bt['shortage_days_ai_pct'] - bt['shortage_days_current_pct']:+.1f} pts", delta_color="inverse")
    st.caption("Simulated on the held-out future period: what if kitchens had cooked the AI-recommended quantity?")

# ---------------- Tab 5 ----------------
with tab5:
    df = pd.read_csv(C.DATA_PATH, parse_dates=["date"])
    df["waste_pct"] = 100 * df["food_waste_kg"] / df["food_prepared_kg"]
    f1, f2 = st.columns(2)
    f1.write("**Average waste % by day of week**")
    f1.bar_chart(df.groupby("day_of_week")["waste_pct"].mean().reindex(C.DAYS))
    f2.write("**Average waste % by location**")
    f2.bar_chart(df.groupby("location")["waste_pct"].mean())
    f2.write("**Average waste % by venue**")
    f2.bar_chart(df.groupby("venue_type")["waste_pct"].mean())
    f1.write("**Average waste % by meal**")
    f1.bar_chart(df.groupby("meal_type")["waste_pct"].mean())
    f2.write("**Monthly total waste (kg)**")
    f2.line_chart(df.set_index("date")["food_waste_kg"].resample("ME").sum())
    with st.expander("Raw data"):
        st.dataframe(df.head(200))
