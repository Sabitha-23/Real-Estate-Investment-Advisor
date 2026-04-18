
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 25px; border-radius: 12px;
        text-align: center; margin-bottom: 20px;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 5px 0 0; }
    .good-box {
        background: linear-gradient(135deg, #0f9b58, #00c851);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white;
    }
    .bad-box {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ──
@st.cache_resource
def load_models():
    clf_model    = joblib.load("models/clf_model.pkl")
    reg_model    = joblib.load("models/reg_model.pkl")
    clf_features = joblib.load("models/clf_feature_cols.pkl")
    reg_features = joblib.load("models/reg_feature_cols.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    return clf_model, reg_model, clf_features, reg_features, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("data/india_housing_prices.csv")
    df["Age_of_Property"]   = 2025 - df["Year_Built"]
    df["Future_Price_5yr"]  = df["Price_in_Lakhs"] * (1.08 ** 5)
    t_map = {"Low":1,"Medium":2,"High":3}
    df["Transport_Num"]     = df["Public_Transport_Accessibility"].map(t_map)
    df["Infrastructure_Score"] = (
        df["Nearby_Schools"]   * 0.3 +
        df["Nearby_Hospitals"] * 0.3 +
        df["Transport_Num"]    * 0.4
    ).round(2)
    city_med = df.groupby("City")["Price_per_SqFt"].transform("median")
    r1 = df["Price_per_SqFt"] <= city_med
    r2 = df["BHK"] >= 2
    r3 = df["Infrastructure_Score"] >= df["Infrastructure_Score"].median()
    r4 = df["Age_of_Property"] <= 20
    appr = (df["Future_Price_5yr"] - df["Price_in_Lakhs"]) / df["Price_in_Lakhs"]
    r5 = appr >= 0.40
    df["Good_Investment"] = (
        (r1.astype(int)+r2.astype(int)+r3.astype(int)+
         r4.astype(int)+r5.astype(int)) >= 4
    ).astype(int)
    return df

clf_model, reg_model, clf_features, reg_features, scaler = load_models()
df_raw = load_data()

# ── Helper: Scale clf input correctly ──
def prepare_clf_input(bhk, size, price, price_per_sqft,
                      floor_no, total_floors, age, schools, hospitals):
    """Scale input using the same scaler used during training"""
    raw = pd.DataFrame([[bhk, size, price, price_per_sqft,
                         floor_no, total_floors, age,
                         schools, hospitals]],
                       columns=clf_features)
    # Only scale columns that scaler knows about
    scale_cols = [c for c in clf_features
                  if c in scaler.feature_names_in_]
    raw[scale_cols] = scaler.transform(raw[scale_cols])
    return raw

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🏠 Real Estate Investment Advisor</h1>
    <p>AI-powered Property Profitability & Future Value Predictor</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Property Analyzer",
    "🔎 Filter & Browse",
    "📊 Visual Insights",
    "🤖 Model Performance"
])

# ══════════════════════════════════════════════
# TAB 1 — Property Analyzer
# ══════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Property Investment Analyzer")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### 📋 Property Details")
        bhk          = st.slider("BHK", 1, 5, 3)
        size         = st.number_input("Size (SqFt)", 500, 5000, 1500, step=100)
        price        = st.number_input("Price (Lakhs)", 10.0, 500.0, 150.0, step=5.0)
        age          = st.slider("Age of Property (yrs)", 2, 35, 5)
        floor_no     = st.slider("Floor Number", 0, 30, 5)
        total_floors = st.slider("Total Floors", 1, 30, 10)

        st.markdown("#### 🏙️ Location & Amenities")
        schools   = st.slider("Nearby Schools",   1, 10, 4)
        hospitals = st.slider("Nearby Hospitals", 1, 10, 3)
        transport = st.selectbox("Public Transport",
                                 ["Low","Medium","High"], index=1)

        analyze = st.button("🔍 Analyze Property",
                            use_container_width=True, type="primary")

    with col_r:
        if analyze:
            # Derived features
            t_num          = {"Low":1,"Medium":2,"High":3}[transport]
            price_per_sqft = round(price / size, 4)
            infra_score    = round(schools*0.3 + hospitals*0.3 + t_num*0.4, 2)
            school_den     = 1 if schools<=2 else (2 if schools<=5 else
                             (3 if schools<=8 else 4))
            hosp_den       = 1 if hospitals<=2 else (2 if hospitals<=5 else
                             (3 if hospitals<=8 else 4))
            floor_ratio    = round(floor_no / max(total_floors,1), 3)
            value_score    = round(infra_score / (price_per_sqft + 1), 4)

            # ── FIXED: Scale clf input ──
            clf_input = prepare_clf_input(
                bhk, size, price, price_per_sqft,
                floor_no, total_floors, age,
                schools, hospitals)

            # Regression input (uses raw engineered features)
            reg_input = pd.DataFrame(
                [[bhk, size, price, price_per_sqft,
                  floor_no, total_floors, age, schools,
                  hospitals, infra_score, school_den,
                  hosp_den, floor_ratio, value_score]],
                columns=reg_features)

            invest       = clf_model.predict(clf_input)[0]
            proba        = clf_model.predict_proba(clf_input)[0]
            confidence   = proba[invest] * 100
            future_price = reg_model.predict(reg_input)[0]
            gain         = future_price - price
            gain_pct     = (gain / price) * 100

            # Result box
            if invest == 1:
                st.markdown(f"""
                <div class="good-box">
                    <h2>✅ GOOD INVESTMENT</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bad-box">
                    <h2>❌ NOT RECOMMENDED</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💰 Future Price (5yr)", f"₹{future_price:.1f}L")
            m2.metric("📈 Expected Gain",
                      f"₹{gain:.1f}L", f"+{gain_pct:.1f}%")
            m3.metric("🏗️ Infra Score", f"{infra_score:.2f}/7")
            m4.metric("💎 Value Score", f"{value_score:.3f}")

            st.divider()

            # Charts
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🎯 Prediction Confidence")
                fig, ax = plt.subplots(figsize=(5, 3))
                labels = ["Not Recommended","Good Investment"]
                colors = ["#e74c3c","#2ecc71"]
                bars   = ax.barh(labels, proba*100,
                                 color=colors, height=0.5)
                ax.set_xlim(0,100)
                ax.set_xlabel("Confidence (%)")
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                for bar, val in zip(bars, proba*100):
                    ax.text(val+1, bar.get_y()+bar.get_height()/2,
                            f"{val:.1f}%", va="center",
                            color="white", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with c2:
                st.markdown("#### 📊 Property Score Card")
                categories = ["BHK","Size","Affordability",
                              "Infrastructure","Newness"]
                scores = [
                    min(bhk/5*100, 100),
                    min(size/5000*100, 100),
                    max(100 - price/500*100, 10),
                    min(infra_score/7*100, 100),
                    max(100 - age/35*100, 10)
                ]
                fig, ax = plt.subplots(figsize=(5, 3))
                clrs = ["#2ecc71" if s>=60 else
                        "#f39c12" if s>=40 else
                        "#e74c3c" for s in scores]
                ax.barh(categories, scores,
                        color=clrs, height=0.5)
                ax.set_xlim(0,100)
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                ax.tick_params(colors="white")
                for i, s in enumerate(scores):
                    ax.text(s+1, i, f"{s:.0f}%",
                            va="center", color="white", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            # Feature Importance
            st.markdown("#### 🔑 Feature Importance")
            if hasattr(clf_model, "feature_importances_"):
                feat_imp = pd.Series(
                    clf_model.feature_importances_,
                    index=clf_features
                ).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, 3))
                clrs_fi = plt.cm.RdYlGn(
                    np.linspace(0.2, 0.9, len(feat_imp)))
                ax.barh(feat_imp.index, feat_imp.values,
                        color=clrs_fi)
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#1a1a2e")
                ax.tick_params(colors="white")
                ax.set_xlabel("Importance", color="white")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        else:
            st.info("👈 Fill in property details and click **Analyze Property**")
            o1,o2,o3,o4 = st.columns(4)
            o1.metric("🏠 Properties Trained","2,50,000")
            o2.metric("🎯 Accuracy","95.44%")
            o3.metric("📊 F1 Score","0.9518")
            o4.metric("📈 R²","1.00")

# ══════════════════════════════════════════════
# TAB 2 — Filter & Browse
# ══════════════════════════════════════════════
with tab2:
    st.subheader("🔎 Filter & Browse Properties")

    f1, f2, f3 = st.columns(3)
    with f1:
        bhk_f  = st.multiselect("BHK", [1,2,3,4,5], default=[2,3])
        city_f = st.multiselect("City",
                   sorted(df_raw["City"].unique()),
                   default=sorted(df_raw["City"].unique())[:3])
    with f2:
        price_r = st.slider("Price (Lakhs)",
                    int(df_raw["Price_in_Lakhs"].min()),
                    int(df_raw["Price_in_Lakhs"].max()), (50,300))
        prop_f  = st.multiselect("Property Type",
                    list(df_raw["Property_Type"].unique()),
                    default=list(df_raw["Property_Type"].unique())[:2])
    with f3:
        area_r   = st.slider("Size (SqFt)",
                    int(df_raw["Size_in_SqFt"].min()),
                    int(df_raw["Size_in_SqFt"].max()), (800,3000))
        inv_only = st.checkbox("Good Investments Only")

    df_f = df_raw[
        (df_raw["BHK"].isin(bhk_f)) &
        (df_raw["City"].isin(city_f)) &
        (df_raw["Price_in_Lakhs"].between(*price_r)) &
        (df_raw["Property_Type"].isin(prop_f)) &
        (df_raw["Size_in_SqFt"].between(*area_r))
    ]
    if inv_only:
        df_f = df_f[df_f["Good_Investment"]==1]

    st.markdown(f"**{len(df_f):,} properties found**")

    show = df_f[["City","Property_Type","BHK","Size_in_SqFt",
                 "Price_in_Lakhs","Age_of_Property",
                 "Infrastructure_Score","Future_Price_5yr",
                 "Good_Investment"]].copy()
    show["Good_Investment"] = show["Good_Investment"].map(
        {1:"✅ Good", 0:"❌ Not Good"})
    st.dataframe(show.head(100),
                 use_container_width=True, height=400)

    if len(df_f) > 0:
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Avg Price",
                  f"₹{df_f['Price_in_Lakhs'].mean():.0f}L")
        s2.metric("Avg Size",
                  f"{df_f['Size_in_SqFt'].mean():.0f} SqFt")
        s3.metric("Good Investment %",
                  f"{df_f['Good_Investment'].mean()*100:.1f}%")
        s4.metric("Avg Future Price",
                  f"₹{df_f['Future_Price_5yr'].mean():.0f}L")

# ══════════════════════════════════════════════
# TAB 3 — Visual Insights
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Visual Insights")

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("#### 🏙️ Top 10 Cities — Avg Price")
        city_avg = (df_raw.groupby("City")["Price_in_Lakhs"]
                    .mean().sort_values(ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(city_avg.index[::-1], city_avg.values[::-1],
                color=plt.cm.RdYlGn(np.linspace(0.2,0.9,10)))
        ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.set_xlabel("Avg Price (Lakhs)", color="white")
        for bar, val in zip(ax.patches, city_avg.values[::-1]):
            ax.text(val+1, bar.get_y()+bar.get_height()/2,
                    f"₹{val:.0f}L", va="center",
                    color="white", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with v2:
        st.markdown("#### ✅ City — Investment Rate")
        city_inv = (df_raw.groupby("City")["Good_Investment"]
                    .mean().sort_values(ascending=False).head(10)*100)
        fig, ax = plt.subplots(figsize=(6,4))
        clrs = ["#2ecc71" if v>=50 else "#e74c3c"
                for v in city_inv.values]
        ax.bar(range(len(city_inv)), city_inv.values, color=clrs)
        ax.set_xticks(range(len(city_inv)))
        ax.set_xticklabels(city_inv.index, rotation=35,
                           ha="right", color="white", fontsize=8)
        ax.axhline(50, color="yellow", linestyle="--")
        ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.set_ylabel("Rate (%)", color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    v3, v4 = st.columns(2)
    with v3:
        st.markdown("#### 🏠 BHK vs Future Price")
        bhk_fp = df_raw.groupby("BHK")["Future_Price_5yr"].mean()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(bhk_fp.index, bhk_fp.values,
               color=plt.cm.Blues(np.linspace(0.4,0.9,len(bhk_fp))))
        ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.set_xlabel("BHK", color="white")
        ax.set_ylabel("Avg Future Price (L)", color="white")
        for i,v in enumerate(bhk_fp.values):
            ax.text(bhk_fp.index[i], v+1, f"₹{v:.0f}L",
                    ha="center", color="white", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with v4:
        st.markdown("#### 🏗️ Infrastructure vs Investment Rate")
        df_raw["Infra_Group"] = pd.cut(
            df_raw["Infrastructure_Score"], bins=5,
            labels=["Very Low","Low","Medium","High","Very High"])
        infra_inv = (df_raw.groupby("Infra_Group", observed=True)
                     ["Good_Investment"].mean()*100)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(infra_inv.index.astype(str), infra_inv.values,
               color=plt.cm.RdYlGn(np.linspace(0.1,0.9,5)))
        ax.axhline(50, color="yellow", linestyle="--")
        ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.set_ylabel("Investment Rate (%)", color="white")
        for i,v in enumerate(infra_inv.values):
            ax.text(i, v+0.5, f"{v:.0f}%",
                    ha="center", color="white", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### 🔥 Correlation Heatmap")
    corr_cols = ["BHK","Size_in_SqFt","Price_in_Lakhs",
                 "Price_per_SqFt","Age_of_Property",
                 "Nearby_Schools","Nearby_Hospitals",
                 "Infrastructure_Score","Future_Price_5yr",
                 "Good_Investment"]
    corr = df_raw[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                mask=np.triu(np.ones_like(corr, dtype=bool)),
                linewidths=0.5, ax=ax, annot_kws={"size":8})
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════
with tab4:
    st.subheader("🤖 Model Performance Dashboard")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### 🎯 Classification — XGBoost")
        st.markdown("""
| Metric | Score |
|--------|-------|
| ✅ Accuracy | **95.44%** |
| 🎯 F1 Score | **0.9518** |
| 📊 CV Score | **95.33%** |
| 📈 Precision | **0.95** |
| 🔁 Recall | **0.96** |
        """)
        cm_data = np.array([[25216,1432],[846,22506]])
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["Not Good","Good"],
                    yticklabels=["Not Good","Good"], ax=ax)
        ax.set_ylabel("Actual", color="white")
        ax.set_xlabel("Predicted", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with p2:
        st.markdown("#### 📈 Regression — XGBoost")
        st.markdown("""
| Metric | Score |
|--------|-------|
| 📈 R² Score | **1.00** |
| 🏠 Target | Future Price (5yr) |
| 🔢 Features | 14 engineered |
        """)
        if hasattr(clf_model, "feature_importances_"):
            feat_imp = pd.Series(
                clf_model.feature_importances_,
                index=clf_features).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.barh(feat_imp.index, feat_imp.values,
                    color=plt.cm.RdYlGn(
                        np.linspace(0.2,0.9,len(feat_imp))))
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.set_xlabel("Importance", color="white")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.markdown("#### 📋 All Models Compared")
    st.dataframe(pd.DataFrame({
        "Model"    : ["Logistic Regression","Random Forest","XGBoost"],
        "Accuracy" : ["82.00%","95.21%","95.44% 🏆"],
        "F1 Score" : ["0.8054","0.9493","0.9518 🏆"],
        "CV Score" : ["82.38%","95.19%","95.33% 🏆"]
    }), use_container_width=True, hide_index=True)
