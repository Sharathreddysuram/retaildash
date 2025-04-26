import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib
from sqlalchemy import create_engine, text
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Retail Analytics on Azure", layout="wide")

# --- Utility ---
def find_col(df, *candidates):
    cols = {c.strip().lower(): c.strip() for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"None of {candidates} found in columns")

# --- User Signup ---
st.sidebar.header("\U0001F464 User Signup")
with st.sidebar.form("signup_form", clear_on_submit=False):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email    = st.text_input("Email")
    registered = st.form_submit_button("Register")
if registered:
    st.sidebar.success(f"Registered as **{username}** ({email})")

# --- Azure SQL Engine ---
@st.cache_resource
def get_engine():
    try:
        # Connection parameters
        driver = "ODBC Driver 17 for SQL Server"
        server = "sharathfinal.database.windows.net"
        database = "finalproject"
        username = "siddu@sharathfinal"
        password = "Cloud@123"

        # Build ODBC string
        odbc_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )

        params = urllib.parse.quote_plus(odbc_str)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine

    except Exception as e:
        st.sidebar.error(f"❌ Failed to connect to Azure SQL: {e}")
        return None

# --- Data loading function
@st.cache_data
def load_data():
    engine = get_engine()

    if engine is not None:
        try:
            # Load from Azure SQL
            hh = pd.read_sql_table("Households", engine)
            trans = pd.read_sql_table("Transactions", engine)
            prod = pd.read_sql_table("Products", engine)

            # Strip columns
            hh.columns, trans.columns, prod.columns = map(lambda c: c.str.strip(), [hh.columns, trans.columns, prod.columns])

            st.sidebar.success("✅ Loaded from Azure SQL Database")
            return hh, trans, prod

        except Exception as e:
            st.sidebar.warning(f"⚠️ Connection succeeded but table loading failed: {e}")
    
    # Fallback to CSV
    st.sidebar.warning("⚠️ Loading from CSVs instead")
    hh = pd.read_csv("400_households.csv")
    trans = pd.read_csv("400_transactions.csv")
    prod = pd.read_csv("400_products.csv")
    hh.columns, trans.columns, prod.columns = map(lambda c: c.str.strip(), [hh.columns, trans.columns, prod.columns])

    return hh, trans, prod

hh, prod, trans, merged = load_data()


# 4) Controls
st.sidebar.header("⚙️ Analytics Controls")
hcol     = find_col(merged,"hshd_num")
bcol     = find_col(merged,"basket_num")
icol     = find_col(merged,"income_range")
min_sup  = st.sidebar.slider("Min Support",    0.0,0.10,0.01,step=0.005)
min_conf = st.sidebar.slider("Min Confidence", 0.10,1.00,0.30,step=0.05)
churn_w  = st.sidebar.slider("Churn window (days)",30,180,90,step=10)
sel      = st.sidebar.selectbox(
    "Household #", sorted(merged[hcol].unique()),
    index=9, format_func=lambda x: f"{int(x):04d}"
)

# Header & KPIs
st.title("🛒 Retail Analytics on Azure")
total_spend = merged["SPEND"].sum()
avg_basket  = merged.groupby([hcol,bcol])["SPEND"].sum().mean()
c1,c2,c3    = st.columns([1,1,2])
with c1: st.metric("💰 Total Spend", f"${total_spend:,.0f}")
with c2: st.metric("🛍️ Avg Spend/Basket", f"${avg_basket:,.2f}")
with c3: st.markdown(f"**As of:** {merged['DATE'].max().strftime('%b %d, %Y')}")
st.markdown("---")

# 5) Sample Pull
st.subheader(f"📋 Data Pull for Household #{int(sel):04d}")
dfp = (merged[merged[hcol]==sel]
       .sort_values([hcol,bcol,"DATE",find_col(merged,"product_num"),"DEPARTMENT","COMMODITY"]))
st.dataframe(dfp, height=200, use_container_width=True)
st.markdown("---")

# 6) Time Series & Spend/Visit
colA,colB = st.columns(2, gap="large")
with colA:
    st.subheader("📈 Spend Over Time")
    ts = merged.groupby("DATE")["SPEND"].sum().reset_index()
    st.line_chart(ts.rename(columns={"DATE":"index"}).set_index("index")["SPEND"])
with colB:
    st.subheader("🏷️ Avg Spend/Visit by Income")
    grp = merged.groupby([hcol,bcol,icol])["SPEND"].sum().reset_index()
    avg_inc = grp.groupby(icol)["SPEND"].mean().sort_values(ascending=False)
    st.bar_chart(avg_inc)
st.markdown("---")

# 7) Basket Analysis (Rules)
st.subheader("🛒 Basket Analysis")
baskets = (merged[merged[hcol]==sel]
           .groupby(bcol)[find_col(merged,"product_num")]
           .apply(lambda x: list(map(str,x))).tolist())
te    = TransactionEncoder()
df_tf = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
freq  = apriori(df_tf, min_support=min_sup, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=min_conf).sort_values("lift",ascending=False)
if rules.empty:
    st.info("No rules at these thresholds.")
else:
    rules["antecedents"] = rules["antecedents"].apply(lambda s:", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s:", ".join(sorted(s)))
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].head(10), height=200)
st.markdown(f"• Total baskets: **{len(baskets)}** • Avg items/basket: **{np.mean([len(b) for b in baskets]):.1f}**")

# 7b) Basket Analysis (Random Forest ML)
st.subheader("🔧 ML Basket Analysis (Random Forest)")
if len(baskets)>10:
    # target = most frequent product
    target = merged[merged[hcol]==sel]["PRODUCT_NUM"].astype(str).value_counts().idxmax()
    y = df_tf[target]
    X = df_tf.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    st.markdown(f"**Predicting presence of product {target} → accuracy: {acc:.2%}**")

    imp = pd.Series(clf.feature_importances_, index=X.columns) \
            .nlargest(10)
    fig, ax = plt.subplots(figsize=(6,3))
    imp.plot(kind="bar", ax=ax)
    ax.set_xlabel("Product Num")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("Not enough baskets for ML analysis.")
st.markdown("---")

# 8) Churn Prediction
st.subheader("🚨 Churn Prediction")
maxd = merged["DATE"].max()
rfm  = merged.groupby(hcol).agg(
    recency   = ("DATE", lambda x: (maxd-x.max()).days),
    frequency = (bcol,    "nunique"),
    monetary  = ("SPEND",  "sum")
).reset_index()
rfm["churn"] = (rfm["recency"]>churn_w).astype(int)
rate = rfm["churn"].mean()
st.markdown(f"**Overall churn (> {churn_w}d):** **{rate:.1%}**")
if rate>0:
    Xc = rfm[["recency","frequency","monetary"]]
    yc = rfm["churn"]
    m  = LogisticRegression(max_iter=500).fit(Xc,yc)
    row = rfm.loc[rfm[hcol]==sel,["recency","frequency","monetary"]]
    p   = m.predict_proba(row)[0,1]
    st.metric("Predicted Churn Risk",f"{p*100:.1f}%",delta=f"{rate*100:.1f}%")
else:
    st.warning("No churn cases; try changing the window.")
st.markdown("""
**How it works**  
- Recency = days since last purchase  
- Frequency = # of baskets  
- Monetary = total spend  
- Logistic regression on RFM
""")
st.markdown("---")
st.header("🔎 Retail Insights Dashboard")

# 9) Demographics & Engagement
with st.expander("👪 Demographics & Engagement", expanded=True):
    size_col  = find_col(merged,"hh_size","hshd_size")
    child_col = find_col(merged,"children")
    dem = merged.groupby([size_col, child_col, icol]).agg(
        total_spend=("SPEND","sum"),
        visits=("BASKET_NUM","nunique")
    ).reset_index()
    dem["spend_per_visit"] = dem["total_spend"]/dem["visits"]
    pivot = dem.pivot_table(index=icol,columns=child_col,values="spend_per_visit").fillna(0)
    fig, ax = plt.subplots(figsize=(6,3))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Income Range"); ax.set_ylabel("Avg Spend/Visit"); ax.legend(title="Has Children")
    fig.tight_layout()
    st.pyplot(fig)

# 10) Engagement Over Time by Commodity
with st.expander("📊 Engagement Over Time by Commodity"):
    top5 = merged.groupby("COMMODITY")["SPEND"].sum().nlargest(5).index
    ts_c = merged[merged["COMMODITY"].isin(top5)].groupby(
        [pd.Grouper(key="DATE",freq="M"), "COMMODITY"]
    )["SPEND"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(6,3))
    for c in top5:
        dfc = ts_c[ts_c["COMMODITY"]==c]
        ax.plot(dfc["DATE"], dfc["SPEND"], label=c)
    ax.set_xlabel("Month"); ax.set_ylabel("Total Spend"); ax.legend(title="Commodity",ncol=2)
    fig.tight_layout()
    st.pyplot(fig)

# 11) Seasonal Trends
with st.expander("❄️ Seasonal Spend Patterns"):
    merged["Month"] = merged["DATE"].dt.month
    mn = merged.groupby("Month")["SPEND"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.plot(mn["Month"], mn["SPEND"], marker="o")
    ax.set_xticks(range(1,13)); ax.set_xlabel("Month"); ax.set_ylabel("Avg Spend/Visit")
    fig.tight_layout()
    st.pyplot(fig)

# 12) Brand & Organic Preferences
with st.expander("🌿 Brand & Organic Preferences"):
    brand_col = find_col(merged,"brand_ty","brand_type")
    org_col   = find_col(merged,"natural_organic_flag","organic_flag","natural/organic flag")
    bt = merged.groupby(brand_col)["SPEND"].sum().reset_index()
    of = merged.groupby(org_col)["SPEND"].sum().reset_index()
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Private vs National Brand Spend")
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(bt[brand_col], bt["SPEND"]); ax.set_ylabel("Total Spend")
        fig.tight_layout(); st.pyplot(fig)
    with c2:
        st.subheader("Organic vs Non-Organic Spend")
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(of[org_col], of["SPEND"]); ax.set_ylabel("Total Spend")
        fig.tight_layout(); st.pyplot(fig)
