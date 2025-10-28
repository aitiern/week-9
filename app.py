import streamlit as st
import pandas as pd
import numpy as np

from apputil import GroupEstimate

# ----------------------------
# Helpers (camelCase style)
# ----------------------------
def buildSampleData() -> pd.DataFrame:
    return pd.DataFrame({
        "loc_country": ["Guatemala", "Mexico", "Mexico", "Brazil", "Guatemala", "Brazil"],
        "roast":       ["Light",     "Medium", "Dark",   "Medium", "Light",    "Dark"  ],
        "rating":      [88.4,         91.0,     90.5,     85.3,     87.9,      83.7   ]
    })

def inferCategoricalColumns(df: pd.DataFrame):
    # Prefer object/category columns; if none, fall back to all non-target columns.
    cats = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    return cats

def trainModel(df: pd.DataFrame, xCols: list[str], yCol: str, estimate: str, defaultCategory: str | None):
    gm = GroupEstimate(estimate=estimate)
    gm.fit(df[xCols], df[yCol], default_category=defaultCategory)
    return gm

def makePredictionFrame(xCols: list[str]) -> pd.DataFrame:
    # Empty starter rows for user to fill in via data editor
    return pd.DataFrame([{c: "" for c in xCols} for _ in range(3)])


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="GroupEstimate (mean/median) by categories", layout="wide")
st.title("GroupEstimate: mean/median by categorical groups")

with st.sidebar:
    st.header("1) Load Data")
    source = st.radio("Choose a data source:", ["Sample coffee data", "Upload CSV"], index=0)
    if source == "Upload CSV":
        up = st.file_uploader("Upload a CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
        else:
            st.info("Upload a CSV to continue, or switch to sample data.")
            df = buildSampleData()
    else:
        df = buildSampleData()

    st.caption("Preview:")
    st.dataframe(df.head(), use_container_width=True)

    st.header("2) Configure")
    guessCats = inferCategoricalColumns(df)
    yCol = st.selectbox("Target column (numeric y)", [c for c in df.columns if c not in guessCats] or df.columns, index=0)
    xCols = st.multiselect("Grouping columns (categorical X)", guessCats if guessCats else [c for c in df.columns if c != yCol],
                           default=guessCats[:2] if guessCats else [])
    estimate = st.radio("Estimate", ["mean", "median"], horizontal=True)

    defaultCategory = st.selectbox(
        "Optional fallback category (for missing combos)",
        ["(none)"] + xCols,
        index=0
    )
    defaultCategory = None if defaultCategory == "(none)" else defaultCategory

    trainBtn = st.button("Train", use_container_width=True)

if "model" not in st.session_state:
    st.session_state.model = None
if "xCols" not in st.session_state:
    st.session_state.xCols = None

# Train model
if trainBtn:
    if not xCols:
        st.error("Please choose at least one grouping (X) column.")
    elif yCol is None:
        st.error("Please choose a target (y) column.")
    else:
        try:
            st.session_state.model = trainModel(df, xCols, yCol, estimate, defaultCategory)
            st.session_state.xCols = xCols
            st.success("Model trained successfully.")
        except Exception as e:
            st.exception(e)

# Main area: prediction editor
st.header("3) Predict")
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Enter observations to predict")
    if st.session_state.xCols:
        xColsCurr = st.session_state.xCols
    else:
        xColsCurr = xCols if xCols else inferCategoricalColumns(df)[:2]

    if not xColsCurr:
        st.info("Select grouping columns in the sidebar and press **Train**.")
    else:
        if "predict_df" not in st.session_state or st.session_state.predict_df is None:
            st.session_state.predict_df = makePredictionFrame(xColsCurr)

        # Ensure editor has current columns
        missing = set(xColsCurr) - set(st.session_state.predict_df.columns)
        if missing:
            for m in missing:
                st.session_state.predict_df[m] = ""
            st.session_state.predict_df = st.session_state.predict_df[xColsCurr]

        st.caption("Fill in the category values (strings) below. Add/remove rows as needed.")
        edited = st.data_editor(
            st.session_state.predict_df,
            num_rows="dynamic",
            use_container_width=True,
            key="pred_editor"
        )
        st.session_state.predict_df = edited

        predictBtn = st.button("Predict on rows above", type="primary", use_container_width=True)

with right:
    st.subheader("Results")
    if predictBtn:
        model = st.session_state.model
        if model is None:
            st.error("Train the model first from the sidebar.")
        else:
            # Convert empty strings to NaN so they won't match any seen group
            Xnew = st.session_state.predict_df.copy()
            Xnew = Xnew.replace("", np.nan)

            try:
                preds = model.predict(Xnew)
                out = Xnew.copy()
                out["estimate"] = preds
                nMissing = int(np.isnan(preds).sum())
                if nMissing > 0:
                    st.warning(f"{nMissing} observation(s) fell into unseen group(s). "
                               f"Use a fallback category in the sidebar to reduce NaNs.")
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="group_estimate_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.exception(e)

st.markdown("---")
with st.expander("Notes & Tips"):
    st.markdown(
        """
- **X** should be categorical columns (strings or `category` dtypes).  
- **y** must be numeric with **no missing values** and the same number of rows as **X**.  
- If an unseen combination appears at prediction time, the app returns **NaN**.  
  - Set a **fallback category** (one of the X columns) to use that single column's mean/median when the full combo is missing.
- Example: Group by `loc_country` and `roast`, predict average coffee `rating`.
        """
    )
