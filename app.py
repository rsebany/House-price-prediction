# Streamlit app, House Price Predictor and results showcase.
# Run with: streamlit run app.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ---------- custom transformer (must match the training notebook, or joblib.load fails) ----------
from sklearn.base import BaseEstimator, TransformerMixin

class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def __init__(self, col="LotFrontage", group_col="Neighborhood"):
        self.col = col
        self.group_col = group_col

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.col in X.columns and self.group_col in X.columns:
            self.global_median_ = X[self.col].median()
            self.group_medians_ = X.groupby(self.group_col)[self.col].median()
        else:
            self.global_median_ = None
            self.group_medians_ = None
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns and self.group_medians_ is not None:
            fill_values = X[self.group_col].map(self.group_medians_).fillna(self.global_median_)
            X[self.col] = X[self.col].fillna(fill_values)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)

# ---------- feature engineering (same logic as the training notebook) ----------

NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
NA_MEANS_ZERO = [
    "GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]

def fill_semantic_na(dataframe):
    df = dataframe.copy()
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in NA_MEANS_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS = {
    "ExterQual": QUALITY, "ExterCond": QUALITY, "BsmtQual": QUALITY,
    "BsmtCond": QUALITY, "HeatingQC": QUALITY, "KitchenQual": QUALITY,
    "FireplaceQu": QUALITY, "GarageQual": QUALITY, "GarageCond": QUALITY,
    "PoolQC": QUALITY,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3,
                   "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Street": {"Grvl": 0, "Pave": 1},
    "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

def engineer_features(dataframe):
    df = dataframe.copy()
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["QualxSF"] = df["OverallQual"] * df["TotalSF"]
    df["LivAreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"].clip(lower=1)
    df["GarageScore"] = df["GarageArea"] * df["GarageCars"]
    df = df.drop(columns=[c for c in ["Id", "Utilities"] if c in df.columns])
    return df

def log_transform(dataframe, columns):
    df = dataframe.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df

# ---------- artifacts ----------

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "house_price_model.joblib")
    meta = json.loads((BASE / "app_meta.json").read_text(encoding="utf-8"))
    results_path = BASE / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else None
    return model, meta, results

model, meta, results = load_artifacts()

def find_csv(name):
    candidates = [
        BASE / name,
        Path("data") / name,
        Path("archive/data") / name,
        Path("/kaggle/input/competitions/house-prices-advanced-regression-techniques") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob(name))
        if matches:
            return matches[0]
    return None

def load_csv(name):
    p = BASE / name
    return pd.read_csv(p) if p.exists() else None

def money(v):
    return f"${v:,.0f}"

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model comparison", "Final model evaluation", "Price predictor", "Submission"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Tuned ElasticNet on log(SalePrice) | Ames Housing | trained on 1,458 labelled rows, refit on all data for the submission.")

# ================= Overview =================
if page == "Overview":
    st.title("House Prices, Advanced Regression Techniques")
    st.caption("Kaggle competition | Ames, Iowa | 1,460 labelled rows, 79 features | my Practical Work 1 pipeline")

    train_path = find_csv("train.csv")
    if results:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Labelled rows", f"{results['train_rows']:,}")
        c2.metric("Features used", f"{results['n_features']}")
        c3.metric("Mean sale price", money(results["target_mean"]))
        c4.metric("Median sale price", money(results["target_median"]))
        st.caption(f"Target is right-skewed (skew = {results['target_skew']:.2f}), so the model trains on log1p(SalePrice) and predictions are converted back.")

    if train_path is not None:
        df_raw = pd.read_csv(train_path)
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_raw["SalePrice"], bins=40, color="steelblue", edgecolor="white")
            ax.set_title("Sale price distribution")
            ax.set_xlabel("SalePrice ($)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            medians = df_raw.groupby("OverallQual")["SalePrice"].median()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(medians.index.astype(str), medians.values, color="steelblue")
            ax.set_title("Median price by overall quality")
            ax.set_xlabel("OverallQual")
            ax.set_ylabel("Median SalePrice ($)")
            st.pyplot(fig)
            plt.close(fig)
        st.caption("OverallQual is the strongest single predictor (Spearman r = 0.80), followed by TotalSF (0.82 with the engineered feature) and GrLivArea (0.73).")
    else:
        st.info("train.csv not found next to the app; running in artifact-only mode.")

# ================= Model comparison =================
elif page == "Model comparison":
    st.title("Model comparison")
    st.caption("5-fold cross-validation on the training set, hold-out evaluated once at the end. Lower is better for RMSE/MAE.")
    if not results:
        st.warning("results.json not found; run the notebook first.")
    else:
        models_df = pd.DataFrame(results["models"]).sort_values("CV RMSE").reset_index(drop=True)
        show = models_df.copy()
        for col in ["CV RMSE", "CV RMSE std", "CV MAE", "Hold-out RMSE", "Hold-out MAE"]:
            show[col] = show[col].map(money)
        for col in ["CV R2", "Hold-out R2", "Hold-out MAPE"]:
            show[col] = show[col].map(lambda v: f"{v:.3f}" if col != "Hold-out MAPE" else f"{v:.1%}")
        show["Fit time (s)"] = show["Fit time (s)"].map(lambda v: f"{v:.1f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ordered = models_df.sort_values("CV RMSE")
        ax.barh(ordered["Model"], ordered["CV RMSE"], color="steelblue")
        ax.invert_yaxis()
        ax.set_xlabel("CV RMSE ($)")
        ax.set_title("Cross-validated RMSE by model")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Tuned ElasticNet vs stacking ensemble vs linear baseline")
        base = models_df[models_df["Model"] == "Linear Regression"].iloc[0]
        comparison = pd.DataFrame([
            {"Model": "ElasticNet (tuned, selected by CV)",
             "CV RMSE": money(results["tuned_cv_rmse"]),
             "Hold-out RMSE": money(results["final_metrics"]["RMSE"]),
             "Hold-out MAE": money(results["final_metrics"]["MAE"]),
             "Hold-out R2": f"{results['final_metrics']['R2']:.4f}"},
            {"Model": "Stacking (Ridge + RF + GB)",
             "CV RMSE": money(results["stacking"]["cv_rmse"]),
             "Hold-out RMSE": money(results["stacking"]["holdout_rmse"]),
             "Hold-out MAE": money(results["stacking"]["holdout_mae"]),
             "Hold-out R2": f"{results['stacking']['holdout_r2']:.4f}"},
            {"Model": "Linear Regression (baseline)",
             "CV RMSE": money(base["CV RMSE"]),
             "Hold-out RMSE": money(base["Hold-out RMSE"]),
             "Hold-out MAE": money(base["Hold-out MAE"]),
             "Hold-out R2": f"{base['Hold-out R2']:.4f}"},
        ])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown(
            "- **Why ElasticNet and not the stacking ensemble:** stacking was marginally better on the hold-out "
            "(R2 0.937 vs 0.936), but worse on cross-validation ($21,589 vs $21,482). The difference is far inside "
            "noise, so the simpler, better-CV model was kept.\n"
            "- **Tuned parameters:** " + ", ".join(f"{k} = {v:.4g}" for k, v in results.get("tuned_params", {}).items())
        )

# ================= Final model evaluation =================
elif page == "Final model evaluation":
    st.title("Final model evaluation (hold-out, 292 houses)")
    if results:
        m = results["final_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", money(m["MAE"]))
        c2.metric("RMSE", money(m["RMSE"]))
        c3.metric("R2", f"{m['R2']:.4f}")
        c4.metric("MAPE", f"{m['MAPE']:.1%}")

    holdout = load_csv("holdout_predictions.csv")
    if holdout is None:
        st.warning("holdout_predictions.csv not found; run the notebook first.")
    else:
        residuals = holdout["actual"] - holdout["predicted"]
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(holdout["actual"], holdout["predicted"], alpha=0.5, color="steelblue")
            lims = [holdout[["actual", "predicted"]].min().min(), holdout[["actual", "predicted"]].max().max()]
            ax.plot(lims, lims, color="crimson", linestyle="--", linewidth=1)
            ax.set_xlabel("Actual price ($)")
            ax.set_ylabel("Predicted price ($)")
            ax.set_title("Actual vs predicted")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(residuals, bins=40, color="steelblue", edgecolor="white")
            ax.axvline(0, color="crimson", linestyle="--", linewidth=1)
            ax.set_xlabel("Residual (actual - predicted, $)")
            ax.set_ylabel("Count")
            ax.set_title("Residual distribution")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Error by price quintile")
        hp = holdout.copy()
        hp["abs_error"] = (hp["actual"] - hp["predicted"]).abs()
        hp["band"] = pd.qcut(hp["actual"], q=5, labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 priciest"])
        by_band = hp.groupby("band", observed=True).agg(
            count=("actual", "size"),
            MAE=("abs_error", "mean"),
        ).reset_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            show = by_band.copy()
            show["MAE"] = show["MAE"].map(money)
            st.dataframe(show, hide_index=True, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(by_band["band"], by_band["MAE"], color="steelblue")
            ax.set_ylabel("MAE ($)")
            ax.set_title("The model underpredicts the priciest quintile")
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Residuals are heteroscedastic: error grows in the top quintile. Expected for a linear model on a log-target; a weighted loss would be the next step.")

# ================= Price predictor =================
elif page == "Price predictor":
    st.title("Price predictor")
    st.caption("Enter the main characteristics. The remaining ~63 raw features are filled with typical-house values from the training data, and the full tuned model scores the request.")

    template = meta["template"]
    qualities = meta["qualities"]
    year_min, year_max = meta["year_min"], meta["year_max"]

    with st.form("house_features"):
        left, right = st.columns(2)
        with left:
            overall_qual = st.slider("Overall quality (1 to 10)", 1, 10, int(template["OverallQual"]))
            gr_liv_area = st.number_input("Living area above ground (sq ft)", 300, 6000, int(template["GrLivArea"]), step=50)
            total_bsmt = st.number_input("Basement area (sq ft)", 0, 3000, int(template["TotalBsmtSF"]), step=50)
            garage_cars = st.slider("Garage capacity (cars)", 0, 4, int(template["GarageCars"]))
            garage_area = st.number_input("Garage area (sq ft)", 0, 1500, int(template["GarageArea"]), step=25)
            year_built = st.number_input("Year built", year_min, year_max, int(template["YearBuilt"]))
            year_remod = st.number_input("Year remodelled", year_min, year_max, int(template["YearRemodAdd"]))
            lot_area = st.number_input("Lot area (sq ft)", 1000, 100000, int(template["LotArea"]), step=500)
        with right:
            neighborhood = st.selectbox("Neighborhood", meta["neighborhoods"], index=meta["neighborhoods"].index(template["Neighborhood"]))
            kitchen_qual = st.selectbox("Kitchen quality", qualities, index=qualities.index(template["KitchenQual"]))
            exter_qual = st.selectbox("Exterior quality", qualities, index=qualities.index(template["ExterQual"]))
            bsmt_qual = st.selectbox("Basement quality", qualities, index=qualities.index(template["BsmtQual"]))
            full_bath = st.slider("Full bathrooms", 0, 4, int(template["FullBath"]))
            tot_rms = st.slider("Rooms above ground", 2, 14, int(template["TotRmsAbvGrd"]))
            fireplace = st.selectbox("Fireplace quality", qualities, index=qualities.index(template["FireplaceQu"]))
            central_air = st.selectbox("Central air", ["Y", "N"], index=["Y", "N"].index(template["CentralAir"]))
        submitted = st.form_submit_button("Estimate the price")

    if submitted:
        row = dict(template)
        row.update({
            "OverallQual": overall_qual, "GrLivArea": gr_liv_area, "TotalBsmtSF": total_bsmt,
            "GarageCars": garage_cars, "GarageArea": garage_area,
            "YearBuilt": year_built, "YearRemodAdd": year_remod, "LotArea": lot_area,
            "FullBath": full_bath, "TotRmsAbvGrd": tot_rms,
            "Neighborhood": neighborhood, "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual, "BsmtQual": bsmt_qual,
            "FireplaceQu": fireplace, "CentralAir": central_air,
        })
        x = pd.DataFrame([row])
        x = fill_semantic_na(x)
        x = engineer_features(x)
        x = log_transform(x, meta["skewed_features"])
        x = x[meta["feature_columns"]]
        price = float(model.predict(x)[0])
        st.metric("Estimated sale price", money(price))
        st.caption(f"Typical error on unseen houses: about {money(meta['holdout_mae'])} MAE (R2 = {meta['holdout_r2']:.3f}). Estimate only, not an appraisal.")
        with st.expander("What the app filled in for you"):
            st.json({k: v for k, v in template.items() if k not in row or row[k] == template[k]})

# ================= Submission =================
elif page == "Submission":
    st.title("Kaggle submission")
    st.caption("submission.csv, generated with the tuned model refit on all 1,458 labelled rows.")
    submission = load_csv("submission.csv")
    if submission is None:
        st.warning("submission.csv not found; run the notebook first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(submission):,}")
        c2.metric("Min", money(submission["SalePrice"].min()))
        c3.metric("Mean", money(submission["SalePrice"].mean()))
        c4.metric("Max", money(submission["SalePrice"].max()))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(submission["SalePrice"], bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("Predicted SalePrice ($)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of predicted prices")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(submission.head(25), hide_index=True, use_container_width=True)
    st.download_button(
        "Download submission.csv",
        submission.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Add a submission CSV")
    st.caption("Upload a submission-style CSV (Id, SalePrice) — e.g. sample_submission.csv or a file from another run — to validate it, compare it with the current submission.csv, and download a clean copy for the Kaggle upload.")

    upload = st.file_uploader("Choose a CSV file", type=["csv"], key="submission_upload")
    if upload is None:
        st.info("Nothing uploaded yet. The current submission.csv is shown above.")
    else:
        try:
            up = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            up = None
        if up is not None and ("Id" not in up.columns or "SalePrice" not in up.columns):
            st.error(f"The file must contain 'Id' and 'SalePrice' columns; found: {list(up.columns)}")
            up = None
        if up is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(up):,}")
            c2.metric("NaN prices", int(up["SalePrice"].isna().sum()))
            c3.metric("Price range", f"${up['SalePrice'].min():,.0f} – ${up['SalePrice'].max():,.0f}")
            c4.metric("Duplicate Ids", int(up["Id"].duplicated().sum()))

            checks = [
                ("1459 rows (test set size)", len(up) == 1459),
                ("No missing SalePrice", int(up["SalePrice"].isna().sum()) == 0),
                ("No negative prices", bool((up["SalePrice"] >= 0).all())),
                ("No duplicate Ids", int(up["Id"].duplicated().sum()) == 0),
            ]
            current = load_csv("submission.csv")
            if current is not None:
                overlap = len(set(up["Id"]).intersection(set(current["Id"])))
                checks.append((f"Ids match current submission ({overlap}/{len(current)})", overlap == len(current)))
            for label, ok in checks:
                st.write(("✅" if ok else "❌"), label)

            if current is not None and overlap == len(current):
                merged = current.merge(up, on="Id", suffixes=("_notebook", "_uploaded"))
                diff = (merged["SalePrice_notebook"] - merged["SalePrice_uploaded"]).abs()
                corr = merged["SalePrice_notebook"].corr(merged["SalePrice_uploaded"])
                st.caption(
                    f"Compared with current submission.csv over {len(merged)} Ids — "
                    f"correlation: {corr:.4f}, mean |difference|: {money(diff.mean())}, "
                    f"max |difference|: {money(diff.max())}"
                )
                worst = merged.loc[diff.idxmax(), "Id"]
                st.caption(f"Largest disagreement: Id {int(worst)} — {money(merged.loc[diff.idxmax(), 'SalePrice_notebook'])} vs {money(merged.loc[diff.idxmax(), 'SalePrice_uploaded'])}")
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.hist(current["SalePrice"], bins=40, alpha=0.6, label="current", color="steelblue")
                ax.hist(up["SalePrice"], bins=40, alpha=0.6, label="uploaded", color="darkorange")
                ax.set_xlabel("SalePrice ($)")
                ax.set_ylabel("Count")
                ax.set_title("Predicted price distribution, current vs uploaded")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            st.download_button(
                "Download this submission CSV",
                up.to_csv(index=False).encode("utf-8"),
                file_name="uploaded_submission.csv",
                mime="text/csv",
                use_container_width=True,
            )
