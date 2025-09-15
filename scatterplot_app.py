import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Scatterplot with Regression", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def to_datetime_safe(series: pd.Series, *, dayfirst: bool, excel_serial: bool):
    """Try to convert a column to datetime robustly."""
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    if excel_serial and pd.api.types.is_numeric_dtype(s):
        s = s.astype("float64")
        with pd.option_context("mode.use_inf_as_na", True):
            s = s.where((s > 20000) & (s < 90000))
        base = pd.Timestamp("1899-12-30")
        return base + pd.to_timedelta(s, unit="D")

    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)

def finite_std(x: pd.Series):
    std = x.std()
    return 1e-12 if pd.isna(std) or std == 0 else std

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your file",
    type=["csv", "xlsx", "xlsm"]  # <--- added xlsm support
)

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            # Works for both .xlsx and .xlsm
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox(
                "Select a sheet from the Excel file",
                xls.sheet_names,
                key="sheet_select"
            )
            df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.write("### Data Preview")
    st.dataframe(df.head())
    st.caption("Tip: If dates look wrong, use the toggles below to control parsing.")

    # ----------------------------
    # Date filter UI
    # ----------------------------
    st.subheader("Date Filter")
    enable_date = st.checkbox("Enable date filter", value=False, key="enable_date")

    if enable_date:
        date_col = st.selectbox(
            "Select a column to treat as a date",
            options=list(df.columns),
            index=0,
            key="date_col_select",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            dayfirst = st.checkbox("Day-first (DD/MM/YYYY)?", value=True, key="dayfirst")
        with col2:
            excel_serial = st.checkbox("Excel serial numbers?", value=False, key="excel_serial")
        with col3:
            drop_na_dates = st.checkbox("Drop rows with invalid dates", value=True, key="drop_na_dates")

        parsed_dates = to_datetime_safe(df[date_col], dayfirst=dayfirst, excel_serial=excel_serial)
        if drop_na_dates:
            mask_valid = parsed_dates.notna()
            if not mask_valid.any():
                st.error("No valid dates after parsing. Try toggling Day-first/Excel serial.")
                st.stop()
            df = df.loc[mask_valid].copy()
            parsed_dates = parsed_dates.loc[mask_valid]

        df["_dt_filter_col"] = parsed_dates

        if df.empty:
            st.error("All rows were dropped after date parsing.")
            st.stop()

        min_ts = pd.to_datetime(df["_dt_filter_col"].min())
        max_ts = pd.to_datetime(df["_dt_filter_col"].max())

        if pd.isna(min_ts) or pd.isna(max_ts):
            st.error("Could not determine min/max date. Check the parsing options.")
            st.stop()

        min_date = min_ts.date()
        max_date = max_ts.date()
        if min_date > max_date:
            min_date, max_date = max_date, min_date

        date_range = st.date_input(
            "Choose date range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            df = df[(df["_dt_filter_col"] >= start_date) & (df["_dt_filter_col"] <= end_date)].copy()
        else:
            st.warning("Please select a start and end date.")
            st.stop()

        if df.empty:
            st.warning("No rows in the selected date range.")
            st.stop()

    # ----------------------------
    # X/Y selection
    # ----------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for X/Y.")
        st.stop()

    x_col = st.selectbox("Select X-axis column (numeric)", numeric_cols, key="x_select")
    y_col = st.selectbox("Select Y-axis column (numeric)", numeric_cols, key="y_select")

    # ----------------------------
    # Coloring categories
    # ----------------------------
    non_numeric_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != "_dt_filter_col"]

    cat_cols = st.multiselect(
        "Select categorical columns to color by",
        non_numeric_cols,
        key="cat_cols"
    )

    if cat_cols:
        df["__combined_color__"] = df[cat_cols].astype(str).agg(" | ".join, axis=1)
    else:
        df["__combined_color__"] = "All Data"

    # ----------------------------
    # Drop NA on axes
    # ----------------------------
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        st.warning("All rows dropped after removing NA for selected X/Y.")
        st.stop()

    # ----------------------------
    # Outlier removal (3σ)
    # ----------------------------
    x_std = finite_std(df[x_col])
    y_std = finite_std(df[y_col])
    x_mean = df[x_col].mean()
    y_mean = df[y_col].mean()

    mask = (
        (df[x_col] >= x_mean - 3 * x_std) & (df[x_col] <= x_mean + 3 * x_std) &
        (df[y_col] >= y_mean - 3 * y_std) & (df[y_col] <= y_mean + 3 * y_std)
    )
    df = df.loc[mask].copy()
    if df.empty:
        st.warning("All rows removed by outlier filter (3σ).")
        st.stop()

    # ----------------------------
    # Category filter for regression
    # ----------------------------
    categories = df["__combined_color__"].unique().tolist()
    selected_categories = st.multiselect(
        "Filter categories for regression (lines drawn only for selected categories)",
        categories,
        key="reg_filter"
    )

    plot_df = df[df["__combined_color__"].isin(selected_categories)] if selected_categories else df

    # ----------------------------
    # Plot
    # ----------------------------
    fig = px.scatter(plot_df, x=x_col, y=y_col, color="__combined_color__", opacity=0.7)
    eq_texts = []

    # If no category is selected, draw the global regression line
    if not selected_categories and len(df) > 1:
        X_all = df[[x_col]].values
        y_all = df[y_col].values
        model_all = LinearRegression().fit(X_all, y_all)
        y_pred_all = model_all.predict(X_all)
        slope_all = model_all.coef_[0]
        intercept_all = model_all.intercept_
        r2_all = r2_score(y_all, y_pred_all)

        line_x_all = np.linspace(df[x_col].min(), df[x_col].max(), 100).reshape(-1, 1)
        line_y_all = model_all.predict(line_x_all)

        fig.add_trace(go.Scatter(
            x=line_x_all.flatten(),
            y=line_y_all,
            mode="lines",
            name="Regression (All Data)",
            line=dict(color="black", width=3, dash="dot")
        ))

        eq_texts.append(f"**All Data:** y = {slope_all:.3f}x + {intercept_all:.3f} (R² = {r2_all:.3f})")

    # Draw individual regression lines only for selected categories
    for category in (selected_categories or []):
        sub_df = df[df["__combined_color__"] == category]
        if len(sub_df) > 1:
            X = sub_df[[x_col]].values
            y = sub_df[y_col].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            slope = model.coef_[0]
            intercept = model.intercept_
            r2 = r2_score(y, y_pred)

            line_x = np.linspace(sub_df[x_col].min(), sub_df[x_col].max(), 100).reshape(-1, 1)
            line_y = model.predict(line_x)

            fig.add_trace(go.Scatter(
                x=line_x.flatten(),
                y=line_y,
                mode="lines",
                name=f"Regression ({category})",
                line=dict(color="red", width=2)  # choose any color here
            ))

            eq_texts.append(f"**{category}:** y = {slope:.5f}x + {intercept:.3f} (R² = {r2:.3f})")

    st.plotly_chart(fig, use_container_width=True)

    if eq_texts:
        st.markdown("### Regression Equations (after outlier removal)")
        for eq in eq_texts:
            st.markdown(eq)
    else:
        st.info("No regression equations to display.")

else:
    st.info("Upload a CSV, XLSX, or XLSM file to begin.")



