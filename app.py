"""
app.py
Wafer Yield Intelligence Platform
Enterprise Root-Cause Analysis & Risk Simulation System
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import (
    load_and_validate,
    compute_summary,
    train_risk_model,
    predict_new_lot,
    get_feature_columns,
    compute_spc_limits,
    generate_wafer_map,
    batch_predict_lots,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wafer Yield Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Enterprise Light Design System (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Clean White Styling */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #111827;
    }
    
    /* Main container adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }
    
    /* Top Header Banner */
    .enterprise-header {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 1.2rem;
        margin-bottom: 1.8rem;
    }
    .enterprise-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #0f62fe;
        background: #edf5ff;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .enterprise-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .enterprise-subtitle {
        font-size: 0.92rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Professional KPI Card */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.35rem;
    }
    .kpi-sub-positive {
        color: #059669;
        font-weight: 600;
    }
    .kpi-sub-alert {
        color: #dc2626;
        font-weight: 600;
    }

    /* Section Card */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1rem;
    }

    /* Status Badges */
    .status-pill {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
    }
    .pill-low {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .pill-medium {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .pill-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .pill-neutral {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
    }

    /* Simulation Output Box */
    .sim-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-header {
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .sidebar-desc {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    
    /* Streamlit button and input refinements */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Plotly Standard Clean White Theme Helper
# ---------------------------------------------------------------------------
def apply_clean_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#1e293b", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto", size=12, color="#475569"),
        margin=dict(l=45, r=25, t=45 if title else 20, b=40),
        height=height,
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        gridwidth=1,
        linecolor="#cbd5e1",
        linewidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        gridwidth=1,
        linecolor="#cbd5e1",
        linewidth=1,
        zeroline=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: System Configuration & Data Source
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <div class="enterprise-tag">FAB 12 NODE</div>
        <h3 class="sidebar-title">Wafer Yield Platform</h3>
        <p class="sidebar-desc">Process monitoring & lot excursion intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Lot Process CSV",
    type=["csv"],
    help="Upload a CSV with Temperature, Pressure, Power, Defects, Yield, and optional GasFlow columns.",
)

use_demo = st.sidebar.checkbox(
    "Load standard demo dataset",
    value=uploaded_file is None,
    help="Loads synthetic 500-lot baseline dataset from wafer_data.csv",
)

if uploaded_file is not None and not use_demo:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as err:
        st.sidebar.error(f"Error reading uploaded CSV: {err}")
        st.stop()
elif use_demo:
    try:
        raw_df = pd.read_csv("wafer_data.csv")
    except Exception as err:
        st.sidebar.error(f"Error reading wafer_data.csv: {err}")
        st.stop()
else:
    st.info("Please upload a CSV file or check 'Load standard demo dataset' in the sidebar.")
    st.stop()

df, warnings = load_and_validate(raw_df)
for w in warnings:
    st.sidebar.warning(w)

if df.empty:
    st.error("No valid numeric records remained after validation. Please check file format.")
    st.stop()

# Model training cached per dataset
@st.cache_resource(show_spinner="Training predictive ensemble model...")
def get_trained_model(data: pd.DataFrame):
    return train_risk_model(data)

trained = get_trained_model(df)
summary = compute_summary(df)

# Sidebar System Metadata
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Module Selection",
    [
        "Overview Dashboard",
        "Yield & Defect Analytics",
        "Root Cause Analysis",
        "Lot Risk Simulator",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Specs")
st.sidebar.markdown(
    f"""
    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
        <div><b>Dataset Size:</b> {summary['total_lots']} lots</div>
        <div><b>Model Type:</b> Random Forest Regressor</div>
        <div><b>Model Fit (R²):</b> {trained.get('r2', 'N/A')}</div>
        <div><b>Mean Error:</b> {trained.get('mae', 'N/A')}% yield</div>
        <div><b>Status:</b> Operational</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Download clean sample CSV template
st.sidebar.markdown("---")
sample_csv = df.head(5).to_csv(index=False)
st.sidebar.download_button(
    label="Download Sample CSV Template",
    data=sample_csv,
    file_name="sample_wafer_lots.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Global Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="enterprise-header">
        <div class="enterprise-tag">SEMICONDUCTOR MANUFACTURING INTELLIGENCE</div>
        <h1 class="enterprise-title">Wafer Lot Yield Platform</h1>
        <div class="enterprise-subtitle">Statistical process monitoring, automated root-cause attribution, and planned lot risk simulation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MODULE 1: Overview Dashboard
# ---------------------------------------------------------------------------
if page == "Overview Dashboard":
    # Top KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    
    yield_delta = summary["average_yield"] - summary["target_yield"]
    delta_class = "kpi-sub-positive" if yield_delta >= 0 else "kpi-sub-alert"
    delta_prefix = "+" if yield_delta >= 0 else ""

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Monitored Lots</div>
                <div class="kpi-value">{summary['total_lots']}</div>
                <div class="kpi-sub">Historical production runs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Yield</div>
                <div class="kpi-value">{summary['average_yield']}%</div>
                <div class="kpi-sub <span class='{delta_class}'>{delta_prefix}{yield_delta:.1f}% vs 90.0% Target</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        excursion_rate = round((summary["low_yield_lots"] / summary["total_lots"]) * 100, 1)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Low-Yield Excursions</div>
                <div class="kpi-value">{summary['low_yield_lots']}</div>
                <div class="kpi-sub"><span class="kpi-sub-alert">{excursion_rate}% of runs (≤ {summary['low_yield_threshold']}%)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Mean Defect Count</div>
                <div class="kpi-value">{summary['avg_defects']}</div>
                <div class="kpi-sub">Classified dies per wafer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Yield Range (Min / Max)</div>
                <div class="kpi-value">{summary['worst_yield']}% / {summary.get('best_yield', 99.0)}%</div>
                <div class="kpi-sub">Critical lot: {summary['worst_lot']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Visual Analytics (Yield Distribution + Interactive Wafer Die Map)
    col_dist, col_wafer = st.columns([1.1, 1.0])

    with col_dist:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Yield Distribution & Specification Limits</div>
                <div class="section-desc">Historical yield distribution relative to lower quartile threshold and mean specification target.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_dist = px.histogram(
            df,
            x="Yield",
            nbins=35,
            color_discrete_sequence=["#0f62fe"],
            opacity=0.85,
        )
        # Add Low-Yield Threshold vertical line
        fig_dist.add_vline(
            x=summary["low_yield_threshold"],
            line_dash="dash",
            line_color="#da1e28",
            line_width=1.5,
            annotation_text=f"25th Pct Threshold ({summary['low_yield_threshold']}%)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#da1e28"),
        )
        # Add Mean Yield line
        fig_dist.add_vline(
            x=summary["average_yield"],
            line_dash="dot",
            line_color="#007d79",
            line_width=1.5,
            annotation_text=f"Mean ({summary['average_yield']}%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#007d79"),
        )
        fig_dist.update_layout(xaxis_title="Yield Percentage (%)", yaxis_title="Lot Count")
        fig_dist = apply_clean_theme(fig_dist, height=360)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_wafer:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Spatial Wafer Die Map Inspection</div>
                <div class="section-desc">Physical 300mm wafer die topology simulation for selected lot, showing edge vs center defect concentration.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Lot selector for wafer map
        lot_options = df["Lot"].tolist() if "Lot" in df.columns else [f"Lot-{i+1}" for i in range(len(df))]
        default_idx = lot_options.index(summary["worst_lot"]) if summary["worst_lot"] in lot_options else 0
        selected_lot = st.selectbox("Select Lot for Spatial Die Map", lot_options, index=default_idx)

        selected_row = df[df["Lot"] == selected_lot].iloc[0] if "Lot" in df.columns else df.iloc[0]
        wafer_data = generate_wafer_map(
            selected_lot,
            defect_count=int(selected_row["Defects"]),
            yield_pct=float(selected_row["Yield"]),
        )

        pass_count = int((wafer_data["status"] == "Pass").sum())
        defect_count_dies = int((wafer_data["status"] == "Defect").sum())
        die_yield = round((pass_count / len(wafer_data)) * 100, 1)

        fig_wafer = px.scatter(
            wafer_data,
            x="x",
            y="y",
            color="status",
            color_discrete_map={"Pass": "#10b981", "Defect": "#ef4444"},
            hover_data={"x": True, "y": True, "status": True, "r": ":.2f"},
        )
        fig_wafer.update_traces(marker=dict(size=11, symbol="square", opacity=0.9))
        
        # Add circular wafer outline boundary
        fig_wafer.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=-8.3,
            y0=-8.3,
            x1=8.3,
            y1=8.3,
            line=dict(color="#94a3b8", width=1.5, dash="solid"),
        )
        fig_wafer.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-9.5, 9.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-9.5, 9.5], scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        )
        fig_wafer = apply_clean_theme(fig_wafer, height=310)
        st.plotly_chart(fig_wafer, use_container_width=True)

        st.markdown(
            f"""
            <div style="display: flex; gap: 1rem; font-size: 0.8rem; color: #475569; background: #f8fafc; padding: 6px 12px; border-radius: 4px; border: 1px solid #e2e8f0;">
                <div><b>Lot:</b> {selected_lot}</div>
                <div><b>Reported Yield:</b> {selected_row['Yield']}%</div>
                <div><b>Good Dies:</b> {pass_count} / {len(wafer_data)}</div>
                <div><b>Defective Dies:</b> {defect_count_dies}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Recent Lots Log Table
    st.markdown(
        """
        <div class="section-card" style="margin-top: 1.5rem;">
            <div class="section-title">Production Lot Execution Records</div>
            <div class="section-desc">Tabular process sensor logs and yield outputs with automated classification tags.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 2])
    with col_filter1:
        status_filter = st.selectbox("Filter by Yield Status", ["All Records", "Normal Yield (> 25th Pct)", "Excursion Lots (≤ 25th Pct)"])
    with col_filter2:
        lot_sort = st.selectbox("Sort Order", ["Latest Run First", "Lowest Yield First", "Highest Defect First"])

    filtered_df = df.copy()
    if status_filter == "Normal Yield (> 25th Pct)":
        filtered_df = filtered_df[filtered_df["Yield"] > summary["low_yield_threshold"]]
    elif status_filter == "Excursion Lots (≤ 25th Pct)":
        filtered_df = filtered_df[filtered_df["Yield"] <= summary["low_yield_threshold"]]

    if lot_sort == "Lowest Yield First":
        filtered_df = filtered_df.sort_values("Yield", ascending=True)
    elif lot_sort == "Highest Defect First":
        filtered_df = filtered_df.sort_values("Defects", ascending=False)
    else:
        if "Lot" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("Lot", ascending=False)

    st.dataframe(
        filtered_df.head(25),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# MODULE 2: Yield & Defect Analytics
# ---------------------------------------------------------------------------
elif page == "Yield & Defect Analytics":
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Statistical Process Control (SPC) Run Chart</div>
            <div class="section-desc">Run-sequence chart with 3-Sigma upper and lower control limits (UCL / LCL) and 10-lot rolling mean trendline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spc = compute_spc_limits(df["Yield"])
    df_sorted = df.sort_values("Lot") if "Lot" in df.columns else df.copy()
    df_sorted["Rolling_Mean"] = df_sorted["Yield"].rolling(window=10, min_periods=1).mean().round(2)

    fig_spc = go.Figure()

    # Yield points
    fig_spc.add_trace(
        go.Scatter(
            x=df_sorted["Lot"],
            y=df_sorted["Yield"],
            mode="lines+markers",
            name="Lot Yield",
            line=dict(color="#0f62fe", width=1.5),
            marker=dict(size=4, color="#0f62fe"),
        )
    )

    # 10-Lot Rolling Mean
    fig_spc.add_trace(
        go.Scatter(
            x=df_sorted["Lot"],
            y=df_sorted["Rolling_Mean"],
            mode="lines",
            name="10-Lot Rolling Mean",
            line=dict(color="#0284c7", width=2.2, dash="dash"),
        )
    )

    # Center Line (Mean)
    fig_spc.add_hline(
        y=spc["mean"],
        line=dict(color="#059669", width=1.5, dash="dot"),
        annotation_text=f"Center Line: {spc['mean']}%",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#059669"),
    )

    # Lower Control Limit (LCL = Mean - 3 Sigma)
    fig_spc.add_hline(
        y=spc["lcl"],
        line=dict(color="#dc2626", width=1.5, dash="dash"),
        annotation_text=f"LCL (Mean - 3σ): {spc['lcl']}%",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#dc2626"),
    )

    # Upper Control Limit (UCL = Min(100, Mean + 3 Sigma))
    fig_spc.add_hline(
        y=spc["ucl"],
        line=dict(color="#64748b", width=1, dash="dash"),
        annotation_text=f"UCL: {spc['ucl']}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#64748b"),
    )

    fig_spc.update_layout(
        xaxis_title="Production Lot Identifier",
        yaxis_title="Yield Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_spc = apply_clean_theme(fig_spc, height=400)
    st.plotly_chart(fig_spc, use_container_width=True)

    # Parameter vs Yield Scatter Correlations
    st.markdown(
        """
        <div class="section-card" style="margin-top: 1.5rem;">
            <div class="section-title">Process Parameter vs Yield Response</div>
            <div class="section-desc">Bivariate relationship between chamber physical parameters, defect formation, and resulting yield.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = get_feature_columns(df)
    param_cols = st.columns(2)
    for i, feat in enumerate(features):
        with param_cols[i % 2]:
            fig_p = px.scatter(
                df,
                x=feat,
                y="Yield",
                color="Defects",
                color_continuous_scale="Blues",
                trendline="ols",
                trendline_color_override="#dc2626",
                hover_data=["Lot", feat, "Defects", "Yield"] if "Lot" in df.columns else [feat, "Defects", "Yield"],
            )
            fig_p = apply_clean_theme(fig_p, title=f"{feat} vs Yield", height=320)
            st.plotly_chart(fig_p, use_container_width=True)

    # Correlation Matrix Section
    st.markdown(
        """
        <div class="section-card" style="margin-top: 1.5rem;">
            <div class="section-title">Parameter Cross-Correlation Heatmap</div>
            <div class="section-desc">Pearson correlation coefficients quantifying pairwise relationships across process settings and defect metrics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    corr_cols = features + ["Yield"]
    corr = df[corr_cols].corr().round(2)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        aspect="auto",
    )
    fig_corr = apply_clean_theme(fig_corr, height=380)
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------------------
# MODULE 3: Root Cause Analysis
# ---------------------------------------------------------------------------
elif page == "Root Cause Analysis":
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Ranked Factor Contribution (Feature Importance)</div>
            <div class="section-desc">Random Forest feature importance ranking quantifying each parameter's statistical association with lot yield variance. Note: Highlights statistical associations rather than validated physical causality.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    imp_df = trained["importance_df"]

    col_imp_chart, col_imp_stats = st.columns([1.3, 1.0])

    with col_imp_chart:
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Factor",
            orientation="h",
            text=imp_df["Importance"].round(1).astype(str) + "%",
            color="Importance",
            color_continuous_scale=["#cbd5e1", "#0f62fe"],
        )
        fig_imp.update_traces(textposition="outside")
        fig_imp.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Relative Importance (%)",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        fig_imp = apply_clean_theme(fig_imp, height=300)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_imp_stats:
        top_factor = imp_df.iloc[0]["Factor"]
        top_pct = imp_df.iloc[0]["Importance"]
        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top: 10px;">
                <div class="kpi-label">Primary Variance Driver</div>
                <div class="kpi-value">{top_factor}</div>
                <div class="kpi-sub">Accounts for <b>{top_pct:.1f}%</b> of yield model variance</div>
            </div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5; padding: 10px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;">
                <div><b>Model Quality:</b> R² = {trained.get('r2', 'N/A')}</div>
                <div><b>Mean Residual Error:</b> {trained.get('mae', 'N/A')}% yield</div>
                <div><b>Baseline Reference:</b> 75th Percentile healthy lots</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Parameter Deep-Dive: Normal vs Excursion Lots Distribution
    st.markdown(
        """
        <div class="section-card" style="margin-top: 1.5rem;">
            <div class="section-title">Parameter Distribution: Healthy Lots vs Excursions</div>
            <div class="section-desc">Comparing process parameter distributions between top quartile healthy lots (≥ 75th percentile yield) and bottom quartile excursions (≤ 25th percentile yield).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_feat = st.selectbox("Select Parameter to Inspect", features, index=0)

    p75 = df["Yield"].quantile(0.75)
    p25 = df["Yield"].quantile(0.25)
    
    df_compare = df.copy()
    df_compare["Lot_Class"] = "Middle Range (25th - 75th Pct)"
    df_compare.loc[df_compare["Yield"] >= p75, "Lot_Class"] = "Healthy Lots (≥ 75th Pct)"
    df_compare.loc[df_compare["Yield"] <= p25, "Lot_Class"] = "Excursion Lots (≤ 25th Pct)"

    fig_box = px.box(
        df_compare,
        x="Lot_Class",
        y=selected_feat,
        color="Lot_Class",
        color_discrete_map={
            "Healthy Lots (≥ 75th Pct)": "#059669",
            "Middle Range (25th - 75th Pct)": "#64748b",
            "Excursion Lots (≤ 25th Pct)": "#dc2626",
        },
        points="all",
    )
    fig_box.update_layout(showlegend=False, xaxis_title="", yaxis_title=f"{selected_feat} Value")
    fig_box = apply_clean_theme(fig_box, height=350)
    st.plotly_chart(fig_box, use_container_width=True)

    # Formal 8D Corrective Action Protocol (RCCA)
    st.markdown(
        """
        <div class="section-card" style="margin-top: 1.5rem;">
            <div class="section-title">Engineering Root-Cause Corrective Action (RCCA) Protocol</div>
            <div class="section-desc">Standard operating procedures (SOP) and diagnostic containment checks for flagged process excursions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rcca_data = [
        {
            "Parameter": "Temperature",
            "Nominal Reference": f"{trained['baseline'].get('Temperature', {}).get('median', 450.0):.1f} °C",
            "Common Failure Mode": "Furnace thermocouple drift, heater element aging, PID loop oscillation",
            "Immediate Containment": "Pause lot dispatch to chamber; verify pyrometer/thermocouple calibration",
            "Corrective Action": "Inspect preventive maintenance records; calibrate heater zones against master probe",
        },
        {
            "Parameter": "Pressure",
            "Nominal Reference": f"{trained['baseline'].get('Pressure', {}).get('median', 2.2):.2f} Torr",
            "Common Failure Mode": "Foreline throttle valve stick, vacuum seal micro-leak, roughing pump degradation",
            "Immediate Containment": "Run helium leak check; verify capacitance manometer zero point",
            "Corrective Action": "Service throttle valve O-rings; check turbo pump backing pressure logs",
        },
        {
            "Parameter": "Power",
            "Nominal Reference": f"{trained['baseline'].get('Power', {}).get('median', 800.0):.1f} W",
            "Common Failure Mode": "RF matching network capacitor wear, generator reflected power imbalance",
            "Immediate Containment": "Review RF match tuning curves; verify delivered forward vs reflected power",
            "Corrective Action": "Recalibrate RF generator match network; inspect plasma dark-space shields",
        },
        {
            "Parameter": "Defects",
            "Nominal Reference": f"{trained['baseline'].get('Defects', {}).get('median', 3.0):.0f} / wafer",
            "Common Failure Mode": "Chamber wall flaking, electrostatic chuck particle shedding, robot arm friction",
            "Immediate Containment": "Initiate automated SEM review for defect spatial signature classification",
            "Corrective Action": "Perform chamber wet-clean cycle; swap gas delivery filter elements",
        },
        {
            "Parameter": "GasFlow",
            "Nominal Reference": f"{trained['baseline'].get('GasFlow', {}).get('median', 120.0):.1f} sccm",
            "Common Failure Mode": "Mass Flow Controller (MFC) sensor zero shift, valve solenoid sticking",
            "Immediate Containment": "Verify line supply delivery pressure; perform MFC rate-of-rise test",
            "Corrective Action": "Recalibrate MFC transducer; replace gas inlet filter manifold",
        },
    ]

    st.dataframe(pd.DataFrame(rcca_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# MODULE 4: Lot Risk Simulator
# ---------------------------------------------------------------------------
elif page == "Lot Risk Simulator":
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Planned Lot Risk & Yield Simulator</div>
            <div class="section-desc">Evaluate upcoming production run parameters against historical baseline models to preemptively identify risk and excursions before wafer processing.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = trained["features"]
    defaults = {f: float(df[f].median()) for f in features}
    stds = {f: float(df[f].std()) for f in features}

    # Recipe Quick-Presets Row
    st.markdown("##### Recipe Presets")
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns([1, 1, 1, 1])

    if "preset_values" not in st.session_state:
        st.session_state["preset_values"] = defaults.copy()

    with col_pre1:
        if st.button("Nominal Recipe (Healthy)", use_container_width=True):
            st.session_state["preset_values"] = {
                "Temperature": 450.0,
                "Pressure": 2.20,
                "Power": 800.0,
                "Defects": 2.0,
                "GasFlow": 120.0 if "GasFlow" in features else None,
            }
            st.rerun()

    with col_pre2:
        if st.button("Thermal Excursion Recipe", use_container_width=True):
            st.session_state["preset_values"] = {
                "Temperature": 475.0,
                "Pressure": 2.20,
                "Power": 805.0,
                "Defects": 14.0,
                "GasFlow": 120.0 if "GasFlow" in features else None,
            }
            st.rerun()

    with col_pre3:
        if st.button("High Stress Process Run", use_container_width=True):
            st.session_state["preset_values"] = {
                "Temperature": 472.0,
                "Pressure": 2.90,
                "Power": 845.0,
                "Defects": 22.0,
                "GasFlow": 124.0 if "GasFlow" in features else None,
            }
            st.rerun()

    with col_pre4:
        if st.button("Reset to Median Baseline", use_container_width=True):
            st.session_state["preset_values"] = defaults.copy()
            st.rerun()

    # Input Form
    st.markdown("---")
    st.markdown("##### Planned Process Setpoints")

    input_values = {}
    cols = st.columns(len(features))

    for i, feat in enumerate(features):
        with cols[i]:
            nom_val = defaults.get(feat, 0.0)
            cur_preset = st.session_state["preset_values"].get(feat, nom_val)
            unit = "°C" if feat == "Temperature" else ("Torr" if feat == "Pressure" else ("W" if feat == "Power" else ("sccm" if feat == "GasFlow" else "dies")))
            
            st.markdown(
                f"""
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 2px;">
                    Nominal: <b>{nom_val:.1f}</b> {unit}
                </div>
                """,
                unsafe_allow_html=True,
            )
            input_values[feat] = st.number_input(
                f"{feat} ({unit})",
                value=float(cur_preset),
                format="%.2f",
                key=f"sim_input_{feat}",
            )

    run_sim = st.button("RUN RISK SIMULATION", type="primary", use_container_width=False)

    # Run Prediction Automatically or on Click
    if run_sim or "last_sim_result" in st.session_state:
        result = predict_new_lot(trained, input_values)
        st.session_state["last_sim_result"] = result

        risk_level = result["risk"]
        pred_yield = result["predicted_yield"]

        if risk_level == "LOW":
            pill_class = "pill-low"
            risk_desc = "Process setpoints are within nominal operating tolerances. Low excursion probability."
        elif risk_level == "MEDIUM":
            pill_class = "pill-medium"
            risk_desc = "Moderate parameter deviation detected. Process monitoring recommended."
        else:
            pill_class = "pill-high"
            risk_desc = "Significant parameter excursions detected. High risk of yield degradation below threshold."

        # Risk Scoreboard
        st.markdown(
            f"""
            <div class="sim-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="kpi-label">Simulation Assessment</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem;">
                            Predicted Yield: <span style="color: #0f62fe;">{pred_yield}%</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #475569; margin-top: 0.35rem;">
                            {risk_desc}
                        </div>
                    </div>
                    <div>
                        <span class="status-pill {pill_class}" style="font-size: 0.95rem; padding: 6px 14px;">
                            {risk_level} RISK
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Findings & Parameter Deviations
        findings = result.get("findings", [])
        if findings:
            st.markdown("##### Parameter Excursion Diagnostics")
            
            dev_rows = []
            for f in findings:
                factor = f["factor"]
                dev_rows.append({
                    "Parameter": factor,
                    "Planned Input": input_values.get(factor),
                    "Healthy Median": f"{trained['baseline'][factor]['median']:.2f}",
                    "Deviation (Z-Score)": f["z_score"],
                    "Direction": f["direction"].upper(),
                    "First-Line Recommendation": f["actions"][0] if f["actions"] else "Monitor process closely",
                })
            
            st.dataframe(pd.DataFrame(dev_rows), use_container_width=True, hide_index=True)

            with st.expander("Detailed Remediation Protocols", expanded=True):
                for f in findings:
                    st.markdown(f"**{f['factor']} Excursion Protocol**")
                    for act in f["actions"]:
                        st.markdown(f"- {act}")
        else:
            st.markdown(
                """
                <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 6px; padding: 12px 16px; color: #065f46; font-size: 0.88rem;">
                    All process parameters are within expected historical operating limits (|Z| < 1.5). No excursions identified.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Simulation Report Export
        sim_summary_df = pd.DataFrame([{
            "Predicted_Yield": pred_yield,
            "Risk_Level": risk_level,
            **input_values,
            "Flagged_Excursions": len(findings),
        }])
        sim_csv = sim_summary_df.to_csv(index=False)
        st.download_button(
            label="Export Simulation Run Report (CSV)",
            data=sim_csv,
            file_name="simulated_lot_assessment.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #94a3b8; padding: 0.5rem 0;">
        <div>Wafer Yield Intelligence Platform | Fab Operations Decision Support</div>
        <div>Standard Process Capability Monitoring</div>
    </div>
    """,
    unsafe_allow_html=True,
)
