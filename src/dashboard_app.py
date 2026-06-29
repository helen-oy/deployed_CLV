from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

SEGMENT_SUMMARY_PATH = MODELS_DIR / "current_state_segment_summary.csv"
LIGHTGBM_PREDICTIONS_PATH = MODELS_DIR / "lightgbm_predictions.csv"
RECOMMENDATIONS_PATH = MODELS_DIR / "customer_recommendations.csv"
RETENTION_IMAGE_PATH = BASE_DIR / "retention_image.webp"

HIGH_CHURN_THRESHOLD = 0.50
TOP_PRIORITY_COUNT = 100

COLOR_SEQUENCE = ["#22577A", "#38A3A5", "#57CC99", "#F9C74F", "#F94144", "#7B2CBF"]


st.set_page_config(
    page_title="Executive CLV & Retention Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricLabel"] {
            color: #475569;
            font-size: 0.82rem;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 700;
        }
        .executive-note {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.45;
            margin: -0.45rem 0 0.85rem 0;
        }
        .section-caption {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: -0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segment_summary = pd.read_csv(SEGMENT_SUMMARY_PATH)
    predictions = pd.read_csv(LIGHTGBM_PREDICTIONS_PATH)
    recommendations = pd.read_csv(RECOMMENDATIONS_PATH)

    segment_summary["customer_segment"] = segment_summary["customer_segment"].str.strip()
    return segment_summary, predictions, recommendations


def currency(value: float) -> str:
    return f"£{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def model_health_label(metrics_path: Path = MODELS_DIR / "lightgbm_metrics.json") -> str:
    if not metrics_path.exists():
        return "Model metrics unavailable"

    try:
        with metrics_path.open() as f:
            metrics = json.load(f)["metrics"]["churn_roc_auc"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "Model metrics unavailable"

    return f"Churn AUC {metrics:.2f}"


df_segments, df_predictions, df_recommendations = load_data()

segment_options = df_segments["customer_segment"].tolist()

st.sidebar.header("Executive Filters")
selected_segments = st.sidebar.multiselect(
    "Customer segments",
    options=segment_options,
    default=segment_options,
)
priority_count = st.sidebar.slider(
    "Priority customer group",
    min_value=25,
    max_value=min(500, len(df_recommendations)),
    value=min(TOP_PRIORITY_COUNT, len(df_recommendations)),
    step=25,
)

filtered_segments = df_segments[df_segments["customer_segment"].isin(selected_segments)]
if filtered_segments.empty:
    st.warning("Select at least one customer segment to show the dashboard.")
    st.stop()

total_customers = int(df_segments["Customer Count"].sum())
selected_customers = int(filtered_segments["Customer Count"].sum())
total_predicted_clv = df_predictions["lightgbm_clv_90d"].sum()
average_predicted_clv = df_predictions["lightgbm_clv_90d"].mean()
average_churn_risk = df_predictions["lightgbm_churn_90d_probability"].mean()
high_churn_customers = int(
    (df_predictions["lightgbm_churn_90d_probability"] >= HIGH_CHURN_THRESHOLD).sum()
)

largest_segment = df_segments.loc[df_segments["Customer Count"].idxmax()]
highest_value_segment = df_segments.loc[df_segments["Monetary"].idxmax()]
highest_frequency_segment = df_segments.loc[df_segments["Frequency"].idxmax()]

top_priority = df_recommendations.head(priority_count).copy()
top_priority_clv = top_priority["CLV"].sum()
top_priority_avg_clv = top_priority["CLV"].mean()

st.title("Executive CLV & Retention Dashboard")
st.markdown(
    """
    <p class="executive-note">
    Strategic view of predicted customer value, churn exposure, segment quality,
    and immediate retention priorities for a non-contractual retail business.
    </p>
    """,
    unsafe_allow_html=True,
)

if RETENTION_IMAGE_PATH.exists():
    with st.expander("Retention context", expanded=False):
        st.write(
            "Customers are not tied to subscriptions, so churn is inferred from purchase inactivity and predicted future buying behavior."
        )
        st.image(Image.open(RETENTION_IMAGE_PATH), width="stretch")

st.subheader("Board-Level Metrics")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric(
    "90-Day Predicted CLV",
    currency(total_predicted_clv),
    f"Avg {currency(average_predicted_clv)} per customer",
)
kpi2.metric(
    "90-Day Churn Risk",
    pct(average_churn_risk),
    f"{high_churn_customers:,} customers above 50%",
)
kpi3.metric(
    "Segment Distribution",
    largest_segment["customer_segment"],
    f"{int(largest_segment['Customer Count']):,} customers",
)
kpi4.metric(
    "RFM Health Leader",
    highest_value_segment["customer_segment"],
    f"Avg spend {currency(highest_value_segment['Monetary'])}",
)
kpi5.metric(
    "Priority Retention Value",
    currency(top_priority_clv),
    f"Top {priority_count} avg {currency(top_priority_avg_clv)}",
)

st.caption(
    f"Portfolio customer base: {total_customers:,}. Selected segment view: {selected_customers:,}. "
    f"Model confidence signal: {model_health_label()}."
)

st.divider()

left_col, right_col = st.columns([1.12, 0.88])

with left_col:
    st.subheader("Value and Churn Exposure")
    st.markdown(
        '<p class="section-caption">Identifies where future revenue is concentrated and where inactivity risk needs management.</p>',
        unsafe_allow_html=True,
    )

    value_risk_fig = px.scatter(
        df_predictions,
        x="lightgbm_churn_90d_probability",
        y="lightgbm_clv_90d",
        color="lightgbm_churn_90d_probability",
        color_continuous_scale=["#22577A", "#57CC99", "#F9C74F", "#F94144"],
        hover_data={
            "CustomerID": True,
            "lightgbm_churn_90d_probability": ":.1%",
            "lightgbm_clv_90d": ":,.2f",
        },
        labels={
            "lightgbm_churn_90d_probability": "90-Day Churn Probability",
            "lightgbm_clv_90d": "Predicted 90-Day CLV",
        },
    )
    value_risk_fig.add_vline(
        x=HIGH_CHURN_THRESHOLD,
        line_width=2,
        line_dash="dash",
        line_color="#F94144",
    )
    value_risk_fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
    )
    value_risk_fig.update_yaxes(tickprefix="£", rangemode="tozero")
    value_risk_fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(value_risk_fig, width="stretch")

with right_col:
    st.subheader("Customer Base Mix")
    st.markdown(
        '<p class="section-caption">Shows the scale of each customer group for campaign planning and resource allocation.</p>',
        unsafe_allow_html=True,
    )

    segment_mix = filtered_segments.sort_values("Customer Count", ascending=False)
    mix_fig = px.bar(
        segment_mix,
        x="customer_segment",
        y="Customer Count",
        color="customer_segment",
        color_discrete_sequence=COLOR_SEQUENCE,
        text="Customer Count",
        labels={
            "customer_segment": "Segment",
            "Customer Count": "Customers",
        },
    )
    mix_fig.update_layout(
        height=420,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    mix_fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(mix_fig, width="stretch")

st.divider()

segment_col, priority_col = st.columns([1.05, 0.95])

with segment_col:
    st.subheader("RFM Segment Health")
    st.markdown(
        '<p class="section-caption">Lower recency and higher frequency or monetary value indicate stronger customer health.</p>',
        unsafe_allow_html=True,
    )

    rfm_fig = px.scatter(
        filtered_segments,
        x="Recency",
        y="Monetary",
        size="Customer Count",
        color="Frequency",
        hover_name="customer_segment",
        hover_data={
            "Recency": ":.1f",
            "Frequency": ":.1f",
            "Monetary": ":,.2f",
            "Customer Count": ":,",
        },
        size_max=72,
        color_continuous_scale=["#F94144", "#F9C74F", "#38A3A5", "#22577A"],
        labels={
            "Recency": "Days Since Last Purchase",
            "Monetary": "Average Monetary Value",
            "Frequency": "Purchase Frequency",
        },
    )
    rfm_fig.update_xaxes(autorange="reversed")
    rfm_fig.update_yaxes(tickprefix="£")
    rfm_fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(rfm_fig, width="stretch")

with priority_col:
    st.subheader("Priority Retention Opportunity")
    st.markdown(
        '<p class="section-caption">Top-ranked customers combine value potential with campaign urgency.</p>',
        unsafe_allow_html=True,
    )

    priority_band = (
        top_priority.groupby(["CLV_Segment", "Churn_Risk"], observed=True)
        .agg(
            Customers=("CustomerID", "count"),
            Total_CLV=("CLV", "sum"),
            Avg_Priority=("Priority_Score", "mean"),
        )
        .reset_index()
        .sort_values("Total_CLV", ascending=True)
    )

    priority_fig = px.bar(
        priority_band,
        x="Total_CLV",
        y="CLV_Segment",
        color="Churn_Risk",
        orientation="h",
        color_discrete_map={
            "Low": "#38A3A5",
            "Moderate": "#F9C74F",
            "High": "#F94144",
        },
        hover_data={
            "Customers": ":,",
            "Total_CLV": ":,.2f",
            "Avg_Priority": ":.2f",
        },
        labels={
            "Total_CLV": "Total CLV",
            "CLV_Segment": "CLV Segment",
            "Churn_Risk": "Churn Risk",
        },
    )
    priority_fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="Churn Risk",
    )
    priority_fig.update_xaxes(tickprefix="£")
    st.plotly_chart(priority_fig, width="stretch")

st.divider()

detail_col, action_col = st.columns([1, 1])

with detail_col:
    st.subheader("Segment Operating View")
    segment_table = filtered_segments.copy()
    segment_table["Customer Share"] = segment_table["Customer Count"] / total_customers
    segment_table = segment_table[
        [
            "customer_segment",
            "Customer Count",
            "Customer Share",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ].sort_values("Customer Count", ascending=False)

    st.dataframe(
        segment_table.style.format(
            {
                "Customer Count": "{:,.0f}",
                "Customer Share": "{:.1%}",
                "Recency": "{:.1f}",
                "Frequency": "{:.1f}",
                "Monetary": "£{:,.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with action_col:
    st.subheader("Executive Actions")
    action_rows = [
        {
            "Decision Area": "Protect",
            "Focus": highest_value_segment["customer_segment"],
            "Why It Matters": f"Highest average value at {currency(highest_value_segment['Monetary'])}.",
        },
        {
            "Decision Area": "Reactivate",
            "Focus": largest_segment["customer_segment"],
            "Why It Matters": f"Largest reachable group with {int(largest_segment['Customer Count']):,} customers.",
        },
        {
            "Decision Area": "Grow",
            "Focus": highest_frequency_segment["customer_segment"],
            "Why It Matters": f"Strongest buying cadence at {highest_frequency_segment['Frequency']:.1f} purchases.",
        },
        {
            "Decision Area": "Prioritize",
            "Focus": f"Top {priority_count} customers",
            "Why It Matters": f"{currency(top_priority_clv)} in ranked CLV opportunity.",
        },
        {
            "Decision Area": "Risk Manage",
            "Focus": "High churn customers",
            "Why It Matters": f"{high_churn_customers:,} customers are above {pct(HIGH_CHURN_THRESHOLD)} churn risk.",
        },
    ]
    st.dataframe(pd.DataFrame(action_rows), width="stretch", hide_index=True)

st.divider()

st.subheader("Top Priority Customers")
priority_table = top_priority[
    [
        "CustomerID",
        "CLV",
        "churn_probability",
        "CLV_Segment",
        "Churn_Risk",
        "Priority_Score",
    ]
].copy()
priority_table["CLV"] = priority_table["CLV"].round(2)

st.dataframe(
    priority_table.style.format(
        {
            "CLV": "£{:,.2f}",
            "churn_probability": "{:.1%}",
            "Priority_Score": "{:.2f}",
        }
    ),
    width="stretch",
    hide_index=True,
)
