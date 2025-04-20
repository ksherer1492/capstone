import os, json, time
from pathlib import Path

import geopandas as gpd
import plotly.express as px
import streamlit as st
import openai

# ────────────────────── Config ──────────────────────
# Streamlit Secrets File Setup:
# 1. Create or edit ~/.streamlit/secrets.toml (Windows: C:\Users\<user>\.streamlit\secrets.toml)
# 2. Include only these key/value pairs:
#
# openai_api_key = "sk-<YOUR_OPENAI_KEY>"
# mapbox_token    = "pk-<YOUR_MAPBOX_TOKEN>"  # optional


# API keys
openai.api_key = st.secrets['openai_key']
MAPBOX_TOKEN = ""

# Other settings via env vars with defaults
DEBOUNCE_MS = int(os.getenv("DEBOUNCE_MS", 400))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.8))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 600))

# ───────────── Session‑state defaults ─────────────
for k, v in {
    'selected_geoid': None,
    'last_click_ts': 0.0,
    'advisor_md': 'Click a tract →',
    'error_msg': ''
}.items():
    st.session_state.setdefault(k, v)

# ─────────────────── Load data ────────────────────
TRACTS = gpd.read_file("data/all/clustered_tracts_filtered.geojson").to_crs(4326)

# Cast cluster to string for true categorical coloring
TRACTS['cluster_str'] = TRACTS['cluster'].astype(str)

centroids = TRACTS.to_crs(3857).geometry.centroid.to_crs(4326)
TRACTS['lon'] = centroids.x
TRACTS['lat'] = centroids.y

# only keep the GEOIDFQ & geometry for the geojson
TRACTS['GEOIDFQ'] = TRACTS['GEOIDFQ'].str.replace(
    '1400000US6',
    '1400000US06'
)
minimal = TRACTS[['GEOIDFQ','geometry']]

geojson = json.loads(minimal.to_json())



# ─────────────────── Prompt helper ──────────────── ────────────────
def σ(v): return f"{v:+.2f}σ"

def build_prompt(row):
    tid, cid = row['GEOIDFQ'], row['cluster']
    freq = TRACTS[TRACTS.cluster == cid]['driver_1'].value_counts().head(1)
    most = freq.idxmax() if not freq.empty else 'N/A'
    cnt = int(freq.max()) if not freq.empty else 0
    lines = [
        f"*(Please mention **{tid}** in your recommendation.)*", '',
        "## 1. Tract Overview",
        f"- Population index: {σ(row['cr_Total'])}",
        f"- Median-income index: {σ(row['ci_EstimateMean_income_dollars_HOUSEHOLD_INCOME_All_households'])}",
        f"- Wage-income index: {σ(row['ci_EstimatePercent_Distribution_HOUSEHOLD_INCOME_All_households_With_earnings_With_wages_or_salary_income'])}",
        f"- Self-employment index: {σ(row['ci_EstimatePercent_Distribution_HOUSEHOLD_INCOME_All_households_With_earnings_With_self_employment_income'])}",
        f"- Retirement-income index: {σ(row['ci_EstimatePercent_Distribution_HOUSEHOLD_INCOME_All_households_With_retirement_income'])}", '',
        "## 2. Top 3 Drivers",
        f"1. {row['driver_1']}",
        f"2. {row['driver_2']}",
        f"3. {row['driver_3']}", '',
        f"## 3. Cluster {cid} Snapshot",
        f"- “{most}” appears in **{cnt}** tracts in this cluster.", '',
        "## 4. Recommendation",
        "Provide **5–7 concise, data-driven bullets** for each section above, then close with a **3–4 sentence restaurant concept**."
    ]
    return "\n".join(lines)

# ─────────────────── Layout ───────────────────────
st.set_page_config(page_title='Census-Tract Explorer', layout='wide')

# Custom CSS for advisor and map
st.markdown("""
<style>
.advisor-box {
  background: #212121;
  color: #fff;
  padding: 1rem;
  border-radius: 8px;
  height: 90vh;
  overflow-y: auto;
}
.error-msg {
  color: #f88;
  font-weight: bold;
}
.plotly-graph-div {
  height: 90vh !important;
}
</style>
""", unsafe_allow_html=True)

# Columns for layout: give map more width (5 parts)
left, right = st.columns([1, 3], gap='small')

# Plotly Express Map in right column
with right:

# right before you build the figure, convert cluster → string
    TRACTS['cluster_cat'] = TRACTS['cluster'].astype(str)

    fig = px.choropleth_map(
        TRACTS,
        geojson=geojson,
        locations='GEOIDFQ',
        featureidkey='properties.GEOIDFQ',
        color='cluster_cat',                               # use the string column
        color_discrete_sequence=px.colors.qualitative.Set3,  # your multicolor palette
        hover_data={
            'cluster_cat': False,  # hide the string itself if you like
            'cluster': True,
            'driver_1': True,
            'driver_2': True,
            'driver_3': True
        },
        center=dict(lat=TRACTS.lat.mean(), lon=TRACTS.lon.mean()),
        zoom=11,
        opacity=0.7,
        height=1050
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox_style='carto-darkmatter' if MAPBOX_TOKEN else 'open-street-map',
        mapbox_accesstoken=MAPBOX_TOKEN or None
    )

    fig.update_traces(
        selected=dict(
            marker=dict(opacity=0.7)                 # keep the clicked tract fully opaque :contentReference[oaicite:1]{index=1}
        ),
        unselected=dict(
            marker=dict(opacity=0.3)               # only slightly dim the others :contentReference[oaicite:2]{index=2}
        )
)
    fig.update_traces(marker_line_color='#000', marker_line_width=0.5)

    map_event = st.plotly_chart(
        fig,
        key='main-map',
        on_select='rerun',
        selection_mode='points',
        use_container_width=True
    )


# Advisor panel and streaming in left column
with left:
    if st.session_state.error_msg:
        st.markdown(f"<div class='error-msg'>{st.session_state.error_msg}</div>", unsafe_allow_html=True)
    advisor_placeholder = st.empty()
    advisor_placeholder.markdown(f"<div class='advisor-box'>{st.session_state.advisor_md}</div>", unsafe_allow_html=True)

# React to map clicks and stream AI response
if isinstance(map_event, dict) and map_event.selection.points:
    gid = map_event.selection.points[0]['location']
    now_ms = time.time() * 1000
    if now_ms - st.session_state.last_click_ts > DEBOUNCE_MS:
        st.session_state.last_click_ts = now_ms
        row = TRACTS[TRACTS.GEOIDFQ == gid].iloc[0]
        prompt = build_prompt(row)
        st.session_state.error_msg = ''
        response = f"### Tract {gid}\n"
        for chunk in openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{'role': 'system', 'content': 'You are a data-driven advisor.'}, {'role': 'user', 'content': prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True
        ):
            response += chunk.choices[0].delta.content or ''
            st.session_state.advisor_md = response
            advisor_placeholder.markdown(f"<div class='advisor-box'>{response}</div>", unsafe_allow_html=True)
