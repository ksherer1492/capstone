import os, json, time
from pathlib import Path

import geopandas as gpd
import plotly.express as px
import streamlit as st
import openai
import gzip

# ───────────── Config ─────────────

openai.api_key = st.secrets['openai_key']
MAPBOX_TOKEN = ""

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



@st.cache_data(show_spinner="Loading and processing GeoJSON...")
def load_clustered_geojson(path="data/all/clustered_tracts_knee_filtered.geojson.gz"):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        geojson_data = json.load(f)

    tracts = gpd.GeoDataFrame.from_features(geojson_data["features"])
    tracts = tracts.set_crs(4326)
    tracts['cluster_str'] = tracts['cluster'].astype(str)

    centroids = tracts.to_crs(3857).geometry.centroid.to_crs(4326)
    tracts['lon'] = centroids.x
    tracts['lat'] = centroids.y

    tracts['GEOIDFQ'] = tracts['GEOIDFQ'].str.replace('1400000US6', '1400000US06')
    
    minimal = tracts[['GEOIDFQ', 'geometry']]
    geojson_minimal = json.loads(minimal.to_json())

    return tracts, geojson_minimal

if "TRACTS" not in st.session_state or "geojson" not in st.session_state:
    tracts, geojson = load_clustered_geojson()
    st.session_state["TRACTS"] = tracts
    st.session_state["geojson"] = geojson

# Use the cached versions from session state
TRACTS = st.session_state["TRACTS"]
geojson = st.session_state["geojson"]


# ───────────── Prompt helper (Markdown) ─────────────

def σ(v, cap=5):
    try:
        z = (v if isinstance(v, (int, float)) else 0.0)
        z = max(min(z, cap), -cap)
        return z
    except:
        return 0.0

def build_prompt(row):
    tid, cid = row['GEOIDFQ'], row['cluster']
    freq = TRACTS[TRACTS.cluster == cid]['driver_1'].value_counts().head(1)
    most = freq.idxmax() if not freq.empty else 'N/A'
    cnt = int(freq.max()) if not freq.empty else 0

    total_hh = row['ci_EstimateNumber_HOUSEHOLD_INCOME_All_households']
    hh_with_earnings = row['ci_EstimateNumber_HOUSEHOLD_INCOME_All_households_With_earnings']
    wage_hh = row['ci_EstimateNumber_HOUSEHOLD_INCOME_All_households_With_earnings_With_wages_or_salary_income']
    selfemp_hh = row['ci_EstimateNumber_HOUSEHOLD_INCOME_All_households_With_earnings_With_self_employment_income']
    percap_income = row['ci_EstimateMean_income_dollars_PER_CAPITA_INCOME_BY_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Total_population']

    hh_with_earnings_pct = (hh_with_earnings / total_hh * 100) if total_hh else 0
    wage_hh_pct = (wage_hh / hh_with_earnings * 100) if hh_with_earnings else 0
    selfemp_pct = (selfemp_hh / hh_with_earnings * 100) if hh_with_earnings else 0

    lines = [
        f"**Census Tract {tid} Overview**", '',
        "## 1. Tract Overview",
        f"- Total households: {int(total_hh):,}",
        f"- Households with earnings: {int(hh_with_earnings):,} ({hh_with_earnings_pct:.1f}% of total)",
        f"- Wage-earning households: {int(wage_hh):,} ({wage_hh_pct:.1f}% of earning households)",
        f"- Self-employed households: {int(selfemp_hh):,} ({selfemp_pct:.1f}% of earning households)",
        f"- Per-capita income: ${percap_income:,.0f}", '',
        "## 2. Top 5 Cluster Membership Drivers",
        f"1. {row['driver_1']}",
        f"2. {row['driver_2']}",
        f"3. {row['driver_3']}",
        f"4. {row['driver_4']}",
        f"5. {row['driver_5']}", '',
        f"## 3. Cluster {cid} Snapshot",
        f"- “{most}” appears in **{cnt}** tracts in this cluster.", '',
        "## 4. Recommendation",
        "Provide **6–8 concise, data-driven bullets** for each section above, then close with a **4–6 sentence restaurant concept**."
    ]
    return "\n".join(lines)

# ───────────── Layout ─────────────
# st.set_page_config(page_title='Census-Tract Cluster Explorer', layout='wide')
st.markdown("""
<style>
.advisor-box { background: #212121; color: #fff; padding: 1rem; border-radius: 8px; height: 90vh; overflow-y: auto; }
.error-msg { color: #f88; font-weight: bold; }
.plotly-graph-div { height: 90vh !important; }
</style>
""", unsafe_allow_html=True)

left, right = st.columns([1,3], gap='small')
with right:
    TRACTS['cluster_cat'] = TRACTS['cluster'].astype(str)
    fig = px.choropleth_map(
        TRACTS, geojson=geojson, locations='GEOIDFQ', featureidkey='properties.GEOIDFQ',
        color='cluster_cat', color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data={'cluster_cat':False,'cluster':True,'driver_1':True,'driver_2':True,'driver_3':True},
        center=dict(lat=41.8781, lon=-87.6298), zoom=11, opacity=0.7, height=1150
    )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), mapbox_style='carto-darkmatter' if MAPBOX_TOKEN else 'open-street-map', mapbox_accesstoken=MAPBOX_TOKEN or None, showlegend=False)
    fig.update_traces(selected=dict(marker=dict(opacity=0.7)), unselected=dict(marker=dict(opacity=0.3)))
    fig.update_traces(marker_line_color='#000', marker_line_width=0.5)
    map_event = st.plotly_chart(fig, key='main-map', on_select='rerun', selection_mode='points', use_container_width=True)

with left:
    if st.session_state.error_msg:
        st.markdown(f"<div class='error-msg'>{st.session_state.error_msg}</div>", unsafe_allow_html=True)
    advisor_placeholder = st.empty()
    advisor_placeholder.markdown(f"<div class='advisor-box'>{st.session_state.advisor_md}</div>", unsafe_allow_html=True)

if isinstance(map_event, dict) and map_event.selection.points:
    gid = map_event.selection.points[0]['location']
    now_ms = time.time() * 1000
    if now_ms - st.session_state.last_click_ts > DEBOUNCE_MS:
        st.session_state.last_click_ts = now_ms
        row = TRACTS[TRACTS.GEOIDFQ==gid].iloc[0]
        prompt = build_prompt(row)
        st.session_state.error_msg = ''
        response = '''

'''
        for chunk in openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {'role':'system','content':'You are a data-driven advisor. Output plain Markdown. Do not include triple backticks or any code blocks. DON'T BE RACIST'},
                {'role':'user','content':prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True
        ):
            response += (chunk.choices[0].delta.content or '')
            clean = response.replace('```','')
            st.session_state.advisor_md = clean
            advisor_placeholder.markdown(f"<div class='advisor-box'>{clean}</div>", unsafe_allow_html=True)
