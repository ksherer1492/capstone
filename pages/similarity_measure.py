import sys
import os
#sys.path.append(os.path.abspath(".."))  # <-- add the capstone folder to import path

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from similarity_engine import run_dot_product_similarity

st.set_page_config(page_title="Search Census Tract", layout="wide")

st.markdown("""
    <style>
    html, body, .main, .block-container {
        background-color: black !important;
        color: white !important;
    }
    input, label, textarea, .stTextInput, .stMarkdown, .stDataFrame, .stTable {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

income_path = "data/all/census/census_income_FIPS.csv"
race_path = "data/all/census/census_race_FIPS.csv"
matched_path = "data/all/census/matched_tracts_padded.csv"
geojson_path = "data/all/census/simplified_cities_00005.geojson"

@st.cache_data
def load_matched_geoids():
    matched_df = pd.read_csv(matched_path)
    return set(matched_df['GEOID'].astype(str).str.zfill(11))

@st.cache_data
def load_geojson():
    return gpd.read_file(geojson_path)

st.title("Search Census Tract From Matched Data")

matched_geoids = load_matched_geoids()
tracts_gdf = load_geojson()

search_geoid = st.text_input("Enter Census Tract GEOID (11 digits) - Hit Enter Key When Done:", max_chars=11)

# default zoom location (USA center)
map_center = {"lat": 39.5, "lon": -98.35}
zoom_level = 4

# always default all tracts to "Other"
tracts_gdf["tract_type"] = "Other"

markers = []

if search_geoid:
    padded = search_geoid.zfill(11)
    if padded in matched_geoids:
        st.success(f"Census Tract {padded} found in matched dataset.")
        st.session_state.selected_geoid = padded

        target_tract = tracts_gdf[tracts_gdf['GEOID'] == padded]
        tracts_gdf["tract_type"] = tracts_gdf["GEOID"].apply(lambda x: "Selected" if x == padded else "Other")

        if not target_tract.empty:
            centroid = target_tract.geometry.centroid.iloc[0]
            lat, lon = centroid.y, centroid.x

            top10_df = run_dot_product_similarity(padded, income_path, race_path, return_df=True)
            top10_with_geom = pd.merge(
                top10_df, tracts_gdf[['GEOID', 'geometry']],
                left_on='FIPS', right_on='GEOID'
            )
            top10_with_geom = gpd.GeoDataFrame(top10_with_geom, geometry='geometry')
            top10_with_geom['centroid'] = top10_with_geom.geometry.centroid
            top10_with_geom['lat'] = top10_with_geom['centroid'].y
            top10_with_geom['lon'] = top10_with_geom['centroid'].x

            markers.append(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(size=12, color='green'),
                text=f"Selected: {padded}",
                name="Selected Tract"
            ))

            for _, row in top10_with_geom.iterrows():
                markers.append(go.Scattermapbox(
                    lat=[row['lat']],
                    lon=[row['lon']],
                    mode='markers',
                    marker=dict(size=10, color='orange'),
                    text=f"Similar: {row['FIPS']} ({row['dot_similarity']:.2f})",
                    name="Similar Tract"
                ))
    else:
        st.warning(f"Census Tract {padded} not found in matched dataset.")

# always render the map (default or updated)
fig = px.choropleth_mapbox(
    tracts_gdf,
    geojson=tracts_gdf.geometry.__geo_interface__,
    locations=tracts_gdf.index,
    color="tract_type",
    color_discrete_map={"Selected": "yellow", "Other": "gray"},
    center=map_center,
    zoom=zoom_level,
    opacity=0.5
)

fig.update_layout(
    mapbox_style="carto-darkmatter",
    margin=dict(l=0, r=0, t=0, b=0)
)

for trace in markers:
    fig.add_trace(trace)

st.markdown("### Census Tract Map (Zoomed Out)")
st.plotly_chart(fig, use_container_width=True)

if search_geoid and padded in matched_geoids:
    st.markdown("### Similarity Results")
    run_dot_product_similarity(padded, income_path, race_path)
