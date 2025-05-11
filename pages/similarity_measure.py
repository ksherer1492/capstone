import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import requests
from similarity_engine import run_dot_product_similarity

# Set up Streamlit
st.set_page_config(page_title="Search Census Tract", layout="wide")

# Styling
st.markdown("""
    <style>
    html, body, .main, .block-container {
        background-color: black !important;
        color: white !important;
    }
    input, label, textarea, .stMarkdown, .stDataFrame, .stTable {
        color: white !important;
    }
    input[type="text"] {
        color: black !important;
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# File paths
income_path = "data/all/census/census_income_FIPS.csv"
race_path = "data/all/census/census_race_FIPS.csv"
matched_path = "data/all/census/matched_tracts_padded.csv"
geojson_path = "data/all/census/simplified_cities_00005.geojson"

city_map = {"06": "LA", "17": "CHI", "48": "ATX", "36": "NYC"}

# Load data
@st.cache_data
def load_matched_geoids():
    df = pd.read_csv(matched_path)
    return set(df['GEOID'].astype(str).str.zfill(11))

@st.cache_data
def load_geojson():
    return gpd.read_file(geojson_path)

matched_geoids = load_matched_geoids()
tracts_gdf = load_geojson()

# Title
st.title("Search Census Tract From Address or GEOID")

# Example
st.markdown("Example: `1234 S California Ave, Chicago, IL 60608`")
address = st.text_input("Enter a U.S. address (or leave blank to use a GEOID):")
search_geoid = None

if address:
    try:
        api_key = st.secrets["google"]["api_key"]
    except Exception:
        st.error("Google API key not found in Streamlit secrets.")
        st.stop()

    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {"address": address, "key": api_key}
    geocode_resp = requests.get(geocode_url, params=geocode_params).json()

    if geocode_resp["status"] != "OK":
        st.error(f"Google Geocoding failed: {geocode_resp['status']}")
        st.stop()

    location = geocode_resp["results"][0]["geometry"]["location"]
    lat, lon = location["lat"], location["lng"]
    st.write(f"Latitude: {lat}, Longitude: {lon}")

    # U.S. Census Geocoder
    census_url = (
        "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        f"?x={lon}&y={lat}&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
    )
    census_resp = requests.get(census_url).json()

    try:
        tract_info = census_resp["result"]["geographies"]["Census Tracts"][0]
        search_geoid = tract_info["GEOID"]
        st.success(f"Census Tract GEOID: {search_geoid}")
    except Exception as e:
        st.error("Could not extract census tract info from Census Geocoder.")
        st.exception(e)
        st.stop()
else:
    default_geoid = "17031251300"
    search_geoid = st.text_input("Or enter a Census Tract GEOID:", value=default_geoid, max_chars=11)

# --- Map + Similarity ---
map_center = {"lat": 39.5, "lon": -98.35}
zoom_level = 3.2
markers = []
tracts_gdf["tract_type"] = "Unmatched"

if search_geoid:
    padded = str(search_geoid).zfill(11)
    if padded in matched_geoids:
        st.success(f"Census Tract {padded} found in matched dataset.")
        st.session_state["selected_geoid"] = padded
        tracts_gdf["tract_type"] = "Unmatched"
        tracts_gdf.loc[tracts_gdf["GEOID"] == padded, "tract_type"] = "Selected"

        target_tract = tracts_gdf[tracts_gdf["GEOID"] == padded]
        if not target_tract.empty:
            centroid = target_tract.geometry.centroid.iloc[0]
            map_center = {"lat": centroid.y, "lon": centroid.x}
            zoom_level = 10

            top10_df, contrib_df = run_dot_product_similarity(padded, income_path, race_path, return_df=True)
            top10_df = top10_df.sort_values(by="dot_similarity", ascending=False).reset_index(drop=True)
            st.session_state["similar_tracts"] = dict(zip(top10_df["FIPS"], top10_df["dot_similarity"].round(3)))

            tract_geom_map = tracts_gdf.set_index("GEOID")["geometry"]
            top10_df["geometry"] = top10_df["FIPS"].map(tract_geom_map)
            top10_with_geom = gpd.GeoDataFrame(top10_df, geometry="geometry")
            top10_with_geom['centroid'] = top10_with_geom.geometry.centroid
            top10_with_geom['lat'] = top10_with_geom['centroid'].y
            top10_with_geom['lon'] = top10_with_geom['centroid'].x

            tracts_gdf.loc[tracts_gdf["GEOID"].isin(top10_df["FIPS"]), "tract_type"] = "Similar"

            city_abbr = city_map.get(padded[:2], "")
            prefixed_label = f"{city_abbr}-{padded}"
            markers.append(go.Scattermapbox(
                lat=[centroid.y],
                lon=[centroid.x],
                mode='markers',
                marker=dict(size=12, color='green'),
                text=f"Selected Tract: {prefixed_label}",
                name=f"Selected: {prefixed_label}",
                legendgroup="Selected Tract",
                showlegend=True
            ))

            for i, row in top10_with_geom.iterrows():
                city_abbr_sim = city_map.get(row['FIPS'][:2], "")
                label = f"{city_abbr_sim}-{row['FIPS']}"
                similarity = f"{row['dot_similarity']:.3f}"
                markers.append(go.Scattermapbox(
                    lat=[row['lat']],
                    lon=[row['lon']],
                    mode='markers',
                    marker=dict(size=10, color='orange'),
                    text=f"Rank {i+1}: {label}<br>Similarity: {similarity}",
                    name=f"{i+1}. {label} Similarity: {similarity}",
                    legendgroup="Similar Tracts",
                    showlegend=True
                ))
    else:
        st.warning(f"Census Tract {padded} not found in matched dataset.")

# Color mapping + Map Plot
color_levels = {"Unmatched": 0, "Similar": 1, "Selected": 2}
color_scale = [[0.0, "gray"], [0.5, "orange"], [1.0, "yellow"]]
fig = go.Figure()

fig.add_trace(go.Choroplethmapbox(
    geojson=tracts_gdf.geometry.__geo_interface__,
    locations=tracts_gdf.index,
    z=tracts_gdf["tract_type"].map(color_levels),
    colorscale=color_scale,
    zmin=0,
    zmax=2,
    showscale=False,
    marker_opacity=0.5,
    marker_line_width=0,
    hoverinfo="skip"
))

for trace in markers:
    fig.add_trace(trace)

fig.update_layout(
    mapbox_style="carto-darkmatter",
    mapbox=dict(center=map_center, zoom=zoom_level),
    margin=dict(l=0, r=0, t=0, b=0),
    uirevision="keep-zoom",
    dragmode='zoom',
    hovermode='closest'
)

st.markdown("### Census Tract Map")
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

if search_geoid and padded in matched_geoids:
    top10_df["City"] = top10_df["FIPS"].str[:2].map(city_map)
    if "geometry" in top10_df.columns:
        top10_df = top10_df.drop(columns=["geometry"])
    reordered_cols = ["City"] + [col for col in top10_df.columns if col != "City"]
    st.markdown("### Similarity Results")
    st.dataframe(top10_df[reordered_cols])

    if not contrib_df.empty:
        st.markdown("### Top Contributing Features to Similarity")
        st.dataframe(contrib_df)

# Updated Description Box
st.markdown("""
<div style="background-color: white; color: black; padding: 15px; border-radius: 10px; margin-top: 30px;">
    <h4 style="color: #DAA520;">How This Tool Works</h4>
    <p>This app allows you to search for a specific address or 11-digit Census Tract GEOID from matched data in <strong>Los Angeles, Chicago, Austin, or NYC</strong>.</p>
    <p>When a valid tract is entered, the tool:
    <ul>
        <li>Highlights the tract's area in <span style="color: goldenrod;"><strong>yellow</strong></span> and places a <span style="color: green;"><strong>green dot</strong></span> at its center</li>
        <li>Computes similarity using the <strong>dot product</strong> of standardized income and race features</li>
        <li>Shows the <strong>10 most similar census tracts</strong> in <span style="color: orange;"><strong>orange</strong></span> with similarity scores</li>
        <li>Provides a table listing the contributing features that drove the similarity score</li>
    </ul>
    </p>
    <p><strong>Note:</strong> If you want to search for a different Census Tract, please refresh the page.</p>
</div>
""", unsafe_allow_html=True)
