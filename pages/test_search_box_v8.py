import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from similarity_engine import run_dot_product_similarity  

# wide layout for more space
st.set_page_config(page_title="Search Census Tract", layout="wide")

# optional global page padding
st.markdown("""
    <style>
    .main {
        padding-left: 50px;
        padding-right: 50px;
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# file paths
income_path = "/Users/Kevin/Desktop/capstone/data/all/census/census_income_FIPS.csv"
race_path = "/Users/Kevin/Desktop/capstone/data/all/census/census_race_FIPS.csv"
matched_path = "/Users/Kevin/Desktop/capstone/data/all/census/matched_tracts_padded.csv"
geojson_path = "/Users/Kevin/Desktop/Mizzou Documents/Capstone/streamlit/simplified_cities_00005.geojson"

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

if search_geoid:
    padded = search_geoid.zfill(11)
    if padded in matched_geoids:
        st.success(f"Census Tract {padded} found in matched dataset.")
        st.session_state.selected_geoid = padded

        # separate target tract and all others
        target_tract = tracts_gdf[tracts_gdf['GEOID'] == padded]
        other_tracts = tracts_gdf[tracts_gdf['GEOID'] != padded]

        if not target_tract.empty:
            centroid = target_tract.geometry.centroid.iloc[0]
            lat, lon = centroid.y, centroid.x

            # create map centered on selected tract
            m = folium.Map(location=[lat, lon], zoom_start=12, tiles="cartodb positron")

            # overlay all other tracts in gray
            folium.GeoJson(
                other_tracts.__geo_interface__,
                name="All Census Tracts",
                style_function=lambda x: {
                    'fillColor': 'gray',
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.1
                }
            ).add_to(m)

            # highlight selected tract in yellow
            folium.GeoJson(
                target_tract.__geo_interface__,
                name="Selected Tract",
                style_function=lambda x: {
                    'fillColor': 'yellow',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.6
                }
            ).add_to(m)

            # add green marker at center of tract
            folium.Marker(
                location=[lat, lon],
                popup=f"Census Tract {padded}",
                icon=folium.Icon(color='green', icon='star')
            ).add_to(m)

            st.markdown("<br>", unsafe_allow_html=True)

            # 2-column layout: map wide, table narrower
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Selected Tract and Nearby Tracts")
                st_folium(m, width=950, height=600)

            with col2:
                run_dot_product_similarity(padded, income_path, race_path)

    else:
        st.warning(f"Census Tract {padded} not found in matched dataset.")
