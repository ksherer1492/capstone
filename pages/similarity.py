import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium import GeoJson, GeoJsonTooltip, GeoJsonPopup

# Set page config
st.set_page_config(page_title="Census Tracts Map", layout="wide")

st.title("Simplified Census Tracts - Interactive Map")

# Cache the GeoJSON loading
@st.cache_data
def load_geojson(filepath):
    return gpd.read_file(filepath)

# Since your file is in the same directory as your Streamlit app
geojson_path = "simplified_cities_00005.geojson"
simplified_gdf = load_geojson(geojson_path)

# User text input for GEOID search
search_geoid = st.text_input("Enter Census Tract GEOID to locate (e.g., 17031280100):")

# Initialize folium map
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='cartodb positron')

# Define colors for cities
city_colors = {
    'Austin': 'green',
    'Chicago': 'blue',
    'NYC': 'purple',
    'LA': 'orange'
}

# Add layers to the map
for city in simplified_gdf['city'].unique():
    city_gdf = simplified_gdf[simplified_gdf['city'] == city]

    # Filter out invalid or empty geometries
    city_gdf = city_gdf[city_gdf.is_valid & ~city_gdf.is_empty].copy()

    geojson = folium.GeoJson(
        data=city_gdf.__geo_interface__,
        name=f"{city} Census Tracts",
        style_function=lambda feature, color=city_colors[city]: {
            'fillColor': color,
            'color': color,
            'weight': 1,
            'fillOpacity': 0.2
        },
        tooltip=GeoJsonTooltip(
            fields=['GEOID'],
            aliases=['Census Tract:'],
            sticky=True
        ),
        popup=GeoJsonPopup(
            fields=['GEOID', 'city'],
            aliases=['Census Tract:', 'City:']
        )
    )
    geojson.add_to(m)

# If user entered a GEOID, attempt to find and zoom to it
if search_geoid:
    tract = simplified_gdf[simplified_gdf['GEOID'] == search_geoid]

    if not tract.empty:
        centroid = tract.geometry.iloc[0].centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            popup=f"Census Tract: {search_geoid}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

        # Move map to the centroid of the found tract
        m.location = [centroid.y, centroid.x]
        m.zoom_start = 13  # Adjust zoom level as desired

        st.success(f"Census Tract {search_geoid} found and highlighted on the map!")
    else:
        st.warning("Census Tract not found. Please check the GEOID and try again.")

# Add layer control
folium.LayerControl().add_to(m)

# Display map in Streamlit
st_data = st_folium(m, width=1200, height=700)

# Optionally, print what the user clicked
if st_data and st_data.get("last_clicked"):
    st.write("You clicked at:", st_data["last_clicked"])

