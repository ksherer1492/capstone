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
tab1, tab2= st.tabs(["🗺️ Census Track Cluster Explorer", "ℹ️ About"])

with tab1:
    # ───────────── Inject CSS ─────────────
    st.markdown("""
    <style>
      /* style every markdown container as a dark, scrollable box */
      div[data-testid="stMarkdownContainer"] {
        background: #212121;
        color: #fff;
        background-color: transparent !important;
        padding: 1rem;
        border-radius: 8px;
        max-height: 80vh;
        overflow-y: auto;
      }
      .error-msg {
        color: #f88;
        font-weight: bold;
      }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 3], gap='small')

    # ───────────── Advisor pane ─────────────
    with left:
        # Show any error above the box
        if st.session_state.get("error_msg"):
            st.markdown(
                f"<div class='error-msg'>{st.session_state.error_msg}</div>",
                unsafe_allow_html=True
            )

        # One placeholder for entire header + GPT output
        advisor_placeholder = st.empty()

        # Initialize state if missing
        if "last_gid" not in st.session_state:
            st.session_state.last_gid = ""
        if "advisor_content" not in st.session_state:
            st.session_state.advisor_content = "Click a tract →"

        # Compose the Markdown
        header_md = f"## Census Tract {st.session_state.last_gid} Overview\n\n"
        body_md   = st.session_state.advisor_content
        full_md   = header_md + body_md

        # Render
        advisor_placeholder.markdown(full_md)

    # ───────────── Map pane ─────────────
    with right:
        TRACTS['cluster_cat'] = TRACTS['cluster'].astype(str)
        fig = px.choropleth_map(
            TRACTS,
            geojson=geojson,
            locations='GEOIDFQ',
            featureidkey='properties.GEOIDFQ',
            color='cluster_cat',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data={
                'cluster_cat': False,
                'cluster': True,
                'driver_1': True,
                'driver_2': True,
                'driver_3': True
            },
            center=dict(lat=41.8781, lon=-87.6298),
            zoom=11,
            opacity=0.7,
            height=1025
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            mapbox_style='carto-darkmatter' if MAPBOX_TOKEN else 'open-street-map',
            mapbox_accesstoken=MAPBOX_TOKEN or None,
            showlegend=False
        )
        fig.update_traces(
            selected=dict(marker=dict(opacity=0.7)),
            unselected=dict(marker=dict(opacity=0.3)),
            marker_line_color='#000',
            marker_line_width=0.5
        )
        map_event = st.plotly_chart(
            fig,
            key='main-map',
            on_select='rerun',
            selection_mode='points',
            use_container_width=True
        )

    # ───────── Click handler for streaming GPT ─────────
    if isinstance(map_event, dict) and map_event.selection.points:
        gid = map_event.selection.points[0]['location']
        now_ms = time.time() * 1000
        if now_ms - st.session_state.get("last_click_ts", 0) > DEBOUNCE_MS:
            # store the new tract ID
            st.session_state.last_click_ts = now_ms
            st.session_state.last_gid = gid

            # clear out the old content
            st.session_state.advisor_content = ""
            row = TRACTS[TRACTS.GEOIDFQ == gid].iloc[0]
            prompt = build_prompt(row)
            st.session_state.error_msg = ""

            # stream the new response
            for chunk in openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                      "role": "system",
                      "content": (
                        "You are a data-driven advisor. Output plain Markdown. "
                        "Do not include triple backticks or ANY code blocks. "
                        "DON'T BE RACIST. AVOID DIRECTLY CALLING OUT RACIAL GROUPS OR ETHNICITIES, "
                        "AVOID RACIAL AND RACIST BIAS, DO NOT HALLUCINATE"
                      )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=True
            ):
                st.session_state.advisor_content += chunk.choices[0].delta.content or ""
                
                # re-compose & re-render header + body in the box
                header_md = f"## Census Tract {st.session_state.last_gid} Overview\n\n"
                full_md   = header_md + st.session_state.advisor_content
                advisor_placeholder.markdown(full_md)


    with tab2:
        # ───────── About Page ─────────
        st.markdown("# Analysis of Census Tract Similarity Matrix")
        st.markdown(''' 
                    ### This section will discuss both the analysis done, and how they contributed to the overall success of our project.
                    #### Analysis Performed:
                    1. Clustering of the census tract via hierarchical clustering.
                    2. Programmatic thresold selection via the plotting of sil score vs threshold vs 4-city cluster count.
                    3. Cluster filtering via the knee method.
                    4. Clustering quality evaluation via silhouette score, and cluster visualization via MDS.
                    5. Cluster membership driver analysis via centroid feature importance.
                    6. Exporting of clustered census tracts, census feature data, and cluster membership driver data to GeoJSON for use in the Streamlit dashboard.
                    7. Streamlit dashboard page creation for the exploration of clustered census tracts, and cluster membership driver data.

                    ### Following are some important figures from the analysis along with explanations of their significance:
                    #### Hierarchical Clustering Dendrogram:
                    ''')
        st.image("images/dendrogram.png")
        st.markdown('''
                    The dendrogram represents a visualization of the hierarchical clustering process. The y-axis represents the distance between clusters, while the x-axis represents the individual census tracts. The height at which two clusters are merged indicates their similarity. A lower height indicates a higher similarity between the clusters.
                    The colors represent rudamentry cluster membership, however this was later refined via programmatic threshold selection via the plotting of sil score vs threshold vs 4-city cluster count which will be discussed next.
                    
                    #### Programmatic Threshold Selection via the plotting of sil score vs threshold vs 4-city cluster count:
                    ''')
        st.image("images/threshold.png")
        st.markdown('''
                    The plot above shows the relationship between the silhouette score, threshold, and the number of clusters with membership in all four cities. The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined clusters.
                    The optimal threshold is where the silhouette score, and 4-city cluster count are highest.
                    This analysis was used to select a threshold for clustering that maximizes the silhouette score while also promoting maximum cluster city diversity.
                    
                    #### Cluster Filtering via the Knee Method:
                    ''')
        st.image("images/knee.png")
        st.markdown('''
                    The plot above shows the relationship between cluster size and cluster rank. The knee method was used to filter out small clusters that are not representative of the overall data. The knee point is where the rate of change in cluster size begins to level off, indicating that the clusters following are not meaningful in classification.
                    
                    #### Clustering quality evaluation via silhouette score, and cluster visualization via MDS:
                    ''')
        st.image("images/silsample.png")
        st.image("images/mds.png")
        st.markdown('''
                    The horizonrtal bar plot shows the silhouette score for each cluster. The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined clusters. There are some clusters with silhouette scores close to 0, indicating that they are not well-defined. However, some of the clusters have silouette scores above 0.2, which given our dense feature space is a decent score.
                    The MDS plot projects all observations into two dimensions, with each point colored by its cluster label. Distances in this 2D view approximate the original pairwise dissimilarities: points that lie close together were deemed similar by the clustering algorithm, while those farther apart belong to more distinct clusters.
                    
                    #### Cluster membership driver analysis via centroid feature importance:
                    ''')
        st.image("images/clusters.png")
        st.markdown('''
                    The following steps were used to determine the cluster membership drivers:
                    1. Standardize all numeric ethnicity and income census features with StandardScaler()

                    2. Calculate the mean value of those features within each cluster to form cluster centroids

                    3. Rank the features with the largest absolute values at each centroid to determine the top 5 features most distinctive to each cluster.

                    4. Translated complicated census feature names into human readable descriptions.

                    5. Exported cluster, driver, census and geographic data into a geojson file for input into the dashboard.

                    #### Final Result / Success Contribution:
                    ''')
        st.markdown('''
                    The generated GeoJSON file was then used as the data source for the Streamlit dashboard. The dashboard allows users to explore the clustered census tracts and their cluster membership drivers. Users can click on a tract to view its GPT generated Deep Insights overview, including a range of different census data features, and the top 5 cluster membership drivers. The dashboard also provides a recommendation section that summarizes the data-driven insights and suggests a restaurant concept based on the analysis.
                    This portion of the project contributed to the overall success of the project by enhancing our geospatial analysis of the census tracts and census features in the four cities. The clustering and analysis provided valuable insights into the similarities and differences between the census tracts, which can be used to inform business decisions and strategies. The dashboard also provides an interactive and user-friendly way to explore the data, making it accessible to a wider audience.


                    #### Known Issues:
                    1. Clusters have cluster drivers that are the same for all displayed mouseover drivers. Cluster drivers were filtered to show only unique values for each cluster, however for unknown reasons some clusters have all the same drivers for their top 5 drivers. 
                    2. The GPT generated Deep Insights overview is not always accurate or relevant to the data. This is a known issue with the OpenAI API and is not specific to this project. The API is trained on a large dataset and may not always produce accurate or relevant results for specific use cases.
                    3. The dashboard performance is slow, even though we cache the data for quick reloading. This is an area of improvement for furutre iterations of the project.
                    4. The GPT generated Deep Insights overview will sometimes name the tract name twice, this can be fixed by clicking another tract, but the cause of this issue is unknown.
                    5. On initial load, the map and deep insights advisor appear to be squished together. The cause of this issue is not currently known. 
                    ''')