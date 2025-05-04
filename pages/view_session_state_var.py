

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Session State Viewer", layout="wide")

st.title("Session State Viewer")

# Selected GEOID
selected_geoid = st.session_state.get("selected_geoid", None)
if selected_geoid:
    st.markdown(f"**Selected Census Tract:** `{selected_geoid}`")
else:
    st.warning("No selected tract found in session state.")

# Similar Tracts
similar_tracts = st.session_state.get("similar_tracts", {})
if similar_tracts:
    st.markdown("### Similar Tracts and Scores")
    df = pd.DataFrame(list(similar_tracts.items()), columns=["FIPS", "Similarity Score"])
    df = df.sort_values("Similarity Score", ascending=False).reset_index(drop=True)
    st.dataframe(df)
else:
    st.warning("No similar tracts stored in session state.")
