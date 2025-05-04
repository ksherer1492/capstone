import streamlit as st

st.title("So, You Want To Be A Restaurant Owner? THIS IS A TEST")
st.subheader("You as a restaurant owner have a need for actionable insights to understand factors affecting business performance in a competitive market.")
st.text("This dashboard focuses on providing operational enhancements and location selection based on data-driven insights for Chicago, Austin, New York, and Los Angeles.")
st.markdown("""
We define our success by being able to answer the following key questions:
- What separates highly reviewed restaurants from poorly reviewed ones?
- What key factors can help restaurant owners determine the best locations for business among New York, Chicago, Austin, and Los Angeles?
""")
st.text("To achieve these answers data was gathered from sources such as Google Maps and the Census Bureau. Below is an outline of the different pages utlizing this data that are available on this dashboard. To visit a page select it from the bar on the left.")
st.subheader("Page Guide")
st.markdown("""
- **City Category Analysis**
  - Identifies localized strengths, weaknesses, and preferences.
  - Allows for market comparisons across cities.
- **similarity measure**
  - identifies the most similar census tracts to a selected tract based on income and demographic data
  - maps similar census tracts and allows users to see the top features contributing to the similarity
""")

##ADD ADDITIONAL PAGE SECTIONS HERE

st.text("In addition to the above details each page contains an about section that describes the analysis strategies that went into it along with how it helps to answer our key questions.")
st.subheader("Disclaimer of Data Bias")
st.markdown("""
Due to the nature of the analysis subject and the methods of data collection employed, there are several data biases that have affected the analysis of restaurants.
- **Survivorship**
    - Data is biased towards restaurants that have been successful enough to persist long term.
- **Nonresponse**
    - Only people responding to the census inquiry are included in the data. (The Census Bureau accounts for this with weights and supplies a margin of error.)
""")