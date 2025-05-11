# So, You Want To Be A Restaurant Owner?

**Capstone Project – Master of Science in Data Science & Analytics**

This Streamlit dashboard is designed for aspiring or current restaurant owners seeking data-driven insights into optimal locations and key factors influencing restaurant success in four major U.S. cities: **Chicago, Austin, New York, and Los Angeles**.

## Objective

To answer these key business questions:

- What separates highly reviewed restaurants from poorly reviewed ones?
- What key factors help determine the best restaurant locations across different cities?

## Dashboard Features

This interactive dashboard provides four main analysis pages:

### 1. **City Category Analysis**
- Identify localized strengths, weaknesses, and preferences.
- Compare market conditions across cities.

### 2. **Similarity Measure**
- Select a census tract and view the top 10 most similar tracts based on income and demographic features.
- Visualize similarity via dot product and see key contributing variables.

### 3. **Census Tract Clustering Analysis**
- Explore clusters of census tracts grouped by income and race characteristics.
- Understand how different areas relate and differ from each other.

### 4. **Search Census Tract**
- Input an address and automatically retrieve its corresponding census tract using Google Maps and the Census Geocoder APIs.

Each page includes an **About** section explaining the methodology and how it supports answering the project’s key questions.

## Data Sources

- [Google Maps API](https://developers.google.com/maps/documentation)
- [U.S. Census Bureau](https://www.census.gov/data.html)
- Public restaurant inspection datasets
- GeoJSON files and shapefiles for census tracts

## Disclaimer of Data Bias

- **Survivorship Bias**: Data may favor restaurants that survived long enough to generate reviews.
- **Nonresponse Bias**: Census data includes only respondents; statistical weights and margins of error are applied to compensate.

## Getting Started

### Run Locally

1. Clone the repo:
    ```bash
    git clone https://github.com/ksherer1492/capstone.git
    cd capstone
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your API keys in `.streamlit/secrets.toml`:
    ```toml
    [google]
    api_key = "YOUR_GOOGLE_API_KEY"

    openai_key = "YOUR_OPENAI_KEY"
    ```

4. Run the app:
    ```bash
    streamlit run home.py
    ```

Or view the hosted version here: [https://capstone-dashboard-mizzou.streamlit.app](https://capstone-dashboard-mizzou.streamlit.app)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Kevin Sherer, Reid Lawson, Ryan Russell, Daniel Bassett
MSDSA Capstone Project – University of Missouri  
GitHub: [@ksherer1492](https://github.com/ksherer1492)

