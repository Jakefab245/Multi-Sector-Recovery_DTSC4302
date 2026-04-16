"""
Education Map: Bachelor's Degree Attainment by State (2011-2024)
DTSC 4302 - Multi-Sector Recovery After Natural Disasters
Author: Wendy Ceja-Huerta

Requirements:
    pip install folium pandas requests

Run:
    python education_map.py
    -> Outputs: education_map.html
"""

import pandas as pd
import folium
import requests
from functools import reduce

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────

def clean_census_csv(filepath, year):
    df = pd.read_csv(filepath, skiprows=5, encoding='utf-8-sig')
    df.columns = ['State', 'Pct_Bachelors', 'MoE']
    df = df.dropna(subset=['State'])
    df = df[df['State'].str.strip() != '']
    df['Pct_Bachelors'] = df['Pct_Bachelors'].str.replace('%', '').str.strip().astype(float)
    df = df[['State', 'Pct_Bachelors']].copy()
    df.columns = ['State', f'pct_{year}']
    df['State'] = df['State'].str.strip()
    df = df[~df['State'].str.contains('Puerto Rico|United States', na=False)]
    return df

# ── Update these paths to match where your files live ──
files = {
    2011: '2011gradpercent.csv',
    2017: '2017gradpercent.csv',
    2019: '2019gradpercent.csv',
    2020: '2020gradpercent.csv',
    2024: '2024gradpercent.csv',
}

dfs = [clean_census_csv(path, year) for year, path in files.items()]
df = reduce(lambda l, r: pd.merge(l, r, on='State', how='outer'), dfs)
df['change_2011_2024'] = (df['pct_2024'] - df['pct_2011']).round(1)

state_abbr = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC',
}
df['abbr'] = df['State'].map(state_abbr)

# Top 15 high-exposure states from the paper (Fig. 1)
HIGH_EXPOSURE_STATES = {
    'Texas', 'Florida', 'Louisiana', 'California', 'Oklahoma',
    'Tennessee', 'Mississippi', 'Missouri', 'Georgia', 'Kentucky',
    'North Carolina', 'Virginia', 'South Carolina', 'Arkansas', 'Iowa'
}

STATE_CENTROIDS = {
    'Alabama': [32.78, -86.83], 'Alaska': [64.20, -153.49],
    'Arizona': [34.27, -111.66], 'Arkansas': [34.89, -92.44],
    'California': [37.18, -119.47], 'Colorado': [38.99, -105.55],
    'Connecticut': [41.62, -72.73], 'Delaware': [38.99, -75.51],
    'Florida': [28.63, -82.45], 'Georgia': [32.64, -83.44],
    'Hawaii': [20.29, -156.37], 'Idaho': [44.35, -114.61],
    'Illinois': [40.04, -89.20], 'Indiana': [39.89, -86.28],
    'Iowa': [42.07, -93.50], 'Kansas': [38.49, -98.38],
    'Kentucky': [37.53, -85.30], 'Louisiana': [31.07, -91.99],
    'Maine': [45.37, -69.24], 'Maryland': [39.05, -76.79],
    'Massachusetts': [42.26, -71.81], 'Michigan': [44.35, -85.41],
    'Minnesota': [46.28, -94.31], 'Mississippi': [32.74, -89.67],
    'Missouri': [38.36, -92.46], 'Montana': [47.05, -109.63],
    'Nebraska': [41.54, -99.80], 'Nevada': [39.33, -116.63],
    'New Hampshire': [43.68, -71.58], 'New Jersey': [40.19, -74.67],
    'New Mexico': [34.41, -106.11], 'New York': [42.95, -75.53],
    'North Carolina': [35.56, -79.39], 'North Dakota': [47.45, -100.47],
    'Ohio': [40.29, -82.79], 'Oklahoma': [35.59, -97.49],
    'Oregon': [43.93, -120.56], 'Pennsylvania': [40.88, -77.80],
    'Rhode Island': [41.68, -71.56], 'South Carolina': [33.92, -80.90],
    'South Dakota': [44.44, -100.23], 'Tennessee': [35.86, -86.35],
    'Texas': [31.48, -99.33], 'Utah': [39.32, -111.09],
    'Vermont': [44.07, -72.67], 'Virginia': [37.52, -78.85],
    'Washington': [47.38, -120.45], 'West Virginia': [38.64, -80.62],
    'Wisconsin': [44.62, -89.99], 'Wyoming': [42.99, -107.55],
    'District of Columbia': [38.91, -77.02],
}

# ─────────────────────────────────────────────
# 2. FETCH GEOJSON
# ─────────────────────────────────────────────

print("Fetching US state boundaries...")
geo_url = "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
state_geo = requests.get(geo_url).json()
print("GeoJSON loaded.")

# ─────────────────────────────────────────────
# 3. BUILD MAP
# ─────────────────────────────────────────────

m = folium.Map(location=[38, -97], zoom_start=4, tiles='CartoDB positron')

def add_choropleth(map_obj, col, label, color='YlOrRd', show=False):
    """Add a choropleth layer with a hover tooltip showing that year's percent."""
    layer_data = df[['abbr', col]].dropna()
    value_lookup = dict(zip(layer_data['abbr'], layer_data[col]))

    cp = folium.Choropleth(
        geo_data=state_geo,
        name=label,
        data=layer_data,
        columns=['abbr', col],
        key_on='feature.id',
        fill_color=color,
        fill_opacity=0.75,
        line_opacity=0.3,
        legend_name=label,
        highlight=True,
        show=show,
    )
    cp.add_to(map_obj)

    # Inject the correct year's value into each state feature for tooltip
    for feature in cp.geojson.data['features']:
        abbr = feature['id']
        state_name = feature['properties'].get('name', abbr)
        val = value_lookup.get(abbr)
        feature['properties']['tooltip'] = state_name

    folium.GeoJsonTooltip(
        fields=['tooltip'],
        aliases=[''],
        style=(
            "background-color:white;color:#333;font-family:Arial;"
            "font-size:13px;padding:6px;border-radius:4px;"
        ),
        sticky=True,
    ).add_to(cp.geojson)

# Each layer gets its own tooltip with that year's data
add_choropleth(m, 'pct_2011',
               "2011: % Bachelor's Degree+ (Tornado Outbreak / MS River Floods)",
               show=True)
add_choropleth(m, 'pct_2017',
               "2017: % Bachelor's Degree+ (Hurricane Harvey & Irma)")
add_choropleth(m, 'pct_2019',
               "2019: % Bachelor's Degree+ (Midwest Floods & Hurricane Dorian)")
add_choropleth(m, 'pct_2020',
               "2020: % Bachelor's Degree+ (Hurricane Laura, CA Wildfires & COVID-19)")
add_choropleth(m, 'pct_2024',
               "2024: % Bachelor's Degree+ (Present Day Comparison)")
add_choropleth(m, 'change_2011_2024',
               "Change 2011-2024 (percentage points)", color='BuGn')

# ─────────────────────────────────────────────
# 4. STATE POPUP MARKERS
# ─────────────────────────────────────────────

markers = folium.FeatureGroup(name="State Details (click a marker)", show=True)

for _, row in df.iterrows():
    state = row['State']
    if state not in STATE_CENTROIDS:
        continue

    lat, lon = STATE_CENTROIDS[state]
    is_high_exposure = state in HIGH_EXPOSURE_STATES
    marker_color = "#e02424" if is_high_exposure else "#1a56db"
    exposure_label = "Top 15 High-Exposure State" if is_high_exposure else "Lower disaster exposure"

    popup_html = f"""
<div style="font-family:Arial,sans-serif;width:270px;font-size:13px;">
  <h4 style="margin:0 0 6px;padding-bottom:4px;border-bottom:2px solid #444;">
    {state} ({row.get('abbr', '')})
  </h4>
  <table style="width:100%;border-collapse:collapse;">
    <tr style="background:#dbeafe">
      <td colspan="2" style="padding:3px 4px;font-weight:bold;">
        % Population 25+ with Bachelor's Degree or Higher
      </td>
    </tr>
    <tr><td style="padding:2px 4px">2011</td><td><b>{row['pct_2011']:.1f}%</b></td></tr>
    <tr style="background:#f3f4f6">
      <td style="padding:2px 4px">2017</td><td><b>{row['pct_2017']:.1f}%</b></td>
    </tr>
    <tr><td style="padding:2px 4px">2019</td><td><b>{row['pct_2019']:.1f}%</b></td></tr>
    <tr style="background:#f3f4f6">
      <td style="padding:2px 4px">2020</td><td><b>{row['pct_2020']:.1f}%</b></td>
    </tr>
    <tr><td style="padding:2px 4px">2024</td><td><b>{row['pct_2024']:.1f}%</b></td></tr>
    <tr style="background:#dcfce7">
      <td style="padding:2px 4px">Change (2011-2024)</td>
      <td><b>+{row['change_2011_2024']:.1f}%</b></td>
    </tr>
    <tr style="background:#fff3e0">
      <td colspan="2" style="padding:3px 4px;">{exposure_label}</td>
    </tr>
  </table>
</div>"""

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color='white',
        weight=1.5,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.9,
        popup=folium.Popup(popup_html, max_width=290),
        tooltip=f"<b>{state}</b> | 2024: {row['pct_2024']:.1f}% | Change: +{row['change_2011_2024']:.1f}%",
    ).add_to(markers)

markers.add_to(m)

# ─────────────────────────────────────────────
# 5. TITLE + LEGEND
# ─────────────────────────────────────────────

m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
     background:white;padding:9px 18px;border-radius:8px;border:2px solid #444;
     z-index:9999;font-family:Arial;text-align:center;
     box-shadow:2px 3px 8px rgba(0,0,0,.25);">
  <b style="font-size:15px;">College-Level Educational Attainment by State</b><br>
  <span style="font-size:11px;color:#666;">
    % Population 25+ with Bachelor's Degree or Higher &bull;
    2011-2024 &bull; Source: U.S. Census Bureau ACS S1501
  </span>
</div>"""))

m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;bottom:28px;right:22px;z-index:9999;background:white;
     padding:10px 14px;border-radius:8px;border:1.5px solid #999;
     font-family:Arial;font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,.2);">
  <b>State Markers</b><br>
  <span style="color:#e02424;">&#9632;</span> Top 15 High-Exposure State<br>
  <span style="color:#1a56db;">&#9632;</span> Lower Exposure State<br>
  <hr style="margin:5px 0;">
  <small>Hover over a state to see its %.<br>
  Click a marker for all years of data.<br>
  Use layer control (top-right) to switch years.</small>
</div>"""))

folium.LayerControl(collapsed=False).add_to(m)

# ─────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────

out = "education_map.html"
m.save(out)
print(f"\nMap saved to: {out}")
print("Open education_map.html in your browser to view!")
