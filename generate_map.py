"""Standalone script to generate the FEMA + Election Folium map as HTML.
Does NOT modify any existing notebook or data files."""

import pandas as pd
import numpy as np
import requests
import folium
import warnings
warnings.filterwarnings("ignore")

# Disaster Exposure - LAUS
df_laus = pd.read_csv("FEMA_Disaster_2006_2025/laus_with_fema_disaster_exposure_2006_2025.csv")
df_laus["state"] = df_laus["state"].str.strip().str.title()
laus_state = df_laus.groupby("state").agg(
    total_disaster_months=("disaster_count_month", "sum"),
    avg_exposure_12m=("disaster_exposure_12m", "mean")
).reset_index()

#FEMA Aid
df_d = pd.read_csv("disaster_data_export.csv")
df_d["year"] = pd.to_numeric(df_d["year"], errors="coerce")
df_d["state"] = df_d["state"].str.strip().str.title()

us_states_list = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire",
    "New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio",
    "Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota",
    "Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia",
    "Wisconsin","Wyoming","District Of Columbia"
]
df_d = df_d[df_d["state"].isin(us_states_list)].copy()

for col in ["ihp_total","pa_total","cdbg_dr_allocation","sba_total_approved_loan_amount"]:
    df_d[col] = pd.to_numeric(df_d[col], errors="coerce").fillna(0)

df_d["total_fema_aid"] = (df_d["ihp_total"] + df_d["pa_total"] +
                           df_d["cdbg_dr_allocation"] + df_d["sba_total_approved_loan_amount"])
df_d["recovery_days"] = pd.to_numeric(df_d["frn1_days_since_disaster"], errors="coerce")

# Compute declaration response time (fallback for states without CDBG-DR data)
for col in ["incident_start", "declaration_date"]:
    df_d[col] = pd.to_datetime(df_d[col], errors="coerce")
df_d["declaration_response_days"] = (df_d["declaration_date"] - df_d["incident_start"]).dt.days

def get_admin(y):
    if y <= 2008: return "Bush (R) 2001-08"
    elif y <= 2016: return "Obama (D) 2009-16"
    elif y <= 2020: return "Trump (R) 2017-20"
    else: return "Biden (D) 2021-25"
df_d["administration"] = df_d["year"].apply(get_admin)

state_aid = df_d.groupby("state").agg(
    disaster_count=("year","count"),
    total_fema_aid=("total_fema_aid","sum"),
    ihp_total=("ihp_total","sum"),
    pa_total=("pa_total","sum"),
    avg_recovery_days=("recovery_days","mean"),
    avg_declaration_response=("declaration_response_days","mean"),
    incident_types=("incident_type", lambda x: ", ".join(sorted(x.dropna().unique())))
).reset_index()
state_aid["aid_per_disaster"] = state_aid["total_fema_aid"] / state_aid["disaster_count"]

# Election Information
df_e = pd.read_csv("1976-2020-president.csv")
df_e["candidatevotes"] = pd.to_numeric(df_e["candidatevotes"], errors="coerce")
df_e = df_e.dropna(subset=["candidatevotes"])
df_e["state"] = df_e["state"].str.strip().str.title()
df_e = df_e[df_e["year"].isin([2008, 2012, 2016, 2020])].copy()

#Party Simplification
def simp(p):
    p = str(p).upper()
    if "DEMOCRAT" in p: return "DEM"
    if "REPUBLICAN" in p: return "REP"
    return "OTH"

#Election Winners
winners = df_e.groupby(["year","state"]).apply(
    lambda x: simp(x.loc[x["candidatevotes"].idxmax(), "party_detailed"])
).reset_index()
winners.columns = ["year","state","winner"]

# Pivoting of Election winners
epiv = winners.pivot(index="state", columns="year", values="winner").reset_index()
epiv.columns = ["state"] + [f"w{int(c)}" for c in epiv.columns[1:]]

# Political Leaning Classification
def get_lean(row):
    vals = [row.get(c, None) for c in ["w2008","w2012","w2016","w2020"]]
    d = sum(1 for v in vals if v == "DEM")
    r = sum(1 for v in vals if v == "REP")
    if d == 4: return "Solid DEM"
    if r == 4: return "Solid REP"
    if d == 3: return "Lean DEM"
    if r == 3: return "Lean REP"
    return "Swing"
epiv["political_lean"] = epiv.apply(get_lean, axis=1)

#Merging datasets with outerjoin on state, then filtering by states with valid abbreviations
state_codes = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
    "Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
    "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
    "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    "District Of Columbia":"DC"
}

#Merging datasets with outerjoin on state, then filter by abbreviations
full = laus_state.merge(state_aid, on="state", how="outer").merge(epiv, on="state", how="left")
full["state_abbr"] = full["state"].map(state_codes)
full = full[full["state_abbr"].notna()].copy()
for c in ["political_lean","w2008","w2012","w2016","w2020"]:
    if c not in full.columns: full[c] = "N/A"
    else: full[c] = full[c].fillna("N/A")
full["total_fema_aid"] = full["total_fema_aid"].fillna(0)
full["disaster_count"] = full["disaster_count"].fillna(0)

# Hybrid recovery: use CDBG-DR days where available, fall back to declaration response time
full["hybrid_recovery_days"] = full["avg_recovery_days"].combine_first(full["avg_declaration_response"])
full["recovery_source"] = np.where(
    full["avg_recovery_days"].notna(), "CDBG-DR Grant Timeline",
    np.where(full["avg_declaration_response"].notna(), "Declaration Response Time", "No Data"))

print(f"  {len(full)} states merged. Building map...")
print(f"  Recovery coverage: {full['avg_recovery_days'].notna().sum()} CDBG-DR + "
      f"{(full['avg_declaration_response'].notna() & full['avg_recovery_days'].isna()).sum()} declaration fallback = "
      f"{full['hybrid_recovery_days'].notna().sum()} total")

# Starting to build Map
state_geo = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
).json()
print("  GeoJSON fetched.")

info_dict = full.set_index("state_abbr").to_dict("index")

m = folium.Map(location=[38, -97], zoom_start=4, tiles="CartoDB positron")

# Layer 1: Total FEMA Aid 2015-2025
folium.Choropleth(
    geo_data=state_geo, name="Total FEMA Aid 2015-2025",
    data=full[["state_abbr","total_fema_aid"]].dropna(),
    columns=["state_abbr","total_fema_aid"], key_on="feature.id",
    fill_color="YlOrRd", fill_opacity=0.75, line_opacity=0.3,
    legend_name="Total FEMA Aid 2015-2025 ($)", highlight=True, show=True
).add_to(m)

# Layer 2: Avg 12-Month Disaster Exposure 2006-2025
folium.Choropleth(
    geo_data=state_geo, name="Avg 12-Month Disaster Exposure 2006-2025",
    data=full[["state_abbr","avg_exposure_12m"]].dropna(),
    columns=["state_abbr","avg_exposure_12m"], key_on="feature.id",
    fill_color="BuPu", fill_opacity=0.75, line_opacity=0.3,
    legend_name="Avg 12-Month Disaster Exposure 2006-2025", highlight=True, show=False
).add_to(m)

# Layer 3: Hybrid Recovery Time (CDBG-DR where available, declaration response as fallback)
_rec_df = full[["state_abbr","hybrid_recovery_days"]].dropna()
if len(_rec_df) > 1:
    folium.Choropleth(
        geo_data=state_geo,
        name="Avg Response/Recovery Time (Hybrid)",
        data=_rec_df,
        columns=["state_abbr","hybrid_recovery_days"], key_on="feature.id",
        fill_color="RdYlGn_r", fill_opacity=0.75, line_opacity=0.3,
        legend_name="Avg Response Time (days) — CDBG-DR or Declaration Response fallback",
        highlight=True, show=False
    ).add_to(m)

# Layer 4: Political Leaning GeoJSON
lean_pal = {
    "Solid DEM":"#1a56db","Lean DEM":"#76a9fa","Swing":"#9ca3af",
    "Lean REP":"#f98080","Solid REP":"#e02424","N/A":"#d1d5db"
}
def style_fn(feat):
    lean = info_dict.get(feat.get("id",""), {}).get("political_lean", "N/A")
    return {"fillColor": lean_pal.get(lean, "#d1d5db"), "fillOpacity": 0.70,
            "color": "white", "weight": 1.2}
folium.GeoJson(
    state_geo, name="Political Leaning 2008-2020",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["name"], aliases=["State:"],
        style="background-color:white;color:#333;font-family:Arial;font-size:12px;padding:8px;"
    ),
    show=False
).add_to(m)

# State Markers with Detailed Popups
state_centroids = {
    "AL":[32.78,-86.83],"AK":[64.20,-153.49],"AZ":[34.27,-111.66],"AR":[34.89,-92.44],
    "CA":[37.18,-119.47],"CO":[38.99,-105.55],"CT":[41.62,-72.73],"DE":[38.99,-75.51],
    "FL":[28.63,-82.45],"GA":[32.64,-83.44],"HI":[20.29,-156.37],"ID":[44.35,-114.61],
    "IL":[40.04,-89.20],"IN":[39.89,-86.28],"IA":[42.07,-93.50],"KS":[38.49,-98.38],
    "KY":[37.53,-85.30],"LA":[31.07,-91.99],"ME":[45.37,-69.24],"MD":[39.05,-76.79],
    "MA":[42.26,-71.81],"MI":[44.35,-85.41],"MN":[46.28,-94.31],"MS":[32.74,-89.67],
    "MO":[38.36,-92.46],"MT":[47.05,-109.63],"NE":[41.54,-99.80],"NV":[39.33,-116.63],
    "NH":[43.68,-71.58],"NJ":[40.19,-74.67],"NM":[34.41,-106.11],"NY":[42.95,-75.53],
    "NC":[35.56,-79.39],"ND":[47.45,-100.47],"OH":[40.29,-82.79],"OK":[35.59,-97.49],
    "OR":[43.93,-120.56],"PA":[40.88,-77.80],"RI":[41.68,-71.56],"SC":[33.92,-80.90],
    "SD":[44.44,-100.23],"TN":[35.86,-86.35],"TX":[31.48,-99.33],"UT":[39.32,-111.09],
    "VT":[44.07,-72.67],"VA":[37.52,-78.85],"WA":[47.38,-120.45],"WV":[38.64,-80.62],
    "WI":[44.62,-89.99],"WY":[42.99,-107.55],"DC":[38.91,-77.02]
}
#Creation of State Info Markers
mkr = folium.FeatureGroup(name="State Info Markers", show=True)

def pc(p):
    return "#1a56db" if p == "DEM" else ("#e02424" if p == "REP" else "#9ca3af")

for _, row in full.iterrows():
    ab = row.get("state_abbr")
    if not ab or ab not in state_centroids:
        continue
    lat, lon = state_centroids[ab]
    aid  = row.get("total_fema_aid", 0) or 0
    nd   = int(row.get("disaster_count", 0) or 0)
    exp  = row.get("avg_exposure_12m", np.nan)
    rec_cdbg = row.get("avg_recovery_days", np.nan)
    rec_decl = row.get("avg_declaration_response", np.nan)
    rec_src  = row.get("recovery_source", "No Data")
    lean = row.get("political_lean", "N/A")
    w08, w12, w16, w20 = [row.get(f"w{y}", "N/A") for y in [2008, 2012, 2016, 2020]]
    apd  = row.get("aid_per_disaster", np.nan)

    # Build recovery row with hybrid metric and source label
    if pd.notna(rec_cdbg):
        rec_row = (f'    <tr><td colspan="2">Avg Recovery (CDBG-DR):</td>'
                   f'<td colspan="2">{rec_cdbg:.0f} days</td></tr>')
    elif pd.notna(rec_decl):
        rec_row = (f'    <tr><td colspan="2">Avg Response (Declaration):</td>'
                   f'<td colspan="2">{rec_decl:.0f} days</td></tr>')
    else:
        rec_row = ""
    a_s = f"${apd/1e6:.1f}M" if pd.notna(apd) and apd > 0 else "N/A"
    e_s = f"{exp:.1f}" if pd.notna(exp) else "N/A"

    # Constructing HTML for popup with style for readability
    popup_html = f"""
<div style="font-family:Arial,sans-serif;width:295px;font-size:13px;">
  <h4 style="margin:0 0 5px;padding-bottom:4px;border-bottom:2px solid #444;">
    {row["state"]} ({ab})</h4>
  <table style="width:100%;border-collapse:collapse;">
    <tr style="background:#dbeafe">
      <td colspan="4" style="padding:2px 4px;font-weight:bold;">Presidential Elections 2008-2020</td>
    </tr>
    <tr>
      <td style="padding:2px 4px">2008</td>
      <td style="color:{pc(w08)};font-weight:bold">{w08}</td>
      <td style="padding:2px 4px">2012</td>
      <td style="color:{pc(w12)};font-weight:bold">{w12}</td>
    </tr>
    <tr>
      <td style="padding:2px 4px">2016</td>
      <td style="color:{pc(w16)};font-weight:bold">{w16}</td>
      <td style="padding:2px 4px">2020</td>
      <td style="color:{pc(w20)};font-weight:bold">{w20}</td>
    </tr>
    <tr>
      <td colspan="2" style="padding:2px 4px">Overall Lean:</td>
      <td colspan="2" style="font-weight:bold">{lean}</td>
    </tr>
    <tr style="background:#fff3e0">
      <td colspan="4" style="padding:2px 4px;font-weight:bold;">FEMA Aid 2015-2025</td>
    </tr>
    <tr><td colspan="2">Total Aid:</td><td colspan="2"><b>${aid/1e9:.3f}B</b></td></tr>
    <tr><td colspan="2">Disaster Events:</td><td colspan="2">{nd}</td></tr>
    <tr><td colspan="2">Aid per Event:</td><td colspan="2">{a_s}</td></tr>
    {rec_row}
    <tr style="background:#dcfce7">
      <td colspan="4" style="padding:2px 4px;font-weight:bold;">Disaster Exposure 2006-2025</td>
    </tr>
    <tr><td colspan="2">Avg 12-Mo Exposure:</td><td colspan="2">{e_s}</td></tr>
  </table>
</div>"""

    folium.CircleMarker(
        location=[lat, lon], radius=7, color="white", weight=1.5,
        fill=True, fill_color=lean_pal.get(lean, "#9ca3af"), fill_opacity=0.9,
        popup=folium.Popup(popup_html, max_width=310),
        tooltip=f"<b>{row['state']}</b> | {lean} | Aid: ${aid/1e9:.2f}B | Events: {nd}"
    ).add_to(mkr)

mkr.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# Title
m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);background:white;
     padding:9px 18px;border-radius:8px;border:2px solid #444;z-index:9999;
     font-family:Arial;text-align:center;box-shadow:2px 3px 8px rgba(0,0,0,.25);">
  <b style="font-size:15px;">FEMA Disaster Aid &amp; Election Policy Impact</b><br>
  <span style="font-size:11px;color:#666;">2006-2025 Disaster Exposure &bull;
  2015-2025 FEMA Aid &bull; 2008-2020 Presidential Elections</span>
</div>"""))

# Legend
m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;bottom:28px;right:22px;z-index:9999;background:white;
     padding:10px 14px;border-radius:8px;border:1.5px solid #999;
     font-family:Arial;font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,.2);">
  <b>Political Lean (2008-2020)</b><br>
  <span style="color:#1a56db">&#9632;</span> Solid DEM (4/4 DEM)<br>
  <span style="color:#76a9fa">&#9632;</span> Lean DEM (3/4 DEM)<br>
  <span style="color:#9ca3af">&#9632;</span> Swing State (split)<br>
  <span style="color:#f98080">&#9632;</span> Lean REP (3/4 REP)<br>
  <span style="color:#e02424">&#9632;</span> Solid REP (4/4 REP)<br>
  <hr style="margin:5px 0;">
  <small>Use layer control (top-right) to toggle views</small>
</div>"""))

# Policy impact summary panel (bottom-left)
admin_order = {"Bush (R) 2001-08":0,"Obama (D) 2009-16":1,"Trump (R) 2017-20":2,"Biden (D) 2021-25":3}
adm = (df_d.groupby("administration")
       .agg(n=("year","count"), aid=("total_fema_aid","sum"),
            avg_apd=("total_fema_aid","mean"), avg_rec=("recovery_days","mean"))
       .reset_index())
adm["ord"] = adm["administration"].map(admin_order)
adm = adm.sort_values("ord")

rows_html = ""
admin_colors = {
    "Bush (R) 2001-08":   "#dc2626",
    "Obama (D) 2009-16":  "#1d4ed8",
    "Trump (R) 2017-20":  "#dc2626",
    "Biden (D) 2021-25":  "#1d4ed8",
}
for _, r in adm.iterrows():
    rec = f"{r['avg_rec']:.0f}d" if pd.notna(r['avg_rec']) else "N/A"
    c = admin_colors.get(r['administration'], '#555')
    rows_html += f"""
  <tr>
    <td style="padding:2px 5px;color:{c};font-weight:bold">{r['administration']}</td>
    <td style="padding:2px 5px;text-align:right">{int(r['n'])}</td>
    <td style="padding:2px 5px;text-align:right">${r['aid']/1e9:.1f}B</td>
    <td style="padding:2px 5px;text-align:right">${r['avg_apd']/1e6:.0f}M</td>
    <td style="padding:2px 5px;text-align:right">{rec}</td>
  </tr>"""

m.get_root().html.add_child(folium.Element(f"""
<div style="position:fixed;bottom:28px;left:10px;z-index:9999;background:white;
     padding:10px 12px;border-radius:8px;border:1.5px solid #999;
     font-family:Arial;font-size:11px;box-shadow:2px 2px 6px rgba(0,0,0,.2);max-width:420px;">
  <b style="font-size:12px;">Policy Impact by Presidential Administration</b>
  <table style="border-collapse:collapse;margin-top:4px;width:100%">
    <tr style="background:#f3f4f6">
      <th style="padding:2px 5px;text-align:left">Administration</th>
      <th style="padding:2px 5px;text-align:right">Events</th>
      <th style="padding:2px 5px;text-align:right">Total Aid</th>
      <th style="padding:2px 5px;text-align:right">Avg/Event</th>
      <th style="padding:2px 5px;text-align:right">Avg Recov</th>
    </tr>
    {rows_html}
  </table>
  <small style="color:#888">Recovery = days to first CDBG-DR grant; Declaration = days from incident to FEMA declaration</small>
</div>"""))

out_path = "fema_map.html"
m.save(out_path)
print(f"Map saved to {out_path}")
