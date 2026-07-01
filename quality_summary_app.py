from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st


BASE_DIR = Path(__file__).parent

FILES = {
    "caracteristiques": "caract-2024.csv",
    "lieux": "lieux-2024.csv",
    "vehicules": "vehicules-2024.csv",
    "usagers": "usagers-2024.csv",
}

MISSING_CODES = {
    "atm": ["-1"],
    "col": ["-1"],
    "circ": ["-1"],
    "vosp": ["-1"],
    "prof": ["-1"],
    "pr": ["-1"],
    "pr1": ["-1"],
    "plan": ["-1"],
    "larrout": ["-1"],
    "surf": ["-1"],
    "infra": ["-1"],
    "situ": ["-1"],
    "vma": ["-1"],
    "senc": ["-1"],
    "catv": ["-1"],
    "obs": ["-1"],
    "obsm": ["-1"],
    "choc": ["-1"],
    "manv": ["-1"],
    "motor": ["-1"],
    "place": ["-1"],
    "grav": ["-1"],
    "sexe": ["-1"],
    "trajet": ["-1", "0"],
    "secu1": ["-1"],
    "secu2": ["-1"],
    "secu3": ["-1"],
    "locp": ["-1"],
    "actp": ["-1"],
    "etatp": ["-1"],
}

GRAVITY_LABELS = {
    "1": "Uninjured",
    "2": "Killed",
    "3": "Hospitalized injured",
    "4": "Slightly injured",
}

GRAVITY_RANK = {
    "1": 1,
    "4": 2,
    "3": 3,
    "2": 4,
}

SEVERITY_COLORS = {
    "Killed": [190, 30, 45],
    "Hospitalized injured": [235, 128, 35],
    "Slightly injured": [244, 190, 65],
    "Uninjured": [60, 150, 105],
    "Unknown": [130, 130, 130],
}

VEHICLE_LABELS = {
    "1": "Bicycle",
    "01": "Bicycle",
    "2": "Moped <50cc",
    "02": "Moped <50cc",
    "7": "Car",
    "07": "Car",
    "10": "Light utility vehicle",
    "13": "Truck 3.5-7.5t",
    "14": "Truck >7.5t",
    "30": "Scooter <50cc",
    "31": "Motorcycle 50-125cc",
    "32": "Scooter 50-125cc",
    "33": "Motorcycle >125cc",
    "34": "Scooter >125cc",
    "37": "Bus",
    "38": "Coach",
    "40": "Tramway",
    "50": "Motorized personal mobility device",
    "60": "Non-motorized personal mobility device",
    "80": "Electric bicycle",
    "99": "Other vehicle",
}


@st.cache_data(show_spinner=False)
def load_tables():
    tables = {}
    for table_name, file_name in FILES.items():
        df = pd.read_csv(
            BASE_DIR / file_name,
            sep=";",
            dtype="string",
            encoding="utf-8",
            na_values=["", "N/A"],
            keep_default_na=False,
        )
        for col in df.columns:
            df[col] = (
                df[col]
                .str.strip()
                .str.replace("\u00a0", "", regex=False)
                .str.replace("\u00c2", "", regex=False)
            )
        tables[table_name] = df
    return tables


def business_missing_mask(df, col):
    return df[col].isna() | df[col].isin(MISSING_CODES.get(col, []))


def percent(value, total):
    return round(value / total * 100, 2) if total else 0


@st.cache_data(show_spinner=False)
def prepare_model(tables):
    car = tables["caracteristiques"].copy()
    usa = tables["usagers"].copy()

    car["latitude"] = pd.to_numeric(car["lat"].str.replace(",", ".", regex=False), errors="coerce")
    car["longitude"] = pd.to_numeric(car["long"].str.replace(",", ".", regex=False), errors="coerce")
    car["accident_datetime"] = pd.to_datetime(
        car["an"] + "-" + car["mois"] + "-" + car["jour"] + " " + car["hrmn"],
        errors="coerce",
    )
    car["hour"] = car["accident_datetime"].dt.hour
    car["month"] = car["accident_datetime"].dt.month

    usa["gravity_rank"] = usa["grav"].map(GRAVITY_RANK).astype("Int64")
    accident_rank = usa.groupby("Num_Acc")["gravity_rank"].max().reset_index()
    accident_rank["severity"] = accident_rank["gravity_rank"].map(
        {rank: label for label, rank in GRAVITY_RANK.items()}
    )
    accident_rank["severity"] = accident_rank["severity"].map(GRAVITY_LABELS).fillna("Unknown")

    accidents = car.merge(accident_rank[["Num_Acc", "severity"]], on="Num_Acc", how="left")
    accidents["severity"] = accidents["severity"].fillna("Unknown")
    accidents["color"] = accidents["severity"].map(SEVERITY_COLORS)
    return accidents


def build_missingness_table(tables):
    rows = []
    for table_name, df in tables.items():
        for col in df.columns:
            raw_missing = df[col].isna()
            business_missing = business_missing_mask(df, col)
            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "field": f"{table_name}.{col}",
                    "raw_missing_%": percent(raw_missing.sum(), len(df)),
                    "business_missing_%": percent(business_missing.sum(), len(df)),
                }
            )
    return pd.DataFrame(rows).sort_values("business_missing_%", ascending=False)


def build_anomalies(tables):
    valid_categories = {
        ("caracteristiques", "lum"): {"1", "2", "3", "4", "5"},
        ("caracteristiques", "agg"): {"1", "2"},
        ("caracteristiques", "atm"): {"-1", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
        ("caracteristiques", "col"): {"-1", "1", "2", "3", "4", "5", "6", "7"},
        ("usagers", "catu"): {"1", "2", "3"},
        ("usagers", "grav"): {"1", "2", "3", "4"},
        ("usagers", "sexe"): {"1", "2"},
        ("usagers", "actp"): {"-1", "0", "1", "2", "3", "4", "5", "6", "9", "A", "B"},
        ("vehicules", "motor"): {"-1", "0", "1", "2", "3", "4", "5", "6"},
    }

    rows = []
    for (table_name, col), valid_values in valid_categories.items():
        values = tables[table_name][col].dropna()
        invalid_values = values[~values.isin(valid_values)]
        if len(invalid_values):
            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "unexpected_values": ", ".join(sorted(invalid_values.unique())),
                    "count": int(len(invalid_values)),
                    "share_%": percent(len(invalid_values), len(tables[table_name])),
                }
            )
    return pd.DataFrame(rows)


def build_quality_report(tables, anomalies):
    car = tables["caracteristiques"]
    lieux = tables["lieux"]
    veh = tables["vehicules"]
    usa = tables["usagers"]

    birth_year = pd.to_numeric(usa["an_nais"], errors="coerce")
    lat = pd.to_numeric(car["lat"].str.replace(",", ".", regex=False), errors="coerce")
    lon = pd.to_numeric(car["long"].str.replace(",", ".", regex=False), errors="coerce")
    invalid_coordinates = lat.isna() | lon.isna() | ~lat.between(-90, 90) | ~lon.between(-180, 180)

    vehicle_ids = set(veh["id_vehicule"].dropna())
    user_vehicle_ids = set(usa["id_vehicule"].dropna())

    return pd.DataFrame(
        [
            {
                "Issue": "Conditional fields look almost empty",
                "Evidence": (
                    f"lartpc {percent(business_missing_mask(lieux, 'lartpc').sum(), len(lieux))}%, "
                    f"occutc {percent(veh['occutc'].isna().sum(), len(veh))}%, "
                    f"etatp {percent(business_missing_mask(usa, 'etatp').sum(), len(usa))}%"
                ),
                "Impact": "Can make the dataset look worse than it is.",
                "Action": "Tag these fields as conditional or not applicable.",
                "Priority": "Medium",
            },
            {
                "Issue": "User profile values are sometimes missing",
                "Evidence": (
                    f"birth year {percent(birth_year.isna().sum(), len(usa))}%, "
                    f"sex unknown {percent(business_missing_mask(usa, 'sexe').sum(), len(usa))}%"
                ),
                "Impact": "Can bias age and gender comparisons.",
                "Action": "Keep rows and create Unknown categories.",
                "Priority": "Medium",
            },
            {
                "Issue": "Unexpected category values",
                "Evidence": f"{int(anomalies['count'].sum()) if len(anomalies) else 0} records affected",
                "Impact": "Can break labels, filters, and grouped charts.",
                "Action": "Map known codes, then flag the rest as Out of dictionary.",
                "Priority": "Medium",
            },
            {
                "Issue": "Exact duplicates",
                "Evidence": f"{int(lieux.duplicated().sum())} duplicate rows in lieux",
                "Impact": "Can slightly overcount road-location records.",
                "Action": "Remove exact duplicates in Silver.",
                "Priority": "Low",
            },
            {
                "Issue": "Vehicles without matching users",
                "Evidence": f"{len(vehicle_ids - user_vehicle_ids)} vehicles",
                "Impact": "Inner joins can silently drop these vehicles.",
                "Action": "Use controlled joins and document hit-and-run cases.",
                "Priority": "Medium",
            },
            {
                "Issue": "Coordinates are valid here",
                "Evidence": f"{int(invalid_coordinates.sum())} invalid coordinates",
                "Impact": "Mapping is reliable for this file.",
                "Action": "Keep this validation as a pipeline check.",
                "Priority": "Low",
            },
        ]
    )


def build_impact_analysis():
    return pd.DataFrame(
        [
            {
                "Downstream use": "Executive dashboard",
                "Risk": "KPIs can be wrong if conditional missing fields are treated as real errors.",
                "What to do": "Show missingness by context and document non-applicable fields.",
            },
            {
                "Downstream use": "Maps",
                "Risk": "Invalid coordinates would misplace accidents.",
                "What to do": "Validate coordinates and fallback to municipality/department.",
            },
            {
                "Downstream use": "Severity analytics",
                "Risk": "Wrong severity mapping can hide serious accidents.",
                "What to do": "Create an explicit severity label and check unknown values.",
            },
            {
                "Downstream use": "User segmentation",
                "Risk": "Dropping unknown age or sex rows can bias comparisons.",
                "What to do": "Keep Unknown classes in dimensions.",
            },
            {
                "Downstream use": "Relational model",
                "Risk": "Vehicle-user inner joins can lose hit-and-run vehicles.",
                "What to do": "Use left joins and add a data-quality flag.",
            },
        ]
    )


def bar_chart(data, x, y, color=None, title=None):
    chart = alt.Chart(data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=x,
        y=y,
        tooltip=list(data.columns),
    )
    if color:
        chart = chart.encode(color=color)
    if title:
        chart = chart.properties(title=title)
    return chart


st.set_page_config(page_title="BAAC 2024 - Quality Dashboard", layout="wide")

tables = load_tables()
accidents = prepare_model(tables)
missingness = build_missingness_table(tables)
anomalies = build_anomalies(tables)
quality_report = build_quality_report(tables, anomalies)
impact_analysis = build_impact_analysis()

valid_map = accidents.dropna(subset=["latitude", "longitude"]).copy()

st.sidebar.title("Filters")
departments = ["All"] + sorted(accidents["dep"].dropna().unique().tolist())
selected_dep = st.sidebar.selectbox("Department", departments)
severity_options = sorted(accidents["severity"].dropna().unique().tolist())
selected_severity = st.sidebar.multiselect("Severity", severity_options, default=severity_options)
max_points = st.sidebar.slider("Map points", 1_000, 30_000, 12_000, step=1_000)

filtered_accidents = accidents.copy()
if selected_dep != "All":
    filtered_accidents = filtered_accidents[filtered_accidents["dep"] == selected_dep]
if selected_severity:
    filtered_accidents = filtered_accidents[filtered_accidents["severity"].isin(selected_severity)]

filtered_map = filtered_accidents.dropna(subset=["latitude", "longitude"])
if len(filtered_map) > max_points:
    filtered_map = filtered_map.sample(max_points, random_state=42)

st.title("D. Data Quality Summary")
st.caption("BAAC 2024 - road accident data quality dashboard")

st.markdown(
    """
This app answers the two requested points, but as a dashboard:
**Quality report** summarizes what is wrong or risky in the data.
**Impact analysis** explains what these issues can break in downstream analytics.
"""
)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Accidents", f"{tables['caracteristiques']['Num_Acc'].nunique():,}")
metric_2.metric("Vehicles", f"{len(tables['vehicules']):,}")
metric_3.metric("Users", f"{len(tables['usagers']):,}")
metric_4.metric("Duplicate lieux", int(tables["lieux"].duplicated().sum()))
metric_5.metric("Unexpected categories", int(anomalies["count"].sum()) if len(anomalies) else 0)

tab_overview, tab_quality, tab_visuals, tab_map, tab_impact = st.tabs(
    ["Overview", "Quality report", "Visual diagnostics", "Accident map", "Impact analysis"]
)

with tab_overview:
    st.header("Overview")
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Accident severity")
        severity_counts = (
            accidents["severity"]
            .value_counts()
            .rename_axis("severity")
            .reset_index(name="accidents")
        )
        severity_chart = (
            alt.Chart(severity_counts)
            .mark_arc(innerRadius=55)
            .encode(
                theta="accidents:Q",
                color=alt.Color("severity:N", legend=alt.Legend(title="Severity")),
                tooltip=["severity", "accidents"],
            )
        )
        st.altair_chart(severity_chart, use_container_width=True)

    with right:
        st.subheader("Analytical readiness")
        st.success("Usable dataset")
        st.write(
            "The dataset is strong enough for analysis: keys are coherent and coordinates are valid. "
            "The main work is to make raw codes understandable and avoid dropping useful rows."
        )
        st.write("Recommended stance: build a Silver layer before BI dashboards.")

    st.subheader("Accidents by hour")
    by_hour = (
        accidents.dropna(subset=["hour"])
        .groupby("hour")
        .size()
        .reset_index(name="accidents")
    )
    st.altair_chart(
        alt.Chart(by_hour)
        .mark_line(point=True)
        .encode(x="hour:O", y="accidents:Q", tooltip=["hour", "accidents"]),
        use_container_width=True,
    )

with tab_quality:
    st.header("Quality report")
    st.write("Main issues discovered during profiling.")
    st.dataframe(quality_report, use_container_width=True, hide_index=True)

    st.subheader("Top missing fields")
    top_missing = missingness.head(15)
    missing_chart = (
        alt.Chart(top_missing)
        .mark_bar()
        .encode(
            x=alt.X("business_missing_%:Q", title="Business missing %"),
            y=alt.Y("field:N", sort="-x", title="Field"),
            color=alt.Color("table:N", title="Table"),
            tooltip=["field", "raw_missing_%", "business_missing_%"],
        )
    )
    st.altair_chart(missing_chart, use_container_width=True)

    st.subheader("Unexpected categories")
    if len(anomalies):
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
    else:
        st.success("No unexpected categories found in the checked fields.")

with tab_visuals:
    st.header("Visual diagnostics")

    left, right = st.columns(2)

    with left:
        st.subheader("Vehicle categories")
        vehicle_categories = tables["vehicules"]["catv"].map(VEHICLE_LABELS).fillna("Other / unmapped")
        vehicle_counts = vehicle_categories.value_counts().head(12).reset_index()
        vehicle_counts.columns = ["vehicle_category", "vehicles"]
        st.altair_chart(
            bar_chart(
                vehicle_counts,
                x=alt.X("vehicles:Q", title="Vehicles"),
                y=alt.Y("vehicle_category:N", sort="-x", title="Vehicle category"),
                title="Top vehicle categories",
            ),
            use_container_width=True,
        )

    with right:
        st.subheader("User injury severity")
        user_gravity = tables["usagers"]["grav"].map(GRAVITY_LABELS).fillna("Unknown")
        user_gravity_counts = user_gravity.value_counts().reset_index()
        user_gravity_counts.columns = ["injury_severity", "users"]
        st.altair_chart(
            bar_chart(
                user_gravity_counts,
                x=alt.X("users:Q", title="Users"),
                y=alt.Y("injury_severity:N", sort="-x", title="Injury severity"),
                title="Users by injury severity",
            ),
            use_container_width=True,
        )

    st.subheader("Accidents by month and severity")
    by_month = (
        accidents.dropna(subset=["month"])
        .groupby(["month", "severity"])
        .size()
        .reset_index(name="accidents")
    )
    month_chart = (
        alt.Chart(by_month)
        .mark_bar()
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("accidents:Q", title="Accidents"),
            color=alt.Color("severity:N", title="Severity"),
            tooltip=["month", "severity", "accidents"],
        )
    )
    st.altair_chart(month_chart, use_container_width=True)

with tab_map:
    st.header("Accident map")
    st.write(
        "Each point is an accident. Color represents the worst user severity recorded for that accident. "
        "Use the sidebar filters to explore a department or a severity level."
    )

    if len(filtered_map) == 0:
        st.warning("No accident matches the selected filters.")
    else:
        center_lat = float(filtered_map["latitude"].mean())
        center_lon = float(filtered_map["longitude"].mean())
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_map,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=45,
            pickable=True,
            opacity=0.75,
        )
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5.2)
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Accident:</b> {Num_Acc}<br/><b>Severity:</b> {severity}<br/><b>Department:</b> {dep}<br/><b>Date:</b> {accident_datetime}",
                "style": {"backgroundColor": "white", "color": "black"},
            },
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption(f"Showing {len(filtered_map):,} points after filters/sample limit.")

with tab_impact:
    st.header("Impact analysis")
    st.write("How the detected issues affect downstream analytics.")
    st.dataframe(impact_analysis, use_container_width=True, hide_index=True)

    st.subheader("Recommended Silver-layer design")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Standardize")
        st.write("Convert dates, coordinates, ages, and coded categories into clear analytical fields.")
    with c2:
        st.markdown("### Clean")
        st.write("Remove exact duplicates, keep unknown values, and flag out-of-dictionary categories.")
    with c3:
        st.markdown("### Protect analytics")
        st.write("Use left joins for vehicle-user analysis and preserve hit-and-run vehicle records.")

    