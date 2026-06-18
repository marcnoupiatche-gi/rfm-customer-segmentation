"""
Application Streamlit — Segmentation Client RFM
Permet de visualiser les segments clients (RFM + K-Means) et de prédire
le segment d'un client à partir de ses valeurs Recency / Frequency / Monetary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --------------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentation Client RFM",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------
# CHARGEMENT DES ARTEFACTS (mis en cache pour éviter de recharger à chaque clic)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans.pkl"))
    labels_map = joblib.load(os.path.join(BASE_DIR, "labels_map.pkl"))
    return scaler, kmeans, labels_map

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "rfm_clustered.csv"))
    return df

SEGMENT_COLORS = {
    "Premium / Champions": "#2E86AB",
    "Fidèles": "#5DA271",
    "Occasionnels / Nouveaux": "#F4A261",
    "À risque / Perdus": "#E76F51",
}

SEGMENT_ADVICE = {
    "Premium / Champions": "Vos meilleurs clients : programme VIP, accès anticipé aux nouveautés, service client prioritaire.",
    "Fidèles": "Clients engagés et réguliers : cross-sell / up-sell, programme de points pour les faire progresser vers le segment Premium.",
    "Occasionnels / Nouveaux": "Clients récents mais peu fréquents : offre de 2e achat, email de bienvenue, recommandations personnalisées.",
    "À risque / Perdus": "Clients inactifs depuis longtemps : campagne de réactivation, remise ciblée, enquête de satisfaction avant qu'ils ne soient définitivement perdus.",
}

try:
    scaler, kmeans, labels_map = load_artifacts()
    rfm_data = load_data()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)

# --------------------------------------------------------------------------------
# SIDEBAR — NAVIGATION
# --------------------------------------------------------------------------------
st.sidebar.title("🛍️ Segmentation RFM")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Vue d'ensemble", "🔍 Exploration des segments", "🎯 Prédire un segment"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Application développée pour exploiter une segmentation client basée sur le "
    "modèle RFM (Recency, Frequency, Monetary) et K-Means."
)

if not artifacts_ok:
    st.error(
        "Impossible de charger les artefacts du modèle "
        f"(scaler.pkl, kmeans.pkl, labels_map.pkl, rfm_clustered.csv). Détail : {load_error}\n\n"
        "Assurez-vous que ces fichiers se trouvent dans le même dossier que app.py "
        "(générés par le notebook d'analyse)."
    )
    st.stop()

# --------------------------------------------------------------------------------
# FONCTION DE PRÉDICTION
# --------------------------------------------------------------------------------
def predict_segment(recency, frequency, monetary):
    X = pd.DataFrame([{
        "Recency": recency,
        "Frequency": np.log1p(frequency),
        "Monetary": np.log1p(monetary),
    }])
    X_scaled = scaler.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])
    segment = labels_map.get(cluster, f"Cluster {cluster}")
    return cluster, segment

# ==================================================================================
# PAGE 1 — VUE D'ENSEMBLE
# ==================================================================================
if page == "🏠 Vue d'ensemble":
    st.title("Vue d'ensemble de la base client")
    st.markdown(
        "Cette application exploite une segmentation **RFM (Recency, Frequency, Monetary)** "
        "couplée à un modèle de clustering **K-Means** pour regrouper les clients selon leur "
        "valeur et leur comportement d'achat."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de clients", f"{len(rfm_data):,}".replace(",", " "))
    col2.metric("Nombre de segments", rfm_data["Segment"].nunique())
    col3.metric("Recency moyenne", f"{rfm_data['Recency'].mean():.0f} jours")
    col4.metric("Monetary moyen", f"£{rfm_data['Monetary'].mean():,.0f}".replace(",", " "))

    st.markdown("### Répartition des clients par segment")
    seg_counts = rfm_data["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Nombre de clients"]
    fig_bar = px.bar(
        seg_counts, x="Segment", y="Nombre de clients", color="Segment",
        color_discrete_map=SEGMENT_COLORS, text="Nombre de clients"
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("### Part du chiffre d'affaires par segment")
        rev = rfm_data.groupby("Segment")["Monetary"].sum().reset_index()
        fig_pie = px.pie(
            rev, names="Segment", values="Monetary",
            color="Segment", color_discrete_map=SEGMENT_COLORS, hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col6:
        st.markdown("### Profil moyen par segment")
        profile = rfm_data.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
        st.dataframe(profile.style.format({"Recency": "{:.0f}", "Frequency": "{:.1f}", "Monetary": "£{:,.0f}"}),
                     use_container_width=True)

# ==================================================================================
# PAGE 2 — EXPLORATION DES SEGMENTS
# ==================================================================================
elif page == "🔍 Exploration des segments":
    st.title("Exploration interactive des segments")

    segments_filter = st.multiselect(
        "Filtrer par segment",
        options=sorted(rfm_data["Segment"].unique()),
        default=sorted(rfm_data["Segment"].unique())
    )
    filtered = rfm_data[rfm_data["Segment"].isin(segments_filter)]

    st.markdown(f"**{len(filtered)} clients** affichés sur {len(rfm_data)} au total.")

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Axe X", ["Recency", "Frequency", "Monetary"], index=0)
    with col2:
        y_axis = st.selectbox("Axe Y", ["Recency", "Frequency", "Monetary"], index=2)

    fig_scatter = px.scatter(
        filtered, x=x_axis, y=y_axis, color="Segment",
        color_discrete_map=SEGMENT_COLORS, opacity=0.7,
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"]
    )
    if y_axis == "Monetary":
        fig_scatter.update_yaxes(type="log")
    if x_axis == "Monetary":
        fig_scatter.update_xaxes(type="log")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Distribution des variables par segment")
    var = st.selectbox("Variable à comparer", ["Recency", "Frequency", "Monetary"], key="boxvar")
    fig_box = px.box(filtered, x="Segment", y=var, color="Segment", color_discrete_map=SEGMENT_COLORS)
    if var == "Monetary":
        fig_box.update_yaxes(type="log")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Détail des clients filtrés")
    st.dataframe(
        filtered[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]]
        .sort_values("Monetary", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

# ==================================================================================
# PAGE 3 — PRÉDICTION
# ==================================================================================
elif page == "🎯 Prédire un segment":
    st.title("Prédire le segment d'un client")
    st.markdown(
        "Saisissez les valeurs **Recency**, **Frequency** et **Monetary** d'un client "
        "pour prédire instantanément son segment et obtenir des recommandations marketing."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input(
                "Recency (jours depuis le dernier achat)",
                min_value=0, max_value=800, value=30, step=1
            )
        with col2:
            frequency = st.number_input(
                "Frequency (nombre d'achats)",
                min_value=1, max_value=500, value=5, step=1
            )
        with col3:
            monetary = st.number_input(
                "Monetary (montant total dépensé, £)",
                min_value=0.0, max_value=300000.0, value=500.0, step=10.0
            )
        submitted = st.form_submit_button("🔮 Prédire le segment", use_container_width=True)

    if submitted:
        cluster, segment = predict_segment(recency, frequency, monetary)
        color = SEGMENT_COLORS.get(segment, "#999999")

        st.markdown("---")
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:0.6rem;background-color:{color}22;"
            f"border-left:6px solid {color};'>"
            f"<h3 style='margin:0;color:{color};'>Segment prédit : {segment}</h3></div>",
            unsafe_allow_html=True
        )

        st.markdown("#### Profil du client")
        st.write(SEGMENT_ADVICE.get(segment, ""))

        # Comparaison avec les moyennes du segment et globales
        seg_avg = rfm_data[rfm_data["Segment"] == segment][["Recency", "Frequency", "Monetary"]].mean()
        comp_df = pd.DataFrame({
            "Variable": ["Recency", "Frequency", "Monetary"],
            "Client saisi": [recency, frequency, monetary],
            f"Moyenne segment ({segment})": [seg_avg["Recency"], seg_avg["Frequency"], seg_avg["Monetary"]],
            "Moyenne globale": [rfm_data["Recency"].mean(), rfm_data["Frequency"].mean(), rfm_data["Monetary"].mean()],
        })
        st.dataframe(comp_df.round(1), use_container_width=True, hide_index=True)

        st.markdown("#### Positionnement du client parmi la base existante")
        fig = px.scatter(
            rfm_data, x="Recency", y="Monetary", color="Segment",
            color_discrete_map=SEGMENT_COLORS, opacity=0.5,
            hover_data=["CustomerID", "Frequency"]
        )
        fig.update_yaxes(type="log")
        fig.add_scatter(
            x=[recency], y=[max(monetary, 1)], mode="markers",
            marker=dict(size=18, color="black", symbol="star"),
            name="Client saisi"
        )
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("© Projet Data Science — Segmentation Client RFM")
