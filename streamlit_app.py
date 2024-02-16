import pandas as pd
import pandas_gbq  # noqa: F401
import streamlit as st

from src.dashboards.chamados_em_um_dia import dashboard as chamados_em_um_dia
from src.dashboards.chamados_por_subtipo import (
    dashboard as chamados_por_subtipo,
)
from src.mypages.homepage import page as homepage


@st.cache_data
def load_call_data():
    return pd.read_parquet(
        "data/chamado_1746.parquet",
        columns=[
            "id_chamado",
            "data_inicio",
            "tipo",
            "subtipo",
            "id_bairro",
        ],
    )


@st.cache_data
def load_neighborhoods():
    return pd.read_parquet(
        "data/bairro.parquet",
        columns=["id_bairro", "nome", "subprefeitura", "geometry"],
    )


@st.cache_data
def load_event_data():
    return pd.read_parquet("data/rede_hoteleira_ocupacao_eventos.parquet")


def create_button(text, name):
    return st.sidebar.button(
        text,
        use_container_width=True,
        on_click=change_page,
        args=(name,),
        type="primary" if st.session_state["page"] == name else "secondary",
    )


def change_page(name):
    st.session_state["page"] = name


st.set_page_config(
    page_title="Chamados ao 1746 Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

if "page" not in st.session_state:
    st.session_state["page"] = "home"

calls = load_call_data()
neighborhoods = load_neighborhoods()
events = load_event_data()

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <a href="https://www.1746.rio/hc/pt-br" target="_blank">
        <img src="app/static/logo-1746.png" class="sidebar_logo">
    </a>
    <h1 class="sidebar_title"> Análise de Chamados - <span class="highlighted">Dashboard</span></h1>
    <p class="sidebar_subtitle">Análise de chamados abertos nos anos de 2022 e 2023</p>
    """,
    unsafe_allow_html=True,
)

pages = {
    "home": homepage,
    "dashboard_1": lambda: chamados_em_um_dia(calls, neighborhoods),
    "dashboard_2": lambda: chamados_por_subtipo(calls, events),
}


create_button("🏠 Página Inicial", "home")
st.sidebar.markdown("### 📊 Dashboards")
create_button("🗓️ Chamados em um dia", "dashboard_1")
create_button("🔊 Chamados por subtipo", "dashboard_2")

pages[st.session_state["page"]]()