# layout/introduction.py

import streamlit as st

def mostrar_pestana_introduccion():
    st.markdown("""
        <style>
            .header {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                padding: 10px;
                background-color: #1f77b4;
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .info-box {
                padding: 15px;
                border-radius: 10px;
                background-color: #f4f4f4;
                margin-bottom: 15px;
            }
            .highlight {
                font-weight: bold;
                color: #1f77b4;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="header">Reconstrucción histórica y estado actual de la proliferación de cianobacterias en embalses españoles (HIBLOOMS)</div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box"><b>Alineación con estrategias nacionales:</b><br>📌 Plan Nacional de Adaptación al Cambio Climático (PNACC 2021-2030)<br>📌 Directiva Marco del Agua 2000/60/EC<br>📌 Objetivo de Desarrollo Sostenible 6: Agua limpia y saneamiento</div>',
        unsafe_allow_html=True)

    st.subheader("Justificación")
    st.markdown("""
        La proliferación de cianobacterias en embalses es una preocupación ambiental y de salud pública.
        El proyecto **HIBLOOMS** busca evaluar la evolución histórica y actual de estos eventos en los embalses de España, contribuyendo a:
        - La monitorización de parámetros clave del cambio climático y sus efectos en los ecosistemas acuáticos.
        - La identificación de factores ambientales y de contaminación que influyen en la proliferación de cianobacterias.
        - La generación de información para mejorar la gestión y calidad del agua en España.
    """)

    st.subheader("Hipótesis y Relevancia del Proyecto")
    st.markdown("""
        Se estima que **40% de los embalses españoles** son susceptibles a episodios de proliferación de cianobacterias.
        En un contexto de cambio climático, donde las temperaturas y la eutrofización aumentan, el riesgo de proliferaciones tóxicas es mayor.

        🛰 **¿Cómo abordamos este desafío?**
        - Uso de **teledetección satelital** para monitoreo en tiempo real.
        - Implementación de **técnicas avanzadas de análisis ambiental** para evaluar las causas y patrones de proliferación.
        - Creación de modelos para predecir episodios de blooms y sus impactos en la salud y el medio ambiente.
    """)

    st.subheader("Impacto esperado")
    st.markdown("""
        El proyecto contribuirá significativamente a la gestión sostenible de embalses, proporcionando herramientas innovadoras para:
        - Evaluar la **calidad del agua** con técnicas avanzadas.
        - Diseñar estrategias de mitigación para **minimizar el riesgo de toxicidad**.
        - Colaborar con administraciones públicas y expertos para la **toma de decisiones basada en datos**.
    """)

    st.subheader("Equipo de Investigación")

    st.markdown("""
        <div class="info-box">
            <b>Equipo de Investigación:</b><br>
            🔬 <b>David Elustondo (DEV)</b> - BIOMA/UNAV, calidad del agua, QA/QC y biogeoquímica.<br>
            🔬 <b>Yasser Morera Gómez (YMG)</b> - BIOMA/UNAV, geoquímica isotópica y geocronología con <sup>210</sup>Pb.<br>
            🔬 <b>Esther Lasheras Adot (ELA)</b> - BIOMA/UNAV, técnicas analíticas y calidad del agua.<br>
            🔬 <b>Jesús Miguel Santamaría (JSU)</b> - BIOMA/UNAV, calidad del agua y técnicas analíticas.<br>
            🔬 <b>Carolina Santamaría Elola (CSE)</b> - BIOMA/UNAV, técnicas analíticas y calidad del agua.<br>
            🔬 <b>Adriana Rodríguez Garraus (ARG)</b> - MITOX/UNAV, análisis toxicológico.<br>
            🔬 <b>Sheila Izquieta Rojano (SIR)</b> - BIOMA/UNAV, SIG y teledetección, datos FAIR, digitalización.<br>
        </div>

        <div class="info-box">
            <b>Equipo de Trabajo:</b><br>
            🔬 <b>Aimee Valle Pombrol (AVP)</b> - BIOMA/UNAV, taxonomía de cianobacterias e identificación de toxinas.<br>
            🔬 <b>Carlos Manuel Alonso Hernández (CAH)</b> - Laboratorio de Radioecología/IAEA, geocronología con <sup>210</sup>Pb.<br>
            🔬 <b>David Widory (DWI)</b> - GEOTOP/UQAM, geoquímica isotópica y calidad del agua.<br>
            🔬 <b>Ángel Ramón Moreira González (AMG)</b> - CEAC, taxonomía de fitoplancton y algas.<br>
            🔬 <b>Augusto Abilio Comas González (ACG)</b> - CEAC, taxonomía de cianobacterias y ecología acuática.<br>
            🔬 <b>Lorea Pérez Babace (LPB)</b> - BIOMA/UNAV, técnicas analíticas y muestreo de campo.<br>
            🔬 <b>José Miguel Otano Calvente (JOC)</b> - BIOMA/UNAV, técnicas analíticas y muestreo de campo.<br>
            🔬 <b>Alain Suescun Santamaría (ASS)</b> - BIOMA/UNAV, técnicas analíticas.<br>
            🔬 <b>Leyre López Alonso (LLA)</b> - BIOMA/UNAV, análisis de datos.<br>
            🔬 <b>María José Rodríguez Pérez (MRP)</b> - Confederación Hidrográfica del Ebro, calidad del agua.<br>
            🔬 <b>María Concepción Durán Lalaguna (MDL)</b> - Confederación Hidrográfica del Júcar, calidad del agua.<br>
        </div>
    """, unsafe_allow_html=True)

    st.success("🔬 HIBLOOMS no solo estudia el presente, sino que reconstruye el pasado para entender el futuro de la calidad del agua en España.")
