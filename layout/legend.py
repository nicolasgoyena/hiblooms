# layout/legend.py

import streamlit as st

def generar_leyenda(indices_seleccionados):
    parametros = {
        "MCI": {"min": -0.1, "max": 0.4, "palette": ['blue', 'green', 'yellow', 'red']},
        "B5_div_B4": {"min": 0.5, "max": 1.5, "palette": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]},
        "NDCI_ind": {"min": -0.1, "max": 0.4, "palette": ['blue', 'green', 'yellow', 'red']},
        "PC_Val_cal": {"min": 0, "max": 7, "palette": ["#ADD8E6", "#008000", "#FFFF00", "#FF0000"]},
        "Chla_Val_cal": {"min": 0, "max": 150, "palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']},
        "Chla_Bellus_cal": {"min": 5, "max": 55, "palette": ['#2171b5', '#75ba82', '#fdae61', '#e31a1c']}
    }

    leyenda_html = "<div style='border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: white;'>"
    leyenda_html += "<h4 style='text-align: center;'>📌 Leyenda de Índices y Capas</h4>"

    scl_palette = {
        1: ('#ff0004', 'Píxeles saturados/defectuosos'),
        2: ('#000000', 'Píxeles de área oscura'),
        3: ('#8B4513', 'Sombras de nube'),
        4: ('#00FF00', 'Vegetación'),
        5: ('#FFD700', 'Suelo desnudo'),
        6: ('#0000FF', 'Agua'),
        7: ('#F4EEEC', 'Probabilidad baja de nubes / No clasificada'),
        8: ('#C8C2C0', 'Probabilidad media de nubes'),
        9: ('#706C6B', 'Probabilidad alta de nubes'),
        10: ('#87CEFA', 'Cirro'),
        11: ('#00FFFF', 'Nieve o hielo')
    }

    leyenda_html += "<b>Capa SCL (Clasificación de Escena):</b><br>"
    for val, (color, desc) in scl_palette.items():
        leyenda_html += f"<div style='display: flex; align-items: center;'><div style='width: 15px; height: 15px; background-color: {color}; border: 1px solid black; margin-right: 5px;'></div> {desc}</div>"

    leyenda_html += "<br>"

    msk_palette = ["blue", "green", "yellow", "red", "black"]
    leyenda_html += "<b>Capa MSK_CLDPRB (Probabilidad de Nubes):</b><br>"
    leyenda_html += f"<div style='background: linear-gradient(to right, {', '.join(msk_palette)}); height: 20px; border: 1px solid #000;'></div>"
    leyenda_html += "<div style='display: flex; justify-content: space-between; font-size: 12px;'><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div><br>"

    for indice in indices_seleccionados:
        if indice in parametros:
            min_val = parametros[indice]["min"]
            max_val = parametros[indice]["max"]
            palette = parametros[indice]["palette"]
            gradient_colors = ", ".join(palette)
            gradient_style = f"background: linear-gradient(to right, {gradient_colors}); height: 20px; border: 1px solid #000;"

            leyenda_html += f"<b>{indice}:</b><br>"
            leyenda_html += f"<div style='{gradient_style}'></div>"

            markers_html = "<div style='display: flex; justify-content: space-between; margin-top: 5px;'>"
            num_colores = len(palette)
            for i in range(num_colores):
                valor = min_val + (max_val - min_val) * i / (num_colores - 1) if num_colores > 1 else min_val
                valor_formateado = f"{valor:.2f}" if isinstance(valor, float) else str(valor)

                markers_html += (
                    "<div style='display: flex; flex-direction: column; align-items: center;'>"
                    "<div style='width: 1px; height: 8px; background-color: black;'></div>"
                    f"<span style='font-size: 12px;'>{valor_formateado}</span>"
                    "</div>"
                )
            markers_html += "</div>"
            leyenda_html += markers_html + "<br>"

    leyenda_html += "</div>"
    st.markdown(leyenda_html, unsafe_allow_html=True)
