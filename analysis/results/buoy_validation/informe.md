# Validación Sentinel-2 ↔ sonda · El Val

Generado 2026-09-22 11:45. Imágenes S2 sobre la boya: 1044; válidas (agua limpia, sin nubes a 300 m): 324; con sonda en ±2 h: 302.

- Pares con clorofila: 195
- Pares con ficocianina: 112

## Resultados por índice

rho = correlación de Spearman; R2_log = R² de log10(sonda) frente al índice; AUC = capacidad de separar muestreos por encima del umbral (0,5 = azar, >0,7 útil, >0,8 buena).

| variable                       | indice             |   n |   rho_spearman |       p |   R2_log |   AUC ≥10 |   n ≥10 |   AUC ≥20 |   n ≥20 |   AUC ≥4.25 |   n ≥4.25 |   AUC ≥23.1 |   n ≥23.1 |
|:-------------------------------|:-------------------|----:|---------------:|--------:|---------:|----------:|--------:|----------:|--------:|------------:|----------:|------------:|----------:|
| Clorofila sonda 0–1,5 m (µg/L) | NDCI               | 195 |          0.758 | 1.1e-37 |    0.517 |     0.861 |     111 |     0.877 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | PCI                | 195 |          0.758 | 1.1e-37 |    0.017 |     0.861 |     111 |     0.877 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | MCI                | 195 |          0.666 | 2.1e-26 |    0.484 |     0.781 |     111 |     0.824 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | 3BDA               | 188 |          0.608 | 2.2e-20 |    0.011 |     0.745 |     110 |     0.842 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | B5-B4              | 195 |          0.736 | 1.8e-34 |    0.55  |     0.823 |     111 |     0.871 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | Turbidez(B4)       | 195 |         -0.103 | 0.15    |    0.006 |     0.428 |     111 |     0.46  |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | SABI               | 195 |         -0.008 | 0.91    |    0.003 |     0.504 |     111 |     0.506 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | FAI                | 195 |          0.181 | 0.011   |    0.03  |     0.6   |     111 |     0.604 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | CI_cyano           | 195 |         -0.666 | 2.1e-26 |    0.484 |     0.219 |     111 |     0.176 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | B3/B4              | 195 |         -0.223 | 0.0018  |    0     |     0.411 |     111 |     0.347 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | B5/B3              | 195 |          0.618 | 6.9e-22 |    0.211 |     0.774 |     111 |     0.822 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | NDVI               | 195 |          0.255 | 0.00032 |    0.088 |     0.604 |     111 |     0.691 |      68 |     nan     |       nan |     nan     |       nan |
| Clorofila sonda 0–1,5 m (µg/L) | Sombra 620 (B3,B4) | 195 |         -0.081 | 0.26    |    0.024 |     0.538 |     111 |     0.465 |      68 |     nan     |       nan |     nan     |       nan |
| Ficocianina sonda superficie   | NDCI               | 112 |          0.132 | 0.16    |    0.013 |   nan     |     nan |   nan     |     nan |       0.572 |        28 |       0.677 |        14 |
| Ficocianina sonda superficie   | PCI                | 112 |          0.132 | 0.16    |    0.049 |   nan     |     nan |   nan     |     nan |       0.572 |        28 |       0.677 |        14 |
| Ficocianina sonda superficie   | MCI                | 112 |          0.095 | 0.32    |    0.055 |   nan     |     nan |   nan     |     nan |       0.554 |        28 |       0.657 |        14 |
| Ficocianina sonda superficie   | 3BDA               |  99 |          0.401 | 3.9e-05 |    0.093 |   nan     |     nan |   nan     |     nan |       0.733 |        22 |       0.741 |        12 |
| Ficocianina sonda superficie   | B5-B4              | 112 |          0.32  | 0.00058 |    0.14  |   nan     |     nan |   nan     |     nan |       0.709 |        28 |       0.762 |        14 |
| Ficocianina sonda superficie   | Turbidez(B4)       | 112 |         -0.182 | 0.055   |    0.006 |   nan     |     nan |   nan     |     nan |       0.395 |        28 |       0.434 |        14 |
| Ficocianina sonda superficie   | SABI               | 112 |         -0.166 | 0.079   |    0.012 |   nan     |     nan |   nan     |     nan |       0.453 |        28 |       0.244 |        14 |
| Ficocianina sonda superficie   | FAI                | 112 |          0.082 | 0.39    |    0.002 |   nan     |     nan |   nan     |     nan |       0.602 |        28 |       0.509 |        14 |
| Ficocianina sonda superficie   | CI_cyano           | 112 |         -0.095 | 0.32    |    0.055 |   nan     |     nan |   nan     |     nan |       0.446 |        28 |       0.343 |        14 |
| Ficocianina sonda superficie   | B3/B4              | 112 |          0.054 | 0.57    |    0.007 |   nan     |     nan |   nan     |     nan |       0.51  |        28 |       0.42  |        14 |
| Ficocianina sonda superficie   | B5/B3              | 112 |          0.073 | 0.44    |    0.03  |   nan     |     nan |   nan     |     nan |       0.584 |        28 |       0.684 |        14 |
| Ficocianina sonda superficie   | NDVI               | 112 |         -0.221 | 0.019   |    0.023 |   nan     |     nan |   nan     |     nan |       0.431 |        28 |       0.446 |        14 |
| Ficocianina sonda superficie   | Sombra 620 (B3,B4) | 112 |          0.086 | 0.37    |    0     |   nan     |     nan |   nan     |     nan |       0.554 |        28 |       0.609 |        14 |

## Cómo leerlo

- Si NDCI/MCI tienen rho ≥ 0,5 y AUC ≥ 0,75 para clorofila, Sentinel-2 sigue la biomasa algal en El Val.
- Si PCI no mejora a NDCI para la ficocianina, el PCI no aporta información propia de cianobacterias (esperable: Sentinel-2 no tiene banda a 620 nm).
- Mira las figuras de dispersión: la forma (umbral, saturación, nube sin estructura) dice más que un número.

## Búsqueda exhaustiva · Ficocianina

Se prueban todas las diferencias normalizadas, diferencias, índices de tres bandas y alturas de línea entre las 10 bandas de S2, más los índices de la bibliografía. «En muestra» es optimista (se elige y se evalúa con los mismos datos); lo que vale es lo **validado dejando fuera un año**. Compara siempre con «solo estacionalidad»: si el satélite no la supera, no aporta información propia.

- **variable**: Ficocianina
- **n**: 112
- **candidatos**: 303
- **mejor_en_muestra**: altura B5 sobre B3–B8A (rho 0.43)
- **rho_validado**: -0.122
- **AUC_validado_p75**: 0.39
- **AUC_validado_p90**: 0.488
- **elegido_por_año**: 2024: (1/B2−1/B11)·B8; 2025: altura B4 sobre B1–B5; 2026: altura B5 sobre B3–B7
- **RF bandas S2: rho**: -0.009
- **RF bandas S2: AUC p90**: 0.474
- **solo estacionalidad (mes): rho**: 0.612
- **solo estacionalidad (mes): AUC p90**: 0.902
- **RF bandas + estacionalidad: rho**: 0.634
- **RF bandas + estacionalidad: AUC p90**: 0.859

Top 20 en muestra:

| indice                  |   rho_en_muestra |
|:------------------------|-----------------:|
| altura B5 sobre B3–B8A  |            0.43  |
| altura B5 sobre B3–B8   |            0.427 |
| B2−B5                   |           -0.407 |
| altura B5 sobre B3–B11  |            0.382 |
| altura B5 sobre B3–B7   |            0.363 |
| altura B6 sobre B2–B8A  |            0.355 |
| altura B8 sobre B6–B11  |           -0.355 |
| altura B5 sobre B2–B11  |            0.349 |
| B3−B5                   |           -0.349 |
| altura B8A sobre B6–B11 |           -0.346 |
| altura B7 sobre B2–B8A  |            0.341 |
| altura B6 sobre B2–B8   |            0.324 |
| altura B5 sobre B2–B8A  |            0.324 |
| altura B8 sobre B7–B11  |           -0.322 |
| B4−B5                   |           -0.32  |
| [B5-B4]                 |            0.32  |
| B6−B8                   |            0.315 |
| altura B8A sobre B7–B11 |           -0.31  |
| (1/B4−1/B2)·B8          |           -0.307 |
| (1/B2−1/B4)·B8          |            0.307 |

## Búsqueda exhaustiva · Clorofila

Se prueban todas las diferencias normalizadas, diferencias, índices de tres bandas y alturas de línea entre las 10 bandas de S2, más los índices de la bibliografía. «En muestra» es optimista (se elige y se evalúa con los mismos datos); lo que vale es lo **validado dejando fuera un año**. Compara siempre con «solo estacionalidad»: si el satélite no la supera, no aporta información propia.

- **variable**: Clorofila
- **n**: 195
- **candidatos**: 357
- **mejor_en_muestra**: altura B4 sobre B1–B5 (rho -0.78)
- **rho_validado**: 0.716
- **AUC_validado_p75**: 0.928
- **AUC_validado_p90**: 0.948
- **elegido_por_año**: 2018: altura B4 sobre B1–B5; 2019: altura B4 sobre B1–B5; 2020: altura B4 sobre B1–B5; 2021: altura B4 sobre B1–B5; 2022: altura B4 sobre B1–B5; 2023: altura B5 sobre B4–B7; 2024: altura B4 sobre B1–B5
- **RF bandas S2: rho**: 0.733
- **RF bandas S2: AUC p90**: 0.882
- **solo estacionalidad (mes): rho**: 0.614
- **solo estacionalidad (mes): AUC p90**: 0.739
- **RF bandas + estacionalidad: rho**: 0.73
- **RF bandas + estacionalidad: AUC p90**: 0.896

Top 20 en muestra:

| indice                 |   rho_en_muestra |
|:-----------------------|-----------------:|
| altura B4 sobre B1–B5  |           -0.783 |
| [PCI]                  |            0.758 |
| [NDCI]                 |            0.758 |
| ND(B4,B5)              |           -0.758 |
| altura B5 sobre B4–B8  |            0.752 |
| altura B5 sobre B4–B8A |            0.748 |
| altura B5 sobre B4–B7  |            0.738 |
| [B5-B4]                |            0.736 |
| B4−B5                  |           -0.736 |
| altura B5 sobre B4–B11 |            0.733 |
| (1/B4−1/B5)·B2         |            0.73  |
| (1/B5−1/B4)·B2         |           -0.73  |
| altura B4 sobre B2–B5  |           -0.726 |
| (1/B5−1/B3)·B4         |           -0.722 |
| (1/B3−1/B5)·B4         |            0.722 |
| (1/B5−1/B4)·B3         |           -0.72  |
| (1/B4−1/B5)·B3         |            0.72  |
| (1/B4−1/B5)·B11        |            0.719 |
| (1/B5−1/B4)·B11        |           -0.719 |
| (1/B3−1/B4)·B5         |           -0.701 |