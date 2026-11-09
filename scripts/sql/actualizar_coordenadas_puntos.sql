-- Coordenadas corregidas de los puntos de muestreo de HIBLOOMS
-- Fuente: correo de Yasser Morera (IP), 6-11-2026.
-- Los códigos de Yasser (p. ej. VAL-1-2) se emparejan con sampling.sampling_points por
-- embalse (water_body_name de sus eventos) y número de punto (último número del código).
--
-- USO: ejecútalo en pgAdmin sobre la base hiblooms. Primero revisa la consulta de
-- comprobación (paso 2): cada código debe casar con UN solo punto y la distancia al
-- punto actual debe ser razonable. Si todo cuadra, cambia el ROLLBACK final por COMMIT.
-- OJO: Yasser escribe «BLL-2-3» entre BLL-3-x; se toma como el punto 3 de Bellús.

BEGIN;

-- 1. Coordenadas nuevas (WGS84)
CREATE TEMP TABLE nuevas_coords (codigo text, embalse_like text, punto int, lat double precision, lon double precision) ON COMMIT DROP;
INSERT INTO nuevas_coords VALUES
  ('EU-2-1', '%Eugui%', 1, 42.9721999, -1.51534907),
  ('EU-2-2', '%Eugui%', 2, 42.9782823, -1.51619259),
  ('EU-2-3', '%Eugui%', 3, 42.9987883, -1.51812609),
  ('All-2-1', '%Alloz%', 1, 42.7094166, -1.9472765),
  ('All-2-2', '%Alloz%', 2, 42.722137, -1.948824),
  ('All-2-3', '%Alloz%', 3, 42.7282146, -1.93702794),
  ('All-2-4', '%Alloz%', 4, 42.740911, -1.937479),
  ('VAL-1-1', '%El Val%', 1, 41.8789317, -1.80117482),
  ('VAL-1-2', '%El Val%', 2, 41.8765336, -1.79241916),
  ('VAL-1-3', '%El Val%', 3, 41.87613, -1.78941722),
  ('IT-1-1', '%Itoiz%', 1, 42.8139449, -1.35762427),
  ('IT-1-2', '%Itoiz%', 2, 42.8133203, -1.36578415),
  ('IT-1-3', '%Itoiz%', 3, 42.8069907, -1.36347659),
  ('G-1-1', '%Lacasa%', 1, 42.1744269, -2.67403677),
  ('G-1-2', '%Lacasa%', 2, 42.180739, -2.684612),
  ('G-1-3', '%Lacasa%', 3, 42.183041, -2.681427),
  ('Urr-1-1', '%Urrunaga%', 1, 42.964047, -2.653221),
  ('Urr-1-2', '%Urrunaga%', 2, 42.9833779, -2.64783979),
  ('Urr-1-3', '%Urrunaga%', 3, 42.9800045, -2.67496119),
  ('BLL-3-1', '%Bell%', 1, 38.9369, -0.478583),
  ('BLL-3-2', '%Bell%', 2, 38.915056, -0.471467),
  ('BLL-2-3', '%Bell%', 3, 38.91985, -0.462267),
  ('BLL-3-4', '%Bell%', 4, 38.933081, -0.459956);

-- 2. Comprobación: qué punto de la base recibe cada coordenada y cuánto se mueve
CREATE TEMP TABLE emparejados ON COMMIT DROP AS
SELECT n.codigo, sp.sampling_point_id, sp.point_code, sp.point_number, ev.water_body_name,
       n.lat, n.lon,
       ROUND(ST_Distance(ST_Transform(ST_PointOnSurface(sp.geometry), 25830),
                         ST_Transform(ST_SetSRID(ST_MakePoint(n.lon, n.lat), 4326), 25830))::numeric, 0) AS se_mueve_m
FROM nuevas_coords n
JOIN sampling.sampling_points sp ON sp.point_number = n.punto
JOIN (SELECT DISTINCT sampling_point_id, water_body_name FROM sampling.sampling_events) ev
  ON ev.sampling_point_id = sp.sampling_point_id AND ev.water_body_name ILIKE n.embalse_like;

SELECT * FROM emparejados ORDER BY codigo;
-- Códigos sin pareja o con más de una (deben salir 0 filas):
SELECT n.codigo, COUNT(e.sampling_point_id) AS n_puntos
FROM nuevas_coords n LEFT JOIN emparejados e ON e.codigo = n.codigo
GROUP BY n.codigo HAVING COUNT(e.sampling_point_id) <> 1;

-- 3. Actualización (mantiene el SRID que ya tenga la columna geometry)
UPDATE sampling.sampling_points sp
SET geometry = ST_Transform(ST_SetSRID(ST_MakePoint(e.lon, e.lat), 4326), ST_SRID(sp.geometry))
FROM emparejados e
WHERE e.sampling_point_id = sp.sampling_point_id;

ROLLBACK;   -- ← cámbialo por COMMIT cuando la comprobación esté bien
