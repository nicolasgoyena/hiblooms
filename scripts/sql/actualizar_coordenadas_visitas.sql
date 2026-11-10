-- Coordenadas de las visitas (sampling_events) alineadas con la posición oficial de cada punto
-- (ejecutar DESPUÉS de actualizar_coordenadas_puntos.sql). 10-11-2026.
--
-- En el Excel de campo cada punto lleva la misma coordenada en todas las campañas (posición de la
-- estación copiada, no GPS del día). Yasser corrigió esa posición en varios puntos, así que las
-- visitas pasan a llevar la posición corregida del punto. Se dejan con su coordenada propia las
-- visitas que en el Excel tienen una posición distinta de verdad:
--   EU-1-2 (Eugui-1 / Eugui-P2), BLL-1-1 y BLL-1-2 (Bellus-1), ALL-B-1 (Alloz-B / Alloz-P1, salida con bloom).
-- De paso corrige la errata de ElVal-4 / ElVal-P3 (llevaba la longitud del punto 2).
-- Resultado esperado: solo Alloz-P1, Bellus-P1, Eugui-P2 y Bellus-P2 con max_gps_offset_m > 0.

BEGIN;

UPDATE sampling.sampling_events e
SET geometry = sp.geometry
FROM sampling.sampling_points sp, sampling.sampling_campaigns c
WHERE e.sampling_point_id = sp.sampling_point_id AND e.campaign_id = c.campaign_id
  AND sp.point_code IN ('ElVal-P1','ElVal-P2','ElVal-P3','Eugui-P1','Eugui-P2','Eugui-P3',
        'Itoiz-P1','Itoiz-P2','Itoiz-P3','Alloz-P1','Alloz-P2','Alloz-P3','Alloz-P4',
        'GonzalezLacasa-P1','GonzalezLacasa-P2','GonzalezLacasa-P3',
        'Urrunaga-P1','Urrunaga-P2','Urrunaga-P3','Bellus-P1','Bellus-P2','Bellus-P3','Bellus-P4')
  AND (c.campaign_code, sp.point_code) NOT IN (('Eugui-1','Eugui-P2'), ('Bellus-1','Bellus-P1'),
                                              ('Bellus-1','Bellus-P2'), ('Alloz-B','Alloz-P1'))
  AND NOT ST_Equals(e.geometry, sp.geometry);

UPDATE sampling.sampling_points sp
SET max_gps_offset_m = d.max_m
FROM (SELECT e.sampling_point_id,
             ROUND(MAX(ST_Distance(e.geometry::geography, p.geometry::geography))::numeric, 1) AS max_m
      FROM sampling.sampling_events e JOIN sampling.sampling_points p USING (sampling_point_id)
      WHERE e.geometry IS NOT NULL GROUP BY e.sampling_point_id) d
WHERE d.sampling_point_id = sp.sampling_point_id;

COMMIT;

SELECT point_code, max_gps_offset_m FROM sampling.sampling_points
WHERE max_gps_offset_m > 0 ORDER BY max_gps_offset_m DESC;
