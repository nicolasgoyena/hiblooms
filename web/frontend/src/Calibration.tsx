import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { api, CalResult, CsvPreview, downloadBlob, fmt, Poi, POI_COLORS, toCsv } from './api'
import { locale, t } from './i18n'

const MODEL_LABEL: Record<string, string> = {
  linear: 'Lineal', ridge: 'Ridge', lasso: 'Lasso', elastic_net: 'ElasticNet', logistic: 'Logística (saturación)',
  poly2: 'Polinómico 2', svr_rbf: 'SVR', random_forest: 'Random Forest', gradient_boosting: 'Gradient Boosting',
}
const RASTER = ['linear', 'ridge', 'lasso', 'elastic_net', 'logistic']
export const PRED_INFO: Record<string, { name: string; group: string }> = {
  R705_R665: { name: 'PCI · cociente B5/B4', group: 'Ficocianina / cianobacterias' },
  NDCI_705_665: { name: 'NDCI · clorofila normalizado', group: 'Ficocianina / cianobacterias' },
  B5_B4_diff: { name: 'Diferencia B5 − B4', group: 'Ficocianina / cianobacterias' },
  MCI_705: { name: 'MCI · pico a 705 nm', group: 'Clorofila / biomasa' },
  NDRE_740_705: { name: 'NDRE 740/705', group: 'Clorofila / biomasa' },
  NDRE_783_705: { name: 'NDRE 783/705', group: 'Clorofila / biomasa' },
  TB_740: { name: 'Borde rojo B6 − B5', group: 'Clorofila / biomasa' },
  R740_R665: { name: 'Cociente B6/B4', group: 'Otros' },
  R783_R665: { name: 'Cociente B7/B4', group: 'Otros' },
  TB_783: { name: 'B7 − B5', group: 'Otros' },
  B6_B5_diff: { name: 'Diferencia B6 − B5', group: 'Otros' },
  B7_B5_diff: { name: 'Diferencia B7 − B5', group: 'Otros' },
}
const KINDS = { phycocyanin: { label: 'Ficocianina', unit: 'µg/L' }, chlorophyll: { label: 'Clorofila-a', unit: 'µg/L' }, other: { label: 'Otra variable', unit: '' } } as const
type Kind = keyof typeof KINDS
const guessKind = (col: string): Kind => /fico|phyco|pc\b|cian|cyan/i.test(col) ? 'phycocyanin' : /cloro|chl/i.test(col) ? 'chlorophyll' : 'other'
export const predName = (x: string) => (PRED_INFO[x] ? t(PRED_INFO[x].name) : x)
const short = (s: string) => new Date(s + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: '2-digit' })
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)} %`)

// ── Panel lateral: formulario ───────────────────────────────────────────────
export function CalibrationForm(p: {
  reservoir: string | null
  reservoirLabel: string
  onResult: (r: CalResult, unit: string, target: string) => void
  onError: (msg: string | null) => void
  running: boolean
  setRunning: (b: boolean) => void
  onProgress?: (p: { progress: number; step: string } | null) => void
}) {
  const [opts, setOpts] = useState<{ predictors: string[]; models: string[]; rasterizable: string[] } | null>(null)
  const [file, setFile] = useState<{ name: string; text: string } | null>(null)
  const [prev, setPrev] = useState<CsvPreview | null>(null)
  const [dateCol, setDateCol] = useState(''); const [timeCol, setTimeCol] = useState(''); const [valueCol, setValueCol] = useState('')
  const [unit, setUnit] = useState('µg/L')
  const [kind, setKind] = useState<Kind>('other')
  const [tz, setTz] = useState('Europe/Madrid')
  // punto de medida
  const [pois, setPois] = useState<Poi[]>([])
  const [where, setWhere] = useState<string>('__aoi')
  const [lat, setLat] = useState(''); const [lon, setLon] = useState('')
  const [siteCol, setSiteCol] = useState(''); const [latCol, setLatCol] = useState(''); const [lonCol, setLonCol] = useState('')
  const perRow = !!siteCol || (!!latCol && !!lonCol)
  // emparejamiento
  const [maxCloud, setMaxCloud] = useState(30); const [minWater, setMinWater] = useState(50)
  const [maxHours, setMaxHours] = useState(3); const [winDays, setWinDays] = useState(0)
  // experto
  const [expert, setExpert] = useState(false)
  const [preds, setPreds] = useState<string[]>(['R705_R665', 'NDCI_705_665'])
  const [models, setModels] = useState<string[]>(['linear', 'logistic'])
  const [transform, setTransform] = useState<'auto' | 'none' | 'log'>('auto')
  const [threshold, setThreshold] = useState('')
  const [criterion, setCriterion] = useState<'balanced' | 'peaks' | 'general' | 'alerts'>('balanced')
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => { api.calOptions().then(setOpts).catch(() => {}) }, [])
  useEffect(() => {
    setPois([]); setWhere('__aoi')
    if (!p.reservoir) return
    api.pois(p.reservoir).then(r => {
      setPois(r.points)
      const s = r.points.find(x => /sonda|saica|probe/i.test(x.name)) ?? r.points[0]
      setWhere(s ? s.name : '__aoi')
    }).catch(() => {})
  }, [p.reservoir])

  const pickValue = (col: string) => { setValueCol(col); const k = guessKind(col); setKind(k); setUnit(KINDS[k].unit) }
  const onFile = async (f?: File) => {
    if (!f) return
    p.onError(null)
    try {
      const text = await f.text()
      const pv = await api.calPreview(text)
      setFile({ name: f.name, text }); setPrev(pv)
      setDateCol(pv.guess.date ?? pv.columns[0]); setTimeCol(pv.guess.time ?? '')
      setSiteCol(pv.guess.site ?? ''); setLatCol(pv.guess.lat ?? ''); setLonCol(pv.guess.lon ?? '')
      if (pv.guess.site || (pv.guess.lat && pv.guess.lon)) setWinDays(1)
      const skip = [pv.guess.date, pv.guess.time, pv.guess.lat, pv.guess.lon]
      pickValue(pv.numeric.find(c => !skip.includes(c)) ?? '')
    } catch (e: any) { p.onError(e.message) }
    if (ref.current) ref.current.value = ''
  }
  const toggle = (arr: string[], v: string, set: (a: string[]) => void) => set(arr.includes(v) ? arr.filter(x => x !== v) : [...arr, v])

  const point = where === '__aoi' ? null
    : where === '__xy' ? (lat && lon ? { lat: parseFloat(lat.replace(',', '.')), lon: parseFloat(lon.replace(',', '.')), name: t('coordenadas') } : null)
    : (() => { const x = pois.find(q => q.name === where); return x ? { lat: x.lat, lon: x.lon, name: x.name } : null })()

  const run = async () => {
    if (!p.reservoir || !file || !valueCol || !dateCol) return
    p.setRunning(true); p.onError(null)
    try {
      const r = await api.calibrate({
        reservoir: p.reservoir, csv_text: file.text, date_col: dateCol, time_col: timeCol || null, value_col: valueCol, unit, tz,
        point: perRow ? null : point, site_col: siteCol || null, lat_col: latCol && lonCol ? latCol : null, lon_col: latCol && lonCol ? lonCol : null, pois,
        max_cloud: maxCloud, min_water: minWater, max_hours: maxHours, window_days: winDays,
        auto: !expert, kind, predictors: preds, models, transform, threshold: threshold ? parseFloat(threshold.replace(',', '.')) : null, criterion,
      }, j => p.onProgress?.({ progress: j.progress, step: j.step }))
      p.onResult(r, unit, valueCol)
    } catch (e: any) { p.onError(e.message) } finally { p.setRunning(false); p.onProgress?.(null) }
  }

  return (
    <>
      <section>
        <label className="lbl">{t('Datos in situ (CSV)')}</label>
        <button className="ghost solid wide" onClick={() => ref.current?.click()}>
          {file ? `📄 ${file.name} · ${t('{n} filas', { n: prev?.n_rows ?? 0 })}` : t('Subir CSV de muestras o sonda')}
        </button>
        <input ref={ref} className="file-in" type="file" accept=".csv,.txt,text/csv" onChange={e => onFile(e.target.files?.[0])} />
        {!file && <p className="muted small" style={{ marginTop: 6 }}>{t('Necesita una columna de fecha (y a ser posible hora) y otra con el valor medido.')}</p>}
      </section>

      {prev && (
        <>
          <section className="row2">
            <div><label className="lbl">{t('Fecha')}</label>
              <select value={dateCol} onChange={e => setDateCol(e.target.value)}>{prev.columns.map(c => <option key={c}>{c}</option>)}</select></div>
            <div><label className="lbl">{t('Hora')}</label>
              <select value={timeCol} onChange={e => setTimeCol(e.target.value)}>
                <option value="">{t('— (en la fecha / 11:00)')}</option>{prev.columns.map(c => <option key={c}>{c}</option>)}</select></div>
          </section>
          <section>
            <label className="lbl">{t('Columna con el valor medido')}</label>
            <select value={valueCol} onChange={e => pickValue(e.target.value)}>
              <option value="" disabled>{t('Elige…')}</option>{(prev.numeric.length ? prev.numeric : prev.columns).map(c => <option key={c}>{c}</option>)}
            </select>
          </section>
          <section className="row2">
            <div><label className="lbl">{t('Tipo de parámetro')}</label>
              <select value={kind} onChange={e => { const k = e.target.value as Kind; setKind(k); if (KINDS[k].unit) setUnit(KINDS[k].unit) }}>
                {(Object.keys(KINDS) as Kind[]).map(k => <option key={k} value={k}>{t(KINDS[k].label)}</option>)}
              </select></div>
            <div><label className="lbl">{t('Unidad')}</label><input className="txt" value={unit} onChange={e => setUnit(e.target.value)} /></div>
          </section>

          <section>
            <label className="lbl">{t('¿Dónde se midió cada fila?')}</label>
            <div className="row2">
              <select value={siteCol} onChange={e => setSiteCol(e.target.value)}>
                <option value="">{t('— sin columna de punto')}</option>{prev.columns.map(x => <option key={x} value={x}>{t('Punto: {x}', { x })}</option>)}
              </select>
              <select value={latCol && lonCol ? `${latCol}|${lonCol}` : ''} onChange={e => { const [a, b] = e.target.value.split('|'); setLatCol(a ?? ''); setLonCol(b ?? '') }}>
                <option value="">{t('— sin coordenadas')}</option>
                {prev.columns.flatMap(a2 => prev.columns.filter(b2 => b2 !== a2 && /lat/i.test(a2) && /lon|lng/i.test(b2)).map(b2 => <option key={a2 + b2} value={`${a2}|${b2}`}>{t('Coord.: {a} / {b}', { a: a2, b: b2 })}</option>))}
              </select>
            </div>
            <p className="muted small" style={{ marginTop: 6 }}>
              {perRow
                ? (siteCol && !(latCol && lonCol)
                  ? t('Cada fila usa su propio punto; los nombres se buscan entre los puntos de interés del embalse ({list}).', { list: pois.map(x => x.name).join(', ') || t('ninguno') })
                  : t('Cada fila usa sus propias coordenadas.')) + ' ' + t('Varios puntos = más pares y el modelo aprende la variabilidad espacial.')
                : t('El CSV no indica ubicación: todas las filas se asignan al punto de abajo (p. ej. una sonda fija).')}
            </p>
          </section>

          {!perRow && <section>
            <label className="lbl">{t('Punto de medida')}</label>
            <select value={where} onChange={e => setWhere(e.target.value)}>
              {pois.map(x => <option key={x.name} value={x.name}>📍 {x.name}</option>)}
              <option value="__xy">📍 {t('Otras coordenadas…')}</option>
              <option value="__aoi">▭ {t('Media de todo el embalse')}</option>
            </select>
            {where === '__xy' && (
              <div className="row2" style={{ marginTop: 8 }}>
                <input className="txt" placeholder={t('Latitud (41.87…)')} value={lat} onChange={e => setLat(e.target.value)} />
                <input className="txt" placeholder={t('Longitud (-1.78…)')} value={lon} onChange={e => setLon(e.target.value)} />
              </div>
            )}
            <p className="muted small" style={{ marginTop: 6 }}>
              {where === '__aoi' ? t('Se compara la medida con la media de todo el embalse. Úsalo solo si la muestra es integrada.')
                : t('El satélite se lee en los píxeles de agua despejada a ±45 m del punto (sin exigir que Sen2Cor los clasifique como agua, para no perder floraciones).')}
            </p>
          </section>
          }

          {!expert && <p className="muted small">
            {t('La app prueba ~90 modelos (combinaciones de 1–3 índices, en escala lineal y logarítmica, y curvas de saturación) y elige el mejor con validación temporal por bloques.')}
          </p>}

          <button className="link" style={{ textAlign: 'left' }} onClick={() => setExpert(e => !e)}>{expert ? '▾ ' + t('Modo experto (activado)') : '▸ ' + t('Modo experto')}</button>
          {expert && (
            <>
              <section>
                <label className="lbl">{t('Índices')} <b>{preds.length}</b></label>
                {['Ficocianina / cianobacterias', 'Clorofila / biomasa', 'Otros'].map(g => (
                  <div key={g} style={{ marginBottom: 8 }}>
                    <p className="muted small" style={{ margin: '0 0 4px' }}>{t(g)}</p>
                    <div className="poi-list">
                      {(opts?.predictors ?? []).filter(x => (PRED_INFO[x]?.group ?? 'Otros') === g).map(x => (
                        <button key={x} title={x} className={'tag nice' + (preds.includes(x) ? ' on' : '')} onClick={() => toggle(preds, x, setPreds)}>{predName(x)}</button>
                      ))}
                    </div>
                  </div>
                ))}
              </section>
              <section>
                <label className="lbl">{t('Modelos')} <b>{models.length}</b></label>
                <div className="poi-list">
                  {(opts?.models ?? []).map(x => (
                    <button key={x} title={RASTER.includes(x) ? t('Se puede pintar en el mapa') : t('No se puede pintar en el mapa')}
                      className={'tag nice' + (models.includes(x) ? ' on' : '')} onClick={() => toggle(models, x, setModels)}>
                      {MODEL_LABEL[x] ? t(MODEL_LABEL[x]) : x}{RASTER.includes(x) ? ' 🗺️' : ''}
                    </button>
                  ))}
                </div>
                <p className="muted small" style={{ marginTop: 6 }}>🗺️ = {t('se puede pintar como índice en el mapa. La logística usa un solo índice por modelo.')}</p>
              </section>
              <section className="row2">
                <div><label className="lbl">{t('Escala')}</label>
                  <select value={transform} onChange={e => setTransform(e.target.value as any)}>
                    <option value="auto">{t('Probar lineal y log')}</option><option value="none">{t('Lineal')}</option><option value="log">{t('Logarítmica')}</option></select></div>
                <div><label className="lbl">{t('Umbral de alerta')}</label>
                  <input className="txt" placeholder="P90 (auto)" value={threshold} onChange={e => setThreshold(e.target.value)} /></div>
              </section>
              <section>
                <label className="lbl">{t('Elegir el modelo por')}</label>
                <select value={criterion} onChange={e => setCriterion(e.target.value as any)}>
                  <option value="balanced">{t('Equilibrado: picos y aguas claras (recomendado)')}</option>
                  <option value="peaks">{t('Precisión en picos (R² en µg/L)')}</option>
                  <option value="general">{t('Precisión general (R² en escala log)')}</option>
                  <option value="alerts">{t('Detección de alertas por encima del umbral (F1)')}</option>
                </select>
              </section>
              <section className="row2">
                <div><label className="lbl">{t('± horas')} <b>{maxHours} h</b></label><input type="range" min={0.5} max={6} step={0.5} value={maxHours} onChange={e => setMaxHours(+e.target.value)} /></div>
                <div><label className="lbl">{t('± días extra')} <b>{winDays}</b></label><input type="range" min={0} max={3} value={winDays} onChange={e => setWinDays(+e.target.value)} /></div>
              </section>
              <section className="row2">
                <div><label className="lbl">{t('Nubes embalse')} <b>≤{maxCloud}%</b></label><input type="range" min={0} max={100} step={5} value={maxCloud} onChange={e => setMaxCloud(+e.target.value)} /></div>
                <div><label className="lbl">{t('Agua despejada')} <b>≥{minWater}%</b></label><input type="range" min={10} max={100} step={5} value={minWater} onChange={e => setMinWater(+e.target.value)} /></div>
              </section>
              <section>
                <label className="lbl">{t('Hora del CSV')}</label>
                <select value={tz} onChange={e => setTz(e.target.value)}>
                  <option value="Europe/Madrid">{t('Hora local (España)')}</option><option value="UTC">UTC</option></select>
              </section>
            </>
          )}
        </>
      )}

      <button className="primary" disabled={!p.reservoir || !file || !valueCol || (expert && (!preds.length || !models.length)) || (!perRow && where === '__xy' && !point) || p.running} onClick={run}>
        {p.running ? <><span className="spin" /> {t('Calibrando…')}</> : t('Calibrar en {name}', { name: p.reservoir ? p.reservoirLabel : '…' })}
      </button>
      {!p.reservoir && <p className="muted small">{t('Elige primero un embalse arriba.')}</p>}
    </>
  )
}

// ── Tarjeta de resultados ────────────────────────────────────────────────────
export function CalibrationResult(p: {
  r: CalResult; unit: string; target: string; reservoirLabel: string
  onUse: (id: string) => void; onClose: () => void
}) {
  const { r, unit } = p
  const s = r.summary, h = r.honest
  const oof = r.predictions.filter(x => x.y_oof !== null)
  const lim = useMemo(() => {
    const v = oof.flatMap(x => [x.y_true, x.y_oof as number]); return [0, Math.ceil(Math.max(1, ...v) * 1.05)]
  }, [oof])
  const sites = useMemo(() => Array.from(new Set(r.predictions.map(x => x.site))).sort(), [r.predictions])
  const q = (v: number | null | undefined, g = 'r2') => v === null || v === undefined ? '' : g === 'r2' ? (v >= 0.7 ? 'good' : v >= 0.4 ? 'mid' : 'bad') : ''

  const dl = {
    pairs: () => downloadBlob('calibracion_pares.csv', toCsv(r.pairs), 'text/csv'),
    preds: () => downloadBlob('calibracion_predicciones_validacion.csv', toCsv(r.predictions), 'text/csv'),
    config: () => downloadBlob('calibracion_config.json', JSON.stringify({ summary: s, honest: h, fit_all: r.fit_all, raster: r.raster, ranking: r.ranking }, null, 2)),
  }

  return (
    <div className="card calres">
      <div className="series-head">
        <div>
          <p className="eyebrow">{t('Calibración')} · {p.reservoirLabel} · {s.sites.length > 1 ? `📍 ${t('{n} puntos', { n: s.sites.length })}` : s.point ? `📍 ${s.point.name ?? t('punto')}` : s.sites[0]?.lat != null ? `📍 ${s.sites[0].site}` : t('media del embalse')} · {r.mode === 'demo' ? t('datos simulados') : 'Sentinel-2'}</p>
          <h2 className="serif">{p.target} ~ {s.model_label}</h2>
          <p className="muted small">{t('Índices')}: {s.predictors.map(predName).join(' · ')}</p>
        </div>
        <button className="x" onClick={p.onClose} aria-label={t('Cerrar')}>×</button>
      </div>

      <div className="kpis k5">
        <div className={q(h.r2)}><span>{t('R² validación temporal')}</span><b>{fmt(h.r2, 2)}</b><em>{t('fuera de muestra · {k} bloques', { k: s.k_blocks })}</em></div>
        <div className={q(h.r2_log)}><span>{t('R² en escala log')}</span><b>{fmt(h.r2_log, 2)}</b><em>{t('pesa igual aguas claras y picos')}</em></div>
        <div><span>{t('Incertidumbre (80 %)')}</span><b>×{fmt(r.uncertainty.factor_lo, 2)}–{fmt(r.uncertainty.factor_hi, 2)}</b><em>MAE {fmt(h.mae, 1)} {unit} · RMSE {fmt(h.rmse, 1)}</em></div>
        <div><span>{t('Detección > {thr} {unit}', { thr: fmt(s.threshold, 1), unit })}</span><b>{pct(h.sens)}</b><em>{t('falsas alarmas {pct}', { pct: pct(h.far) })}</em></div>
        <div><span>{t('Pares válidos')}</span><b>{s.n_pairs}</b><em>{t('{d} fechas · {i} imágenes S2 revisadas', { d: s.n_days, i: r.stats.images })}</em></div>
      </div>
      <p className="muted small" style={{ marginTop: -6 }}>
        {t('Métricas honestas: la selección del modelo se repite dentro de cada bloque de entrenamiento y se evalúa en un periodo que el modelo no ha visto.')}{' '}
        {t('Con todos los datos (optimista) el R² sería {r2}. Criterio: {crit}.', { r2: fmt(r.fit_all.r2, 2), crit: s.criterion_label })}{' '}
        {t('Incertidumbre: el valor real suele estar entre ×{lo} y ×{hi} de (1 + predicción) en 8 de cada 10 casos.', { lo: fmt(r.uncertainty.factor_lo, 2), hi: fmt(r.uncertainty.factor_hi, 2) })}
        {s.selection_stability < 0.6 && ' ⚠️ ' + t('El modelo elegido cambia entre bloques: con más datos podría variar.')}
      </p>

      {r.per_site.length > 1 && (
        <div>
          <p className="lbl2">{t('Por punto · validación temporal')}</p>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>{t('Punto')}</th><th>{t('Pares')}</th><th>R²</th><th>R² log</th><th>MAE</th><th>{t('Detección')}</th></tr></thead>
              <tbody>{r.per_site.map((m, i) => (
                <tr key={m.site}><td><span className="dot" style={{ background: POI_COLORS[sites.indexOf(m.site) % POI_COLORS.length] }} /> {m.site}</td>
                  <td>{m.n}</td><td>{fmt(m.r2, 2)}</td><td>{fmt(m.r2_log, 2)}</td><td>{fmt(m.mae, 1)}</td><td>{pct(m.sens)}</td></tr>
              ))}</tbody>
            </table>
          </div>
          <p className="muted small" style={{ marginTop: 6 }}>{t('Si un punto va mucho peor que los demás, el modelo no se generaliza bien a esa zona del embalse (orilla, cola, efecto de adyacencia…).')}</p>
        </div>
      )}

      <div className="calgrid">
        <div>
          <p className="lbl2">{t('Observado vs predicho (fuera de muestra)')}</p>
          <ResponsiveContainer width="100%" height={230}>
            <ScatterChart margin={{ top: 8, right: 12, left: -6, bottom: 4 }}>
              <CartesianGrid stroke="#E3E9E7" />
              <XAxis type="number" dataKey="y_true" name={t('Observado')} domain={lim} tick={{ fontSize: 11 }} />
              <YAxis type="number" dataKey="y_oof" name={t('Predicho')} domain={lim} tick={{ fontSize: 11 }} />
              <ZAxis range={[40, 40]} />
              <Tooltip formatter={(v: any) => fmt(v, 2)} labelFormatter={() => ''} />
              <Scatter data={[{ y_true: lim[0], y_oof: lim[0] }, { y_true: lim[1], y_oof: lim[1] }]} line={{ stroke: '#9FB2AE', strokeDasharray: '4 3' }} shape={() => null as any} legendType="none" />
              {sites.length > 1
                ? sites.map((st, i) => <Scatter key={st} name={st} data={oof.filter(x => x.site === st)} fill={POI_COLORS[i % POI_COLORS.length]} />)
                : <Scatter name={t('Pares')} data={oof} fill="#0F8A78" />}
              {sites.length > 1 && <Legend iconSize={10} wrapperStyle={{ fontSize: 12 }} />}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div>
          <p className="lbl2">{t('Serie temporal')}</p>
          <ResponsiveContainer width="100%" height={230}>
            <ComposedChart data={r.predictions} margin={{ top: 8, right: 12, left: -6, bottom: 4 }}>
              <CartesianGrid stroke="#E3E9E7" vertical={false} />
              <XAxis dataKey="date" tickFormatter={short} tick={{ fontSize: 11 }} minTickGap={30} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip labelFormatter={(l: any) => short(String(l))} formatter={(v: any, n: any) => [`${fmt(v, 2)} ${unit}`, n]} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 12 }} />
              {sites.length === 1 && <Area name={t('Intervalo 80 %')} dataKey={(d: any) => (d.lo80 != null && d.hi80 != null ? [d.lo80, d.hi80] : null)} stroke="none" fill="#0F8A78" fillOpacity={0.15} connectNulls legendType="square" />}
              <Line name={t('Observado')} dataKey="y_true" stroke="#0A2629" strokeWidth={sites.length > 1 ? 0 : 1.8} dot={{ r: 2.5, fill: "#0A2629" }} />
              <Line name={t('Predicho (fuera de muestra)')} dataKey="y_oof" stroke="#0F8A78" strokeWidth={sites.length > 1 ? 0 : 1.8} strokeDasharray="5 3" dot={{ r: 2.5, fill: "#0F8A78" }} connectNulls />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="autobox">
        <p className="lbl2">{s.auto ? t('Selección automática') : t('Comparativa')} · {t('{n} modelos probados · puntuación de validación temporal ({crit})', { n: s.n_candidates, crit: s.criterion_label.toLowerCase() })}</p>
        <div className="autolist">
          {r.ranking.slice(0, 6).map((x, i) => (
            <div key={i} className={i === 0 ? 'best' : ''}>
              <span>{i === 0 ? '★' : i + 1}</span>
              <span>{x.label}{RASTER.includes(x.model) ? '' : ' ' + t('(no pintable)')} · {x.predictors.map(predName).join(' + ')}</span>
              <b>{fmt(x.cv_r2, 2)}</b>
            </div>
          ))}
        </div>
      </div>

      {r.raster && (
        <p className="formula">
          {r.raster.type === 'logistic'
            ? <>{p.target} = {fmt(r.raster.L, 2)} / (1 + e<sup>−{fmt(r.raster.k, 3)}·({predName(r.raster.predictor)} − {fmt(r.raster.x0, 3)})</sup>)</>
            : <>{r.raster.transform === 'log' ? `ln(1 + ${p.target}) = ln(${fmt(r.raster.smear, 3)}) + ` : `${p.target} = `}{fmt(r.raster.intercept, 4)}
              {(r.raster.predictors as string[]).map((x, i) => {
                const k = r.raster!.coefficients[i] as number
                if (Math.abs(k) < 1e-12) return null
                return <span key={x}> {k < 0 ? '−' : '+'} {fmt(Math.abs(k), 4)}·{x}</span>
              })}</>}
        </p>
      )}

      <div className="calactions">
        {r.rasterizable && r.calibration_id
          ? <button className="primary" onClick={() => p.onUse(r.calibration_id!)}>🗺️ {t('Usar como índice en el mapa')}</button>
          : <p className="muted small">{t('El modelo elegido no se puede pintar en el mapa (no es lineal ni logístico). Incluye un modelo 🗺️ en el modo experto si lo necesitas en el mapa.')}</p>}
        <div className="dl">
          <button onClick={dl.pairs}>{t('Pares satélite–in situ')}</button>
          <button onClick={dl.preds}>{t('Predicciones validación')}</button>
          <button onClick={dl.config}>{t('Informe JSON')}</button>
          {r.calibration_id && <button onClick={() => window.open(api.calModelUrl(r.calibration_id!), '_blank')}>{t('Modelo .joblib')}</button>}
        </div>
      </div>
      <p className="muted small">
        {t('Emparejamiento: medidas a ±{h} h del paso del satélite', { h: s.max_hours })}{s.window_days ? ' ' + t('(o ±{d} días)', { d: s.window_days }) : ''}, {t('una imagen por fecha, máscara de agua: {mask}.', { mask: s.water_mask })}
        {s.transform === 'log' ? ' ' + t('Retransformación con corrección de Duan (×{s}).', { s: fmt(s.smear, 3) }) : ''} {t('{n} medidas in situ en {d} fechas.', { n: r.stats.n_insitu_rows, d: r.stats.n_insitu_dates })}
      </p>
    </div>
  )
}
