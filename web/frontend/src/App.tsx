import { useEffect, useMemo, useRef, useState } from 'react'
import { ComposedChart, Area, Line, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import MapView, { Basemap } from './MapView'
import {
  api, ClassRow, colorAt, DateHit, downloadText, fmt, ImageResult, IndexMeta, niceName,
  parsePoisCsv, Poi, POI_COLORS, SeriesPoint, toCsv,
} from './api'

const today = new Date()
const iso = (d: Date) => d.toISOString().slice(0, 10)
const monthsAgo = (n: number) => { const d = new Date(today); d.setMonth(d.getMonth() - n); return d }
const fmtDate = (s: string) => new Date(s + 'T12:00:00').toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric' })
const short = (s: string) => new Date(s + 'T12:00:00').toLocaleDateString('es-ES', { day: 'numeric', month: 'short' })
const niceMax = (v: number) => { const p = Math.pow(10, Math.floor(Math.log10(v || 1))); return Math.ceil(v / p) * p }

type Tab = 'serie' | 'tabla' | 'clases'

export default function App() {
  const [mode, setMode] = useState<'gee' | 'demo' | null>(null)
  const [reservoirs, setReservoirs] = useState<GeoJSON.FeatureCollection | null>(null)
  const [indices, setIndices] = useState<IndexMeta[]>([])
  const [palette, setPalette] = useState<string[]>([])

  const [reservoir, setReservoir] = useState<string | null>(null)
  const [start, setStart] = useState(iso(monthsAgo(3)))
  const [end, setEnd] = useState(iso(today))
  const [maxCloud, setMaxCloud] = useState(30)
  const [indexId, setIndexId] = useState('PC_Val_cal')

  const [pois, setPois] = useState<Poi[]>([])
  const [adding, setAdding] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const [hits, setHits] = useState<DateHit[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [activeDate, setActiveDate] = useState<string | null>(null)
  const [img, setImg] = useState<ImageResult | null>(null)
  const [loadingImg, setLoadingImg] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [panelOpen, setPanelOpen] = useState(false)
  const [tab, setTab] = useState<Tab>('serie')
  const [series, setSeries] = useState<SeriesPoint[] | null>(null)
  const [loadingSeries, setLoadingSeries] = useState(false)
  const [classes, setClasses] = useState<{ date: string; index: string; rows: ClassRow[] } | null>(null)
  const [loadingClasses, setLoadingClasses] = useState(false)
  const [downloading, setDownloading] = useState<string | null>(null)

  const [basemap, setBasemap] = useState<Basemap>('satellite')
  const [showRgb, setShowRgb] = useState(false)
  const [opacity, setOpacity] = useState(0.9)

  useEffect(() => {
    api.health().then(h => setMode(h.mode)).catch(() => setError('No se puede conectar con el backend (¿está arrancado en :8000?)'))
    api.reservoirs().then(setReservoirs).catch(() => {})
    api.indices().then(r => { setIndices(r.indices); setPalette(r.palette) }).catch(() => {})
  }, [])

  const meta = indices.find(i => i.id === indexId)
  const names = useMemo(() => (reservoirs?.features ?? []).map(f => String(f.properties?.NOMBRE)).sort((a, b) => niceName(a).localeCompare(niceName(b))), [reservoirs])
  const groups = useMemo(() => Array.from(new Set(indices.map(i => i.group))), [indices])

  const resetResults = () => { setHits(null); setActiveDate(null); setImg(null); setSeries(null); setClasses(null); setPanelOpen(false); setError(null) }

  const pickReservoir = (n: string) => {
    if (n === reservoir) return // clic de nuevo en el mismo embalse: no borrar resultados
    setReservoir(n); resetResults(); setAdding(false); setPois([])
    api.pois(n).then(r => setPois(r.points)).catch(() => {})
  }

  // ── puntos de interés ──────────────────────
  const addPoint = (lat: number, lon: number) => {
    const n = pois.filter(p => p.custom).length + 1
    setPois(ps => [...ps, { name: `P${n}`, lat: +lat.toFixed(6), lon: +lon.toFixed(6), custom: true }])
    setAdding(false); setSeries(null)
  }
  const removePoint = (name: string) => { setPois(ps => ps.filter(p => p.name !== name)); setSeries(null) }
  const onCsv = async (f: File | undefined) => {
    if (!f) return
    try {
      const pts = parsePoisCsv(await f.text())
      if (!pts.length) throw new Error('No se encontraron puntos válidos en el CSV')
      setPois(ps => [...ps.filter(p => !pts.some(q => q.name === p.name)), ...pts]); setSeries(null)
    } catch (e: any) { setError(e.message) }
    if (fileRef.current) fileRef.current.value = ''
  }
  // recargar valores en puntos cuando cambian
  useEffect(() => { if (activeDate) loadDate(activeDate) }, [pois]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── búsqueda e imagen ──────────────────────
  const search = async () => {
    if (!reservoir) return
    resetResults(); setSearching(true)
    try {
      const r = await api.search({ reservoir, start, end, max_cloud: maxCloud })
      setHits(r.dates)
      if (r.dates.length) loadDate(r.dates[r.dates.length - 1].date)
    } catch (e: any) { setError(e.message) } finally { setSearching(false) }
  }

  const loadDate = async (d: string, idx = indexId) => {
    if (!reservoir) return
    setActiveDate(d); setLoadingImg(true); setError(null)
    try { setImg(await api.image({ reservoir, date: d, index: idx, max_cloud: maxCloud, points: pois })) }
    catch (e: any) { setImg(null); setError(e.message) } finally { setLoadingImg(false) }
  }

  const changeIndex = (id: string) => {
    setIndexId(id); setSeries(null); setClasses(null)
    if (activeDate) loadDate(activeDate, id)
  }

  // ── análisis ───────────────────────────────
  const runSeries = async () => {
    if (!reservoir || !hits?.length) return
    setPanelOpen(true); setTab(t => (t === 'clases' ? 'serie' : t))
    if (series) return
    setLoadingSeries(true); setError(null)
    try { setSeries((await api.timeseries({ reservoir, index: indexId, dates: hits.map(h => h.date), max_cloud: maxCloud, points: pois })).series) }
    catch (e: any) { setError(e.message) } finally { setLoadingSeries(false) }
  }

  const runClasses = async () => {
    if (!reservoir || !activeDate) return
    setPanelOpen(true); setTab('clases')
    if (classes && classes.date === activeDate && classes.index === indexId) return
    setLoadingClasses(true); setError(null)
    try { setClasses({ date: activeDate, index: indexId, rows: (await api.classes({ reservoir, date: activeDate, index: indexId, max_cloud: maxCloud })).classes }) }
    catch (e: any) { setError(e.message) } finally { setLoadingClasses(false) }
  }
  useEffect(() => { if (panelOpen && tab === 'clases') runClasses() }, [activeDate, indexId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── descargas ──────────────────────────────
  const geotiff = async (all: boolean) => {
    if (!reservoir || !activeDate) return
    setDownloading(all ? 'all' : 'one'); setError(null)
    try {
      const r = await api.download({ reservoir, date: activeDate, indices: all ? indices.map(i => i.id) : [indexId], max_cloud: maxCloud })
      window.open(r.url, '_blank')
    } catch (e: any) { setError(e.message) } finally { setDownloading(null) }
  }

  const csvDate = () => {
    if (!reservoir || !activeDate || !img || !meta) return
    const rows = [
      { embalse: reservoir, fecha: activeDate, pasada_utc: img.datetime, indice: indexId, ubicacion: 'Media_Embalse', lat: '', lon: '', valor: img.mean, nubes_pct: img.cloud, cobertura_pct: img.coverage },
      ...pois.map(p => ({ embalse: reservoir, fecha: activeDate, pasada_utc: img.datetime, indice: indexId, ubicacion: p.name, lat: p.lat, lon: p.lon, valor: img.points?.[p.name] ?? null, nubes_pct: img.cloud, cobertura_pct: img.coverage })),
    ]
    downloadText(`hiblooms_${reservoir}_${activeDate}_${indexId.replace('/', '-')}.csv`, toCsv(rows))
  }

  const csvSeries = () => {
    if (!series || !reservoir) return
    const rows = series.map(r => {
      const o: Record<string, unknown> = { fecha: r.date, Media_Embalse: r.mean }
      pois.forEach(p => { o[p.name] = r[p.name] ?? null })
      return o
    })
    downloadText(`hiblooms_${reservoir}_serie_${indexId.replace('/', '-')}.csv`, toCsv(rows))
  }

  const demoFill = mode === 'demo' && img && meta && img.mean !== null ? colorAt(palette, (img.mean - meta.min) / (meta.max - meta.min)) : null
  const seriesMax = series ? Math.max(0, ...series.flatMap(s => [s.mean, ...pois.map(p => s[p.name])]).map(v => (typeof v === 'number' ? v : 0))) : 0
  const poiColor = (i: number) => POI_COLORS[i % POI_COLORS.length]

  return (
    <div className="app">
      <MapView
        reservoirs={reservoirs} selected={reservoir} onSelect={pickReservoir} basemap={basemap}
        indexTileUrl={img?.tile_url ?? null} rgbTileUrl={img?.rgb_tile_url ?? null}
        showRgb={showRgb} opacity={opacity} demoFill={demoFill}
        points={pois} addingPoint={adding} onMapClick={addPoint}
        onPoiClick={() => { if (hits && hits.length > 1) { setTab('serie'); runSeries() } }}
      />

      {/* ── Panel de búsqueda ─────────────────────────── */}
      <aside className="panel">
        <header className="brand">
          <img src="/logo_hiblooms.png" alt="HIBLOOMS" />
          <div>
            <h1>Visor satelital</h1>
            <p>Sentinel-2 · cianobacterias en embalses</p>
          </div>
        </header>

        {mode === 'demo' && <div className="badge warn">Modo demo · datos simulados (sin credenciales GEE)</div>}

        <section>
          <label className="lbl">Embalse</label>
          <select value={reservoir ?? ''} onChange={e => pickReservoir(e.target.value)}>
            <option value="" disabled>Elige uno o haz clic en el mapa…</option>
            {names.map(n => <option key={n} value={n}>{niceName(n)}</option>)}
          </select>
        </section>

        <section className="row2">
          <div><label className="lbl">Desde</label><input type="date" value={start} max={end} onChange={e => setStart(e.target.value)} /></div>
          <div><label className="lbl">Hasta</label><input type="date" value={end} min={start} max={iso(today)} onChange={e => setEnd(e.target.value)} /></div>
        </section>
        <div className="chips">
          {[1, 3, 6, 12].map(m => <button key={m} className="chip" onClick={() => { setStart(iso(monthsAgo(m))); setEnd(iso(today)) }}>{m === 12 ? '1 año' : `${m} m`}</button>)}
        </div>

        <section>
          <label className="lbl">Nubosidad máxima <b>{maxCloud}%</b></label>
          <input type="range" min={0} max={100} step={5} value={maxCloud} onChange={e => setMaxCloud(+e.target.value)} />
        </section>

        <section>
          <label className="lbl">Índice</label>
          <select value={indexId} onChange={e => changeIndex(e.target.value)}>
            {groups.map(g => <optgroup key={g} label={g}>{indices.filter(i => i.group === g).map(i => <option key={i.id} value={i.id}>{i.label}</option>)}</optgroup>)}
          </select>
        </section>

        {reservoir && (
          <section className="pois">
            <label className="lbl">Puntos de interés <b>{pois.length || ''}</b></label>
            {pois.length > 0 && (
              <div className="poi-list">
                {pois.map((p, i) => (
                  <span key={p.name} className="poi-chip" title={`${p.lat}, ${p.lon}`}>
                    <i style={{ background: poiColor(i) }} />{p.name}
                    <button onClick={() => removePoint(p.name)} aria-label={`Quitar ${p.name}`}>×</button>
                  </span>
                ))}
              </div>
            )}
            <div className="poi-actions">
              <button className={'ghost' + (adding ? ' on' : '')} onClick={() => setAdding(a => !a)}>
                {adding ? 'Haz clic en el mapa…' : '+ Añadir en el mapa'}
              </button>
              <button className="ghost" onClick={() => fileRef.current?.click()}>Subir CSV</button>
              <input ref={fileRef} className="file-in" type="file" accept=".csv,text/csv" onChange={e => onCsv(e.target.files?.[0])} />
            </div>
          </section>
        )}

        <button className="primary" disabled={!reservoir || searching} onClick={search}>
          {searching ? <><span className="spin" /> Buscando imágenes…</> : 'Buscar imágenes'}
        </button>

        {error && <div className="badge err">{error}</div>}

        {hits && (
          <section className="results">
            <div className="res-head">
              <span><b>{hits.length}</b> fechas válidas</span>
            </div>
            {hits.length > 0 && (
              <div className="poi-actions">
                <button className="ghost solid" onClick={() => { setTab('serie'); runSeries() }} disabled={hits.length < 2 || loadingSeries}>
                  {loadingSeries ? 'Calculando…' : '📈 Serie temporal'}
                </button>
                <button className="ghost solid" onClick={() => { setTab('tabla'); runSeries() }} disabled={hits.length < 2 || loadingSeries}>Tabla</button>
                <button className="ghost solid" onClick={runClasses} disabled={!img}>Clases</button>
              </div>
            )}
            {hits.length === 0 && <p className="muted">Sin imágenes con esos filtros. Prueba a subir la nubosidad o ampliar el rango.</p>}
            <div className="dates">
              {hits.slice().reverse().map(h => (
                <button key={h.date} className={'date' + (h.date === activeDate ? ' on' : '')} onClick={() => loadDate(h.date)}>
                  {short(h.date)}<small>{new Date(h.date).getFullYear()}</small>
                </button>
              ))}
            </div>
          </section>
        )}
      </aside>

      {/* ── Controles de capas ───────────────────────── */}
      <div className="layers">
        <div className="seg">
          <button className={basemap === 'satellite' ? 'on' : ''} onClick={() => setBasemap('satellite')}>Satélite</button>
          <button className={basemap === 'light' ? 'on' : ''} onClick={() => setBasemap('light')}>Mapa</button>
        </div>
        {img?.rgb_tile_url && <label className="tog"><input type="checkbox" checked={showRgb} onChange={e => setShowRgb(e.target.checked)} /> RGB Sentinel-2</label>}
        {img?.tile_url && <label className="tog">Opacidad <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={e => setOpacity(+e.target.value)} /></label>}
      </div>

      {/* ── Tarjeta de imagen ────────────────────────── */}
      {activeDate && meta && (
        <div className="card info">
          <div className="info-top">
            <div>
              <p className="eyebrow">{niceName(reservoir!)}</p>
              <h2>{fmtDate(activeDate)}</h2>
            </div>
            {loadingImg && <span className="spin dark" />}
          </div>
          <div className="kpis">
            <div><span>Media embalse</span><b>{fmt(img?.mean)}</b><em>{meta.unit}</em></div>
            <div><span>Nubes</span><b>{fmt(img?.cloud, 1)}</b><em>%</em></div>
            <div><span>Cobertura</span><b>{fmt(img?.coverage, 0)}</b><em>%</em></div>
          </div>
          {pois.length > 0 && (
            <div className="pt-vals">
              {pois.map((p, i) => (
                <div key={p.name}><i style={{ background: poiColor(i) }} /><span>{p.name}</span><b>{fmt(img?.points?.[p.name])}</b></div>
              ))}
            </div>
          )}
          <div className="legend">
            <p>{meta.label}{meta.unit && ` (${meta.unit})`}</p>
            <div className="ramp" style={{ background: `linear-gradient(90deg, ${palette.join(',')})` }} />
            <div className="ticks"><span>{meta.min}</span><span>{(meta.min + meta.max) / 2}</span><span>≥ {meta.max}</span></div>
          </div>
          <div className="dl">
            <button onClick={runClasses} disabled={!img}>Clases</button>
            <button onClick={csvDate} disabled={!img}>CSV</button>
            <button onClick={() => geotiff(false)} disabled={!img || !!downloading || mode === 'demo'} title="GeoTIFF del índice actual">
              {downloading === 'one' ? '…' : 'GeoTIFF'}
            </button>
            <button onClick={() => geotiff(true)} disabled={!img || !!downloading || mode === 'demo'} title="GeoTIFF multibanda con todos los índices">
              {downloading === 'all' ? '…' : 'Todos'}
            </button>
          </div>
          {img?.datetime && <p className="muted small">Pasada: {img.datetime} UTC</p>}
        </div>
      )}

      {/* ── Panel de análisis ────────────────────────── */}
      {panelOpen && meta && (
        <div className="card series">
          <div className="series-head">
            <div className="tabs">
              <button className={tab === 'serie' ? 'on' : ''} onClick={() => { setTab('serie'); runSeries() }}>Serie temporal</button>
              <button className={tab === 'tabla' ? 'on' : ''} onClick={() => { setTab('tabla'); runSeries() }}>Tabla</button>
              <button className={tab === 'clases' ? 'on' : ''} onClick={runClasses} disabled={!activeDate}>Clases</button>
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              {tab !== 'clases' && series && <button className="link dark" onClick={csvSeries}>Descargar CSV</button>}
              <button className="x" onClick={() => setPanelOpen(false)} aria-label="Cerrar">×</button>
            </div>
          </div>

          {tab !== 'clases' && loadingSeries && <p className="muted an-empty"><span className="spin dark" /> Calculando serie… (con GEE tarda ~5–10 s por fecha)</p>}

          {tab === 'serie' && series && (
            <>
              <ResponsiveContainer width="100%" height={190}>
                <ComposedChart data={series} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}
                  onClick={(e: any) => e?.activeLabel && loadDate(e.activeLabel)}>
                  <defs>
                    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#0F8A78" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#0F8A78" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#E3E9E7" vertical={false} />
                  <XAxis dataKey="date" tickFormatter={short} tick={{ fontSize: 11, fill: '#5B6B6C' }} minTickGap={24} />
                  <YAxis tick={{ fontSize: 11, fill: '#5B6B6C' }} domain={[0, niceMax(Math.max(meta.max * 0.2, seriesMax * 1.1))]} tickFormatter={(v: number) => fmt(v, 2)} />
                  <Tooltip labelFormatter={(l: any) => fmtDate(String(l))} formatter={(v: any, n: any) => [`${fmt(v)} ${meta.unit}`, n]} />
                  {pois.length > 0 && <Legend iconSize={10} wrapperStyle={{ fontSize: 12 }} />}
                  {activeDate && <ReferenceLine x={activeDate} stroke="#C8561B" strokeDasharray="4 3" />}
                  <Area name="Media embalse" type="monotone" dataKey="mean" stroke="#0F8A78" strokeWidth={2.5} fill="url(#g)" connectNulls dot={{ r: 3, fill: '#0F8A78' }} activeDot={{ r: 5 }} />
                  {pois.map((p, i) => (
                    <Line key={p.name} name={p.name} type="monotone" dataKey={p.name} stroke={poiColor(i)} strokeWidth={1.8} dot={{ r: 2.5 }} connectNulls />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
              <p className="muted small">Haz clic en un punto de la gráfica para ver su mapa.</p>
            </>
          )}

          {tab === 'tabla' && series && (
            <div className="tbl-wrap">
              <table className="tbl">
                <thead><tr><th>Fecha</th><th>Media embalse</th>{pois.map(p => <th key={p.name}>{p.name}</th>)}</tr></thead>
                <tbody>
                  {series.slice().reverse().map(r => (
                    <tr key={r.date} onClick={() => loadDate(r.date)} style={{ cursor: 'pointer', background: r.date === activeDate ? '#E6F1EE' : undefined }}>
                      <td>{fmtDate(r.date)}</td><td>{fmt(r.mean)}</td>
                      {pois.map(p => <td key={p.name}>{fmt(r[p.name] as number | null)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {tab === 'clases' && (
            loadingClasses ? <p className="muted an-empty"><span className="spin dark" /> Calculando superficie por clases…</p>
            : classes && (
              <>
                <p className="muted small" style={{ marginBottom: 10 }}>Superficie de agua por rangos de <b>{meta.label}</b> · {fmtDate(classes.date)}</p>
                <div className="bars">
                  {classes.rows.map((c, i) => (
                    <div className="bar" key={i}>
                      <span>{c.high === null ? `≥ ${fmt(c.low)}` : `${fmt(c.low)} – ${fmt(c.high)}`} {meta.unit}</span>
                      <div className="track"><div className="fill" style={{ width: `${Math.max(c.pct, 0.5)}%`, background: colorAt(palette, (i + 0.5) / classes.rows.length) }} /></div>
                      <span>{fmt(c.area_ha, 1)} ha · {fmt(c.pct, 1)} %</span>
                    </div>
                  ))}
                </div>
              </>
            )
          )}
        </div>
      )}

      {!reservoir && mode && (
        <div className="hint">Selecciona un embalse en el mapa o en el panel para empezar</div>
      )}
      {adding && <div className="hint">Haz clic en el embalse para añadir un punto · Esc para cancelar</div>}
      <EscListener onEsc={() => setAdding(false)} />
    </div>
  )
}

function EscListener({ onEsc }: { onEsc: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onEsc() }
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h)
  }, [onEsc])
  return null
}
