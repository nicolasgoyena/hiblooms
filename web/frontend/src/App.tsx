import { useEffect, useMemo, useRef, useState } from 'react'
import { ComposedChart, Area, Line, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import MapView, { Basemap } from './MapView'
import { CalibrationForm, CalibrationResult } from './Calibration'
import ProjectPage from './ProjectPage'
import { MonitorForm, MonitorResult } from './Monitor'
import Climatology from './Climatology'
import {
  api, CalResult, ClimResp, MonitorResp, ClassRow, colorAt, DateHit, downloadText, fmt, ImageResult, IndexMeta, niceName,
  parsePoisCsv, Poi, POI_COLORS, SeriesPoint, toCsv,
} from './api'
import { locale, t, useLang } from './i18n'
import LangToggle from './LangToggle'

// Estado inicial de la URL: se lee una sola vez, antes de que nada la reescriba.
const INITIAL_HASH = new URLSearchParams(location.hash.slice(1))

const today = new Date()
const iso = (d: Date) => d.toISOString().slice(0, 10)
const monthsAgo = (n: number) => { const d = new Date(today); d.setMonth(d.getMonth() - n); return d }
const fmtDate = (s: string) => new Date(s + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: 'numeric' })
const short = (s: string) => new Date(s + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short' })
const niceMax = (v: number) => { const p = Math.pow(10, Math.floor(Math.log10(v || 1))); return Math.ceil(v / p) * p }

type Tab = 'serie' | 'tabla' | 'clases' | 'clima'

export default function App({ user, onLogout }: { user?: string | null; onLogout?: () => void }) {
  const [lang, setLang] = useLang()
  const [mode, setMode] = useState<'gee' | 'demo' | null>(null)
  const [reservoirs, setReservoirs] = useState<GeoJSON.FeatureCollection | null>(null)
  const [indices, setIndices] = useState<IndexMeta[]>([])
  const [palette, setPalette] = useState<string[]>([])

  const [reservoir, setReservoir] = useState<string | null>(null)
  const [start, setStart] = useState(iso(monthsAgo(3)))
  const [end, setEnd] = useState(iso(today))
  const [maxCloud, setMaxCloud] = useState(30)
  const [waterOnly, setWaterOnly] = useState(true)
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
  const [clim, setClim] = useState<ClimResp | null>(null)
  const [loadingClim, setLoadingClim] = useState(false)
  const CLIM_OK = ['NDCI_ind', 'PCI_B5/B4']
  const runClim = async (idx = indexId) => {
    if (!reservoir) return
    setPanelOpen(true); setTab('clima')
    if (!CLIM_OK.includes(idx)) { setClim(null); return }
    if (clim && clim.index === idx) return
    setLoadingClim(true); setError(null)
    try { setClim(await api.climatology({ reservoir, index: idx, water_only: waterOnly }, j => setProg(j))) }
    catch (e: any) { setError(e.message) } finally { setLoadingClim(false); setProg(null) }
  }
  useEffect(() => { setClim(null) }, [reservoir])
  useEffect(() => { if (panelOpen && tab === 'clima') runClim(indexId) }, [indexId]) // eslint-disable-line react-hooks/exhaustive-deps

  const [appMode, setAppMode] = useState<'visor' | 'cal' | 'info' | 'mon'>('info')
  const [monDays, setMonDays] = useState(30)
  const [monCloud, setMonCloud] = useState(60)
  const [mon, setMon] = useState<MonitorResp | null>(null)
  const [monRunning, setMonRunning] = useState(false)
  const runMonitor = async () => {
    setMonRunning(true); setError(null)
    try { setMon(await api.monitor({ days: monDays, max_cloud: monCloud }, j => setProg(j))) }
    catch (e: any) { setError(e.message) } finally { setMonRunning(false); setProg(null) }
  }
  const openFromMonitor = (id: string, day: string | null) => {
    setAppMode('visor'); pickReservoir(id)
    if (day) { setStart(day); setEnd(iso(today)); setTimeout(() => searchFrom(id, day), 60) }
  }
  const [calRes, setCalRes] = useState<{ r: CalResult; unit: string; target: string } | null>(null)
  const [calRunning, setCalRunning] = useState(false)
  const [prog, setProg] = useState<{ progress: number; step: string } | null>(null)
  const useCalibration = async (id: string) => {
    const r = await api.indices(); setIndices(r.indices)
    setAppMode('visor'); setCalRes(null)
    changeIndex(id)
  }

  // ── Comparador de dos fechas ──
  const [cmpDate, setCmpDate] = useState<string | null>(null)
  const [cmpImg, setCmpImg] = useState<ImageResult | null>(null)
  const [swipe, setSwipe] = useState(0.5)
  const pickCompare = async (d: string | null) => {
    setCmpDate(d); setCmpImg(null)
    if (!d || !reservoir) return
    try { setCmpImg(await api.image({ reservoir, date: d, index: indexId, max_cloud: maxCloud, points: [], water_only: waterOnly })) }
    catch (e: any) { setError(e.message); setCmpDate(null) }
  }
  useEffect(() => { if (cmpDate) pickCompare(cmpDate) }, [indexId, waterOnly]) // eslint-disable-line react-hooks/exhaustive-deps

  const [copied, setCopied] = useState(false)
  const copyLink = async () => {
    try { await navigator.clipboard.writeText(location.href) } catch { /* sin permiso: la URL ya está en la barra */ }
    setCopied(true); setTimeout(() => setCopied(false), 2000)
  }

  const [basemap, setBasemap] = useState<Basemap>('satellite')
  const [showRgb, setShowRgb] = useState(false)
  const [opacity, setOpacity] = useState(0.9)

  // ── Enlace permanente: el estado va en la URL (#e=EMBALSE&d=FECHA&i=INDICE) ──
  const booted = useRef(false)
  useEffect(() => {
    const q = INITIAL_HASH
    const e = q.get('e'), d = q.get('d'), i = q.get('i'), m = q.get('m')
    if (m === 'visor' || m === 'cal' || m === 'mon' || m === 'info') setAppMode(m)
    if (i) setIndexId(i)
    if (e) {
      setAppMode(m === 'cal' || m === 'mon' ? (m as any) : 'visor')
      setReservoir(e)
      api.pois(e).then(r => setPois(r.points)).catch(() => {})
      if (d) { setStart(d); setTimeout(() => searchFrom(e, d), 200) }
    }
    booted.current = true
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!booted.current) return
    const q = new URLSearchParams()
    q.set('m', appMode)
    if (reservoir) q.set('e', reservoir)
    if (activeDate) q.set('d', activeDate)
    if (indexId) q.set('i', indexId)
    history.replaceState(null, '', '#' + q.toString())
  }, [appMode, reservoir, activeDate, indexId])

  useEffect(() => {
    api.health().then(h => setMode(h.mode)).catch(() => setError(t('No se puede conectar con el backend (¿está arrancado en :8000?)')))
    api.reservoirs().then(setReservoirs).catch(() => {})
    api.indices().then(r => { setIndices(r.indices); setPalette(r.palette) }).catch(() => {})
  }, [])

  const meta = indices.find(i => i.id === indexId)
  const labels = useMemo(() => {
    const m: Record<string, string> = {}
    ;(reservoirs?.features ?? []).forEach(f => { const id = String(f.properties?.NOMBRE); m[id] = f.properties?.LABEL ?? niceName(id) })
    return m
  }, [reservoirs])
  const labelOf = (id: string) => labels[id] ?? niceName(id)
  const names = useMemo(() => (reservoirs?.features ?? []).filter(f => !f.properties?.CUSTOM).map(f => String(f.properties?.NOMBRE)).sort((a, b) => labelOf(a).localeCompare(labelOf(b))), [reservoirs]) // eslint-disable-line react-hooks/exhaustive-deps
  const customNames = useMemo(() => (reservoirs?.features ?? []).filter(f => f.properties?.CUSTOM).map(f => String(f.properties?.NOMBRE)), [reservoirs])
  const shpRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const onShapefile = async (f: File | undefined) => {
    if (!f) return
    setUploading(true); setError(null)
    try {
      const r = await api.uploadShapefile(f)
      setReservoirs(prev => ({ type: 'FeatureCollection', features: [...(prev?.features ?? []), ...r.geojson.features] }))
      const first = String(r.geojson.features[0].properties?.NOMBRE)
      setTimeout(() => pickReservoir(first), 50)
    } catch (e: any) { setError(e.message) } finally {
      setUploading(false); if (shpRef.current) shpRef.current.value = ''
    }
  }
  const removeUpload = async (resId: string) => {
    const uid = resId.split(':')[1]
    const n = (reservoirs?.features ?? []).filter(f => String(f.properties?.NOMBRE).startsWith(`u:${uid}:`)).length
    if (!window.confirm(t('¿Quitar este shapefile ({n} polígonos) de tus embalses?', { n }))) return
    try {
      await api.deleteUpload(uid)
      setReservoirs(prev => prev && ({ ...prev, features: prev.features.filter(f => !String(f.properties?.NOMBRE).startsWith(`u:${uid}:`)) }))
      setReservoir(null); resetResults(); setPois([])
    } catch (e: any) { setError(e.message) }
  }
  // Los índices calibrados para un embalse concreto solo se ofrecen en ese embalse
  // (o cuando aún no hay ninguno elegido, para que se vea que existen).
  const allowed = useMemo(() => indices.filter(i => !i.reservoir || !reservoir || i.reservoir === reservoir), [indices, reservoir])
  const groups = useMemo(() => Array.from(new Set(allowed.map(i => i.group))), [allowed])
  useEffect(() => {
    if (!indices.length || allowed.some(i => i.id === indexId)) return
    changeIndex(allowed.find(i => i.id === 'NDCI_ind')?.id ?? allowed[0]?.id)
  }, [allowed]) // eslint-disable-line react-hooks/exhaustive-deps

  const resetResults = () => { setHits(null); setActiveDate(null); setImg(null); setSeries(null); setClasses(null); setPanelOpen(false); setError(null); setCmpDate(null); setCmpImg(null) }

  const pickReservoir = (n: string) => {
    if (n === reservoir) return // clic de nuevo en el mismo embalse: no borrar resultados
    if (!n) { setReservoir(null); resetResults(); setAdding(false); setPois([]); return }
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
      if (!pts.length) throw new Error(t('No se encontraron puntos válidos en el CSV'))
      setPois(ps => [...ps.filter(p => !pts.some(q => q.name === p.name)), ...pts]); setSeries(null)
    } catch (e: any) { setError(e.message) }
    if (fileRef.current) fileRef.current.value = ''
  }
  // recargar valores en puntos cuando cambian
  useEffect(() => { if (activeDate) loadDate(activeDate) }, [pois])
  useEffect(() => { if (activeDate) { setSeries(null); setClasses(null); loadDate(activeDate) } }, [waterOnly]) // eslint-disable-line react-hooks/exhaustive-deps

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

  /** Búsqueda para un embalse concreto (desde el monitor, sin esperar al estado). */
  const searchFrom = async (res: string, from: string) => {
    setSearching(true); setError(null)
    try {
      const r = await api.search({ reservoir: res, start: from, end: iso(today), max_cloud: maxCloud })
      setHits(r.dates)
      if (r.dates.length) {
        const d = r.dates[r.dates.length - 1].date
        setActiveDate(d); setLoadingImg(true)
        try { setImg(await api.image({ reservoir: res, date: d, index: indexId, max_cloud: maxCloud, points: [], water_only: waterOnly })) }
        finally { setLoadingImg(false) }
      }
    } catch (e: any) { setError(e.message) } finally { setSearching(false) }
  }

  const loadDate = async (d: string, idx = indexId) => {
    if (!reservoir) return
    setActiveDate(d); setLoadingImg(true); setError(null)
    try { setImg(await api.image({ reservoir, date: d, index: idx, max_cloud: maxCloud, points: pois, water_only: waterOnly })) }
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
    try { setSeries((await api.timeseries({ reservoir, index: indexId, dates: hits.map(h => h.date), max_cloud: maxCloud, points: pois, water_only: waterOnly })).series) }
    catch (e: any) { setError(e.message) } finally { setLoadingSeries(false) }
  }

  const runClasses = async () => {
    if (!reservoir || !activeDate) return
    setPanelOpen(true); setTab('clases')
    if (classes && classes.date === activeDate && classes.index === indexId) return
    setLoadingClasses(true); setError(null)
    try { setClasses({ date: activeDate, index: indexId, rows: (await api.classes({ reservoir, date: activeDate, index: indexId, max_cloud: maxCloud, water_only: waterOnly })).classes }) }
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

  const safeName = reservoir ? labelOf(reservoir).replace(/[^\w-]+/g, '_') : ''
  const csvDate = () => {
    if (!reservoir || !activeDate || !img || !meta) return
    const rows = [
      { embalse: labelOf(reservoir), fecha: activeDate, pasada_utc: img.datetime, indice: indexId, ubicacion: 'Media_Embalse', lat: '', lon: '', valor: img.mean, nubes_pct: img.cloud, cobertura_pct: img.coverage },
      ...pois.map(p => ({ embalse: labelOf(reservoir), fecha: activeDate, pasada_utc: img.datetime, indice: indexId, ubicacion: p.name, lat: p.lat, lon: p.lon, valor: img.points?.[p.name] ?? null, nubes_pct: img.cloud, cobertura_pct: img.coverage })),
    ]
    downloadText(`hiblooms_${safeName}_${activeDate}_${indexId.replace('/', '-')}.csv`, toCsv(rows))
  }

  const csvSeries = () => {
    if (!series || !reservoir) return
    const rows = series.map(r => {
      const o: Record<string, unknown> = { fecha: r.date, Media_Embalse: r.mean }
      pois.forEach(p => { o[p.name] = r[p.name] ?? null })
      return o
    })
    downloadText(`hiblooms_${safeName}_serie_${indexId.replace('/', '-')}.csv`, toCsv(rows))
  }

  const demoFill = mode === 'demo' && img && meta && img.mean !== null ? colorAt(palette, (img.mean - meta.min) / (meta.max - meta.min)) : null
  const seriesMax = series ? Math.max(0, ...series.flatMap(s => [s.mean, ...pois.map(p => s[p.name])]).map(v => (typeof v === 'number' ? v : 0))) : 0
  const poiColor = (i: number) => POI_COLORS[i % POI_COLORS.length]

  return (
    <div className="app">
      <MapView lang={lang}
        reservoirs={reservoirs} selected={reservoir} onSelect={pickReservoir} basemap={basemap}
        indexTileUrl={img?.tile_url ?? null} rgbTileUrl={img?.rgb_tile_url ?? null}
        showRgb={showRgb} opacity={opacity} demoFill={demoFill}
        points={pois} addingPoint={adding} onMapClick={addPoint}
        onPoiClick={() => { if (hits && hits.length > 1) { setTab('serie'); runSeries() } }}
        compareTileUrl={cmpImg?.tile_url ?? null} swipe={swipe}
      />

      {/* ── Panel de búsqueda ─────────────────────────── */}
      <aside className="panel">
        <header className="brand">
          <img src="/logo_hiblooms.png" alt="HIBLOOMS" />
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1>{t('Visor satelital')}</h1>
            <p>{t('Sentinel-2 · floraciones algales en embalses')}</p>
          </div>
          <LangToggle lang={lang} setLang={setLang} />
        </header>
        {onLogout && (
          <div className="userbar">
            <span>👤 {user}</span>
            <button className="link" onClick={onLogout}>{t('Cerrar sesión')}</button>
          </div>
        )}

        {mode === 'demo' && <div className="badge warn">{t('Modo demo · datos simulados (sin credenciales GEE)')}</div>}

        <div className="seg dark big">
          <button className={appMode === 'info' ? 'on' : ''} onClick={() => setAppMode('info')} title={t('Información del proyecto')}>ℹ️ {t('Proyecto')}</button>
          <button className={appMode === 'visor' ? 'on' : ''} onClick={() => setAppMode('visor')}>🛰️ {t('Visor')}</button>
          <button className={appMode === 'mon' ? 'on' : ''} onClick={() => { setAppMode('mon'); if (!mon) runMonitor() }} title={t('Estado de todos los embalses')}>📊 {t('Monitor')}</button>
          <button className={appMode === 'cal' ? 'on' : ''} onClick={() => setAppMode('cal')}>🧪 {t('Calibración')}</button>
        </div>

        {appMode !== 'info' && appMode !== 'mon' && (
        <section>
            <label className="lbl">{t('Embalse')}</label>
            <select value={reservoir ?? ''} onChange={e => pickReservoir(e.target.value)}>
              <option value="">{reservoir ? t('— Ningún embalse —') : t('Elige uno o haz clic en el mapa…')}</option>
              {customNames.length > 0 && (
                <optgroup label={t('Tus embalses (shapefile)')}>
                  {customNames.map(n => <option key={n} value={n}>{labelOf(n)}</option>)}
                </optgroup>
              )}
              <optgroup label={t('Embalses HIBLOOMS')}>
                {names.map(n => <option key={n} value={n}>{labelOf(n)}</option>)}
              </optgroup>
            </select>
            <button className="link upl" onClick={() => shpRef.current?.click()} disabled={uploading}>
              {uploading ? t('Subiendo shapefile…') : t('+ Subir shapefile propio (ZIP)')}
            </button>
            {reservoir?.startsWith('u:') && (
              <button className="link upl del" onClick={() => removeUpload(reservoir)}>
                🗑 {t('Quitar este shapefile')}
              </button>
            )}
            <input ref={shpRef} className="file-in" type="file" accept=".zip,application/zip" onChange={e => onShapefile(e.target.files?.[0])} />
          </section>
        )}

        {prog && (
          <div className="prog">
            <div className="prog-bar"><i style={{ width: `${Math.max(3, prog.progress)}%` }} /></div>
            <small>{prog.step || t('Calculando…')} · {prog.progress}%</small>
          </div>
        )}

        {appMode === 'info' ? (
          <div className="pj-nav">
            <p><b style={{ color: '#F2F6F4' }}>HIBLOOMS</b> · {t('proyecto PID2023-153234OB-I00 del Instituto BIOMA (Universidad de Navarra) con las Confederaciones Hidrográficas del Ebro y del Júcar.')}</p>
            <p>🛰️ <b>{t('Visor')}</b>: {t('busca imágenes Sentinel-2 de cualquier embalse, mapas de índices, series temporales, puntos de interés y descargas.')}</p>
            <p>🧪 <b>{t('Calibración')}</b>: {t('sube tus medidas in situ y obtén un modelo validado que se pinta como índice en el mapa.')}</p>
            <button className="primary" onClick={() => setAppMode('visor')}>{t('Empezar')}</button>
          </div>
        ) : appMode === 'mon' ? (
          <>
            <MonitorForm days={monDays} setDays={setMonDays} maxCloud={monCloud} setMaxCloud={setMonCloud}
              onRun={runMonitor} running={monRunning} />
            {error && <div className="badge err">{error}</div>}
          </>
        ) : appMode === 'cal' ? (
          <>
            <CalibrationForm reservoir={reservoir} reservoirLabel={reservoir ? labelOf(reservoir) : ''}
              onResult={(r, unit, target) => setCalRes({ r, unit, target })} onError={setError}
              running={calRunning} setRunning={setCalRunning} onProgress={setProg} />
            {error && <div className="badge err">{error}</div>}
          </>
        ) : (<>
        <section className="row2">
          <div><label className="lbl">{t('Desde')}</label><input type="date" value={start} max={end} onChange={e => setStart(e.target.value)} /></div>
          <div><label className="lbl">{t('Hasta')}</label><input type="date" value={end} min={start} max={iso(today)} onChange={e => setEnd(e.target.value)} /></div>
        </section>
        <div className="chips">
          {[1, 3, 6, 12].map(m => <button key={m} className="chip" onClick={() => { setStart(iso(monthsAgo(m))); setEnd(iso(today)) }}>{m === 12 ? t('1 año') : t('{m} m', { m })}</button>)}
        </div>

        <section>
          <label className="lbl">{t('Nubosidad máxima')} <b>{maxCloud}%</b></label>
          <input type="range" min={0} max={100} step={5} value={maxCloud} onChange={e => setMaxCloud(+e.target.value)} />
        </section>


        <section>
          <label className="lbl">{t('Índice')}</label>
          <select value={indexId} onChange={e => changeIndex(e.target.value)}>
            {groups.map(g => <optgroup key={g} label={t(g)}>{allowed.filter(i => i.group === g).map(i => <option key={i.id} value={i.id}>{t(i.label)}</option>)}</optgroup>)}
          </select>
        </section>

        {reservoir && (
          <section className="pois">
            <label className="lbl">{t('Puntos de interés')} <b>{pois.length || ''}</b></label>
            {pois.length > 0 && (
              <div className="poi-list">
                {pois.map((p, i) => (
                  <span key={p.name} className="poi-chip" title={`${p.lat}, ${p.lon}`}>
                    <i style={{ background: poiColor(i) }} />{p.name}
                    <button onClick={() => removePoint(p.name)} aria-label={t('Quitar {name}', { name: p.name })}>×</button>
                  </span>
                ))}
              </div>
            )}
            <div className="poi-actions">
              <button className={'ghost' + (adding ? ' on' : '')} onClick={() => setAdding(a => !a)}>
                {adding ? t('Haz clic en el mapa…') : t('+ Añadir en el mapa')}
              </button>
              <button className="ghost" onClick={() => fileRef.current?.click()}>{t('Subir CSV')}</button>
              <input ref={fileRef} className="file-in" type="file" accept=".csv,text/csv" onChange={e => onCsv(e.target.files?.[0])} />
            </div>
          </section>
        )}

        <button className="primary" disabled={!reservoir || searching} onClick={search}>
          {searching ? <><span className="spin" /> {t('Buscando imágenes…')}</> : t('Buscar imágenes')}
        </button>

        {error && <div className="badge err">{error}</div>}

        {hits && (
          <section className="results">
            <div className="res-head">
              <span><b>{hits.length}</b> {t('fechas válidas')}</span>
            </div>
            {hits.length > 0 && (
              <div className="poi-actions">
                <button className="ghost solid" onClick={() => { setTab('serie'); runSeries() }} disabled={hits.length < 2 || loadingSeries}>
                  {loadingSeries ? t('Calculando…') : '📈 ' + t('Serie temporal')}
                </button>
                <button className="ghost solid" onClick={() => { setTab('tabla'); runSeries() }} disabled={hits.length < 2 || loadingSeries}>{t('Tabla')}</button>
                <button className="ghost solid" onClick={runClasses} disabled={!img}>{t('Clases')}</button>
                <button className="ghost solid" onClick={() => runClim()}>📊 {t('Climatología')}</button>
                <button className="ghost solid" onClick={copyLink}>{copied ? '✓ ' + t('Enlace copiado') : '🔗 ' + t('Copiar enlace')}</button>
              </div>
            )}
            {hits.length === 0 && <p className="muted">{t('Sin imágenes con esos filtros. Prueba a subir la nubosidad o ampliar el rango.')}</p>}
            {hits.length > 1 && (
              <section className="cmp">
                <label className="lbl">{t('Comparar con otra fecha')}</label>
                <select value={cmpDate ?? ''} onChange={e => pickCompare(e.target.value || null)}>
                  <option value="">{t('Sin comparación')}</option>
                  {hits.slice().reverse().filter(h => h.date !== activeDate).map(h => (
                    <option key={h.date} value={h.date}>{fmtDate(h.date)}</option>
                  ))}
                </select>
                {cmpDate && (
                  <>
                    <input type="range" min={0} max={100} value={Math.round(swipe * 100)}
                      onChange={e => setSwipe(+e.target.value / 100)} />
                    <p className="muted small">{t('Izquierda')}: {activeDate ? fmtDate(activeDate) : '—'} · {t('derecha')}: {fmtDate(cmpDate)}</p>
                  </>
                )}
              </section>
            )}
            <div className="dates">
              {hits.slice().reverse().map(h => (
                <button key={h.date} className={'date' + (h.date === activeDate ? ' on' : '')} onClick={() => loadDate(h.date)}>
                  {short(h.date)}<small>{new Date(h.date).getFullYear()}</small>
                </button>
              ))}
            </div>
          </section>
        )}
        </>)}
      </aside>

      {cmpImg?.tile_url && (
        <div className="swipe-handle" style={{ left: `calc(${swipe * 100}%)` }} aria-hidden>
          <span>{activeDate ? short(activeDate) : ''}</span><i />
          <span>{cmpDate ? short(cmpDate) : ''}</span>
        </div>
      )}

      {/* ── Controles de capas ───────────────────────── */}
      <div className="layers">
        <div className="seg">
          <button className={basemap === 'satellite' ? 'on' : ''} onClick={() => setBasemap('satellite')}>{t('Satélite')}</button>
          <button className={basemap === 'light' ? 'on' : ''} onClick={() => setBasemap('light')}>{t('Mapa')}</button>
        </div>
        {img?.rgb_tile_url && (
          <div className="seg">
            <button className={!showRgb ? 'on' : ''} onClick={() => setShowRgb(false)}>{t('Índice')}</button>
            <button className={showRgb ? 'on' : ''} onClick={() => setShowRgb(true)}>{t('Color real')}</button>
          </div>
        )}
        {img?.tile_url && !showRgb && <label className="tog">{t('Opacidad')} <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={e => setOpacity(+e.target.value)} /></label>}
      </div>

      {/* ── Tarjeta de imagen ────────────────────────── */}
      {appMode === 'visor' && activeDate && meta && (
        <div className="card info">
          <div className="info-top">
            <div>
              <p className="eyebrow">{labelOf(reservoir!)}</p>
              <h2>{fmtDate(activeDate)}</h2>
            </div>
            {loadingImg && <span className="spin dark" />}
          </div>
          <div className="kpis">
            <div><span>{t('Media embalse')}</span><b>{fmt(img?.mean)}</b><em>{meta.unit}</em></div>
            <div><span>{t('Nubes')}</span><b>{fmt(img?.cloud, 1)}</b><em>%</em></div>
            <div><span>{t('Cobertura')}</span><b>{fmt(img?.coverage, 0)}</b><em>%</em></div>
          </div>
          <label className="switch" title={t('Detecta el agua en cada imagen (MNDWI/NDWI) y descarta 20 m de borde, para que las orillas secas no se cuenten como floración.')}>
            <input type="checkbox" checked={waterOnly} onChange={e => setWaterOnly(e.target.checked)} />
            <span>{t('Solo lámina de agua')}<small>{t('quita orillas y píxeles de borde')}</small></span>
          </label>
          {img?.interval80 && meta && img.mean != null && (
            <p className="muted small" style={{ margin: 0 }}>{t('Rango probable (80 %): {lo} – {hi} {unit}', { lo: fmt(img.interval80[0], 1), hi: fmt(img.interval80[1], 1), unit: meta.unit })}</p>
          )}
          {img?.extrapolation_pct != null && img.extrapolation_pct > 5 && (
            <div className="badge warn" style={{ background: '#FFF4E8', color: '#9A4B12' }}>⚠️ {t('{pct} % del embalse está fuera del rango de índices con el que se calibró: esos valores son extrapolación.', { pct: fmt(img.extrapolation_pct, 0) })}</div>
          )}
          {pois.length > 0 && (
            <div className="pt-vals">
              {pois.map((p, i) => (
                <div key={p.name}><i style={{ background: poiColor(i) }} /><span>{p.name}</span><b>{fmt(img?.points?.[p.name])}</b></div>
              ))}
            </div>
          )}
          <div className="legend">
            <p>{t(meta.label)}{meta.unit && ` (${meta.unit})`}</p>
            <div className="ramp" style={{ background: `linear-gradient(90deg, ${palette.join(',')})` }} />
            <div className="ticks"><span>{fmt(meta.min)}</span><span>{fmt((meta.min + meta.max) / 2)}</span><span>≥ {fmt(meta.max)}</span></div>
          </div>
          <div className="dl">
            <button onClick={runClasses} disabled={!img}>{t('Clases')}</button>
            <button onClick={csvDate} disabled={!img}>CSV</button>
            <button onClick={() => geotiff(false)} disabled={!img || !!downloading || mode === 'demo'} title={t('GeoTIFF del índice actual')}>
              {downloading === 'one' ? '…' : 'GeoTIFF'}
            </button>
            <button onClick={() => geotiff(true)} disabled={!img || !!downloading || mode === 'demo'} title={t('GeoTIFF multibanda con todos los índices')}>
              {downloading === 'all' ? '…' : t('Todos')}
            </button>
          </div>
          {img?.datetime && <p className="muted small">{t('Pasada: {dt} UTC', { dt: img.datetime })}</p>}
        </div>
      )}

      {/* ── Panel de análisis ────────────────────────── */}
      {appMode === 'visor' && panelOpen && meta && (
        <div className="card series">
          <div className="series-head">
            <div className="tabs">
              <button className={tab === 'serie' ? 'on' : ''} onClick={() => { setTab('serie'); runSeries() }}>{t('Serie temporal')}</button>
              <button className={tab === 'tabla' ? 'on' : ''} onClick={() => { setTab('tabla'); runSeries() }}>{t('Tabla')}</button>
              <button className={tab === 'clases' ? 'on' : ''} onClick={runClasses} disabled={!activeDate}>{t('Clases')}</button>
              <button className={tab === 'clima' ? 'on' : ''} onClick={() => runClim()}>{t('Climatología')}</button>
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              {(tab === 'serie' || tab === 'tabla') && series && <button className="link dark" onClick={csvSeries}>{t('Descargar CSV')}</button>}
              <button className="x" onClick={() => setPanelOpen(false)} aria-label={t('Cerrar')}>×</button>
            </div>
          </div>

          {tab !== 'clases' && loadingSeries && <p className="muted an-empty"><span className="spin dark" /> {t('Calculando serie… (con GEE tarda ~5–10 s por fecha)')}</p>}

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
                  <Area name={t('Media embalse')} type="monotone" dataKey="mean" stroke="#0F8A78" strokeWidth={2.5} fill="url(#g)" connectNulls dot={{ r: 3, fill: '#0F8A78' }} activeDot={{ r: 5 }} />
                  {pois.map((p, i) => (
                    <Line key={p.name} name={p.name} type="monotone" dataKey={p.name} stroke={poiColor(i)} strokeWidth={1.8} dot={{ r: 2.5 }} connectNulls />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
              <p className="muted small">{t('Haz clic en un punto de la gráfica para ver su mapa.')}</p>
            </>
          )}

          {tab === 'tabla' && series && (
            <div className="tbl-wrap">
              <table className="tbl">
                <thead><tr><th>{t('Fecha')}</th><th>{t('Media embalse')}</th>{pois.map(p => <th key={p.name}>{p.name}</th>)}</tr></thead>
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

          {tab === 'clima' && (loadingClim
            ? <p className="muted an-empty"><span className="spin dark" /> {prog?.step || t('Calculando la climatología…')} {prog ? `· ${prog.progress}%` : ''}</p>
            : !CLIM_OK.includes(indexId)
              ? <div className="an-empty">
                  <p className="muted">{t('La climatología compara con años anteriores, así que solo tiene sentido con índices comparables entre embalses. Elige NDCI o PCI.')}</p>
                  <button className="ghost solid" style={{ maxWidth: 200, margin: '10px auto 0' }}
                    onClick={() => { changeIndex('NDCI_ind'); runClim('NDCI_ind') }}>{t('Ver con NDCI')}</button>
                </div>
              : clim ? <Climatology data={clim} /> : null)}
          {tab === 'clases' && (
            loadingClasses ? <p className="muted an-empty"><span className="spin dark" /> {t('Calculando superficie por clases…')}</p>
            : classes && (
              <>
                <p className="muted small" style={{ marginBottom: 10 }}>{t('Superficie de agua por rangos de')} <b>{t(meta.label)}</b> · {fmtDate(classes.date)}</p>
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

      {appMode === 'cal' && calRes && (
        <CalibrationResult r={calRes.r} unit={calRes.unit} target={calRes.target} reservoirLabel={reservoir ? labelOf(reservoir) : ''}
          onUse={useCalibration} onClose={() => setCalRes(null)} />
      )}
      {appMode === 'cal' && calRunning && <div className="hint">{prog?.step || t('Extrayendo índices de Sentinel-2 y ajustando modelos…')}</div>}

      {appMode === 'mon' && mon && (
        <MonitorResult data={mon} onOpen={openFromMonitor} onClose={() => setAppMode('visor')} />
      )}
      {appMode === 'mon' && monRunning && !mon && <div className="hint">{prog?.step || t('Consultando Sentinel-2 en todos los embalses…')}</div>}
      {appMode === 'info' && <ProjectPage lang={lang} setLang={setLang} onClose={() => setAppMode('visor')} onStart={() => setAppMode('visor')} />}

      {!reservoir && mode && appMode !== 'info' && appMode !== 'mon' && (
        <div className="hint">{t('Selecciona un embalse en el mapa o en el panel para empezar')}</div>
      )}
      {adding && <div className="hint">{t('Haz clic en el embalse para añadir un punto · Esc para cancelar')}</div>}
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
