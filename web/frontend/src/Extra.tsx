import { useEffect, useMemo, useState } from 'react'
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { api, colorAt, CoreData, CoreMeta, fmt, PhytoResp, SensorRes, SensorSeries } from './api'
import { locale, t } from './i18n'

const fmtDate = (s: string | null) => s ? new Date(s.slice(0, 10) + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: 'numeric' }) : '—'
const fmtShort = (s: string) => new Date(s.slice(0, 10) + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: '2-digit' })
const ms = (s: string) => new Date(s.slice(0, 10) + 'T12:00:00').getTime()
const RAMP = ['#2c7bb6', '#00a6ca', '#00ccbc', '#90eb9d', '#ffff8c', '#f9d057', '#f29e2e', '#e76818', '#d7191c']
/** Números grandes legibles: 1,2 M · 35 k. */
const big = (v: number | null | undefined) => v == null ? '—' : Math.abs(v) >= 1e6 ? `${fmt(v / 1e6, 1)} M` : Math.abs(v) >= 1e4 ? `${fmt(v / 1e3, 0)} k` : fmt(v, v < 10 ? 2 : 0)

function Empty({ text }: { text: string }) { return <p className="muted an-empty">{text}</p> }

// ── Fitoplancton ────────────────────────────────────────────────────────────

const GROUP_COLORS: [RegExp, string][] = [
  [/cyano/i, '#1D8A99'], [/chloro/i, '#6BA539'], [/bacillario|diatom/i, '#C9A227'], [/crypto/i, '#B5651D'],
  [/dino/i, '#A34A6B'], [/chryso/i, '#E0B04A'], [/eugleno/i, '#7A63B8'], [/zygnemato|desmid|charo/i, '#3F8F5A'],
]
const EXTRA = ['#5B8DB8', '#D98C5F', '#9C7FB7', '#7FA38A', '#B8A25B']
export function groupColor(g: string, i = 0) {
  return GROUP_COLORS.find(([re]) => re.test(g))?.[1] ?? (g === 'Sin asignar' ? '#B0BCBE' : EXTRA[i % EXTRA.length])
}

export function PhytoResult({ body, sites, onClose }: { body: string; sites: string[]; onClose: () => void }) {
  const [metric, setMetric] = useState<'biovolume' | 'density'>('biovolume')
  const [data, setData] = useState<PhytoResp | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [sel, setSel] = useState<number>(-1)
  const [share, setShare] = useState(false)
  useEffect(() => {
    setData(null); setErr(null)
    api.dbPhyto(body || undefined, sites, metric).then(r => { setData(r); setSel(r.samples.length - 1) }).catch(e => setErr(e.message))
  }, [body, sites.join(','), metric]) // eslint-disable-line react-hooks/exhaustive-deps

  const rows = useMemo(() => (data?.samples ?? []).map((s, i) => {
    const r: Record<string, number | string | null> = { i, label: `${fmtShort(s.date)} · ${s.code}`, cyano: s.cyano_pct }
    for (const g of data!.groups) r[g] = share ? (s.total ? 100 * (s.groups[g] ?? 0) / s.total : 0) : (s.groups[g] ?? 0)
    return r
  }), [data, share])

  const unit = metric === 'density' ? t('células/mL') : 'µm³/mL'
  const s = data && sel >= 0 ? data.samples[sel] : null

  return (
    <div className="card calres dbres xres">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div>
        <p className="eyebrow">{t('Fitoplancton')}</p>
        <h2 className="serif">{data ? t('{n} muestras analizadas', { n: data.samples.length }) : t('Cargando…')}</h2>
        <p className="muted small">{t('Recuentos al microscopio (Utermöhl). Cada barra es una muestra; los colores son grupos taxonómicos y la línea, el % de cianobacterias.')}</p>
      </div>
      <div className="wc-bar">
        <div className="seg">
          <button className={metric === 'biovolume' ? 'on' : ''} onClick={() => setMetric('biovolume')}>{t('Biovolumen')}</button>
          <button className={metric === 'density' ? 'on' : ''} onClick={() => setMetric('density')}>{t('Densidad')}</button>
        </div>
        <div className="seg">
          <button className={!share ? 'on' : ''} onClick={() => setShare(false)}>{t('Absoluto')}</button>
          <button className={share ? 'on' : ''} onClick={() => setShare(true)}>%</button>
        </div>
      </div>
      {err && <div className="badge err">{err}</div>}
      {data && !data.samples.length && <Empty text={t('No hay recuentos de fitoplancton para esta selección.')} />}
      {data && data.samples.length > 0 && (
        <>
          <div className="an-chart">
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
                onClick={(e: any) => { if (e?.activeTooltipIndex != null) setSel(Number(e.activeTooltipIndex)) }}>
                <CartesianGrid stroke="#E3ECEC" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis yAxisId="l" tick={{ fontSize: 11 }} width={52} tickFormatter={v => share ? `${v}%` : big(v)}
                  domain={share ? [0, 100] : [0, 'auto']} />
                <YAxis yAxisId="r" orientation="right" domain={[0, 100]} tick={{ fontSize: 11 }} width={36} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={(v: any, n: any) => n === t('% cianobacterias') ? `${fmt(v, 0)}%` : share ? `${fmt(v, 1)}%` : `${big(v)} ${unit}`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {data.groups.map((g, i) => (
                  <Bar key={g} yAxisId="l" dataKey={g} stackId="a" fill={groupColor(g, i)} name={g} cursor="pointer" />
                ))}
                <Line yAxisId="r" dataKey="cyano" name={t('% cianobacterias')} stroke="#0C2B33" strokeDasharray="4 3" dot={{ r: 2.5 }} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="muted small">{t('Pulsa una barra para ver sus taxones dominantes.')}</p>
          {s && (
            <div className="phy-detail">
              <div className="phy-head">
                <b>{s.code}</b> · {fmtDate(s.date)}{s.depth_m != null ? ` · ${fmt(s.depth_m, 1)} m` : ''}
                <span className="muted"> · {s.n_taxa} {t('taxones')} · {t('total')} {big(s.total)} {unit}</span>
                {s.cyano_pct != null && <span className={'phy-cy' + (s.cyano_pct >= 50 ? ' hi' : '')}>{fmt(s.cyano_pct, 0)}% {t('cianobacterias')}</span>}
                {s.toxic_pct != null && s.toxic_pct > 0 && <span className="phy-tox">☣ {fmt(s.toxic_pct, 0)}% {t('potencialmente tóxicas')}</span>}
              </div>
              <table className="tbl">
                <thead><tr><th>{t('Taxón')}</th><th>{t('Grupo')}</th><th>{t('Densidad')} (cél/mL)</th><th>{t('Biovolumen')} (µm³/mL)</th><th>%</th></tr></thead>
                <tbody>
                  {s.top.map(x => {
                    const v = metric === 'density' ? x.density : x.biovolume
                    return (
                      <tr key={x.name}>
                        <td><i>{x.name}</i>{x.toxic && <span className="phy-toxtag" title={t('Potencialmente tóxica')}> ☣</span>}</td>
                        <td><span className="gdot" style={{ background: groupColor(x.group, data.groups.indexOf(x.group)) }} />{x.group}</td>
                        <td>{big(x.density)}</td><td>{big(x.biovolume)}</td>
                        <td>{v != null && s.total ? fmt(100 * v / s.total, 1) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {s.cyano_density > 0 && (
                <p className="muted small">
                  {t('Cianobacterias: {n} células/mL.', { n: big(s.cyano_density) })}{' '}
                  {s.cyano_density >= 100000 ? t('Por encima del nivel de alerta 2 de la OMS para aguas de baño (100 000 cél/mL).')
                    : s.cyano_density >= 20000 ? t('Por encima del nivel de alerta 1 de la OMS para aguas de baño (20 000 cél/mL).')
                    : t('Por debajo de los niveles de alerta de la OMS para aguas de baño (20 000 cél/mL).')}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── Testigos de sedimento ───────────────────────────────────────────────────

export function CoresResult({ body, onClose }: { body: string; onClose: () => void }) {
  const [cores, setCores] = useState<CoreMeta[] | null>(null)
  const [cid, setCid] = useState<number | null>(null)
  const [cd, setCd] = useState<CoreData | null>(null)
  const [pick, setPick] = useState<string[]>([])
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    setCores(null); setErr(null)
    api.dbCores(body || undefined).then(r => {
      setCores(r.cores)
      setCid(r.cores.find(c => c.n_sections > 0)?.core_id ?? r.cores[0]?.core_id ?? null)
    }).catch(e => setErr(e.message))
  }, [body])
  useEffect(() => {
    setCd(null)
    if (cid == null) return
    api.dbCore(cid).then(r => { setCd(r); setPick(r.parameters.slice(0, 4).map(p => p.parameter_code)) }).catch(e => setErr(e.message))
  }, [cid])
  const meta = cores?.find(c => c.core_id === cid)
  const toggle = (p: string) => setPick(v => v.includes(p) ? v.filter(x => x !== p) : [...v, p].slice(-6))

  return (
    <div className="card calres dbres xres">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div>
        <p className="eyebrow">{t('Testigos de sedimento')}</p>
        <h2 className="serif">{cores ? t('{n} testigos', { n: cores.length }) : t('Cargando…')}</h2>
        <p className="muted small">{t('Cada testigo se corta en secciones; la profundidad está en cm bajo la superficie del sedimento (arriba, lo más reciente).')}</p>
      </div>
      {err && <div className="badge err">{err}</div>}
      {cores && !cores.length && <Empty text={t('No hay testigos de sedimento para esta selección.')} />}
      {cores && cores.length > 0 && (
        <div className="core-list">
          {cores.map(c => (
            <button key={c.core_id} className={'kchip' + (c.core_id === cid ? ' on' : '')} onClick={() => setCid(c.core_id)}>
              <b>{c.code ?? `#${c.core_id}`}</b>
              <small className="muted">{c.water_body ?? ''} {c.date ? `· ${fmtDate(c.date)}` : ''} · {c.n_sections ? t('{n} secciones', { n: c.n_sections }) : t('sin analíticas')}</small>
            </button>
          ))}
        </div>
      )}
      {meta && (
        <p className="core-meta small">
          {meta.length_cm != null && <span>📏 {fmt(meta.length_cm, 1)} cm</span>}
          {meta.water_depth_m != null && <span>🌊 {fmt(meta.water_depth_m, 1)} m {t('de columna de agua')}</span>}
          {meta.interval_cm != null && <span>🔪 {t('cortes cada {n} cm', { n: fmt(meta.interval_cm, 1) })}</span>}
          {meta.site_code && <span>📍 {meta.site_code}</span>}
          {meta.notes && <span className="muted core-notes">{meta.notes}</span>}
        </p>
      )}
      {meta && cd && !cd.rows.length && <Empty text={t('Este testigo aún no tiene resultados de laboratorio en la base de datos.')} />}
      {cd && cd.rows.length > 0 && (
        <>
          <div className="kind-legend">
            {cd.parameters.map(p => (
              <button key={p.parameter_code} className={'kchip' + (pick.includes(p.parameter_code) ? ' on' : ' off')} onClick={() => toggle(p.parameter_code)}>
                {p.name}{p.unit ? ` (${p.unit})` : ''}
              </button>
            ))}
          </div>
          <div className="core-grid">
            {pick.map(pc => {
              const p = cd.parameters.find(x => x.parameter_code === pc)
              const pts = cd.rows.filter(r => r.parameter_code === pc).sort((a, b) => a.depth_cm - b.depth_cm)
              if (!p || !pts.length) return null
              return (
                <div key={pc} className="core-plot">
                  <b className="small">{p.name}{p.unit ? <span className="muted"> · {p.unit}</span> : null}</b>
                  <ResponsiveContainer width="100%" height={260}>
                    <ComposedChart layout="vertical" data={pts} margin={{ top: 6, right: 10, bottom: 0, left: 0 }}>
                      <CartesianGrid stroke="#E3ECEC" />
                      <XAxis type="number" dataKey="value" tick={{ fontSize: 10 }} domain={['auto', 'auto']} tickFormatter={v => big(v)} />
                      <YAxis type="number" dataKey="depth_cm" tick={{ fontSize: 10 }} width={34} unit=" cm" domain={[0, 'dataMax']} />
                      <Tooltip formatter={(v: any) => fmt(v, 3)} labelFormatter={() => ''}
                        content={({ payload }: any) => {
                          const r = payload?.[0]?.payload
                          return r ? <div className="tt">{fmt(r.section_top_cm, 1)}–{fmt(r.section_bottom_cm ?? r.section_top_cm, 1)} cm: <b>{fmt(r.value, 3)}</b>{r.below_lod ? ` (${t('< LD')})` : ''}</div> : null
                        }} />
                      <Line dataKey="value" stroke="#8B6B4A" dot={{ r: 2.5, fill: '#8B6B4A' }} isAnimationActive={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}

// ── Sondas fijas + satélite ─────────────────────────────────────────────────

export function SensorsResult({ onClose }: { onClose: () => void }) {
  const [res, setRes] = useState<SensorRes[] | null>(null)
  const [vars, setVars] = useState<{ key: string; name: string; unit: string }[]>([])
  const [rid, setRid] = useState<number | null>(null)
  const [variable, setVariable] = useState('phycocyanin')
  const [layer, setLayer] = useState<'surface' | 'column'>('surface')
  const [view, setView] = useState<'series' | 'section'>('series')
  const [d, setD] = useState<SensorSeries | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [showIdx, setShowIdx] = useState(true)
  useEffect(() => {
    api.dbSensors().then(r => { setRes(r.reservoirs); setVars(r.variables); setRid(r.reservoirs[0]?.reservoir_id ?? null) }).catch(e => setErr(e.message))
  }, [])
  useEffect(() => {
    if (rid == null) return
    setD(null)
    api.dbSensorSeries(rid, variable, layer).then(setD).catch(e => setErr(e.message))
  }, [rid, variable, layer])

  // Serie combinada en tiempo continuo
  const rows = useMemo(() => {
    if (!d) return []
    const m = new Map<number, Record<string, number | null>>()
    const put = (date: string, k: string, v: number | null) => {
      const x = ms(date); const r = m.get(x) ?? { t: x }; r[k] = v; m.set(x, r)
    }
    d.daily.forEach(r => put(r.date, 'sensor', r.value))
    if (d.variable === 'phycocyanin') d.sat.forEach(r => { if (r.phycocyanin_est != null) put(r.date, 'sat', r.phycocyanin_est) })
    d.idx.forEach(r => { if (r.pci != null) put(r.date, 'pci', r.pci) })
    return Array.from(m.values()).sort((a, b) => (a.t as number) - (b.t as number))
  }, [d])

  const hasSensor = !!d?.daily.length
  const hasPci = rows.some(r => r.pci != null)
  const meta = res?.find(r => r.reservoir_id === rid)

  return (
    <div className="card calres dbres xres">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div>
        <p className="eyebrow">{t('Sondas fijas y satélite')}</p>
        <h2 className="serif">{d?.name ?? meta?.name ?? t('Cargando…')}{d?.unit ? <span className="unit"> · {t(d.var_name)}</span> : null}</h2>
        {meta && <p className="muted small">
          {t('{a} días con datos de sonda · {b} fechas de satélite', { a: meta.sensor_days, b: meta.sat_dates })}
          {meta.first ? ` · ${fmtDate(meta.first)} → ${fmtDate(meta.last)}` : ''}
          {meta.sources?.length ? ` · ${meta.sources.join(', ')}` : ''}
        </p>}
      </div>
      {err && <div className="badge err">{err}</div>}
      {res && !res.length && <Empty text={t('No hay datos de sondas ni estimaciones de satélite en la base de datos.')} />}
      {res && res.length > 0 && (
        <div className="wc-bar">
          <select className="wc-date" value={rid ?? ''} onChange={e => setRid(+e.target.value)}>
            {res.map(r => <option key={r.reservoir_id} value={r.reservoir_id}>{r.name}</option>)}
          </select>
          <select className="wc-date" value={variable} onChange={e => setVariable(e.target.value)}>
            {vars.map(v => <option key={v.key} value={v.key}>{t(v.name)}{v.unit ? ` (${v.unit})` : ''}</option>)}
          </select>
          <div className="seg">
            <button className={view === 'series' ? 'on' : ''} onClick={() => setView('series')}>{t('Serie')}</button>
            <button className={view === 'section' ? 'on' : ''} onClick={() => setView('section')}>{t('Profundidad × tiempo')}</button>
          </div>
          {view === 'series' && (
            <div className="seg">
              <button className={layer === 'surface' ? 'on' : ''} onClick={() => setLayer('surface')}>{t('Superficie (0–2 m)')}</button>
              <button className={layer === 'column' ? 'on' : ''} onClick={() => setLayer('column')}>{t('Media de la columna')}</button>
            </div>
          )}
        </div>
      )}
      {d && view === 'series' && (
        !rows.length ? <Empty text={t('Sin datos de esta variable en este embalse.')} /> : (
          <>
            <div className="an-chart">
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#E3ECEC" />
                  <XAxis dataKey="t" type="number" scale="time" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11 }}
                    tickFormatter={v => new Date(v).toLocaleDateString(locale(), { month: 'short', year: '2-digit' })} />
                  <YAxis yAxisId="l" tick={{ fontSize: 11 }} width={48} label={{ value: d.unit, angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  {hasPci && showIdx && <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 11 }} width={40} label={{ value: 'PCI', angle: 90, position: 'insideRight', fontSize: 11 }} />}
                  <Tooltip labelFormatter={v => fmtDate(new Date(v as number).toISOString())} formatter={(v: any) => fmt(v, 2)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {hasSensor && <Line yAxisId="l" dataKey="sensor" name={t('Sonda (media diaria)')} stroke="#1D6FA8" dot={false} strokeWidth={1.4} connectNulls={false} isAnimationActive={false} />}
                  {d.variable === 'phycocyanin' && <Scatter yAxisId="l" dataKey="sat" name={t('Satélite (ficocianina estimada)')} fill="#C8561B" />}
                  {hasPci && showIdx && <Scatter yAxisId="r" dataKey="pci" name={t('Índice PCI (Sentinel-2)')} fill="#16B3A6" shape="diamond" />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            {hasPci && <label className="chk small"><input type="checkbox" checked={showIdx} onChange={e => setShowIdx(e.target.checked)} /> {t('Mostrar el índice PCI de Sentinel-2')}</label>}
            {d.model && d.variable === 'phycocyanin' && (
              <p className={'small ' + (d.model.valid && (d.model.r2 ?? 0) > 0.3 ? 'muted' : 'badge warn')}>
                {t('Modelo de satélite')}: {d.model.name ?? '—'} {t('sobre')} {d.model.index?.toUpperCase() ?? '—'}
                {d.model.r2 != null ? ` · R² (CV) ${fmt(d.model.r2, 2)}` : ''}{d.model.rmse != null ? ` · RMSE ${fmt(d.model.rmse, 2)}` : ''}
                {!(d.model.valid && (d.model.r2 ?? 0) > 0.3) ? ` · ${t('modelo no validado: las estimaciones son orientativas.')}` : ''}
              </p>
            )}
            {!hasSensor && <p className="muted small">{t('Este embalse no tiene datos de sonda válidos para esta variable; solo se muestra el satélite.')}</p>}
          </>
        )
      )}
      {d && view === 'section' && <SensorSection d={d} />}
      <p className="muted small">{t('Las sondas registran varias veces al día y a varias profundidades; aquí se muestran medias diarias (serie) o semanales por metro (diagrama). Se excluyen los datos marcados como erróneos.')}</p>
    </div>
  )
}

function SensorSection({ d }: { d: SensorSeries }) {
  const p = d.profile
  if (!p.length) return <Empty text={t('Sin perfiles de esta variable.')} />
  const weeks = Array.from(new Set(p.map(r => r.date))).sort()
  const zmax = Math.max(...p.map(r => r.dbin))
  const vs = p.map(r => r.value).sort((a, b) => a - b)
  const lo = vs[Math.floor(vs.length * .02)], hi = vs[Math.floor(vs.length * .98)]
  const W = 900, L = 40, B = 22, H = 260, cw = (W - L - 8) / weeks.length, ch = (H - B - 6) / (zmax + 1)
  const col = (v: number) => colorAt(RAMP, hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : .5)
  const wi = new Map(weeks.map((w, i) => [w, i]))
  const years = weeks.map((w, i) => [w, i] as const).filter(([w], i) => i === 0 || w.slice(0, 7) !== weeks[i - 1].slice(0, 7)).filter(([w]) => ['01', '04', '07', '10'].includes(w.slice(5, 7)))
  return (
    <>
      <svg className="camp-tl" viewBox={`0 0 ${W} ${H}`}>
        {p.map((r, k) => (
          <rect key={k} x={L + (wi.get(r.date) ?? 0) * cw} y={6 + r.dbin * ch} width={cw + .6} height={ch + .6} fill={col(r.value)}>
            <title>{`${fmtDate(r.date)} · ${r.dbin}–${r.dbin + 1} m · ${fmt(r.value, 2)} ${d.unit}`}</title>
          </rect>
        ))}
        {Array.from({ length: zmax + 2 }, (_, z) => z).filter(z => z % Math.max(1, Math.ceil(zmax / 6)) === 0).map(z => (
          <text key={z} x={L - 6} y={6 + z * ch + 4} textAnchor="end" className="tl-lbl">{z} m</text>
        ))}
        {years.map(([w, i]) => <text key={w} x={L + i * cw} y={H - 6} className="tl-lbl">{fmtShort(w).split(' ').slice(1).join(' ')}</text>)}
      </svg>
      <div className="wc-legend"><span>{fmt(lo, 1)}</span><i style={{ background: `linear-gradient(90deg, ${RAMP.join(',')})` }} /><span>{fmt(hi, 1)} {d.unit}</span></div>
    </>
  )
}
