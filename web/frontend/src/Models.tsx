import { useEffect, useMemo, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { api, fmt, LabPairs, LabRow, ModelCard } from './api'
import { locale, t } from './i18n'

const TROFICO = [
  { lo: 0.1, hi: 2.5, label: 'oligotrófico', color: '#E0EDF8' },
  { lo: 2.5, hi: 8, label: 'mesotrófico', color: '#DCF0E4' },
  { lo: 8, hi: 25, label: 'eutrófico', color: '#FBF0C7' },
  { lo: 25, hi: 1000, label: 'hipereutrófico', color: '#FBE6CE' },
]
const MES = (d: string) => new Date(d + 'T12:00:00').getMonth()
const MES_COL = ['#2c7bb6', '#2c7bb6', '#00a6ca', '#00ccbc', '#90eb9d', '#f9d057', '#f29e2e', '#e76818', '#d7191c', '#f29e2e', '#00ccbc', '#2c7bb6']

/** Lista de modelos, en el panel lateral. */
export function ModelList({ models, sel, onSel }: { models: ModelCard[]; sel: string | null; onSel: (id: string) => void }) {
  return (
    <section>
      <label className="lbl">{t('Modelos de la plataforma')}</label>
      <div className="model-list">
        {models.map(m => (
          <button key={m.id} className={'model-item' + (sel === m.id ? ' on' : '')} onClick={() => onSel(m.id)}>
            <span className="model-dot" />
            <span><b>{t(m.variable)}</b><small>{m.reservoir_label}</small></span>
            <Status s={m.status} small />
          </button>
        ))}
        {!models.length && <p className="muted small">{t('Cargando modelos…')}</p>}
      </div>
      <p className="muted small">{t('Cada modelo se valida con datos independientes antes de publicarse en el visor. ¿Tienes medidas de tu embalse? Crea tu propio modelo en «Calibra tu embalse».')}</p>
    </section>
  )
}

/** Ficha grande del modelo. */
export function ModelCardView({ m, onOpen, onClose }: { m: ModelCard; onOpen: () => void; onClose: () => void }) {
  const pts = m.pairs.map(p => ({ ...p, lo: Math.log10(Math.max(p.obs, 0.1)), lp: Math.log10(Math.max(p.pred, 0.1)), mes: MES(p.date) }))
  const lg = (v: number) => Math.log10(v)
  const ticks = [0.3, 1, 3, 10, 30, 100, 300].map(lg)
  const vmax = pts.length ? Math.max(...pts.map(p => Math.max(p.lo, p.lp))) + 0.15 : lg(300)
  const vmin = pts.length ? Math.min(...pts.map(p => Math.min(p.lo, p.lp))) - 0.15 : lg(0.3)
  const tk = ticks.filter(x => x >= vmin && x <= vmax)
  const dom: [number, number] = [vmin, vmax]
  const tf = (v: number) => fmt(10 ** v, v < 0 ? 1 : 0)

  return (
    <div className="card calres model-card">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div className="model-head">
        <div>
          <p className="eyebrow">{t('Modelo')} · {m.reservoir_label} · v{m.version}</p>
          <h2 className="serif">{t(m.variable)} <span className="unit">· {m.unit}</span></h2>
          <p className="muted">{t(m.description)}</p>
        </div>
        <Status s={m.status} />
      </div>

      <div className="model-formula">
        <code>{m.formula}</code>
        <small className="muted">{m.index_formula}</small>
      </div>

      <div className="model-metrics">
        {m.validation.metrics.map(k => (
          <div key={k.label} className="kpi" title={t(k.help)}>
            <b>{k.value}</b><span>{t(k.label)}</span>
          </div>
        ))}
      </div>

      <div className="model-grid">
        <div>
          {m.kind === 'lab' ? <LabPC /> : m.kind === 'seasonal' ? <SeasonalChart m={m} /> : m.kind === 'trend' ? <TrendChart m={m} /> : pts.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
                  {TROFICO.map(c => (
                    <ReferenceArea key={c.label} x1={Math.max(lg(c.lo), vmin)} x2={Math.min(lg(c.hi), vmax)}
                      y1={Math.max(lg(c.lo), vmin)} y2={Math.min(lg(c.hi), vmax)} fill={c.color} fillOpacity={0.9} ifOverflow="hidden" />
                  ))}
                  <CartesianGrid stroke="#E3ECEC" />
                  <XAxis type="number" dataKey="lo" domain={dom} ticks={tk} tickFormatter={tf} tick={{ fontSize: 11 }}
                    label={{ value: `${t('Medido por la sonda')} (${m.unit})`, position: 'bottom', fontSize: 11, offset: 2 }} />
                  <YAxis type="number" dataKey="lp" domain={dom} ticks={tk} tickFormatter={tf} tick={{ fontSize: 11 }} width={44}
                    label={{ value: `${t('Estimado por satélite')}`, angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  <ZAxis range={[26, 26]} />
                  <ReferenceLine segment={[{ x: vmin, y: vmin }, { x: vmax, y: vmax }]} stroke="#0C2B33" strokeDasharray="4 3" />
                  <Tooltip content={({ payload }: any) => {
                    const p = payload?.[0]?.payload
                    return p ? <div className="tt">{new Date(p.date + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: 'numeric' })}<br />
                      {t('Sonda')}: <b>{fmt(p.obs, 1)}</b> · {t('Satélite')}: <b>{fmt(p.pred, 1)}</b> {m.unit}</div> : null
                  }} />
                  <Scatter data={pts} shape={(props: any) => (
                    <circle cx={props.cx} cy={props.cy} r={3.6} fill={MES_COL[props.payload.mes]} stroke="#0C2B33" strokeWidth={0.4} fillOpacity={0.85} />
                  )} />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="muted small">
                {t('Cada punto es una fecha con imagen y sonda a la vez; la estimación es la de validación (el modelo no había visto ese año). Línea discontinua: acierto perfecto. Recuadros: estado trófico (OCDE). Color: mes del año.')}
              </p>
            </>
          ) : (
            <p className="muted an-empty">{t('Gráfico de validación no disponible todavía.')}</p>
          )}
          {!m.kind || m.kind === 'satellite' ? (
            <div className="model-legend">
              {TROFICO.map(c => <span key={c.label}><i style={{ background: c.color }} />{t(c.label)}</span>)}
            </div>
          ) : <ByYear m={m} />}
        </div>
        <div className="model-info">
          <h4>{t('Datos de calibración')}</h4>
          <ul>
            <li><b>{m.training.pairs}</b> {m.kind && m.kind !== 'satellite' ? t('días de sonda') : t('pares imagen–sonda')} · {m.training.period}</li>
            <li>{t(m.training.ground_truth)}</li>
            <li className="muted">{t(m.training.matching)}</li>
          </ul>
          <h4>{t('Validación')}</h4>
          <p>{t(m.validation.method)}</p>
          <p className="muted small">{t(m.validation.compared)}</p>
          <h4>{t('Limitaciones')}</h4>
          <ul>{m.limits.map(l => <li key={l}>{t(l)}</li>)}</ul>
          {m.index_id
            ? <button className="primary" onClick={onOpen}>🛰️ {t('Ver en el visor')}</button>
            : m.kind === 'lab'
              ? <p className="muted small">{t('Los pares salen de la base del proyecto y se actualizan con ella. Nada de lo que pruebes aquí modifica el visor: es un banco de pruebas.')}</p>
              : <p className="muted small">{t('Se calcula con los datos de la sonda y se actualiza solo cuando llegan datos nuevos. El valor de hoy aparece en la pestaña Monitor, junto a El Val.')}</p>}
        </div>
      </div>
    </div>
  )
}

const RIESGO_COL: Record<string, string> = { bajo: '#1C6B4B', medio: '#8A6D0B', alto: '#A8540C' }
const MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const MES_INI = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
const doyLabel = (d: number) => {
  const x = new Date(2025, 0, d)
  return `${x.getDate()} ${t(MESES[x.getMonth()])}`
}

function Status({ s, small }: { s: string; small?: boolean }) {
  const exp = s !== 'validado'
  return <span className={(small ? 'model-status' : 'model-badge') + (exp ? ' exp' : '')}>{exp ? '⚗' : '✓'} {t(s)}</span>
}

/** Curva anual de riesgo, con la frecuencia observada de picos y el día de hoy. */
function SeasonalChart({ m }: { m: ModelCard }) {
  const c = (m.curve || []).map(p => ({ ...p, pp: p.p * 100, oo: p.obs === null ? null : p.obs * 100 }))
  const hoy = m.today
  const lv = m.levels
  const ymax = Math.max(10, ...c.map(p => Math.max(p.pp, p.oo ?? 0))) * 1.1
  return (
    <>
      {hoy && (
        <div className="risk-today" style={{ borderColor: RIESGO_COL[hoy.level], color: RIESGO_COL[hoy.level] }}>
          {t('Hoy')} ({doyLabel(hoy.doy)}): <b>{t('riesgo {n}', { n: t(hoy.level) })}</b> · {fmt(hoy.p * 100, 0)} %
          {m.high_season && <span className="muted"> · {t('época de más riesgo')}: {m.high_season}</span>}
        </div>
      )}
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={c} margin={{ top: 18, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid stroke="#E3ECEC" />
          {lv && <ReferenceArea y1={lv.alto * 100} y2={ymax} fill="#FBE6CE" fillOpacity={0.6} ifOverflow="hidden" />}
          {lv && <ReferenceArea y1={lv.medio * 100} y2={lv.alto * 100} fill="#FBF0C7" fillOpacity={0.6} ifOverflow="hidden" />}
          <XAxis dataKey="doy" type="number" domain={[1, 366]} ticks={MES_INI} tickFormatter={d => t(MESES[MES_INI.indexOf(d)])} tick={{ fontSize: 11 }} />
          <YAxis domain={[0, ymax]} tickFormatter={v => `${fmt(v, 0)}%`} tick={{ fontSize: 11 }} width={44}
            label={{ value: t('Probabilidad de pico'), angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip content={({ payload }: any) => {
            const p = payload?.[0]?.payload
            return p ? <div className="tt">{doyLabel(p.doy)}<br />{t('Modelo')}: <b>{fmt(p.pp, 0)} %</b>
              {p.oo !== null && <><br />{t('Observado')}: {fmt(p.oo, 0)} %</>}</div> : null
          }} />
          <Line dataKey="oo" stroke="#9AB0B3" strokeWidth={1.2} dot={false} connectNulls isAnimationActive={false} />
          <Line dataKey="pp" stroke="#0E7C86" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          {hoy && <ReferenceLine x={hoy.doy} stroke={RIESGO_COL[hoy.level]} strokeDasharray="4 3" label={{ value: t('hoy'), fontSize: 11, position: 'top' }} />}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="muted small">
        {t('Línea gruesa: probabilidad de que un día sea de pico según la época (modelo). Línea gris: proporción de días de pico observada por la sonda en esas fechas. Fondo: riesgo medio y alto.')}
      </p>
    </>
  )
}

/** Últimas semanas de la sonda y pronóstico a 1, 3 y 7 días con su rango. */
function TrendChart({ m }: { m: ModelCard }) {
  const h = (m.history || []).map(p => ({ date: p.date, obs: p.value }))
  const f = (m.forecast || []).map(p => ({ date: p.date, pred: p.value, band: [p.lo, p.hi] as [number, number] }))
  const data = [...h.filter(p => !f.some(q => q.date === p.date)), ...f.map(q => ({ ...q, obs: h.find(p => p.date === q.date)?.obs }))]
  const fd = (d: string) => new Date(d + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short' })
  const last = m.forecast?.[m.forecast.length - 1]
  return (
    <>
      {m.forecast && m.forecast.length > 1 && (
        <div className="risk-today trend-row">
          {m.forecast.slice(1).map(p => (
            <span key={p.h}>+{p.h} {p.h > 1 ? t('días') : t('día')}: <b>{fmt(p.value, 1)}</b> <small className="muted">({fmt(p.lo, 1)}–{fmt(p.hi, 1)})</small></span>
          ))}
          <span className="muted">µg/L</span>
        </div>
      )}
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 18, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid stroke="#E3ECEC" />
          <XAxis dataKey="date" tickFormatter={fd} tick={{ fontSize: 11 }} minTickGap={24} />
          <YAxis tick={{ fontSize: 11 }} width={44} label={{ value: 'µg/L', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip content={({ payload }: any) => {
            const p = payload?.[0]?.payload
            return p ? <div className="tt">{fd(p.date)}<br />
              {p.obs !== undefined && <>{t('Sonda')}: <b>{fmt(p.obs, 1)}</b><br /></>}
              {p.pred !== undefined && <>{t('Esperado')}: <b>{fmt(p.pred, 1)}</b> ({fmt(p.band[0], 1)}–{fmt(p.band[1], 1)})</>}</div> : null
          }} />
          <Area dataKey="band" stroke="none" fill="#F29E2E" fillOpacity={0.22} isAnimationActive={false} />
          <Line dataKey="obs" stroke="#0E7C86" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="pred" stroke="#A8540C" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3 }} isAnimationActive={false} />
          {last && <ReferenceLine x={m.forecast![0].date} stroke="#5E7376" strokeDasharray="2 3" label={{ value: t('último dato'), fontSize: 11, position: 'top' }} />}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="muted small">
        {t('Azul: ficocianina media diaria de la sonda. Naranja: valor esperado; la banda es el rango en el que cayó el 80 % de los casos en años anteriores.')}
      </p>
    </>
  )
}

/** Tabla de validación año a año. */
function ByYear({ m }: { m: ModelCard }) {
  const rows: any[] = (m.validation as any).by_year || []
  if (!rows.length) return null
  if (m.kind === 'seasonal') {
    return (
      <table className="by-year"><thead><tr><th>{t('Año fuera')}</th><th>{t('Días')}</th><th>{t('Picos')}</th><th>AUC</th></tr></thead>
        <tbody>{rows.map(r => <tr key={r.year}><td>{r.year}</td><td>{r.days}</td><td>{r.peaks}</td><td>{r.auc === null ? '–' : fmt(r.auc, 2)}</td></tr>)}</tbody></table>
    )
  }
  const years = [...new Set(rows.map(r => r.year))]
  const hs = [...new Set(rows.map(r => r.h))]
  return (
    <table className="by-year"><thead><tr><th>{t('Año')}</th>{hs.map(h => <th key={h}>R² {h} d</th>)}</tr></thead>
      <tbody>{years.map(y => <tr key={y}><td>{y}</td>{hs.map(h => {
        const r = rows.find(x => x.year === y && x.h === h)
        return <td key={h} className={r && r.r2 < 0.3 ? 'bad' : ''}>{r ? fmt(r.r2, 2) : '–'}</td>
      })}</tr>)}</tbody></table>
  )
}

/* ── Laboratorio: ajustar un índice de Sentinel-2 a la ficocianina de la sonda ── */

type Forma = 'lineal' | 'cuadratica'
type Trans = 'ninguna' | 'log'

/** Mínimos cuadrados por ecuaciones normales (pocas columnas: resolución directa). */
function ajustar(X: number[][], y: number[]): number[] | null {
  const k = X[0].length
  const A: number[][] = Array.from({ length: k }, () => new Array(k + 1).fill(0))
  for (let i = 0; i < X.length; i++) {
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < k; b++) A[a][b] += X[i][a] * X[i][b]
      A[a][k] += X[i][a] * y[i]
    }
  }
  for (let c = 0; c < k; c++) {                       // eliminación de Gauss con pivoteo
    let piv = c
    for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r
    if (Math.abs(A[piv][c]) < 1e-12) return null
    ;[A[c], A[piv]] = [A[piv], A[c]]
    for (let r = 0; r < k; r++) {
      if (r === c) continue
      const f = A[r][c] / A[c][c]
      for (let j = c; j <= k; j++) A[r][j] -= f * A[c][j]
    }
  }
  return A.map((fila, i) => fila[k] / fila[i])
}

const r2 = (y: number[], p: number[]) => {
  const m = y.reduce((s, v) => s + v, 0) / y.length
  const ss = y.reduce((s, v) => s + (v - m) ** 2, 0)
  const sr = y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0)
  return ss > 0 ? 1 - sr / ss : NaN
}

const rangos = (v: number[]) => {                      // rangos medios, para Spearman
  const idx = v.map((x, i) => [x, i] as [number, number]).sort((a, b) => a[0] - b[0])
  const r = new Array(v.length).fill(0)
  for (let i = 0; i < idx.length;) {
    let j = i
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++
    const med = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) r[idx[k][1]] = med
    i = j + 1
  }
  return r
}

const spearman = (a: number[], b: number[]) => {
  const ra = rangos(a), rb = rangos(b)
  const ma = ra.reduce((s, v) => s + v, 0) / ra.length, mb = rb.reduce((s, v) => s + v, 0) / rb.length
  let num = 0, da = 0, db = 0
  for (let i = 0; i < ra.length; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2 }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : NaN
}

/** AUC por Mann-Whitney: ¿ordena el modelo los días de pico por encima del resto? */
const auc = (alto: boolean[], score: number[]) => {
  const r = rangos(score)
  const np = alto.filter(Boolean).length, nn = alto.length - np
  if (np < 3 || nn < 3) return NaN
  const sp = r.reduce((s, v, i) => s + (alto[i] ? v : 0), 0)
  return (sp - np * (np + 1) / 2) / (np * nn)
}

export function LabPC() {
  const [data, setData] = useState<LabPairs | null>(null)
  const [error, setError] = useState(false)
  const [indice, setIndice] = useState('pci')
  const [forma, setForma] = useState<Forma>('lineal')
  const [trans, setTrans] = useState<Trans>('ninguna')
  const [soloVerano, setSoloVerano] = useState(false)
  const [sinExtremos, setSinExtremos] = useState(false)
  const [fuera, setFuera] = useState<number[]>([])
  const [quitados, setQuitados] = useState<string[]>([])   // fechas excluidas a mano

  useEffect(() => { api.labPairs().then(setData).catch(() => setError(true)) }, [])

  const años = useMemo(() => Array.from(new Set((data?.rows ?? []).map(r => r.year))).sort(), [data])

  const res = useMemo(() => {
    if (!data) return null
    let filas = data.rows.filter(r => (r as any)[indice] != null && !fuera.includes(r.year) && !quitados.includes(r.date))
    if (soloVerano) filas = filas.filter(r => r.doy >= 121 && r.doy <= 304)     // may–oct
    if (sinExtremos && filas.length > 10) {
      const orden = [...filas].sort((a, b) => a.pc - b.pc)
      const lim = orden[Math.floor(orden.length * 0.99)].pc
      filas = filas.filter(r => r.pc < lim)
    }
    if (filas.length < 8) return { filas, pocos: true } as any

    const x = filas.map(r => (r as any)[indice] as number)
    const y = filas.map(r => (trans === 'log' ? Math.log10(r.pc) : r.pc))
    const base = (v: number) => (forma === 'cuadratica' ? [1, v, v * v] : [1, v])
    const X = x.map(base)
    const coef = ajustar(X, y)
    const pred = (c: number[], v: number) => base(v).reduce((s, b, i) => s + b * c[i], 0)
    const ajuste = coef ? x.map(v => pred(coef, v)) : null

    // validado: se ajusta sin un año y se predice ese año
    const val = new Array(filas.length).fill(NaN)
    for (const yr of Array.from(new Set(filas.map(r => r.year)))) {
      const tr = filas.map((r, i) => [r, i] as [LabRow, number]).filter(([r]) => r.year !== yr)
      if (tr.length < 8) continue
      const c = ajustar(tr.map(([, i]) => X[i]), tr.map(([, i]) => y[i]))
      if (!c) continue
      filas.forEach((r, i) => { if (r.year === yr) val[i] = pred(c, x[i]) })
    }
    // línea base: solo estacionalidad (seno y coseno del día del año), misma validación
    const S = filas.map(r => [1, Math.sin(2 * Math.PI * r.doy / 365.25), Math.cos(2 * Math.PI * r.doy / 365.25)])
    const est = new Array(filas.length).fill(NaN)
    for (const yr of Array.from(new Set(filas.map(r => r.year)))) {
      const tr = filas.map((_, i) => i).filter(i => filas[i].year !== yr)
      if (tr.length < 8) continue
      const c = ajustar(tr.map(i => S[i]), tr.map(i => y[i]))
      if (!c) continue
      filas.forEach((r, i) => { if (r.year === yr) est[i] = S[i].reduce((s, b, j) => s + b * c[j], 0) })
    }

    const ok = val.map(v => Number.isFinite(v))
    const yv = y.filter((_, i) => ok[i]), pv = val.filter((_, i) => ok[i])
    const pe = est.filter((_, i) => ok[i] && Number.isFinite(est[i]))
    const ordenPc = [...filas].map(r => r.pc).sort((a, b) => a - b)
    const p90 = ordenPc[Math.floor(ordenPc.length * 0.9)]
    const alto = filas.filter((_, i) => ok[i]).map(r => r.pc >= p90)

    const inv = (v: number) => (trans === 'log' ? 10 ** v : v)
    const lo = Math.min(...x), hi = Math.max(...x)
    const curva = coef ? Array.from({ length: 60 }, (_, i) => {
      const v = lo + (hi - lo) * i / 59
      return { x: v, y: inv(pred(coef, v)) }
    }) : []

    const hayVal = val.some(v => Number.isFinite(v))
    return {
      pocos: false, hayVal, n: filas.length, años: Array.from(new Set(filas.map(r => r.year))).sort(),
      puntos: filas.map((r, i) => ({ x: x[i], y: r.pc, date: r.date, year: r.year })),
      curva,
      serie: filas.filter((_, i) => ok[i]).map((r, i) => ({ date: r.date, obs: r.pc, pred: inv(pv[i]) })),
      r2_ajuste: ajuste ? r2(y, ajuste) : NaN,
      r2_val: pv.length >= 8 ? r2(yv, pv) : NaN,
      r2_est: pe.length >= 8 ? r2(yv.slice(0, pe.length), pe) : NaN,
      rho: spearman(x, filas.map(r => r.pc)),
      auc: auc(alto, pv),
      p90,
    }
  }, [data, indice, forma, trans, soloVerano, sinExtremos, fuera, quitados])

  if (error) return <p className="muted an-empty">{t('No se han podido cargar los pares de calibración.')}</p>
  if (!data) return <p className="muted an-empty"><span className="spin" /> {t('Cargando pares imagen–sonda…')}</p>
  if (!res || res.pocos) return (
    <>
      <LabControles {...{ data, indice, setIndice, forma, setForma, trans, setTrans, soloVerano, setSoloVerano, sinExtremos, setSinExtremos, años, fuera, setFuera }} />
      <p className="muted an-empty">{t('Con estos filtros quedan muy pocos pares para ajustar nada.')}</p>
    </>
  )

  const buena = res.hayVal && res.r2_val > 0.3 && res.r2_val > res.r2_est
  const veredicto = !res.hayVal
    ? t('Con un solo año no se puede validar: para comprobar el modelo hay que ajustarlo sin un año y probarlo en ese año. Activa al menos dos años y compara el R² del ajuste con el validado.')
    : buena
    ? t('El modelo aguanta la validación con estos filtros. Antes de creértelo, comprueba que no dependa de un solo año ni de unos pocos puntos extremos.')
    : res.r2_ajuste > 0.2
      ? t('Ojo: el ajuste parece razonable, pero al validarlo en un año que el modelo no ha visto se cae. Eso es sobreajuste: el índice no lleva información de ficocianina.')
      : t('Ni siquiera el ajuste encuentra relación. Es lo esperable: Sentinel-2 no tiene banda en 620 nm, donde absorbe la ficocianina.')

  return (
    <>
      <LabControles {...{ data, indice, setIndice, forma, setForma, trans, setTrans, soloVerano, setSoloVerano, sinExtremos, setSinExtremos, años, fuera, setFuera }} />

      <div className="model-metrics lab-kpis">
        <div className="kpi" title={t('Con todos los datos a la vez. Siempre mejora al complicar el modelo.')}>
          <b>{fmt(res.r2_ajuste, 2)}</b><span>{t('R² del ajuste')}</span>
        </div>
        <div className={'kpi' + (res.r2_val > 0.3 ? ' ok' : ' mal')} title={t('Ajustando sin un año y comprobando en ese año. Es el que cuenta.')}>
          <b>{fmt(res.r2_val, 2)}</b><span>{t('R² validado')}</span>
        </div>
        <div className="kpi" title={t('Lo que consigue un modelo que solo sabe la época del año, con la misma validación.')}>
          <b>{fmt(res.r2_est, 2)}</b><span>{t('solo estacionalidad')}</span>
        </div>
        <div className="kpi" title={t('Correlación de rangos entre el índice y la ficocianina.')}>
          <b>{fmt(res.rho, 2)}</b><span>{t('correlación (Spearman)')}</span>
        </div>
        <div className="kpi" title={t('Capacidad de señalar los días por encima del percentil 90 (0,5 = azar).')}>
          <b>{fmt(res.auc, 2)}</b><span>{t('AUC de picos')}</span>
        </div>
      </div>

      {quitados.length > 0 && (
        <p className="lab-quitados">
          {t('{n} puntos excluidos a mano', { n: quitados.length })}: {quitados.join(' · ')}
          <button className="ghost small" onClick={() => setQuitados([])}>{t('restaurar todos')}</button>
          <br /><span className="muted">{t('Quitar puntos cambia el resultado: hazlo solo si sabes por qué esa medida es mala, y déjalo escrito. Si de verdad es un fallo de la sonda, lo correcto es marcarla en la base de datos con su bandera de calidad.')}</span>
        </p>
      )}
      <p className={'lab-veredicto' + (buena ? ' ok' : '')}>{buena ? '✓ ' : '⚠ '}{veredicto}</p>

      <div className="lab-charts">
        <div>
          <p className="lbl">{t('Ficocianina frente al índice')}</p>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="#E3ECEC" />
              <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => fmt(v, 2)}
                label={{ value: data.indices.find(i => i.key === indice)?.name, position: 'bottom', fontSize: 11, offset: 2 }} />
              <YAxis type="number" dataKey="y" tick={{ fontSize: 11 }} width={44}
                label={{ value: 'µg/L', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <ZAxis range={[26, 26]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }: any) => {
                const p = payload?.find((x: any) => x?.payload?.date)?.payload
                return p ? <div className="tt">{p.date}<br />{t('Sonda')}: <b>{fmt(p.y, 1)}</b> µg/L · {t('índice')} {fmt(p.x, 3)}<br /><span className="muted">{t('pincha para excluirlo')}</span></div> : null
              }} />
              <Scatter data={res.puntos} onClick={(p: any) => p?.date && setQuitados(q => [...q, p.date])}
                cursor="pointer" shape={(pr: any) => (
                  <circle cx={pr.cx} cy={pr.cy} r={3.8} fill="#0E7C86" fillOpacity={0.55} stroke="#0C2B33" strokeWidth={0.3}>
                    <title>{`${pr.payload.date} · ${fmt(pr.payload.y, 1)} µg/L · ${t('índice')} ${fmt(pr.payload.x, 3)}\n${t('pincha para excluirlo')}`}</title>
                  </circle>
                )} />
              <Line data={res.curva} dataKey="y" stroke="#A8540C" strokeWidth={2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <details className="lab-top">
            <summary>{t('Ver los 8 valores más altos (con su fecha)')}</summary>
            <ul>
              {[...res.puntos].sort((a: any, b: any) => b.y - a.y).slice(0, 8).map((p: any) => (
                <li key={p.date}>
                  <code>{p.date}</code> · <b>{fmt(p.y, 1)}</b> µg/L · {t('índice')} {fmt(p.x, 3)}
                  <button className="ghost small" onClick={() => setQuitados(q => [...q, p.date])}>✕ {t('excluir')}</button>
                </li>
              ))}
            </ul>
          </details>
        </div>
        <div>
          <p className="lbl">{t('En el tiempo: sonda y estimación validada')}</p>
          {!res.hayVal ? (
            <p className="muted an-empty lab-sinval">{t('Sin estimación validada: hace falta más de un año activado.')}</p>
          ) : (
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={res.serie} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="#E3ECEC" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={28}
                tickFormatter={(d: string) => new Date(d + 'T12:00:00').toLocaleDateString(locale(), { month: 'short', year: '2-digit' })} />
              <YAxis tick={{ fontSize: 11 }} width={44} label={{ value: 'µg/L', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <Tooltip content={({ payload }: any) => {
                const p = payload?.[0]?.payload
                return p ? <div className="tt">{p.date}<br />{t('Sonda')}: <b>{fmt(p.obs, 1)}</b><br />{t('Modelo')}: <b>{fmt(p.pred, 1)}</b></div> : null
              }} />
              <Line dataKey="obs" name={t('Sonda')} stroke="#0E7C86" strokeWidth={1.8} dot={false} isAnimationActive={false} />
              <Line dataKey="pred" name={t('Modelo')} stroke="#A8540C" strokeWidth={1.6} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          )}
        </div>
      </div>

      <p className="muted small">
        {t('{n} pares de {a} · umbral de pico (percentil 90): {p} µg/L · datos de la sonda {s}.',
          { n: res.n, a: res.años.join(', '), p: fmt(res.p90, 1), s: data.station })}
      </p>
    </>
  )
}

function LabControles(p: any) {
  const { data, indice, setIndice, forma, setForma, trans, setTrans, soloVerano, setSoloVerano,
    sinExtremos, setSinExtremos, años, fuera, setFuera } = p
  return (
    <div className="lab-ctrl">
      <label>{t('Índice')}
        <select value={indice} onChange={e => setIndice(e.target.value)}>
          {data.indices.map((i: any) => <option key={i.key} value={i.key}>{i.name}</option>)}
        </select>
      </label>
      <label>{t('Forma')}
        <select value={forma} onChange={e => setForma(e.target.value as Forma)}>
          <option value="lineal">{t('lineal')}</option>
          <option value="cuadratica">{t('cuadrática')}</option>
        </select>
      </label>
      <label>{t('Ficocianina')}
        <select value={trans} onChange={e => setTrans(e.target.value as Trans)}>
          <option value="ninguna">{t('sin transformar')}</option>
          <option value="log">{t('logaritmo')}</option>
        </select>
      </label>
      <label className="chk"><input type="checkbox" checked={soloVerano} onChange={e => setSoloVerano(e.target.checked)} /> {t('solo mayo–octubre')}</label>
      <label className="chk"><input type="checkbox" checked={sinExtremos} onChange={e => setSinExtremos(e.target.checked)} /> {t('quitar el 1 % más alto')}</label>
      <span className="lab-years">{t('Años')}:
        {años.map((y: number) => (
          <button key={y} className={'tag' + (fuera.includes(y) ? '' : ' on')}
            onClick={() => setFuera((f: number[]) => f.includes(y) ? f.filter(x => x !== y) : [...f, y])}>{y}</button>
        ))}
      </span>
    </div>
  )
}
