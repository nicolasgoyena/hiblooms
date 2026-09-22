import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { fmt, ModelCard } from './api'
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
          {m.kind === 'seasonal' ? <SeasonalChart m={m} /> : m.kind === 'trend' ? <TrendChart m={m} /> : pts.length > 0 ? (
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
