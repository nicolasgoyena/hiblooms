import { CartesianGrid, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
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
            <span className="model-status">✓ {t(m.status)}</span>
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
        <span className="model-badge">✓ {t(m.status)}</span>
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
          {pts.length > 0 ? (
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
          <div className="model-legend">
            {TROFICO.map(c => <span key={c.label}><i style={{ background: c.color }} />{t(c.label)}</span>)}
          </div>
        </div>
        <div className="model-info">
          <h4>{t('Datos de calibración')}</h4>
          <ul>
            <li><b>{m.training.pairs}</b> {t('pares imagen–sonda')} · {m.training.period}</li>
            <li>{t(m.training.ground_truth)}</li>
            <li className="muted">{t(m.training.matching)}</li>
          </ul>
          <h4>{t('Validación')}</h4>
          <p>{t(m.validation.method)}</p>
          <p className="muted small">{t(m.validation.compared)}</p>
          <h4>{t('Limitaciones')}</h4>
          <ul>{m.limits.map(l => <li key={l}>{t(l)}</li>)}</ul>
          <button className="primary" onClick={onOpen}>🛰️ {t('Ver en el visor')}</button>
        </div>
      </div>
    </div>
  )
}
