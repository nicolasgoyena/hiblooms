import { useMemo, useState } from 'react'
import { DbCampaign, DbKind, fmt } from './api'
import { locale, t } from './i18n'

/** Color de cada tipo de muestra (mismo en leyenda, cronograma y visitas). */
export const KIND_COLORS: Record<string, string> = {
  field: '#1D8A99', probe: '#2F6FB0', fluoro: '#3E9E6B', lab: '#8A6DC2',
  phyto: '#C9812E', sed: '#8B6B4A', core: '#5E4A3A', other: '#8A9A9C',
}
const fmtDate = (s: string | null) => s ? new Date(s + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: 'numeric' }) : '—'
const fmtMonth = (d: Date) => d.toLocaleDateString(locale(), { month: 'short', year: '2-digit' })

function KindDots({ kinds }: { kinds: string[] }) {
  return <span className="kdots">{kinds.map(k => <i key={k} style={{ background: KIND_COLORS[k] ?? KIND_COLORS.other }} />)}</span>
}

/** Cronograma: una fila por masa de agua, un punto por campaña. */
function Timeline({ camps, onPick }: { camps: DbCampaign[]; onPick: (c: DbCampaign) => void }) {
  const dated = camps.filter(c => c.start)
  const bodies = Array.from(new Set(dated.map(c => c.water_body ?? '—'))).sort()
  if (!dated.length) return null
  const ts = dated.map(c => new Date(c.start + 'T12:00:00').getTime())
  const t0 = Math.min(...ts) - 20 * 864e5, t1 = Math.max(...ts) + 20 * 864e5
  const W = 920, L = 136, R = 12, ROW = 22, H = bodies.length * ROW + 26
  const x = (ms: number) => L + (ms - t0) / (t1 - t0) * (W - L - R)
  // marcas: principio de cada trimestre
  const ticks: Date[] = []
  const d = new Date(t0); d.setDate(1); d.setMonth(Math.floor(d.getMonth() / 3) * 3 + 3)
  while (d.getTime() < t1) { ticks.push(new Date(d)); d.setMonth(d.getMonth() + 3) }
  return (
    <svg className="camp-tl" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('Cronograma de campañas')}>
      {ticks.map(k => (
        <g key={k.getTime()}>
          <line x1={x(k.getTime())} x2={x(k.getTime())} y1={4} y2={H - 18} className="tl-grid" />
          <text x={x(k.getTime())} y={H - 5} textAnchor="middle" className="tl-lbl">{fmtMonth(k)}</text>
        </g>
      ))}
      {bodies.map((b, i) => (
        <g key={b}>
          <text x={L - 8} y={i * ROW + 17} textAnchor="end" className="tl-body">{b.length > 18 ? b.slice(0, 17) + "…" : b}<title>{b}</title></text>
          <line x1={L} x2={W - R} y1={i * ROW + 13} y2={i * ROW + 13} className="tl-row" />
        </g>
      ))}
      {dated.map(c => {
        const i = bodies.indexOf(c.water_body ?? '—')
        const r = 3.5 + Math.min(4, Math.sqrt(c.n_visits))
        return (
          <circle key={`${c.campaign_id}-${c.code}`} cx={x(new Date(c.start + 'T12:00:00').getTime())} cy={i * ROW + 13} r={r}
            className="tl-dot" onClick={() => onPick(c)}>
            <title>{`${c.code} · ${fmtDate(c.start)} · ${c.n_visits} ${t('visitas')}`}</title>
          </circle>
        )
      })}
    </svg>
  )
}

export function CampaignsResult({ data, onOpenSite, onClose }: {
  data: { campaigns: DbCampaign[]; kinds: DbKind[] }
  onOpenSite: (site: string, waterBody: string | null) => void
  onClose: () => void
}) {
  const [open, setOpen] = useState<string | null>(null)
  const [kind, setKind] = useState<string | null>(null)
  const [year, setYear] = useState<string>('')
  const years = useMemo(() => Array.from(new Set(data.campaigns.map(c => c.start?.slice(0, 4)).filter(Boolean) as string[])).sort().reverse(), [data])
  const camps = data.campaigns.filter(c => (!kind || c.kinds.includes(kind)) && (!year || c.start?.startsWith(year)))
  const used = new Set(data.campaigns.flatMap(c => c.kinds))
  const nVisits = camps.reduce((a, c) => a + c.n_visits, 0)
  const key = (c: DbCampaign) => `${c.campaign_id}-${c.code}`

  const pick = (c: DbCampaign) => {
    setOpen(key(c))
    setTimeout(() => document.getElementById('camp-' + key(c))?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50)
  }

  return (
    <div className="card calres dbres camp">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div>
        <p className="eyebrow">{t('Campañas de muestreo')}</p>
        <h2 className="serif">{t('{n} campañas · {v} visitas', { n: camps.length, v: nVisits })}</h2>
        <p className="muted small">{t('Cada visita es un punto muestreado en una campaña. Los colores indican qué se tomó.')}</p>
      </div>

      <div className="camp-filters">
        <select value={year} onChange={e => setYear(e.target.value)}>
          <option value="">{t('Todos los años')}</option>
          {years.map(y => <option key={y} value={y}>{y}</option>)}
        </select>
        <div className="kind-legend">
          {data.kinds.filter(k => used.has(k.key)).map(k => (
            <button key={k.key} className={'kchip' + (kind === k.key ? ' on' : kind ? ' off' : '')}
              onClick={() => setKind(kind === k.key ? null : k.key)}>
              <i style={{ background: KIND_COLORS[k.key] ?? KIND_COLORS.other }} />{t(k.label)}
            </button>
          ))}
        </div>
      </div>

      <Timeline camps={camps} onPick={pick} />

      <div className="camp-list">
        {camps.map(c => {
          const on = open === key(c)
          return (
            <div key={key(c)} id={'camp-' + key(c)} className={'camp-item' + (on ? ' on' : '')}>
              <button className="camp-head" onClick={() => setOpen(on ? null : key(c))}>
                <span className="camp-code"><b>{c.code}</b><small>{c.water_body}</small></span>
                <span className="camp-date">{fmtDate(c.start)}{c.end && c.end !== c.start ? ` → ${fmtDate(c.end)}` : ''}</span>
                <KindDots kinds={c.kinds} />
                <span className="camp-n">{c.n_visits} {t('visitas')}<small>{fmt(c.n_obs, 0)} {t('medidas')}</small></span>
                <span className="camp-chev">{on ? '▾' : '▸'}</span>
              </button>
              {on && (
                <table className="tbl camp-visits">
                  <thead><tr><th>{t('Punto')}</th><th>{t('Fecha')}</th><th>{t('Hora')}</th><th>{t('Qué se tomó')}</th><th>{t('Parámetros')}</th><th /></tr></thead>
                  <tbody>
                    {c.visits.map(v => (
                      <tr key={v.extraction_point_id}>
                        <td><b>{v.code}</b></td>
                        <td>{fmtDate(v.date)}</td>
                        <td>{v.time ?? '—'}</td>
                        <td className="kinds-cell">
                          {v.kinds.length ? v.kinds.map(k => (
                            <span key={k} className="ktag" style={{ borderColor: KIND_COLORS[k] ?? KIND_COLORS.other, color: KIND_COLORS[k] ?? KIND_COLORS.other }}>
                              {t(data.kinds.find(x => x.key === k)?.label ?? k)}
                            </span>
                          )) : <span className="muted">—</span>}
                        </td>
                        <td>{v.n_params ? `${v.n_params} · ${fmt(v.n_obs, 0)} ${t('medidas')}` : <span className="muted">{v.kinds.length ? t('ver en su vista') : t('sin datos')}</span>}</td>
                        <td>{v.n_obs > 0 && <button className="link" onClick={() => onOpenSite(v.site, c.water_body)}>{t('Ver datos')} →</button>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )
        })}
        {!camps.length && <p className="muted small">{t('No hay campañas con este filtro.')}</p>}
      </div>
    </div>
  )
}
