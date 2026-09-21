import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Cell, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { colorAt, DbDepth, DbParam, DbSeries, DbSite, DbStatus, fmt, POI_COLORS } from './api'
import { locale, t } from './i18n'

const fmtDate = (s: string) => new Date(s + 'T12:00:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: 'numeric' })

/** Controles de la pestaña, en el panel lateral. */
export function DataForm(p: {
  status: DbStatus | null
  bodies: { name: string; n_sites: number }[]
  body: string; setBody: (b: string) => void
  params: DbParam[]
  param: string; setParam: (c: string) => void
  depth: DbDepth; setDepth: (d: DbDepth) => void
  sites: DbSite[]; selSites: string[]; toggleSite: (s: string) => void; clearSites: () => void
  exportUrl: string
}) {
  const groups = useMemo(() => Array.from(new Set(p.params.map(x => x.group))), [p.params])
  const s = p.status

  return (
    <>
      {s && !s.ok && <div className="badge err">{t('No se puede leer la base de datos del proyecto')}: {s.detail}</div>}
      {s?.mode === 'demo' && <div className="badge warn">{t('Base de datos no conectada: se muestran datos simulados con la misma estructura.')}</div>}
      {s?.ok && s.mode === 'db' && (
        <p className="muted small">
          {t('{n} observaciones en {s} puntos, de {a} a {b}.', { n: fmt(s.n_obs, 0), s: s.n_sites ?? 0, a: s.first ? fmtDate(s.first) : '—', b: s.last ? fmtDate(s.last) : '—' })}
        </p>
      )}

      <section>
        <label className="lbl">{t('Masa de agua')}</label>
        <select value={p.body} onChange={e => p.setBody(e.target.value)}>
          <option value="">{t('Todas')}</option>
          {p.bodies.map(b => <option key={b.name} value={b.name}>{b.name} · {b.n_sites} {t('puntos')}</option>)}
        </select>
      </section>

      <section>
        <label className="lbl">{t('Parámetro')}</label>
        <select value={p.param} onChange={e => p.setParam(e.target.value)} disabled={!p.params.length}>
          {!p.params.length && <option>{t('Sin datos')}</option>}
          {groups.map(g => (
            <optgroup key={g} label={t(g)}>
              {p.params.filter(x => x.group === g).map(x => (
                <option key={x.parameter_code} value={x.parameter_code}>
                  {x.name ?? x.parameter_code}{x.unit_name ? ` (${x.unit_name})` : ''} · {t('{n} datos', { n: x.n_here })}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </section>

      <section>
        <label className="lbl" title={t('Muchas sondas miden a varias profundidades; así se comparan muestreos equivalentes.')}>{t('Profundidad')}</label>
        <div className="seg dark depth-seg">
          <button className={p.depth === 'surface' ? 'on' : ''} onClick={() => p.setDepth('surface')}>{t('Superficie')}</button>
          <button className={p.depth === 'bottom' ? 'on' : ''} onClick={() => p.setDepth('bottom')}>{t('Fondo')}</button>
          <button className={p.depth === 'all' ? 'on' : ''} onClick={() => p.setDepth('all')}>{t('Toda la columna')}</button>
        </div>
        <small className="muted">{p.depth === 'surface' ? t('Medidas a 1 m o menos.')
          : p.depth === 'bottom' ? t('La medida más profunda de cada muestreo.')
          : t('Todas las profundidades, sin promediar: perfiles y diagrama profundidad × tiempo.')}</small>
      </section>

      {p.sites.length > 0 && (
        <section className="pois">
          <label className="lbl">{t('Puntos de muestreo')} <b>{p.selSites.length ? `${p.selSites.length}/${p.sites.length}` : p.sites.length}</b></label>
          <div className="poi-list">
            {p.sites.map((x, i) => {
              const on = !p.selSites.length || p.selSites.includes(x.site)
              return (
                <button key={x.site} className={'poi-chip db-chip' + (on ? '' : ' off')} onClick={() => p.toggleSite(x.site)}
                  title={`${x.water_body} · ${x.n_obs} ${t('observaciones')}`}>
                  <i style={{ background: POI_COLORS[i % POI_COLORS.length] }} />{String(x.code)}
                </button>
              )
            })}
          </div>
          {p.selSites.length > 0 && <button className="link" onClick={p.clearSites}>{t('Ver todos los puntos')}</button>}
        </section>
      )}

      <a className="ghost solid wide dl-link" href={p.exportUrl} download>
        ⬇ {p.param ? t('Descargar este parámetro (CSV)') : t('Descargar datos (CSV)')}
      </a>
    </>
  )
}

const RAMP = ['#2c7bb6', '#00a6ca', '#00ccbc', '#90eb9d', '#ffff8c', '#f9d057', '#f29e2e', '#e76818', '#d7191c']

/** Vistas sin promediar: perfil de un muestreo y diagrama profundidad × tiempo. */
function WaterColumn({ data, color }: { data: DbSeries; color: (c: string) => string }) {
  const rows = useMemo(() => data.rows.filter(r => r.depth_m != null), [data])
  const dates = useMemo(() => Array.from(new Set(rows.map(r => r.date))).sort(), [rows])
  const [view, setView] = useState<'profile' | 'section'>('profile')
  const [date, setDate] = useState<string>('')
  const codes = useMemo(() => Array.from(new Set(rows.map(r => r.site_code))), [rows])
  const [site, setSite] = useState<string>('')
  useEffect(() => { if (!codes.includes(site)) setSite(codes[0] ?? '') }, [codes]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!dates.includes(date)) setDate(dates[dates.length - 1] ?? '') }, [dates]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!rows.length) return <p className="muted an-empty">{t('Este parámetro no tiene medidas a distintas profundidades.')}</p>

  const u = data.unit ? ` (${data.unit})` : ''
  const maxDepth = Math.max(...rows.map(r => r.depth_m as number))
  const vmin = Math.min(...rows.map(r => r.value)), vmax = Math.max(...rows.map(r => r.value))
  const col = (v: number) => colorAt(RAMP, vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5)

  // Perfil: una serie por punto, ordenada por profundidad
  const onDate = rows.filter(r => r.date === date)
  const bySite = Array.from(new Set(onDate.map(r => r.site_code))).map(c => ({
    code: c, pts: onDate.filter(r => r.site_code === c).sort((a, b) => (a.depth_m! - b.depth_m!))
      .map(r => ({ value: r.value, depth: r.depth_m, qc: r.qc_flag })),
  }))
  // Profundidad × tiempo: un punto de muestreo cada vez (mezclarlos confunde)
  const section = rows.filter(r => r.site_code === site)
    .map(r => ({ t: new Date(r.date + 'T12:00:00').getTime(), depth: r.depth_m, value: r.value }))
  const t0 = Math.min(...section.map(p => p.t)), t1 = Math.max(...section.map(p => p.t))
  const ticks = t1 > t0 ? Array.from({ length: 6 }, (_, i) => t0 + (t1 - t0) * i / 5) : [t0]

  return (
    <div className="wc">
      <div className="wc-bar">
        <div className="seg">
          <button className={view === 'profile' ? 'on' : ''} onClick={() => setView('profile')}>{t('Perfil')}</button>
          <button className={view === 'section' ? 'on' : ''} onClick={() => setView('section')}>{t('Profundidad × tiempo')}</button>
        </div>
        {view === 'profile' ? (
          <select value={date} onChange={e => setDate(e.target.value)} className="wc-date">
            {dates.slice().reverse().map(d => <option key={d} value={d}>{fmtDate(d)}</option>)}
          </select>
        ) : (
          <select value={site} onChange={e => setSite(e.target.value)} className="wc-date">
            {codes.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        )}
      </div>

      {view === 'profile' ? (
        <div style={{ height: 320 }}>
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="#E4EBEA" />
              <XAxis type="number" dataKey="value" name={data.name} domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#5E7376' }}
                label={{ value: data.name + u, position: 'insideBottom', offset: -8, style: { fontSize: 11, fill: '#5E7376' } }} />
              <YAxis type="number" dataKey="depth" name={t('Profundidad')} reversed domain={[0, Math.ceil(maxDepth)]} width={46}
                tick={{ fontSize: 11, fill: '#5E7376' }} label={{ value: t('Profundidad (m)'), angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#5E7376' } }} />
              <Tooltip formatter={(v: any, n: any) => [fmt(v as number, 2), n]} />
              <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 11 }} />
              {bySite.map(s => (
                <Scatter key={s.code} name={s.code} data={s.pts} fill={color(s.code)} line={{ stroke: color(s.code), strokeWidth: 1.8 }} isAnimationActive={false} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <>
          <div style={{ height: 320 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
                <CartesianGrid stroke="#E4EBEA" />
                <XAxis type="number" dataKey="t" scale="time" domain={[t0, t1]} ticks={ticks} name={t('Fecha')}
                  tickFormatter={v => new Date(v).toLocaleDateString(locale(), { day: 'numeric', month: 'short', year: '2-digit' })} tick={{ fontSize: 11, fill: '#5E7376' }} />
                <YAxis type="number" dataKey="depth" reversed domain={[0, Math.ceil(maxDepth)]} width={46} name={t('Profundidad')}
                  tick={{ fontSize: 11, fill: '#5E7376' }} label={{ value: t('Profundidad (m)'), angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#5E7376' } }} />
                <ZAxis type="number" dataKey="value" range={[60, 60]} name={data.name} />
                <Tooltip formatter={(v: any, n: any) => [n === t('Fecha') ? fmtDate(new Date(v).toISOString().slice(0, 10)) : fmt(v as number, 2), n]} />
                <Scatter data={section} shape="square" isAnimationActive={false}>
                  {section.map((p, i) => <Cell key={i} fill={col(p.value)} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="wc-legend">
            <span>{fmt(vmin, 2)}</span><i style={{ background: `linear-gradient(90deg, ${RAMP.join(',')})` }} /><span>{fmt(vmax, 2)}{data.unit ? ` ${data.unit}` : ''}</span>
          </div>
        </>
      )}
    </div>
  )
}

/** Tarjeta principal: gráfico por punto con el control de calidad marcado y resumen. */
export function DataResult({ data, sites, onClose }: { data: DbSeries; sites: DbSite[]; onClose: () => void }) {
  const codes = useMemo(() => Array.from(new Set(data.rows.map(r => r.site_code))), [data])
  const color = (code: string) => {
    const i = sites.findIndex(s => s.code === code)
    return POI_COLORS[(i < 0 ? codes.indexOf(code) : i) % POI_COLORS.length]
  }
  // El flag más bajo de qc_flags es el "bueno"; el resto se marca en el gráfico
  const goodFlag = data.qc.length ? Math.min(...data.qc.map(q => q.qc_flag)) : null
  const qcLabel = (f: number | null) => data.qc.find(q => q.qc_flag === f)?.label ?? String(f)
  const flagged = data.rows.filter(r => r.qc_flag != null && goodFlag != null && r.qc_flag !== goodFlag)

  // Una fila por fecha, una columna por punto (media si hay varias profundidades)
  const chart = useMemo(() => {
    const m = new Map<string, any>()
    for (const r of data.rows) {
      const k = r.date
      const row = m.get(k) ?? { date: k, t: new Date(k + 'T12:00:00').getTime() }
      const acc = row[`_${r.site_code}`] ?? []
      acc.push(r)
      row[`_${r.site_code}`] = acc
      m.set(k, row)
    }
    return Array.from(m.values()).sort((a, b) => a.t - b.t).map(row => {
      for (const c of codes) {
        const rs: any[] = row[`_${c}`] ?? []
        if (rs.length) {
          row[c] = rs.reduce((s, r) => s + r.value, 0) / rs.length
          row[`${c}__qc`] = rs.some(r => goodFlag != null && r.qc_flag != null && r.qc_flag !== goodFlag)
            ? rs.find(r => r.qc_flag !== goodFlag).qc_flag : null
        }
      }
      return row
    })
  }, [data, codes, goodFlag])

  const Dot = (code: string) => (props: any) => {
    const { cx, cy, payload } = props
    if (cx == null || cy == null || payload[code] == null) return <g />
    const bad = payload[`${code}__qc`]
    return bad != null
      ? <g><circle cx={cx} cy={cy} r={6} fill="#fff" stroke="#C0392B" strokeWidth={2} /><circle cx={cx} cy={cy} r={2.5} fill={color(code)} /></g>
      : <circle cx={cx} cy={cy} r={3.5} fill={color(code)} />
  }

  return (
    <div className="card calres dbres">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div>
        <p className="eyebrow">{t(data.group ?? 'Datos del proyecto')}</p>
        <h2 className="serif">{data.name}{data.unit ? <span className="unit"> · {data.unit}</span> : null}</h2>
        <p className="muted small">
          {t('{n} medidas en {m} muestreos · {s} puntos', { n: data.rows.length, m: data.n_dates, s: codes.length })}
          {flagged.length > 0 && <> · <span className="qc-note">{t('{n} marcadas por control de calidad', { n: flagged.length })}</span></>}
        </p>
      </div>

      {data.rows.length === 0 ? (
        <p className="muted an-empty">{t('No hay datos de este parámetro con estos filtros.')}</p>
      ) : data.depth === 'all' && data.rows.some(r => r.depth_m != null) ? (
        <WaterColumn data={data} color={color} />
      ) : (
        <div style={{ height: 300 }}>
          <ResponsiveContainer>
            <ComposedChart data={chart} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="#E4EBEA" vertical={false} />
              <XAxis dataKey="t" type="number" scale="time" domain={['dataMin', 'dataMax']}
                tickFormatter={v => new Date(v).toLocaleDateString(locale(), { month: 'short', year: '2-digit' })}
                tick={{ fontSize: 11, fill: '#5E7376' }} />
              <YAxis tick={{ fontSize: 11, fill: '#5E7376' }} width={52}
                label={data.unit ? { value: data.unit, angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#5E7376' } } : undefined} />
              <Tooltip labelFormatter={(v: any) => fmtDate(new Date(v).toISOString().slice(0, 10))}
                formatter={(v: any, n: any, it: any) => {
                  const f = it?.payload?.[`${n}__qc`]
                  const k = (it?.payload?.[`_${n}`] ?? []).length
                  const avg = k > 1 ? `  (${t('media de {k} profundidades', { k })})` : ''
                  return [`${fmt(v as number, 3)}${avg}${f != null ? `  ⚠ ${qcLabel(f)}` : ''}`, n]
                }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {codes.map(c => (
                <Line key={c} dataKey={c} name={c} stroke={color(c)} strokeWidth={1.8} connectNulls
                  dot={Dot(c)} activeDot={{ r: 5 }} isAnimationActive={false} />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {flagged.length > 0 && (
        <p className="muted small">
          <span className="qc-ring" /> {t('Los puntos con aro rojo están marcados por control de calidad:')}{' '}
          {Array.from(new Set(flagged.map(r => r.qc_flag))).map(f => `${qcLabel(f)} (${flagged.filter(r => r.qc_flag === f).length})`).join(' · ')}
        </p>
      )}

      {data.summary.length > 0 && (
        <div className="tbl-wrap" style={{ maxHeight: 220 }}>
          <table className="tbl">
            <thead><tr>
              <th>{t('Punto')}</th><th>{t('Muestreos')}</th><th>{t('Medidas')}</th><th>{t('Mín.')}</th><th>{t('Mediana')}</th><th>{t('Máx.')}</th><th>{t('Último')}</th>
            </tr></thead>
            <tbody>
              {data.summary.map(s => (
                <tr key={s.site_code}>
                  <td><span className="dot" style={{ background: color(s.site_code) }} />{s.site_code}</td>
                  <td>{s.n_dates}</td><td>{s.n}</td><td>{fmt(s.min, 2)}</td><td>{fmt(s.median, 2)}</td><td>{fmt(s.max, 2)}</td>
                  <td>{fmt(s.last_value, 2)} <span className="muted small">· {fmtDate(s.last_date)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
