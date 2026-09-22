import { useEffect, useState } from 'react'
import { api, fmt, MonitorRow } from './api'
import { t } from './i18n'

export const LEVELS: Record<string, { color: string; bg: string }> = {
  'muy alto': { color: '#8E1B15', bg: '#F7D9D6' },
  'alto': { color: '#A8540C', bg: '#FBE6CE' },
  'moderado': { color: '#8A6D0B', bg: '#FBF0C7' },
  'bajo': { color: '#1C6B4B', bg: '#DCF0E4' },
  'muy bajo': { color: '#1D6FA8', bg: '#E0EDF8' },
  'sin datos': { color: '#5E7376', bg: '#EBEFEF' },
}
const ORDER = ['muy alto', 'alto', 'moderado', 'bajo', 'muy bajo', 'sin datos']

/** Controles de la pestaña, dentro del panel lateral. */
export function MonitorForm({ days, setDays, maxCloud, setMaxCloud, onRun, running }: {
  days: number; setDays: (n: number) => void; maxCloud: number; setMaxCloud: (n: number) => void
  onRun: () => void; running: boolean
}) {
  return (
    <>
      <section>
        <label className="lbl">{t('Ventana de tiempo')}</label>
        <select value={days} onChange={e => setDays(+e.target.value)}>
          <option value={15}>{t('Últimos 15 días')}</option>
          <option value={30}>{t('Últimos 30 días')}</option>
          <option value={60}>{t('Últimos 60 días')}</option>
        </select>
      </section>
      <section>
        <label className="lbl">{t('Nubes máximas')} <b>{maxCloud}%</b></label>
        <input type="range" min={10} max={100} step={5} value={maxCloud} onChange={e => setMaxCloud(+e.target.value)} />
      </section>
      <button className="primary" onClick={onRun} disabled={running}>
        {running ? <><span className="spin" /> {t('Revisando embalses…')}</> : t('Actualizar estado')}
      </button>
      <p className="muted small">
        {t('Estado orientativo a partir del NDCI, un índice espectral de biomasa algal (clorofila-a). Sirve para priorizar qué embalse mirar, no para dar concentraciones ni distinguir cianobacterias de otras algas.')}
      </p>
    </>
  )
}

/** Tarjeta grande: resumen + lista de embalses ordenada por señal. */
export function MonitorResult({ data, onOpen, onClose }: {
  data: { rows: MonitorRow[]; start: string; end: string; days: number }
  onOpen: (id: string, day: string | null) => void
  onClose: () => void
}) {
  const [filter, setFilter] = useState<string | null>(null)
  const [onlyCombined, setOnlyCombined] = useState(false)
  const nCombined = data.rows.filter(r => r.combined).length
  const counts = ORDER.map(l => [l, data.rows.filter(r => r.level === l).length] as const).filter(([, n]) => n > 0)
  let rows = filter ? data.rows.filter(r => r.level === filter) : data.rows
  if (onlyCombined) rows = rows.filter(r => r.combined)
  const withData = data.rows.filter(r => r.ndci_p90 !== null).length

  return (
    <div className="card calres mon">
      <button className="x" onClick={onClose} aria-label={t('Cerrar')}>×</button>
      <div>
        <p className="eyebrow">{t('Estado general · últimos {n} días', { n: data.days })}</p>
        <h2 className="serif">{t('{n} embalses revisados', { n: data.rows.length })}</h2>
        <p className="muted small">
          {t('{n} con observación despejada en la ventana', { n: withData })} · {data.start} → {data.end}
        </p>
      </div>

      <div className="mon-sum">
        {counts.map(([l, n]) => (
          <button key={l} className={'mon-pill' + (filter === l ? ' on' : '')}
            style={{ background: LEVELS[l].bg, color: LEVELS[l].color }}
            onClick={() => setFilter(filter === l ? null : l)}>
            <b>{n}</b> {t(l)}
          </button>
        ))}
        {nCombined > 0 && (
          <button className={'mon-pill mon-warn' + (onlyCombined ? ' on' : '')} onClick={() => setOnlyCombined(v => !v)}
            title={t('Señal alta de algas y lámina por debajo del 60 % de la habitual')}>
            <b>{nCombined}</b> ⚠ {t('señal alta + embalse bajo')}
          </button>
        )}
      </div>

      <div className="mon-grid">
        {rows.map(r => (
          <button key={r.id} className="mon-row" onClick={() => onOpen(r.id, r.date)}
            title={t('Abrir en el visor')}>
            <span className="mon-bar" style={{ background: LEVELS[r.level].color }} />
            <span className="mon-name">
              <b>{r.combined ? '⚠ ' : ''}{r.label}</b>
              <small>{r.date ? r.date : t('sin imagen despejada')}{r.water_ha !== null ? ` · ${fmt(r.water_ha, 0)} ha` : ''}</small>
              <Fill row={r} />
            </span>
            <span className="mon-vals">
              <em>NDCI</em><b>{fmt(r.ndci_p90, 2)} <Arrow v={r.ndci_trend} thr={0.03} up="bad" fmtv={v => fmt(v, 2)} /></b>
            </span>
            <span className="mon-tags">
              <span className="mon-lvl" style={{ background: LEVELS[r.level].bg, color: LEVELS[r.level].color }}>{t(r.level)}</span>
              {r.cyano && <span className="mon-cy" title={t('Biomasa algal alta (NDCI ≥ 0,15): posible floración. Confirmar con muestreo o sonda.')}>🌿 {t('biomasa alta')}</span>}
            </span>
          </button>
        ))}
      </div>

      <p className="muted small">
        {t('Nivel del embalse: donde existe, el volumen embalsado oficial del Boletín Hidrológico Semanal (MITECO, embalses de más de 5 hm³, dato semanal); en el resto, la superficie de agua vista por el satélite frente a su lámina habitual (JRC Global Surface Water), con la flecha de los {n} días anteriores. El aviso ⚠ marca los embalses con señal alta de algas y nivel por debajo del 60 %, donde la concentración de nutrientes y el calentamiento agravan el episodio.', { n: data.days })}
      </p>
      <p className="muted small">
        {t('Las flechas comparan con el periodo anterior de la misma duración: en el NDCI, ↑ en rojo significa que la señal va a más; en el nivel, ↓ en rojo significa que el embalse sigue bajando.')}
      </p>
      <p className="muted small">
        {t('NDCI es el percentil 90 del píxel despejado más reciente de cada embalse dentro de la ventana, sobre la lámina de agua (NDWI > 0). Un valor alto indica mucha biomasa algal; el aviso 🌿 aparece con NDCI ≥ 0,15. El satélite no distingue cianobacterias de otras algas: confirma con muestreo o sonda.')}
      </p>
    </div>
  )
}

/** Flecha de tendencia. `up`='bad' cuando subir es mala señal (más algas). */
function Arrow({ v, thr, up, fmtv }: { v: number | null; thr: number; up?: 'bad' | 'good'; fmtv?: (n: number) => string }) {
  if (v === null || Math.abs(v) < thr) return <span className="mon-arw flat" title={t('estable')}>→</span>
  const rising = v > 0
  const bad = up === 'bad' ? rising : !rising
  const txt = fmtv ? fmtv(Math.abs(v)) : fmt(Math.abs(v), 0)
  return (
    <span className={'mon-arw ' + (bad ? 'bad' : 'good')}
      title={(rising ? t('subiendo') : t('bajando')) + ' ' + txt + ' ' + t('respecto al periodo anterior')}>
      {rising ? '↑' : '↓'}
    </span>
  )
}

/** Barra de lámina de agua respecto a la habitual, con su tendencia. */
function Fill({ row }: { row: MonitorRow }) {
  const pct = row.nivel_pct
  if (pct === null) return <small className="mon-fill-na">{t('nivel sin estimar')}</small>
  const col = pct < 40 ? '#A8540C' : pct < 60 ? '#8A6D0B' : '#1D6FA8'
  const o = row.oficial
  const label = o
    ? `${fmt(pct, 0)}% ${t('de su capacidad')} · ${fmt(o.hm3, 0)}/${fmt(o.cap_hm3, 0)} hm³`
    : `${fmt(pct, 0)}% ${t('de su lámina habitual')}`
  return (
    <span className="mon-fill" title={o
      ? `${t('Volumen embalsado oficial · Boletín Hidrológico Semanal (MITECO)')} · ${o.fecha}`
      : t('Superficie de agua frente a la lámina habitual (JRC, agua presente más de la mitad del tiempo)')}>
      <span className="mon-fill-track"><i style={{ width: `${Math.min(100, pct)}%`, background: col }} /></span>
      <small style={{ color: col }}>
        {label} <Arrow v={row.nivel_trend} thr={o ? 2 : 8} up="good" fmtv={v => (o ? `${fmt(v, 0)} pts` : `${fmt(v, 0)}%`)} />
        {' '}<span className="mon-src">{o ? t('oficial') : t('satélite')}</span>
      </small>
    </span>
  )
}

/** Carga inicial automática al entrar en la pestaña. */
export function useAutoRun(run: () => void, ready: boolean) {
  useEffect(() => { if (ready) run() }, [ready]) // eslint-disable-line react-hooks/exhaustive-deps
}
