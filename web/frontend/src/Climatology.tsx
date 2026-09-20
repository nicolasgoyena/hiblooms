import { useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ClimResp, fmt } from './api'
import { t } from './i18n'

const MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const etiqueta = (q: number) => `${t(MESES[Math.floor((q - 1) / 2)])} ${q % 2 === 1 ? 'I' : 'II'}`

const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b)
  const i = (s.length - 1) * p
  const lo = Math.floor(i), hi = Math.ceil(i)
  return s[lo] + (s[hi] - s[lo]) * (i - lo)
}

/** Rango habitual de cada quincena (años anteriores) frente al año en curso. */
export default function Climatology({ data }: { data: ClimResp }) {
  const year = new Date().getFullYear()
  const rows = useMemo(() => Array.from({ length: 24 }, (_, k) => {
    const q = k + 1
    const hist = data.points.filter(p => p.q === q && p.year < year).map(p => p.value)
    const now = data.points.filter(p => p.q === q && p.year === year).map(p => p.value)
    const p25 = hist.length >= 4 ? +pct(hist, 0.25).toFixed(3) : null
    const p75 = hist.length >= 4 ? +pct(hist, 0.75).toFixed(3) : null
    return {
      q, etq: etiqueta(q),
      p25, p75,
      banda: p25 !== null && p75 !== null ? +(p75 - p25).toFixed(3) : null,
      mediana: hist.length >= 3 ? +pct(hist, 0.5).toFixed(3) : null,
      actual: now.length ? +pct(now, 0.5).toFixed(3) : null,
      nHist: hist.length,
    }
  }), [data, year])

  const alto = rows.filter(r => r.actual !== null && r.p75 !== null && r.actual > r.p75).length
  const bajo = rows.filter(r => r.actual !== null && r.p25 !== null && r.actual < r.p25).length
  const conDato = rows.filter(r => r.actual !== null && r.p75 !== null).length
  const nImgs = data.points.length

  return (
    <div className="clima">
      <p className="muted small">
        {t('Rango habitual de {a}–{b} (mitad central de los años anteriores) frente a {y}, por quincenas.', {
          a: data.years[0], b: year - 1, y: year })}{' '}
        {conDato > 0 && (alto > bajo
          ? t('Este año va por encima de lo normal en {n} de {m} quincenas comparables.', { n: alto, m: conDato })
          : bajo > alto
            ? t('Este año va por debajo de lo normal en {n} de {m} quincenas comparables.', { n: bajo, m: conDato })
            : t('Este año se mueve dentro de lo normal.'))}
      </p>
      <div style={{ height: 250 }}>
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#E4EBEA" vertical={false} />
            <XAxis dataKey="etq" tick={{ fontSize: 10, fill: '#5E7376' }} interval={1} />
            <YAxis tick={{ fontSize: 11, fill: '#5E7376' }} width={48}
              label={{ value: data.label, angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#5E7376' } }} />
            <Tooltip formatter={(v: any, n: any) => [fmt(v as number, 3), n]}
              labelFormatter={(l: any, pl: any) => {
                const r = pl?.[0]?.payload
                return r && r.p25 !== null
                  ? `${l} · ${t('habitual')} ${fmt(r.p25, 2)}–${fmt(r.p75, 2)} (${r.nHist} ${t('imágenes')})`
                  : String(l)
              }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Area dataKey="p25" stackId="b" stroke="none" fill="none" legendType="none" tooltipType="none" name="" />
            <Area dataKey="banda" stackId="b" stroke="none" fill="#1D6FA8" fillOpacity={0.14} tooltipType="none" name={t('rango habitual')} />
            <Line dataKey="mediana" stroke="#1D6FA8" strokeWidth={2} dot={false} connectNulls name={t('mediana de años anteriores')} />
            <Line dataKey="actual" stroke="#C0392B" strokeWidth={2.5} connectNulls
              dot={{ r: 3, fill: '#C0392B' }} name={String(year)} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="muted small">
        {t('Percentil 90 del índice sobre la lámina de agua de cada imagen (el mismo criterio que el monitor), con {n} imágenes válidas desde {a}. Las quincenas con menos de 4 imágenes en años anteriores se quedan sin rango.', { n: nImgs, a: data.years[0] })}
      </p>
    </div>
  )
}
