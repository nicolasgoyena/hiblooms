export type IndexMeta = { id: string; label: string; unit: string; min: number; max: number; group: string }
export type DateHit = { date: string; cloud: number | null }
export type ImageResult = {
  mode: 'gee' | 'demo'
  tile_url: string | null
  rgb_tile_url: string | null
  datetime: string | null
  cloud: number | null
  coverage: number | null
  mean: number | null
  points?: Record<string, number | null>
}
export type Poi = { name: string; lat: number; lon: number; custom?: boolean }
export type SeriesPoint = { date: string; mean: number | null; [poi: string]: number | string | null }
export type ClassRow = { low: number; high: number | null; area_ha: number; pct: number }

const BASE = (import.meta as any).env?.VITE_API_URL ?? ''

async function req<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(BASE + path, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    let msg = `${r.status}`
    try { msg = (await r.json()).detail ?? msg } catch { /* noop */ }
    throw new Error(msg)
  }
  return r.json()
}

export const api = {
  health: () => req<{ ok: boolean; mode: 'gee' | 'demo'; gee_error: string | null }>('/api/health'),
  indices: () => req<{ indices: IndexMeta[]; palette: string[] }>('/api/indices'),
  reservoirs: () => req<GeoJSON.FeatureCollection>('/api/reservoirs'),
  search: (b: { reservoir: string; start: string; end: string; max_cloud: number }) =>
    req<{ dates: DateHit[]; mode: string }>('/api/search', b),
  pois: (reservoir: string) => req<{ points: Poi[] }>(`/api/pois?reservoir=${encodeURIComponent(reservoir)}`),
  image: (b: { reservoir: string; date: string; index: string; max_cloud: number; points: Poi[] }) =>
    req<ImageResult>('/api/image', b),
  timeseries: (b: { reservoir: string; index: string; dates: string[]; max_cloud: number; points: Poi[] }) =>
    req<{ series: SeriesPoint[] }>('/api/timeseries', b),
  classes: (b: { reservoir: string; date: string; index: string; max_cloud: number; n_classes?: number }) =>
    req<{ classes: ClassRow[] }>('/api/classes', b),
  download: (b: { reservoir: string; date: string; indices: string[]; max_cloud: number }) =>
    req<{ url: string; filename: string }>('/api/download', b),
}

export function colorAt(palette: string[], t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (palette.length - 1)
  const i = Math.floor(x), f = x - i
  if (i >= palette.length - 1) return palette[palette.length - 1]
  const a = hex(palette[i]), b = hex(palette[i + 1])
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}
function hex(h: string) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255] }

export const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toLocaleString('es-ES', { maximumFractionDigits: d })

export const niceName = (n: string) => {
  const m = n.match(/^(.*), (LA|EL|LAS|LOS)$/)
  const s = m ? `${m[2]} ${m[1]}` : n
  return s.toLowerCase().replace(/(^|\s|\()\S/g, c => c.toUpperCase())
}

export const POI_COLORS = ['#F29A5B', '#7FB8E0', '#C7E07A', '#E58FC7', '#FFD166', '#A78BFA', '#FF8A80', '#80CBC4']

/** CSV → puntos. Acepta cabeceras nombre/name, lat/latitud, lon/longitud (coma o punto y coma). */
export function parsePoisCsv(text: string): Poi[] {
  const lines = text.replace(/\r/g, '').split('\n').filter(l => l.trim())
  if (lines.length < 2) return []
  const sep = lines[0].includes(';') ? ';' : ','
  const head = lines[0].split(sep).map(h => h.trim().toLowerCase())
  const iN = head.findIndex(h => ['nombre', 'name', 'punto', 'id'].includes(h))
  const iLa = head.findIndex(h => ['lat', 'latitud', 'latitude', 'y'].includes(h))
  const iLo = head.findIndex(h => ['lon', 'lng', 'longitud', 'longitude', 'x'].includes(h))
  if (iLa < 0 || iLo < 0) throw new Error('El CSV necesita columnas lat/latitud y lon/longitud')
  return lines.slice(1).map((l, k) => {
    const c = l.split(sep).map(v => v.trim())
    return { name: iN >= 0 && c[iN] ? c[iN] : `P${k + 1}`, lat: parseFloat(c[iLa].replace(',', '.')), lon: parseFloat(c[iLo].replace(',', '.')), custom: true }
  }).filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon))
}

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return ''
  const cols = Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  const esc = (v: unknown) => v === null || v === undefined ? '' : /[",;\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v)
  return [cols.join(','), ...rows.map(r => cols.map(c => esc(r[c])).join(','))].join('\n')
}

export function downloadText(name: string, text: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
