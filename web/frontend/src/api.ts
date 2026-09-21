import { locale, t } from './i18n'

export type IndexMeta = { id: string; label: string; unit: string; min: number; max: number; group: string; reservoir?: string | null }
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
  interval80?: [number, number]
  extrapolation_pct?: number
}
export type MonitorRow = {
  id: string; label: string; ndci_mean: number | null; ndci_p90: number | null; pci_p90: number | null
  date: string | null; coverage: number; level: string; cyano: boolean
  water_ha: number | null; fill_pct: number | null; trend_pct: number | null; combined: boolean
  nivel_pct: number | null; nivel_fuente: 'boletin' | 'satelite' | null
  ndci_trend: number | null; pci_trend: number | null; nivel_trend: number | null
  oficial: { pct: number; hm3: number; cap_hm3: number; fecha: string; fuente: string } | null
}
export type MonitorResp = { mode: string; start: string; end: string; days: number; rows: MonitorRow[] }
export type ClimPoint = { date: string; q: number; year: number; value: number }
export type ClimResp = { mode: string; index: string; unit: string; label: string; years: [number, number]; points: ClimPoint[] }
export type Poi = { name: string; lat: number; lon: number; custom?: boolean }
export type SeriesPoint = { date: string; mean: number | null; [poi: string]: number | string | null }
export type ClassRow = { low: number; high: number | null; area_ha: number; pct: number }

const BASE = (import.meta as any).env?.VITE_API_URL ?? ''

// ── Sesión (login) ──
const TOKEN_KEY = 'hiblooms-token'
let token: string | null = (() => { try { return localStorage.getItem(TOKEN_KEY) } catch { return null } })()
export const getToken = () => token
export function setToken(t: string | null) {
  token = t
  try { if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY) } catch { /* noop */ }
}
/** Se dispara cuando el backend responde 401 (sesión caducada) */
export const onUnauthorized = { cb: () => {} }

async function req<T>(path: string, body?: unknown, method?: string): Promise<T> {
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  const r = await fetch(BASE + path, {
    method: method ?? (body === undefined ? 'GET' : 'POST'),
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (r.status === 401 && path !== '/api/login') onUnauthorized.cb()
  if (!r.ok) {
    let msg = `${r.status}`
    try { msg = (await r.json()).detail ?? msg } catch { /* noop */ }
    throw new Error(msg)
  }
  return r.json()
}

export type Prog = (p: { progress: number; step: string; seconds: number }) => void
export type JobState = { status: 'running' | 'done' | 'error'; progress: number; step: string; seconds: number; detail?: string; result?: any }

/** Lanza una tarea larga y sondea hasta que termina, informando del progreso. */
async function job<T>(path: string, body: unknown, onProgress?: (p: { progress: number; step: string; seconds: number }) => void): Promise<T> {
  const { job_id } = await req<{ job_id: string }>(path, body)
  for (;;) {
    await new Promise(r => setTimeout(r, 1500))
    const j = await req<JobState>(`/api/jobs/${job_id}`)
    if (j.status === 'done') return j.result as T
    if (j.status === 'error') throw new Error(j.detail || 'La tarea ha fallado')
    onProgress?.({ progress: j.progress, step: j.step, seconds: j.seconds })
  }
}

export const api = {
  health: () => req<{ ok: boolean; mode: 'gee' | 'demo'; gee_error: string | null; auth: boolean }>('/api/health'),
  login: (username: string, password: string) => req<{ token: string; user: string }>('/api/login', { username, password }),
  me: () => req<{ user: string | null; auth: boolean }>('/api/me'),
  deleteUpload: (id: string) => req<{ ok: boolean }>(`/api/uploads/${id}`, undefined, 'DELETE'),
  indices: () => req<{ indices: IndexMeta[]; palette: string[] }>('/api/indices'),
  reservoirs: () => req<GeoJSON.FeatureCollection>('/api/reservoirs'),
  search: (b: { reservoir: string; start: string; end: string; max_cloud: number }) =>
    req<{ dates: DateHit[]; mode: string }>('/api/search', b),
  uploadShapefile: async (file: File) => {
    const buf = new Uint8Array(await file.arrayBuffer())
    let bin = ''
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000))
    return req<{ upload_id: string; count: number; name_column: string | null; geojson: GeoJSON.FeatureCollection }>(
      '/api/upload-shapefile', { filename: file.name, content_b64: btoa(bin) })
  },
  monitor: (b: { days: number; max_cloud: number }, onProgress?: Prog) => job<MonitorResp>('/api/monitor', b, onProgress),
  climatology: (b: { reservoir: string; index: string; years?: number; max_cloud?: number; water_only?: boolean }, onProgress?: Prog) =>
    job<ClimResp>('/api/climatology', b, onProgress),
  calOptions: () => req<{ predictors: string[]; models: string[]; rasterizable: string[] }>('/api/calibration/options'),
  calPreview: (csv_text: string) => req<CsvPreview>('/api/calibration/preview', { csv_text }),
  calibrate: (b: CalibrateReq, onProgress?: Prog) => job<CalResult>('/api/calibrate', b, onProgress),
  calModelUrl: (id: string) => `${BASE}/api/calibration/${id.replace('cal:', '')}/model${token ? `?token=${encodeURIComponent(token)}` : ''}`,
  pois: (reservoir: string) => req<{ points: Poi[] }>(`/api/pois?reservoir=${encodeURIComponent(reservoir)}`),
  image: (b: { reservoir: string; date: string; index: string; max_cloud: number; points: Poi[]; water_only?: boolean }) =>
    req<ImageResult>('/api/image', b),
  timeseries: (b: { reservoir: string; index: string; dates: string[]; max_cloud: number; points: Poi[]; water_only?: boolean }) =>
    req<{ series: SeriesPoint[] }>('/api/timeseries', b),
  classes: (b: { reservoir: string; date: string; index: string; max_cloud: number; n_classes?: number; water_only?: boolean }) =>
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
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toLocaleString(locale(), { maximumFractionDigits: d })

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
  if (iLa < 0 || iLo < 0) throw new Error(t('El CSV necesita columnas lat/latitud y lon/longitud'))
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

export type CsvPreview = { columns: string[]; numeric: string[]; n_rows: number; guess: { date: string | null; time: string | null; site: string | null; lat: string | null; lon: string | null }; head: Record<string, unknown>[] }
export type CalibrateReq = {
  reservoir: string; csv_text: string; date_col: string; time_col: string | null; value_col: string; unit: string; tz: string
  point: { lat: number; lon: number; name?: string } | null
  site_col: string | null; lat_col: string | null; lon_col: string | null; pois: Poi[]
  max_cloud: number; min_water: number; max_hours: number; window_days: number
  auto: boolean; kind: 'phycocyanin' | 'chlorophyll' | 'other'
  predictors: string[]; models: string[]; transform: 'auto' | 'none' | 'log'; threshold: number | null
  criterion: 'balanced' | 'peaks' | 'general' | 'alerts'
}
export type CalMetrics = { r2: number | null; rmse: number | null; mae: number | null; r2_log: number | null; sens: number | null; far: number | null; tp?: number; fn?: number; fp?: number; tn?: number; n: number }
export type CalResult = {
  calibration_id: string | null; rasterizable: boolean; raster: Record<string, any> | null
  summary: { model: string; model_label: string; transform: string; predictors: string[]; n_pairs: number; k_blocks: number; threshold: number
    point: { lat: number; lon: number; name?: string } | null; sites: { site: string; lat: number | null; lon: number | null }[]; n_days: number; max_hours: number; window_days: number; selection_stability: number; n_candidates: number; auto: boolean
    criterion: string; criterion_label: string; smear: number; water_mask: string }
  honest: CalMetrics; fit_all: CalMetrics;
  uncertainty: { factor_lo: number | null; factor_hi: number | null; coverage: number | null }; per_site: (CalMetrics & { site: string })[]
  ranking: { predictors: string[]; model: string; transform: string; label: string; cv_r2: number }[]
  predictions: { date: string; site: string; y_true: number; y_oof: number | null; block: number; lo80: number | null; hi80: number | null }[]
  pairs: Record<string, any>[]
  stats: { images: number; images_valid: number; pairs: number; n_insitu_rows: number; n_insitu_dates: number }
  mode: 'gee' | 'demo'
}

export function downloadBlob(name: string, text: string, type = 'application/json') {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type }))
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
