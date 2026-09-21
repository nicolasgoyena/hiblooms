import { useEffect, useRef } from 'react'
import maplibregl, { Map as MLMap, LngLatBoundsLike } from 'maplibre-gl'
import { Poi, POI_COLORS } from './api'
import { Lang, t } from './i18n'

export type Basemap = 'satellite' | 'light'

type Props = {
  reservoirs: GeoJSON.FeatureCollection | null
  selected: string | null
  onSelect: (name: string) => void
  basemap: Basemap
  indexTileUrl: string | null
  rgbTileUrl: string | null
  showRgb: boolean
  opacity: number
  demoFill: string | null // color de relleno en modo demo (media del embalse)
  points: Poi[]
  addingPoint: boolean
  onMapClick: (lat: number, lon: number) => void
  onPoiClick: (name: string) => void
  lang?: Lang
  compareTileUrl?: string | null   // segunda fecha, para el comparador
  swipe?: number                   // 0–1: posición de la cortinilla
}

const BASEMAPS: Record<Basemap, { tiles: string[]; attribution: string }> = {
  satellite: {
    tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
    attribution: 'Imagery © Esri',
  },
  // Esri World Light Gray Canvas: sin clave de API (CARTO ahora la exige fuera de su lista blanca)
  light: {
    tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
    attribution: 'Esri, HERE, Garmin, © OpenStreetMap contributors',
  },
}

function bboxOf(f: GeoJSON.Feature): LngLatBoundsLike {
  let [x0, y0, x1, y1] = [180, 90, -180, -90]
  const walk = (c: any) => {
    if (typeof c[0] === 'number') {
      x0 = Math.min(x0, c[0]); y0 = Math.min(y0, c[1]); x1 = Math.max(x1, c[0]); y1 = Math.max(y1, c[1])
    } else c.forEach(walk)
  }
  walk((f.geometry as any).coordinates)
  return [[x0, y0], [x1, y1]]
}

export default function MapView(p: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const refB = useRef<HTMLDivElement>(null)
  const mapB = useRef<MLMap | null>(null)
  const map = useRef<MLMap | null>(null)
  const ready = useRef(false)
  const onSelectRef = useRef(p.onSelect)
  onSelectRef.current = p.onSelect
  const addingRef = useRef(p.addingPoint)
  addingRef.current = p.addingPoint
  const onMapClickRef = useRef(p.onMapClick)
  onMapClickRef.current = p.onMapClick
  const onPoiRef = useRef(p.onPoiClick)
  onPoiRef.current = p.onPoiClick

  // init
  useEffect(() => {
    const m = new maplibregl.Map({
      container: ref.current!,
      style: {
        version: 8,
        sources: { base: { type: 'raster', tiles: BASEMAPS.satellite.tiles, tileSize: 256, attribution: BASEMAPS.satellite.attribution } },
        layers: [{ id: 'base', type: 'raster', source: 'base' }],
      },
      center: [-1.2, 41.9],
      zoom: 6.3,
      attributionControl: { compact: true },
    })
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    m.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right')
    m.on('load', () => {
      ready.current = true; m.fire('hb:ready')
    })
    m.on('click', e => { if (addingRef.current) onMapClickRef.current(e.lngLat.lat, e.lngLat.lng) })
    map.current = m
    return () => m.remove()
  }, [])

  const whenReady = (fn: (m: MLMap) => void) => {
    const m = map.current
    if (!m) return
    if (ready.current) fn(m)
    else m.once('hb:ready' as any, () => fn(m))
  }

  // basemap
  useEffect(() => whenReady(m => {
    const b = BASEMAPS[p.basemap]
    if (m.getLayer('base')) m.removeLayer('base')
    if (m.getSource('base')) m.removeSource('base')
    m.addSource('base', { type: 'raster', tiles: b.tiles, tileSize: 256, attribution: b.attribution, maxzoom: 19 })
    const first = m.getStyle().layers?.[0]?.id
    m.addLayer({ id: 'base', type: 'raster', source: 'base' }, first)
  }), [p.basemap])

  // reservoirs
  useEffect(() => whenReady(m => {
    if (!p.reservoirs) return
    const existing = m.getSource('res') as maplibregl.GeoJSONSource | undefined
    if (existing) { existing.setData(p.reservoirs); return }
    m.addSource('res', { type: 'geojson', data: p.reservoirs, promoteId: 'NOMBRE' })
    m.addLayer({ id: 'res-fill', type: 'fill', source: 'res', paint: {
      'fill-color': ['case', ['boolean', ['feature-state', 'sel'], false], ['coalesce', ['feature-state', 'demo'], '#5ED3BD'],
        ['boolean', ['get', 'CUSTOM'], false], '#F29A5B', '#5ED3BD'],
      'fill-opacity': ['case', ['boolean', ['feature-state', 'sel'], false], ['case', ['boolean', ['feature-state', 'demoOn'], false], 0.8, 0.05], ['boolean', ['feature-state', 'hover'], false], 0.45, 0.25],
    } })
    m.addLayer({ id: 'res-line', type: 'line', source: 'res', paint: {
      'line-color': ['case', ['boolean', ['feature-state', 'sel'], false], '#FFFFFF', ['boolean', ['get', 'CUSTOM'], false], '#F29A5B', '#5ED3BD'],
      'line-width': ['case', ['boolean', ['feature-state', 'sel'], false], 2.5, 1.2],
    } })
    let hover: string | null = null
    m.on('mousemove', 'res-fill', e => {
      const id = e.features?.[0]?.id as string
      if (hover && hover !== id) m.setFeatureState({ source: 'res', id: hover }, { hover: false })
      hover = id; m.setFeatureState({ source: 'res', id }, { hover: true })
      m.getCanvas().style.cursor = 'pointer'
    })
    m.on('mouseleave', 'res-fill', () => {
      if (hover) m.setFeatureState({ source: 'res', id: hover }, { hover: false })
      hover = null; m.getCanvas().style.cursor = ''
    })
    m.on('click', 'res-fill', e => { if (addingRef.current) return; const id = e.features?.[0]?.id; if (id) onSelectRef.current(String(id)) })
  }), [p.reservoirs])

  // selection → zoom
  const prevSel = useRef<string | null>(null)
  useEffect(() => whenReady(m => {
    if (!m.getSource('res')) return
    if (prevSel.current) m.setFeatureState({ source: 'res', id: prevSel.current }, { sel: false, demo: null, demoOn: false })
    prevSel.current = p.selected
    if (!p.selected || !p.reservoirs) return
    m.setFeatureState({ source: 'res', id: p.selected }, { sel: true })
    const f = p.reservoirs.features.find(f => f.properties?.NOMBRE === p.selected)
    if (f) m.fitBounds(bboxOf(f), { padding: { top: 80, bottom: 260, left: 440, right: 380 }, duration: 1200, maxZoom: 14 })
  }), [p.selected, p.reservoirs])

  // demo fill
  useEffect(() => whenReady(m => {
    if (!p.selected || !m.getSource('res')) return
    m.setFeatureState({ source: 'res', id: p.selected }, { demo: p.demoFill, demoOn: !!p.demoFill })
  }), [p.demoFill, p.selected])

  // raster layers (GEE tiles)
  useEffect(() => whenReady(m => {
    const put = (id: string, url: string | null, visible: boolean, opacity: number) => {
      if (m.getLayer(id)) m.removeLayer(id)
      if (m.getSource(id)) m.removeSource(id)
      if (!url || !visible) return
      m.addSource(id, { type: 'raster', tiles: [url], tileSize: 256, attribution: 'Contains modified Copernicus Sentinel data · Google Earth Engine' })
      m.addLayer({ id, type: 'raster', source: id, paint: { 'raster-opacity': opacity } }, m.getLayer('res-line') ? 'res-line' : undefined)
    }
    // Las dos capas se cargan siempre; el conmutador decide cuál se ve.
    put('rgb', p.rgbTileUrl, true, 1)
    put('idx', p.indexTileUrl, true, p.opacity)
    if (m.getLayer('idx')) m.setLayoutProperty('idx', 'visibility', p.showRgb ? 'none' : 'visible')
  }), [p.indexTileUrl, p.rgbTileUrl])

  useEffect(() => whenReady(m => {
    if (m.getLayer('idx')) m.setPaintProperty('idx', 'raster-opacity', p.opacity)
  }), [p.opacity])

  // Índice ↔ color real sin volver a pedir teselas
  useEffect(() => whenReady(m => {
    if (m.getLayer('idx')) m.setLayoutProperty('idx', 'visibility', p.showRgb ? 'none' : 'visible')
  }), [p.showRgb])

  // puntos de interés (marcadores HTML: no necesitan glyphs)
  const markers = useRef<maplibregl.Marker[]>([])
  useEffect(() => whenReady(m => {
    markers.current.forEach(mk => mk.remove())
    markers.current = p.points.map((pt, i) => {
      const el = document.createElement('div')
      el.className = 'poi'
      el.innerHTML = `<span class="poi-dot" style="background:${POI_COLORS[i % POI_COLORS.length]}"></span><span class="poi-lbl"></span>`
      ;(el.querySelector('.poi-lbl') as HTMLElement).textContent = pt.name
      el.title = `${pt.name} · ${t('clic para ver su serie')}`
      el.addEventListener('click', ev => { ev.stopPropagation(); onPoiRef.current(pt.name) })
      return new maplibregl.Marker({ element: el, anchor: 'left', offset: [-8, 0] }).setLngLat([pt.lon, pt.lat]).addTo(m)
    })
  }), [p.points, p.lang])

  useEffect(() => whenReady(m => { m.getCanvas().style.cursor = p.addingPoint ? 'crosshair' : '' }), [p.addingPoint])

  // ── Comparador: segundo mapa encima, recortado por la cortinilla y sincronizado ──
  useEffect(() => whenReady(m => {
    if (!p.compareTileUrl) {
      mapB.current?.remove(); mapB.current = null
      return
    }
    if (!mapB.current) {
      const b = new maplibregl.Map({
        container: refB.current!,
        style: {
          version: 8,
          sources: { base: { type: 'raster', tiles: BASEMAPS[p.basemap].tiles, tileSize: 256 } },
          layers: [{ id: 'base', type: 'raster', source: 'base' }],
        },
        center: m.getCenter(), zoom: m.getZoom(), bearing: m.getBearing(), pitch: m.getPitch(),
        interactive: false, attributionControl: false,
      })
      const sync = () => {
        b.jumpTo({ center: m.getCenter(), zoom: m.getZoom(), bearing: m.getBearing(), pitch: m.getPitch() })
      }
      m.on('move', sync); m.on('resize', sync)
      b.on('load', () => { sync(); setTiles(b, p.compareTileUrl!) })
      mapB.current = b
    } else {
      setTiles(mapB.current, p.compareTileUrl)
    }
  }), [p.compareTileUrl, p.basemap])

  useEffect(() => {
    const el = refB.current
    if (el) el.style.clipPath = `inset(0 0 0 ${Math.round((p.swipe ?? 0.5) * 100)}%)`
  }, [p.swipe, p.compareTileUrl])

  return (
    <>
      <div ref={ref} className="map" />
      <div ref={refB} className="map map-b" style={{ display: p.compareTileUrl ? 'block' : 'none' }} />
    </>
  )
}

function setTiles(m: MLMap, url: string) {
  const id = 'idxB'
  if (m.getLayer(id)) m.removeLayer(id)
  if (m.getSource(id)) m.removeSource(id)
  m.addSource(id, { type: 'raster', tiles: [url], tileSize: 256 })
  m.addLayer({ id, type: 'raster', source: id, paint: { 'raster-opacity': 0.9 } })
}
