import { Lang, t } from './i18n'
import LangToggle from './LangToggle'

const TEAM_RESEARCH = [
  ['David Elustondo', 'DEV', 'BIOMA/UNAV', 'Calidad del agua, QA/QC y biogeoquímica'],
  ['Yasser Morera Gómez', 'YMG', 'BIOMA/UNAV', 'Geoquímica isotópica y geocronología con ²¹⁰Pb'],
  ['Esther Lasheras Adot', 'ELA', 'BIOMA/UNAV', 'Técnicas analíticas y calidad del agua'],
  ['Jesús Miguel Santamaría', 'JSU', 'BIOMA/UNAV', 'Calidad del agua y técnicas analíticas'],
  ['Carolina Santamaría Elola', 'CSE', 'BIOMA/UNAV', 'Técnicas analíticas y calidad del agua'],
  ['Adriana Rodríguez Garraus', 'ARG', 'MITOX/UNAV', 'Análisis toxicológico'],
  ['Sheila Izquieta Rojano', 'SIR', 'BIOMA/UNAV', 'SIG y teledetección, datos FAIR, digitalización'],
]
const TEAM_WORK = [
  ['Aimee Valle Pombrol', 'AVP', 'BIOMA/UNAV', 'Taxonomía de cianobacterias e identificación de toxinas'],
  ['Carlos Manuel Alonso Hernández', 'CAH', 'Lab. de Radioecología/IAEA', 'Geocronología con ²¹⁰Pb'],
  ['David Widory', 'DWI', 'GEOTOP/UQAM', 'Geoquímica isotópica y calidad del agua'],
  ['Ángel Ramón Moreira González', 'AMG', 'CEAC', 'Taxonomía de fitoplancton y algas'],
  ['Augusto Abilio Comas González', 'ACG', 'CEAC', 'Taxonomía de cianobacterias y ecología acuática'],
  ['Lorea Pérez Babace', 'LPB', 'BIOMA/UNAV', 'Técnicas analíticas y muestreo de campo'],
  ['José Miguel Otano Calvente', 'JOC', 'BIOMA/UNAV', 'Técnicas analíticas y muestreo de campo'],
  ['Alain Suescun Santamaría', 'ASS', 'BIOMA/UNAV', 'Técnicas analíticas'],
  ['Leyre López Alonso', 'LLA', 'BIOMA/UNAV', 'Análisis de datos'],
  ['María José Rodríguez Pérez', 'MRP', 'Confederación Hidrográfica del Ebro', 'Calidad del agua'],
  ['María Concepción Durán Lalaguna', 'MDL', 'Confederación Hidrográfica del Júcar', 'Calidad del agua'],
]

const FLOW = [
  { icon: '🛰️', t: 'Sentinel-2', s: 'Imagen cada 2–5 días, 10–20 m' },
  { icon: '☁️', t: 'Google Earth Engine', s: 'Filtrado de nubes y máscara de agua' },
  { icon: '📐', t: 'Índices espectrales', s: 'Clorofila-a (algas) y ficocianina (cianobacterias)' },
  { icon: '🗺️', t: 'Mapas y series', s: 'Concentración por píxel, puntos y embalse' },
  { icon: '🚨', t: 'Posible floración', s: 'Detección temprana, con aviso de floraciones tóxicas' },
]

function Team({ title, rows }: { title: string; rows: string[][] }) {
  return (
    <div className="pj-team">
      <h3>{t(title)}</h3>
      <div className="pj-grid">
        {rows.map(([n, code, org, role]) => (
          <div key={code} className="pj-person">
            <span className="pj-av">{code}</span>
            <div><b>{n}</b><small>{t(org)}</small><p>{t(role)}</p></div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ProjectPage({ onClose, onStart, lang, setLang }: { onClose: () => void; onStart: () => void; lang?: Lang; setLang?: (l: Lang) => void }) {
  return (
    <div className="card pj">
      <button className="x pj-x" onClick={onClose} aria-label={t('Cerrar')}>×</button>

      <section className="pj-hero">
      <div className="pj-logobar">
        <img className="pj-logo-main" src="/logo_hiblooms.png" alt="HIBLOOMS" />
        <span className="pj-logo-sep" />
        <img src="/ministerio.png" alt={t('Ministerio de Ciencia, Innovación y Universidades')} />
        <img src="/logo_bioma_unav.png" alt={t('Instituto BIOMA · Universidad de Navarra')} />
        <img src="/logo_ebro.png" alt={t('Confederación Hidrográfica del Ebro')} />
        <img src="/logo_jucar.png" alt={t('Confederación Hidrográfica del Júcar')} />
      </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span className="pj-live"><i /> {t('Monitorización activa · Sentinel-2')}</span>
          {lang && setLang && <LangToggle lang={lang} setLang={setLang} />}
        </div>
        <h1>{t('Vigilancia satelital de')} <em>{t('floraciones algales')}</em> {t('en embalses')}</h1>
        <p>{t('Reconstrucción histórica y monitorización en tiempo casi real de las floraciones algales en embalses españoles mediante teledetección Sentinel-2, con especial atención a las algas tóxicas como las cianobacterias.')}</p>
        <div className="pj-chips">
          <span>🛰️ {t('31 embalses por defecto')}</span><span>📅 {t('Desde 2017')}</span><span>🧪 {t('Clorofila-a y ficocianina')}</span><span>🔬 {t('Calibración con datos in situ')}</span>
        </div>
        <button className="primary pj-cta" onClick={onStart}>{t('Ver el estado de los embalses →')}</button>
      </section>

      <section className="pj-flow">
        <p className="lbl2">{t('Flujo de datos · del satélite a la alerta')}</p>
        <div className="pj-steps">
          {FLOW.map((f, i) => (
            <div key={f.t} className="pj-step">
              <span className="pj-ic">{f.icon}</span><b>{t(f.t)}</b><small>{t(f.s)}</small>
              {i < FLOW.length - 1 && <span className="pj-arrow">→</span>}
            </div>
          ))}
        </div>
      </section>

      <section className="pj-title">
        <small>PID2023-153234OB-I00</small>
        <h2>{t('Reconstrucción histórica y estado actual de la proliferación de cianobacterias en embalses españoles:')} <em>HIBLOOMS</em></h2>
        <div className="pj-align">
          <span><b>PNACC</b> {t('Plan Nacional de Adaptación al Cambio Climático 2021–2030')}</span>
          <span><b>{t('DMA')}</b> {t('Directiva Marco del Agua 2000/60/CE')}</span>
          <span><b>{t('ODS 6')}</b> {t('Agua limpia y saneamiento')}</span>
        </div>
      </section>

      <section className="pj-cols">
        <div>
          <h3>{t('Justificación')}</h3>
          <p>{t('Las floraciones algales en embalses son una preocupación ambiental y de salud pública, sobre todo cuando las forman algas tóxicas como las cianobacterias. HIBLOOMS evalúa la evolución histórica y actual de estos eventos en los embalses de España, contribuyendo a:')}</p>
          <ul>
            <li>{t('Monitorizar parámetros clave del cambio climático y sus efectos en los ecosistemas acuáticos.')}</li>
            <li>{t('Identificar los factores ambientales y de contaminación que influyen en las floraciones.')}</li>
            <li>{t('Generar información para mejorar la gestión y la calidad del agua.')}</li>
          </ul>
        </div>
        <div>
          <h3>{t('Hipótesis y relevancia')}</h3>
          <p>{t('Se estima que el')} <b>{t('40 % de los embalses españoles')}</b> {t('son susceptibles a episodios de floración algal. Con el aumento de temperaturas y de la eutrofización, el riesgo de floraciones tóxicas de cianobacterias es mayor.')}</p>
          <ul>
            <li><b>{t('Teledetección satelital')}</b> {t('para el seguimiento continuo.')}</li>
            <li><b>{t('Análisis ambiental avanzado')}</b> {t('de causas y patrones.')}</li>
            <li><b>{t('Modelos')}</b> {t('para anticipar episodios y sus impactos.')}</li>
          </ul>
        </div>
        <div>
          <h3>{t('Impacto esperado')}</h3>
          <p>{t('Herramientas para una gestión sostenible de los embalses:')}</p>
          <ul>
            <li>{t('Evaluar la')} <b>{t('calidad del agua')}</b> {t('con técnicas avanzadas.')}</li>
            <li>{t('Diseñar estrategias para')} <b>{t('minimizar el riesgo de toxicidad')}</b>.</li>
            <li>{t('Apoyar a las administraciones en la')} <b>{t('toma de decisiones basada en datos')}</b>.</li>
          </ul>
        </div>
      </section>

      <Team title="Equipo de investigación" rows={TEAM_RESEARCH} />
      <Team title="Equipo de trabajo" rows={TEAM_WORK} />

      <p className="pj-quote pj-last">{t('HIBLOOMS no solo estudia el presente: reconstruye el pasado para entender el futuro de la calidad del agua en España.')}</p>

    </div>
  )
}
