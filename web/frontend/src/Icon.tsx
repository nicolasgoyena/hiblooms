/**
 * Iconos de línea propios, en SVG. Sustituyen a los emojis: se ven igual en todos los
 * sistemas, heredan el color del texto y dan un aspecto más sobrio.
 * Uso: <Icon name="satelite" /> · tamaño por defecto 16 px, alineado con el texto.
 */

type Props = { name: string; size?: number; className?: string; title?: string }

const P: Record<string, React.ReactNode> = {
  // navegación
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 7.6v.4" /></>,
  monitor: <><path d="M4 19V5M4 19h16" /><path d="M8 16V9M12 16v-5M16 16v-9M20 16v-3" /></>,
  satelite: <><path d="M5.5 9.5 9.5 5.5M8 3.5 11.5 7M3.5 8 7 11.5" /><path d="m11 11 8 8" /><circle cx="17" cy="17" r="3" /><path d="M14 6a4 4 0 0 1 4 4M14 2.5A7.5 7.5 0 0 1 21.5 10" /></>,
  datos: <><ellipse cx="12" cy="6" rx="7" ry="3" /><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" /></>,
  modelos: <><path d="M9 3h6M10 3v6.2L5.6 17a2 2 0 0 0 1.7 3h9.4a2 2 0 0 0 1.7-3L14 9.2V3" /><path d="M8 14h8" /></>,
  // acciones y estados
  serie: <><path d="M4 19V5M4 19h16" /><path d="m6 15 4-5 3 3 5-7" /></>,
  tabla: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18M3 14.5h18M9 9v11" /></>,
  descargar: <><path d="M12 4v11m0 0 4-4m-4 4-4-4" /><path d="M4 19h16" /></>,
  enlace: <><path d="M9.5 14.5 14.5 9.5" /><path d="M11 6.5 12.6 5a4 4 0 0 1 5.7 5.7l-1.6 1.6M13 17.5 11.4 19a4 4 0 0 1-5.7-5.7l1.6-1.6" /></>,
  aviso: <><path d="M12 4.5 2.8 19.5h18.4z" /><path d="M12 10v4M12 16.8v.2" /></>,
  ok: <path d="m4.5 12.5 5 5 10-11" />,
  cerrar: <path d="M6 6l12 12M18 6 6 18" />,
  reset: <><path d="M4 12a8 8 0 1 0 2.4-5.7" /><path d="M4 4v4h4" /></>,
  ensayo: <><path d="M10 3v6L6 18a2 2 0 0 0 1.8 3h8.4a2 2 0 0 0 1.8-3l-4-9V3" /><path d="M8.5 3h7M8 15h8" /></>,
  papelera: <><path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13" /><path d="M10 11v6M14 11v6" /></>,
  archivo: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>,
  punto: <><path d="M12 21s7-6.3 7-11a7 7 0 1 0-14 0c0 4.7 7 11 7 11z" /><circle cx="12" cy="10" r="2.5" /></>,
  mapa: <><path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z" /><path d="M9 3v15M15 6v15" /></>,
  calendario: <><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M3 10h18M8 3v4M16 3v4" /></>,
  // datos del proyecto
  fitoplancton: <><circle cx="12" cy="12" r="8" /><circle cx="10" cy="10" r="1.4" /><circle cx="14.5" cy="13" r="1.1" /><path d="M9.5 15.5c1.5 1 3.5 1 5-1" /></>,
  sonda: <><path d="M12 3v9" /><circle cx="12" cy="15" r="3" /><path d="M7.5 6.5a6 6 0 0 1 9 0M5 4a9.5 9.5 0 0 1 14 0" /></>,
  testigo: <><rect x="8" y="3" width="8" height="18" rx="1.5" /><path d="M8 9h8M8 13.5h8M8 18h8" /></>,
  regla: <><rect x="3" y="8" width="18" height="8" rx="1.5" /><path d="M7 8v3M11 8v4M15 8v3M19 8v4" /></>,
  agua: <><path d="M12 3.5S6.5 10 6.5 14a5.5 5.5 0 0 0 11 0c0-4-5.5-10.5-5.5-10.5z" /></>,
  corte: <><path d="M4 6h16M4 12h16M4 18h16" /><path d="M9 3v18" /></>,
  toxico: <><circle cx="12" cy="12" r="9" /><path d="M12 8.5a3.5 3.5 0 0 1 3 1.7M12 8.5a3.5 3.5 0 0 0-3 1.7M12 15.5a3.5 3.5 0 0 1-3-1.7M12 15.5a3.5 3.5 0 0 0 3-1.7" /><circle cx="12" cy="12" r="1.6" /></>,
  // portada
  nube: <><path d="M7 18h10a4 4 0 0 0 .4-8A6 6 0 0 0 6 11.2 3.4 3.4 0 0 0 7 18z" /></>,
  indice: <><path d="M4 4v16h16" /><path d="M7 15c2.5 0 3-8 5.5-8s3 6 5.5 6" /></>,
  alerta: <><path d="M12 4a5 5 0 0 0-5 5v4l-1.5 3h13L17 13V9a5 5 0 0 0-5-5z" /><path d="M10 20a2 2 0 0 0 4 0" /></>,
  lupa: <><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></>,
  usuario: <><circle cx="12" cy="8" r="3.5" /><path d="M5 20c0-3.6 3.1-5.5 7-5.5s7 1.9 7 5.5" /></>,
  biomasa: <><path d="M5 20c8 0 14-4 14-13-7-1-12 2-12 7 0 2 .6 3.5 1.6 4.6" /><path d="M5 20c1.5-3 3.5-5 6-6.5" /></>,
}

export default function Icon({ name, size = 16, className, title }: Props) {
  const d = P[name]
  if (!d) return null
  return (
    <svg className={'ico' + (className ? ' ' + className : '')} width={size} height={size}
      viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden={title ? undefined : true}
      role={title ? 'img' : undefined} focusable="false">
      {title && <title>{title}</title>}
      {d}
    </svg>
  )
}
