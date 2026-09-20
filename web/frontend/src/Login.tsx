import { useEffect, useState } from 'react'
import App from './App'
import LangToggle from './LangToggle'
import { api, getToken, onUnauthorized, setToken } from './api'
import { t, useLang } from './i18n'

/** Puerta de acceso: si el backend tiene usuarios configurados, pide login antes de mostrar la app. */
export default function Gate() {
  const [lang, setLang] = useLang()
  const [state, setState] = useState<'loading' | 'login' | 'app'>('loading')
  const [user, setUser] = useState<string | null>(null)
  const [authOn, setAuthOn] = useState(false)

  const check = async () => {
    try {
      const h = await api.health()
      setAuthOn(h.auth)
      if (!h.auth) { setState('app'); return }
      if (!getToken()) { setState('login'); return }
      const m = await api.me()
      setUser(m.user); setState('app')
    } catch {
      // 401 → onUnauthorized ya manda al login; backend caído → la app muestre su aviso
      if (getToken() !== null) setState('app')
      else setState(s => s === 'loading' ? 'app' : s)
    }
  }
  useEffect(() => {
    onUnauthorized.cb = () => { setToken(null); setUser(null); setState('login') }
    check()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const logout = () => { setToken(null); setUser(null); setState('login') }

  if (state === 'loading') return <div className="login-wrap" />
  if (state === 'app') return <App user={user} onLogout={authOn ? logout : undefined} />
  return <LoginForm lang={lang} setLang={setLang} onOk={(u) => { setUser(u); setState('app') }} />
}

function LoginForm({ lang, setLang, onOk }: { lang: any; setLang: any; onOk: (u: string) => void }) {
  const [u, setU] = useState('')
  const [p, setP] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true); setErr(null)
    try {
      const r = await api.login(u.trim(), p)
      setToken(r.token); onOk(r.user)
    } catch (e: any) {
      setErr(String(e.message).startsWith('401') || /incorrect/i.test(e.message) ? t('Usuario o contraseña incorrectos') : t('No se puede conectar con el servidor'))
    } finally { setBusy(false) }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <div className="login-lang"><LangToggle lang={lang} setLang={setLang} /></div>
        <img src="/logo_hiblooms.png" alt="HIBLOOMS" className="login-logo" />
        <h1>HI<span>BLOOMS</span></h1>
        <p className="login-sub">{t('Sistema de monitorización satelital')}</p>
        <label className="lbl">{t('Usuario')}</label>
        <input value={u} onChange={e => setU(e.target.value)} autoFocus autoComplete="username" placeholder={t('Introduce tu usuario')} />
        <label className="lbl">{t('Contraseña')}</label>
        <input type="password" value={p} onChange={e => setP(e.target.value)} autoComplete="current-password" placeholder="••••••••" />
        {err && <div className="badge err">{err}</div>}
        <button className="primary" type="submit" disabled={busy || !u || !p}>{busy ? t('Entrando…') : t('Iniciar sesión')}</button>
        <p className="login-foot">PID2023-153234OB-I00 · {t('Universidad de Navarra')} · BIOMA</p>
      </form>
    </div>
  )
}
