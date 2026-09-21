import { Component, ReactNode } from 'react'

/** Si una parte de la web falla, enseña el error en vez de dejar la página en blanco. */
export default class ErrorBoundary extends Component<{ children: ReactNode; label?: string; onReset?: () => void }, { err: Error | null }> {
  state = { err: null as Error | null }
  static getDerivedStateFromError(err: Error) { return { err } }
  componentDidCatch(err: Error) { console.error('[HIBLOOMS]', err) }
  render() {
    if (!this.state.err) return this.props.children
    return (
      <div className="card calres dbres err-box">
        <p className="eyebrow">{this.props.label ?? 'Error'}</p>
        <h2 className="serif">Algo ha fallado al mostrar esta parte</h2>
        <pre className="formula" style={{ whiteSpace: 'pre-wrap' }}>{String(this.state.err?.stack || this.state.err)}</pre>
        <button className="ghost solid" style={{ maxWidth: 220 }}
          onClick={() => { this.setState({ err: null }); this.props.onReset?.() }}>Volver a intentarlo</button>
      </div>
    )
  }
}
