import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error) {
    return { error }
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ background: '#050d1f', color: 'white', padding: '60px', fontFamily: 'monospace', minHeight: '100vh' }}>
          <h1 style={{ color: '#ff6b6b', marginBottom: '20px' }}>Render Error</h1>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '14px', color: '#ffaaaa' }}>
            {this.state.error.message}
          </pre>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginTop: '20px' }}>
            {this.state.error.stack}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}
