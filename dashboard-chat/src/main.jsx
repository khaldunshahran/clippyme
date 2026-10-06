import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App.jsx';

function ErrorFallback({ error }) {
  return (
    <div className="nc-error-fallback">
      <h1>Something went wrong</h1>
      <p>{String(error?.message || error)}</p>
      <button onClick={() => window.location.reload()}>Reload</button>
    </div>
  );
}

class Boundary extends React.Component {
  constructor(p) { super(p); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    return this.state.error
      ? <ErrorFallback error={this.state.error} />
      : this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Boundary><App /></Boundary>
  </React.StrictMode>,
);
