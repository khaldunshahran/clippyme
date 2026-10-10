import { X, Plus } from 'lucide-react';

/**
 * Task 17 — the selected-clip identity beside the composer.
 *
 * "Editing · Clip N — {title} · v{version}" with two explicit actions:
 * New source (clears the selection so a pasted link starts a fresh source
 * flow instead of becoming an edit) and Close (drops the selection).
 */
export default function SelectionBar({ clipIndex, title, version, onNewSource, onClose }) {
  const label = title || `Clip ${clipIndex + 1}`;
  return (
    <div className="nc-selection-bar" role="status" aria-label={`Selected clip: ${label}`}>
      <span className="nc-selection-dot" aria-hidden="true" />
      <span className="nc-selection-editing">Editing</span>
      <span className="nc-selection-title" title={label}>
        Clip {clipIndex + 1} — {label}
      </span>
      {version != null && (
        <span className="nc-selection-version" title="Acknowledged project version">
          v{version}
        </span>
      )}
      <span style={{ flex: 1 }} />
      <button
        className="nc-chip nc-selection-btn"
        onClick={onNewSource}
        title="Clear the selection — the next link you paste starts a new source"
      >
        <Plus size={13} /> New source
      </button>
      <button
        className="nc-icon-btn nc-selection-btn"
        onClick={onClose}
        aria-label="Close clip selection"
        title="Close clip selection"
      >
        <X size={15} />
      </button>
    </div>
  );
}
