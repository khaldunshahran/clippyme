// Nugget — LowerThirdControls: broadcast-style name tags / location titles.
//
// Follows the layerControls.jsx / bannerControls.jsx precedent: fully
// controlled `value` + `onChange(partial)`, UI only. The caller (Create
// recipe) owns the on/off toggle and persists the spec with the clip
// options (localStorage `clippyme_create_opts`, like the other layers).
//
// The clean overlay spec object:
//   { preset, lines: { name, title }, style: { font, accent, align },
//     startSec, endSec }
//
// Burn-in rendering is BACKEND work that doesn't exist yet — the control
// says so honestly (see the note at the bottom of the drawer). The spec is
// saved with the recipe so it applies the moment the backend supports it.
import { Segmented } from './primitives';
import { SUB_FONTS } from './data';

export const LOWER_THIRD_PRESETS = [
  { id: 'name_slide', label: 'Name tag', desc: 'Accent bar, name over title — slides in' },
  { id: 'location_fade', label: 'Location', desc: 'Thin rule + place name — fades up' },
  { id: 'minimal_bar', label: 'Minimal bar', desc: 'Small pill with the name only' },
  { id: 'bold_block', label: 'Bold block', desc: 'Full color block, high contrast' },
  { id: 'news_tone', label: 'News two-tone', desc: 'Accent tab + dark bar, desk style' },
  { id: 'elegant_rule', label: 'Elegant line', desc: 'Centered serif with hairline rules' },
];

export const LOWER_THIRD_ACCENTS = [
  { id: 'rust', label: 'Rust', hex: '#C96F4A' },
  { id: 'gold', label: 'Gold', hex: '#D9A441' },
  { id: 'sage', label: 'Sage', hex: '#7A8B6F' },
  { id: 'plum', label: 'Plum', hex: '#7A5C8E' },
  { id: 'ink', label: 'Ink', hex: '#3B332A' },
];

export const LOWER_THIRD_DEFAULT = {
  preset: 'name_slide',
  lines: { name: 'Alex Rivera', title: 'Founder, Brightline Studio' },
  style: { font: 'Montserrat-Black', accent: 'rust', align: 'left' },
  startSec: 0.5,
  endSec: 4.5,
};

/** Normalize any stored value into the canonical spec shape. */
export function buildLowerThirdSpec(value) {
  const v = value || {};
  const presetIds = new Set(LOWER_THIRD_PRESETS.map((p) => p.id));
  const accentIds = new Set(LOWER_THIRD_ACCENTS.map((a) => a.id));
  const start = Math.max(0, Number(v.startSec ?? LOWER_THIRD_DEFAULT.startSec) || 0);
  let end = Number(v.endSec ?? LOWER_THIRD_DEFAULT.endSec) || 0;
  if (!(end > start)) end = start + 1;
  return {
    preset: presetIds.has(v.preset) ? v.preset : LOWER_THIRD_DEFAULT.preset,
    lines: {
      name: String(v.lines?.name ?? LOWER_THIRD_DEFAULT.lines.name).slice(0, 60),
      title: String(v.lines?.title ?? LOWER_THIRD_DEFAULT.lines.title).slice(0, 80),
    },
    style: {
      font: v.style?.font || LOWER_THIRD_DEFAULT.style.font,
      accent: accentIds.has(v.style?.accent) ? v.style.accent : LOWER_THIRD_DEFAULT.style.accent,
      align: v.style?.align === 'center' ? 'center' : 'left',
    },
    startSec: Math.round(start * 10) / 10,
    endSec: Math.round(end * 10) / 10,
  };
}

function accentHex(id) {
  return (LOWER_THIRD_ACCENTS.find((a) => a.id === id) || LOWER_THIRD_ACCENTS[0]).hex;
}

function fontFamily(fontId) {
  const known = SUB_FONTS.find(([v]) => v === fontId);
  // Map the bundled TTF basenames to reasonable web fallbacks for preview.
  const map = {
    'Montserrat-Black': "'Montserrat', 'Arial Black', sans-serif",
    'Anton-Regular': "'Anton', 'Arial Black', sans-serif",
    'Bangers-Regular': "'Bangers', cursive",
    'Poppins-Black': "'Poppins', 'Arial Black', sans-serif",
    'Poppins-Medium': "'Poppins', sans-serif",
    'Verdana': 'Verdana, sans-serif',
  };
  return map[known?.[0]] || "'Montserrat', 'Arial Black', sans-serif";
}

/**
 * Live HTML/CSS preview of the lower third. `mini` renders the static
 * graphic (used inside the preset picker); the full version loops the
 * entrance animation.
 */
export function LowerThirdPreview({ spec, mini = false }) {
  const s = buildLowerThirdSpec(spec);
  const accent = accentHex(s.style.accent);
  const font = fontFamily(s.style.font);
  const { name, title } = s.lines;
  const align = s.style.align;
  const anim = mini ? {} : { animation: `lt-${s.preset} 2.6s ease-in-out infinite` };
  const common = {
    fontFamily: font,
    textAlign: align,
    ...anim,
  };

  let graphic = null;
  switch (s.preset) {
    case 'name_slide':
      graphic = (
        <div className="lt lt-name_slide" style={common}>
          <div className="lt-bar" style={{ background: accent }} />
          <div className="lt-name" style={{ color: '#FFF' }}>{name}</div>
          <div className="lt-title" style={{ color: 'rgba(255,255,255,.82)' }}>{title}</div>
        </div>
      );
      break;
    case 'location_fade':
      graphic = (
        <div className="lt lt-location_fade" style={common}>
          <div className="lt-rule" style={{ background: accent, margin: align === 'center' ? '0 auto 6px' : '0 0 6px' }} />
          <div className="lt-name" style={{ color: '#FFF', letterSpacing: '0.14em' }}>{name.toUpperCase()}</div>
          <div className="lt-title" style={{ color: 'rgba(255,255,255,.75)' }}>{title}</div>
        </div>
      );
      break;
    case 'minimal_bar':
      graphic = (
        <div className="lt lt-minimal_bar" style={{ ...common, justifyContent: align === 'center' ? 'center' : 'flex-start' }}>
          <div className="lt-pill" style={{ background: 'rgba(0,0,0,.55)', borderColor: accent }}>
            <span className="lt-dot" style={{ background: accent }} />
            <span className="lt-name" style={{ color: '#FFF' }}>{name}</span>
          </div>
        </div>
      );
      break;
    case 'bold_block':
      graphic = (
        <div className="lt lt-bold_block" style={{ ...common, background: accent }}>
          <div className="lt-name" style={{ color: '#FFF' }}>{name}</div>
          <div className="lt-title" style={{ color: 'rgba(255,255,255,.9)' }}>{title}</div>
        </div>
      );
      break;
    case 'news_tone':
      graphic = (
        <div className="lt lt-news_tone" style={common}>
          <div className="lt-tab" style={{ background: accent }}><span style={{ color: '#FFF' }}>LIVE</span></div>
          <div className="lt-dark">
            <div className="lt-name" style={{ color: '#FFF' }}>{name}</div>
            <div className="lt-title" style={{ color: 'rgba(255,255,255,.75)' }}>{title}</div>
          </div>
        </div>
      );
      break;
    case 'elegant_rule':
      graphic = (
        <div className="lt lt-elegant_rule" style={common}>
          <div className="lt-name lt-serif" style={{ color: '#FFF' }}>{name}</div>
          <div className="lt-erules"><span style={{ background: accent }} /><span style={{ background: accent }} /></div>
          <div className="lt-title" style={{ color: 'rgba(255,255,255,.8)', letterSpacing: '0.2em' }}>{title.toUpperCase()}</div>
        </div>
      );
      break;
    default:
      graphic = null;
  }

  return (
    <div className={'lt-stage' + (mini ? ' lt-mini' : '')} aria-hidden={mini}>
      {graphic}
    </div>
  );
}

export function LowerThirdControls({ value, onChange }) {
  const spec = buildLowerThirdSpec(value);
  const set = (partial) => {
    const next = buildLowerThirdSpec({ ...spec, ...partial });
    onChange(next);
  };
  const setLines = (partial) => set({ lines: { ...spec.lines, ...partial } });
  const setStyle = (partial) => set({ style: { ...spec.style, ...partial } });
  const duration = Math.max(0, Math.round((spec.endSec - spec.startSec) * 10) / 10);

  return (
    <>
      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Preset</span>
        <div className="lt-presets">
          {LOWER_THIRD_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={'lt-preset' + (spec.preset === p.id ? ' on' : '')}
              onClick={() => set({ preset: p.id })}
              aria-pressed={spec.preset === p.id}
              title={p.desc}
            >
              <LowerThirdPreview mini spec={{ ...spec, preset: p.id }} />
              <span className="lt-preset-label">{p.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Preview</span>
        <LowerThirdPreview spec={spec} />
        <div className="eo-d" style={{ marginTop: 6 }}>
          Loops the entrance animation · shows {spec.startSec}s → {spec.endSec}s ({duration}s)
        </div>
      </div>

      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Name</span>
        <input className="input-field" style={{ width: '100%' }} aria-label="Lower third name"
          placeholder="Alex Rivera" value={spec.lines.name}
          onChange={(e) => setLines({ name: e.target.value })} />
      </div>
      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Title / location</span>
        <input className="input-field" style={{ width: '100%' }} aria-label="Lower third title"
          placeholder="Founder, Brightline Studio" value={spec.lines.title}
          onChange={(e) => setLines({ title: e.target.value })} />
      </div>

      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Color</span>
        <div className="swatches" role="radiogroup" aria-label="Lower third accent color">
          {LOWER_THIRD_ACCENTS.map((a) => (
            <button
              key={a.id} type="button" role="radio" aria-checked={spec.style.accent === a.id}
              title={a.label} aria-label={a.label}
              className={'swatch' + (spec.style.accent === a.id ? ' on' : '')}
              style={{ background: a.hex }}
              onClick={() => setStyle({ accent: a.id })}
            />
          ))}
        </div>
      </div>

      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Font</span>
        <select className="input-field" style={{ width: '100%' }} aria-label="Lower third font"
          value={spec.style.font} onChange={(e) => setStyle({ font: e.target.value })}>
          {SUB_FONTS.map(([v, label]) => (
            <option key={v} value={v}>{label}</option>
          ))}
        </select>
      </div>

      <div className="cf-row">
        <span className="field-label" style={{ marginBottom: 9, display: 'flex' }}>Align</span>
        <Segmented full value={spec.style.align} onChange={(id) => setStyle({ align: id })}
          options={[{ id: 'left', label: 'Left' }, { id: 'center', label: 'Center' }]} />
      </div>

      <div className="cf-row" style={{ marginBottom: 0 }}>
        <span className="field-label" style={{ marginBottom: 9, display: 'flex', justifyContent: 'space-between' }}>
          <span>Timing</span><span className="eo-d">{duration}s long</span>
        </span>
        <div style={{ display: 'flex', gap: 12 }}>
          <label style={{ flex: 1 }}>
            <span className="eo-d">Start (s)</span>
            <input type="number" className="input-field" style={{ width: '100%' }} min="0" step="0.5"
              aria-label="Lower third start seconds" value={spec.startSec}
              onChange={(e) => set({ startSec: Math.max(0, Number(e.target.value) || 0) })} />
          </label>
          <label style={{ flex: 1 }}>
            <span className="eo-d">End (s)</span>
            <input type="number" className="input-field" style={{ width: '100%' }} min="0.5" step="0.5"
              aria-label="Lower third end seconds" value={spec.endSec}
              onChange={(e) => set({ endSec: Math.max(0.5, Number(e.target.value) || 0) })} />
          </label>
        </div>
      </div>

      <div className="od" style={{ marginTop: 10 }}>
        Preview only for now — the backend can&apos;t burn lower thirds into exports yet.
        Your design is saved with the recipe and will apply once rendering support lands.
      </div>
    </>
  );
}
