import { Clapperboard, Link2, SlidersHorizontal, Scissors } from 'lucide-react';

const EXAMPLES = [
  'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
];

export default function WelcomeView({ onExample }) {
  return (
    <div className="nc-scroll nc-anim-fade-in" style={{ flex: 1, overflowY: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
      <div style={{ maxWidth: 640, width: '100%', textAlign: 'center' }}>
        <div
          className="nc-anim-fade-up"
          style={{
            width: 64, height: 64, borderRadius: 20, margin: '0 auto 18px',
            background: 'linear-gradient(140deg, var(--nc-accent), #6cb8f5)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 12px 40px rgba(10,129,217,0.35)',
          }}
        >
          <Clapperboard size={30} color="#fff" />
        </div>
        <h1 className="nc-anim-fade-up" style={{ fontSize: 30, fontWeight: 800, letterSpacing: '-0.02em', margin: '0 0 10px', animationDelay: '0.05s' }}>
          Turn any video into clips
        </h1>
        <p className="nc-anim-fade-up" style={{ fontSize: 15, color: 'var(--nc-text-dim)', lineHeight: 1.6, margin: '0 0 26px', animationDelay: '0.1s' }}>
          Paste a link below. I'll check it's downloadable, you pick the style,
          and the AI cuts scroll-stopping clips while you watch.
        </p>
        <div className="nc-anim-fade-up" style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap', animationDelay: '0.15s' }}>
          {[
            { icon: Link2, label: 'Paste a link' },
            { icon: SlidersHorizontal, label: 'Pick settings' },
            { icon: Scissors, label: 'Get clips' },
          ].map(({ icon: Icon, label }, i) => (
            <div key={label} className="nc-card" style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 8, fontSize: 13.5, color: 'var(--nc-text-dim)' }}>
              <span style={{
                width: 22, height: 22, borderRadius: '50%', background: 'var(--nc-accent-soft)',
                color: 'var(--nc-accent)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700,
              }}>{i + 1}</span>
              <Icon size={14} /> {label}
            </div>
          ))}
        </div>
        <button
          className="nc-chip nc-anim-fade-up"
          onClick={() => onExample && onExample(EXAMPLES[0])}
          style={{ marginTop: 22, fontSize: 12.5, animationDelay: '0.2s' }}
        >
          Try an example link →
        </button>
      </div>
    </div>
  );
}
