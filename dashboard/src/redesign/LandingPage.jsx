/**
 * Nugget — public landing page ("Tactile Clay" design system)
 *
 * Section inventory (in order): sticky nav, hero, proof strip, how-it-works,
 * features, virality-score explainer, caption gallery, pricing, FAQ,
 * final CTA, footer + Terms/Privacy modal.
 *
 * Expected sibling modules (not part of this file):
 *   ./icon        -> export function Icon({ n, ...props })
 *                    kebab-case names: sparkles, check, captions, trending-up,
 *                    crop, calendar-clock, flame, send, plus, x, circle-check,
 *                    clock, bar-chart, play, globe, languages, type, wand-sparkles
 *   ./icon        -> export function Social({ n })  n: "tiktok"|"instagram"|"youtube"
 *   ./primitives  -> export function Btn({ variant, children, onClick, ...props })
 *                    variant: "primary" | "ghost" | "outline"
 *                  -> export function RingGauge({ score, size, accent, label, sublabel })
 *                    accent accepts a CSS var token string, e.g. "var(--gold)"
 *
 * All classes come from phaseA.css (.lp-* family + shared .sk/.empty).
 * No <style> tags, no new CSS, no off-palette colors — inline styles use
 * var(--token) references only.
 */
import { useEffect, useRef, useState } from 'react';
import { Icon, Social } from './icon';
import { Btn, RingGauge } from './primitives';

const scrollToId = (id) => {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

/* ------------------------------------------------------------------ */
/* Data                                                                */
/* ------------------------------------------------------------------ */

const NAV_LINKS = [
  { label: 'Features', id: 'features' },
  { label: 'Examples', id: 'examples' },
  { label: 'Pricing', id: 'pricing' },
  { label: 'FAQ', id: 'faq' },
];

const PROOF_STATS = [
  { val: '2M+', label: 'clips rendered' },
  { val: '94', label: 'avg. top-clip score' },
  { val: '12', label: 'languages' },
  { val: '4.9/5', label: 'creator rating' },
];

const STEPS = [
  {
    n: 1,
    title: 'Drop a link or upload',
    body: 'Paste any YouTube link or upload a file. By uploading, you confirm you own the rights to the video — we take that seriously.',
  },
  {
    n: 2,
    title: 'AI finds the moments',
    body: 'Nugget transcribes your video, scores every segment 1–100, and picks the keepers — the bits people actually stay for.',
  },
  {
    n: 3,
    title: 'Style & post everywhere',
    body: 'Auto-reframe, word-pop captions, hook cards — then schedule straight to TikTok, Reels and Shorts from one place.',
  },
];

const FEATURES = [
  {
    icon: 'trending-up',
    accent: 'var(--grad-rust)',
    title: 'Virality score',
    body: 'Every clip earns a 1–100 score built from hook strength, payoff, pacing and clarity — so you post the keepers, not the maybes.',
  },
  {
    icon: 'captions',
    accent: 'var(--grad-gold)',
    title: 'Word-pop captions',
    body: 'Big, bouncy, Hormozi-style captions that keep thumbs stopped. Every word lands exactly on the beat.',
  },
  {
    icon: 'crop',
    accent: 'var(--grad-sage)',
    title: 'Auto-reframe 9:16',
    body: 'The camera finds the speaker and follows them. Your 16:9 footage becomes native vertical — no manual cropping, ever.',
  },
  {
    icon: 'calendar-clock',
    accent: 'var(--grad-plum)',
    title: 'Smart scheduling',
    body: 'Queue a week of clips in one sitting. Nugget publishes to TikTok, Reels and Shorts when your audience is actually awake.',
  },
  {
    icon: 'flame',
    accent: 'var(--grad-rust)',
    title: 'Trend radar',
    body: 'See which sounds, hooks and formats are heating up in your niche — while there is still time to ride them.',
  },
  {
    icon: 'send',
    accent: 'var(--grad-sage)',
    title: 'Multi-account publishing',
    body: 'One render, every account. Connect as many TikTok, Instagram and YouTube accounts as your plan allows.',
  },
];

const SCORE_FACTORS = [
  { icon: 'sparkles', name: 'Hook strength', weight: 35, desc: 'The first 3 seconds decide everything — this measures how hard yours grab.' },
  { icon: 'flame', name: 'Payoff density', weight: 25, desc: 'Insight per second. Clips that keep delivering keep people watching.' },
  { icon: 'clock', name: 'Pacing & retention', weight: 25, desc: 'Dead air removed, beats tightened — the edit patterns high-retention clips share.' },
  { icon: 'captions', name: 'Clarity & captions', weight: 15, desc: 'Readable at a glance, even on mute. Captions are half the game on the feed.' },
];

const GALLERY = [
  {
    name: 'Hormozi pop',
    text: 'wait for the twist',
    style: { color: 'var(--gold)', textTransform: 'uppercase', fontSize: '26px' },
  },
  {
    name: 'Karaoke',
    text: 'this changed everything',
    style: {
      background: 'var(--grad-plum)',
      padding: '8px 18px',
      borderRadius: 'var(--r-pill)',
      fontSize: '20px',
      textShadow: 'none',
    },
  },
  {
    name: 'Minimal mono',
    text: 'she said no. then this.',
    style: { fontFamily: 'var(--font-mono)', fontSize: '17px', textShadow: 'none', color: 'var(--ink-light)' },
  },
  {
    name: 'Bold outline',
    text: 'never do this',
    style: {
      color: 'var(--ink-light)',
      WebkitTextStroke: '1.5px var(--ink)',
      textShadow: 'none',
      fontSize: '26px',
      textTransform: 'uppercase',
    },
  },
  {
    name: 'Typewriter',
    text: 'the results shocked me ▌',
    style: { fontFamily: 'var(--font-mono)', color: 'var(--gold)', fontSize: '18px', textShadow: 'none' },
  },
  {
    name: 'Lower-third',
    text: 'chapter 3: the comeback',
    style: {
      top: 'auto',
      bottom: '46px',
      textAlign: 'left',
      fontSize: '15px',
      fontFamily: 'var(--font-body)',
      fontWeight: 700,
    },
  },
];

const TIERS = [
  {
    name: 'Starter',
    desc: 'For testing the waters.',
    price: '$12',
    features: ['30 clips / month', '720p renders', '3 caption styles', '1 account per platform'],
    popular: false,
    ctaVariant: 'outline',
  },
  {
    name: 'Creator',
    desc: 'For posting every single day.',
    price: '$24',
    features: ['150 clips / month', '1080p renders', 'All caption styles', 'Smart scheduling', 'Trend radar', '3 accounts per platform'],
    popular: true,
    ctaVariant: 'primary',
  },
  {
    name: 'Studio',
    desc: 'For teams and agencies.',
    price: '$59',
    features: ['Unlimited clips', '4K renders', 'Brand kits', '5 seats', 'Priority render queue', '10 accounts per platform'],
    popular: false,
    ctaVariant: 'outline',
  },
];

const FAQS = [
  {
    q: 'Do I keep the rights to my clips?',
    a: 'Yes — everything Nugget renders from your videos is yours, forever, including the trial clips. The one condition is on the way in: by uploading a source video, you confirm you own it or have the rights to use it. Don’t upload other people’s copyrighted work.',
  },
  {
    q: 'Which platforms can I publish to?',
    a: 'TikTok, Instagram Reels and YouTube Shorts — and you can connect multiple accounts on each, depending on your plan. Schedule once and Nugget posts to all of them at the times your audience is online.',
  },
  {
    q: 'What languages does it support?',
    a: 'Twelve languages for transcription and captions, including auto-translated captions — so one clip can ship in English and land in Spanish too. The full list ships at release.',
  },
  {
    q: 'Can I cancel my subscription?',
    a: 'Anytime, in two clicks, no retention maze. Your clips stay yours and stay downloadable after you cancel — we don’t hold your work hostage.',
  },
  {
    q: 'Is there a free plan?',
    a: 'Yes. The free plan includes a set of watermark-free trial clips every month so you can see your real footage, with your real voice, scored and captioned before you spend a cent.',
  },
  {
    q: 'How does the virality score work?',
    a: 'Every segment gets scored 0–100 from four weighted factors: hook strength (35%), payoff density (25%), pacing & retention (25%) and clarity & captions (15%). It’s honest guidance — a high score means a clip is built to be watched, never a promise it will go viral.',
  },
];

/* ------------------------------------------------------------------ */
/* Terms & Privacy (plain-language v1)                                 */
/* ------------------------------------------------------------------ */

const LEGAL = {
  terms: {
    heading: 'Terms of Service',
    sections: [
      {
        h: 'What Nugget does',
        body: 'Nugget turns your long videos into short clips: auto-reframe to 9:16, word-pop captions, hook cards, a 1–100 virality score, and optional scheduled publishing to TikTok, Instagram Reels and YouTube Shorts.',
      },
      {
        h: 'Your content, your rights',
        body: 'You keep full rights to everything Nugget renders from your videos — including trial clips on the free plan. By uploading a source video, you confirm you own it or have the rights to use it. Don’t upload other people’s copyrighted work; we remove infringing content when we learn of it.',
      },
      {
        h: 'Fair use',
        body: 'Use the render queue like a creator, not a botnet: no spam, no automated abuse of capacity, and no claiming the virality score is a guarantee. We may pause accounts that degrade the service for everyone else.',
      },
      {
        h: 'Plans and cancellation',
        body: 'Launch pricing is shown on this page and confirmed at release. Cancel anytime — your rendered clips remain yours and downloadable. We process rendering on operator-owned hardware and keep pricing honest: no hidden fees, no lock-in.',
      },
      {
        h: 'Contact',
        body: 'Questions about these terms? Write to hello@nugget.app and a human replies.',
      },
    ],
  },
  privacy: {
    heading: 'Privacy Policy',
    sections: [
      {
        h: 'What we collect',
        body: 'Three things, and only three things: your account email, the source videos you upload or link, and basic usage data (renders, scores, scheduling activity) so the product works and gets better.',
      },
      {
        h: 'How we process it',
        body: 'Your videos are processed on operator-owned hardware — not handed around between third-party AI services for fun. We never sell your data, we run no ad tracking on this site, and connected social accounts are used only to publish where you tell us to.',
      },
      {
        h: 'Retention and deletion',
        body: 'Source videos are kept only as long as needed to render and re-render your clips. You can delete your uploads and your account anytime; deletion is real and permanent, not a soft archive.',
      },
      {
        h: 'Your rights',
        body: 'Access, export, or delete everything we hold about you — email hello@nugget.app and we’ll act within 30 days. You’re in charge of your data; we’re just holding it for you.',
      },
      {
        h: 'Security',
        body: 'We use reasonable technical and organizational measures to protect your data. No system is 100% secure, and we won’t pretend otherwise — but we take this seriously.',
      },
    ],
  },
};

function LegalModal({ kind, onClose }) {
  const headingRef = useRef(null);
  const legal = LEGAL[kind];

  useEffect(() => {
    headingRef.current?.focus();
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="lp-modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="lp-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="lp-legal-heading"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="lp-legal-heading" ref={headingRef} tabIndex={-1}>
          {legal.heading}
        </h2>
        <p className="lp-modal-sub">v1 · effective September 2026</p>
        {legal.sections.map((s) => (
          <div key={s.h}>
            <h3>{s.h}</h3>
            <p>{s.body}</p>
          </div>
        ))}
        <Btn variant="primary" onClick={onClose} className="lp-modal-close">
          Got it
        </Btn>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Landing page                                                        */
/* ------------------------------------------------------------------ */

export function LandingPage({ onStart, onSignIn }) {
  const [legalKind, setLegalKind] = useState(null);

  return (
    <div className="lp">
      {/* ---------- Sticky nav ---------- */}
      <header className="lp-nav">
        <div className="lp-nav-inner">
          <button className="lp-brand" onClick={() => scrollToId('top')} aria-label="Nugget — back to top">
            <img src="/logo.svg" alt="Nugget logo" width="34" height="34" />
            <span>Nugget</span>
          </button>
          <nav className="lp-nav-links" aria-label="Page sections">
            {NAV_LINKS.map((l) => (
              <button key={l.id} onClick={() => scrollToId(l.id)}>
                {l.label}
              </button>
            ))}
          </nav>
          <div className="lp-nav-cta">
            <Btn variant="ghost" onClick={onSignIn}>
              Sign in
            </Btn>
            <Btn variant="primary" onClick={onStart}>
              Start free
            </Btn>
          </div>
        </div>
      </header>

      <main id="top">
        <div className="lp-inner">
          {/* ---------- Hero ---------- */}
          <section className="lp-hero" aria-label="Introduction">
            <div>
              <span className="lp-eyebrow">
                <Icon n="sparkles" /> AI clipping studio
              </span>
              <h1 className="lp-h1">
                Find the <span className="accent">moment worth keeping.</span>
              </h1>
              <p className="lp-sub">
                Nugget turns long videos into viral-ready Shorts, Reels &amp; TikToks —
                auto-reframed, captioned, and scored for shareability.
              </p>
              <div className="lp-cta-row">
                <Btn variant="primary" onClick={onStart}>
                  Start clipping free
                </Btn>
                <Btn variant="ghost" onClick={() => scrollToId('how')}>
                  See how it works
                </Btn>
              </div>
              <p className="lp-cta-note">
                <Icon n="check" aria-hidden="true" />
                Free plan · No credit card · Cancel anytime
              </p>
            </div>

            <div className="lp-hero-visual" aria-hidden="true">
              <div className="lp-phone">
                <video autoPlay muted loop playsInline poster="/demo-poster.jpg">
                  <source src="/demo-clip.mp4" type="video/mp4" />
                </video>
                <div className="lp-phone-caption">
                  this part got <span className="pop">2.4M views</span>
                </div>
              </div>
              <div className="lp-float f1">
                <RingGauge score={94} size={64} accent="var(--gold)" />
                <div>
                  <div className="lp-float-val">94</div>
                  <div className="lp-float-label">Virality score</div>
                </div>
              </div>
              <div className="lp-float f2">
                <Icon n="captions" />
                <div>
                  <div className="lp-float-val">Hormozi captions</div>
                  <div className="lp-float-label">style · on</div>
                </div>
              </div>
            </div>
          </section>

          {/* ---------- Proof strip ---------- */}
          <section className="lp-proof" aria-label="Nugget by the numbers">
            <div className="lp-stats">
              {PROOF_STATS.map((s) => (
                <div key={s.label}>
                  <div className="lp-stat-val">{s.val}</div>
                  <div className="lp-stat-label">{s.label}</div>
                </div>
              ))}
            </div>
          </section>

          {/* ---------- How it works ---------- */}
          <section className="lp-section" id="how" aria-label="How it works">
            <div className="lp-sec-head">
              <h2>From long video to posted clip in minutes</h2>
              <p>No timeline scrubbing, no caption software, no second-guessing which part to post. Three steps and you’re done.</p>
            </div>
            <div className="lp-steps">
              {STEPS.map((s) => (
                <article className="lp-step" key={s.n}>
                  <div className="lp-step-num">{s.n}</div>
                  <h3>{s.title}</h3>
                  <p>{s.body}</p>
                </article>
              ))}
            </div>
          </section>

          {/* ---------- Features ---------- */}
          <section className="lp-section alt" id="features" aria-label="Features">
            <div className="lp-sec-head">
              <h2>Everything a clip needs, nothing it doesn’t</h2>
              <p>One studio that takes the raw footage and hands back clips that look like they were edited by someone with too much coffee and great taste.</p>
            </div>
            <div className="lp-feats">
              {FEATURES.map((f) => (
                <article className="lp-feat" key={f.title}>
                  <div className="lp-feat-ic" style={{ background: f.accent }}>
                    <Icon n={f.icon} />
                  </div>
                  <h3>{f.title}</h3>
                  <p>{f.body}</p>
                </article>
              ))}
            </div>
          </section>

          {/* ---------- Score explainer ---------- */}
          <section className="lp-section" id="score" aria-label="How the virality score works">
            <div className="lp-sec-head">
              <h2>Know it’s good <em>before</em> you post</h2>
              <p>Every segment of your video gets scored 1–100 from four weighted factors. Here’s what goes into the number.</p>
            </div>
            <div className="lp-score-wrap">
              <div className="lp-score-card">
                <RingGauge
                  score={94}
                  size={160}
                  accent="var(--gold)"
                  label="Virality score"
                  sublabel="top clip · this render"
                />
              </div>
              <div className="lp-score-factors">
                {SCORE_FACTORS.map((f) => (
                  <div className="lp-factor" key={f.name}>
                    <div className="lp-factor-top">
                      <span className="lp-factor-name">
                        <Icon n={f.icon} aria-hidden="true" />
                        {f.name}
                      </span>
                      <span className="lp-factor-w">{f.weight}%</span>
                    </div>
                    <div className="lp-factor-bar">
                      <i style={{ width: `${f.weight}%` }} />
                    </div>
                    <p>{f.desc}</p>
                  </div>
                ))}
                <p style={{ fontSize: '13.5px', color: 'var(--ink-dim)', margin: '4px 2px 0', lineHeight: 1.6 }}>
                  A high score means a clip is built to be watched — never a promise it’ll go viral.
                </p>
              </div>
            </div>
          </section>

          {/* ---------- Caption gallery ---------- */}
          <section className="lp-section alt" id="examples" aria-label="Caption style examples">
            <div className="lp-sec-head">
              <h2>Captions that stop the scroll</h2>
              <p>Six styles to start with, each tuned for a different feed energy. This is what they look like on real footage.</p>
            </div>
            <div className="lp-gal">
              {GALLERY.map((c) => (
                <div className="lp-gal-card" key={c.name}>
                  <img src="/demo-poster.jpg" alt={`Example of the ${c.name} caption style`} loading="lazy" />
                  <div className="lp-gal-cap" style={c.style}>
                    {c.text}
                  </div>
                  <div className="lp-gal-name">{c.name}</div>
                </div>
              ))}
            </div>
          </section>

          {/* ---------- Pricing ---------- */}
          <section className="lp-section" id="pricing" aria-label="Pricing">
            <div className="lp-sec-head">
              <h2>Pricing that pays for itself</h2>
              <p>Start free, upgrade when your posting schedule outgrows the trial clips. Cancel anytime — your clips stay yours.</p>
            </div>
            <div className="lp-tiers">
              {TIERS.map((t) => (
                <article className={`lp-tier${t.popular ? ' pop' : ''}`} key={t.name}>
                  {t.popular && <span className="lp-tier-flag">Most popular</span>}
                  <h3>{t.name}</h3>
                  <p className="lp-tier-desc">{t.desc}</p>
                  <div className="lp-price">
                    <b>{t.price}</b>
                    <span>/month</span>
                  </div>
                  <ul>
                    {t.features.map((f) => (
                      <li key={f}>
                        <Icon n="check" aria-hidden="true" />
                        {f}
                      </li>
                    ))}
                  </ul>
                  <Btn variant={t.ctaVariant} onClick={onStart}>
                    {t.popular ? 'Start with Creator' : `Choose ${t.name}`}
                  </Btn>
                </article>
              ))}
            </div>
            <p className="lp-tier-note">Launch pricing — final numbers confirmed at release.</p>
          </section>

          {/* ---------- FAQ ---------- */}
          <section className="lp-section alt" id="faq" aria-label="Frequently asked questions">
            <div className="lp-sec-head">
              <h2>Fair questions, straight answers</h2>
              <p>Everything creators usually ask before their first render.</p>
            </div>
            <div className="lp-faq">
              {FAQS.map((f) => (
                <details className="lp-faq-item" key={f.q}>
                  <summary>
                    {f.q}
                    <Icon n="plus" aria-hidden="true" />
                  </summary>
                  <div className="lp-faq-a">{f.a}</div>
                </details>
              ))}
            </div>
          </section>

          {/* ---------- Final CTA ---------- */}
          <section className="lp-section" aria-label="Get started">
            <div className="lp-final">
              <h2>Your next viral moment is already in your footage.</h2>
              <p>Drop a link, let the AI find it, and post your best clip today — free.</p>
              <div className="lp-cta-row">
                <Btn variant="primary" onClick={onStart}>
                  Start clipping free
                </Btn>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* ---------- Footer ---------- */}
      <footer className="lp-footer">
        <div className="lp-inner">
          <div className="lp-foot-grid">
            <div className="lp-foot-brand">
              <button className="lp-brand" onClick={() => scrollToId('top')} aria-label="Nugget — back to top">
                <img src="/logo.svg" alt="Nugget logo" width="34" height="34" />
                <span>Nugget</span>
              </button>
              <p>Find the moment worth keeping. The AI clipping studio that turns long videos into viral-ready Shorts, Reels and TikToks.</p>
              <div style={{ display: 'flex', gap: '10px', marginTop: '14px' }}>
                <Social n="tiktok" />
                <Social n="instagram" />
                <Social n="youtube" />
              </div>
            </div>
            <div className="lp-foot-col">
              <h4>Product</h4>
              <button onClick={() => scrollToId('features')}>Features</button>
              <button onClick={() => scrollToId('examples')}>Examples</button>
              <button onClick={() => scrollToId('pricing')}>Pricing</button>
              <button onClick={() => scrollToId('faq')}>FAQ</button>
            </div>
            <div className="lp-foot-col">
              <h4>Resources</h4>
              <button onClick={() => scrollToId('how')}>How it works</button>
              <a href="mailto:hello@nugget.app">Contact</a>
            </div>
            <div className="lp-foot-col">
              <h4>Legal</h4>
              <button onClick={() => setLegalKind('terms')}>Terms</button>
              <button onClick={() => setLegalKind('privacy')}>Privacy</button>
            </div>
          </div>
          <div className="lp-foot-note">
            <span>© 2026 Nugget · Built for creators</span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <Icon n="circle-check" aria-hidden="true" />
              All systems nominal
            </span>
          </div>
        </div>
      </footer>

      {legalKind && <LegalModal kind={legalKind} onClose={() => setLegalKind(null)} />}
    </div>
  );
}
