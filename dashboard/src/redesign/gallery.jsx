import { useState } from 'react';
import {
  Btn,
  Badge,
  Switch,
  Segmented,
  Stepper,
  Panel,
  PlatPill,
  RingGauge,
  StatusDot,
  PebbleWaveform,
} from './primitives';
import { Hero } from './chrome';

export function ComponentGalleryView() {
  const [swOn, setSwOn] = useState(true);
  const [segVal, setSegVal] = useState('auto');
  const [stepVal, setStepVal] = useState(3);
  const [gaugeVal] = useState(82);

  return (
    <div className="container fade-in">
      <Hero
        eyebrow="Design System"
        line1="Tactile Clay Component Gallery"
        grad="Living Specs"
        sub="The ground-truth visual gallery of all Nugget Tactile Clay UI primitives, states, and elevation levels."
      />

      {/* 1. Squish Buttons */}
      <Panel title="Squish Buttons" sub="Tactile click physics with dual-source clay elevation">
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center' }}>
          <Btn variant="primary">Primary Rust</Btn>
          <Btn variant="ghost">Ghost Clay</Btn>
          <Btn variant="danger">Danger</Btn>
          <Btn loading variant="primary">Loading State</Btn>
          <Btn disabled variant="primary">Disabled</Btn>
          <Btn size="sm" variant="primary">Small</Btn>
          <Btn size="lg" variant="primary">Large Hero</Btn>
        </div>
      </Panel>

      {/* 2. Conic Ring Gauges */}
      <Panel title="Conic Ring Gauges" sub="Animated radial scores with Space Mono numeric readouts">
        <div style={{ display: 'flex', gap: 28, alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <RingGauge score={gaugeVal} max={100} size="big" accent="var(--gold)" label="Overall Virality" sublabel="/100" />
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>Big (150px)</div>
          </div>
          <div>
            <RingGauge score={91} max={100} size="med" accent="var(--rust)" label="Hook Score" />
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>Med Rust (74px)</div>
          </div>
          <div>
            <RingGauge score={76} max={100} size="med" accent="var(--sage)" label="Coherence Score" />
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>Med Sage (74px)</div>
          </div>
          <div>
            <RingGauge score={80} max={100} size="med" accent="var(--plum)" label="Payoff Score" />
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>Med Plum (74px)</div>
          </div>
          <div>
            <RingGauge score={96} max={100} size="sm" accent="var(--gold)" label="Clip Score" />
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>Small (52px)</div>
          </div>
          <div>
            <RingGauge score={88} max={100} size="mini" accent="var(--plum)" label="Mini" />
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>Mini (34px)</div>
          </div>
        </div>
      </Panel>

      {/* 3. Form Controls */}
      <Panel title="Clay Form Controls" sub="Switches, Segmented Rails, Steppers, and Text Inputs">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
          <div>
            <div className="label" style={{ marginBottom: 8 }}>Toggle Switch</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <Switch on={swOn} onChange={setSwOn} label="Demo toggle" />
              <span style={{ fontWeight: 700, fontSize: 14 }}>{swOn ? 'Enabled' : 'Disabled'}</span>
            </div>
          </div>

          <div>
            <div className="label" style={{ marginBottom: 8 }}>Segmented Rail</div>
            <Segmented
              value={segVal}
              onChange={setSegVal}
              options={[
                { id: 'auto', label: 'Auto Tracker' },
                { id: 'subject', label: 'Subject Focus' },
                { id: 'letterbox', label: 'Letterbox' },
              ]}
            />
          </div>

          <div>
            <div className="label" style={{ marginBottom: 8 }}>Stepper</div>
            <Stepper value={stepVal} set={setStepVal} min={1} max={10} label="Clip count" />
          </div>

          <div>
            <div className="label" style={{ marginBottom: 8 }}>Sunken Text Input</div>
            <input type="text" defaultValue="https://youtube.com/watch?v=example" placeholder="Paste link..." />
          </div>
        </div>
      </Panel>

      {/* 4. Badges, Status Dots & Pebble Waveforms */}
      <Panel title="Rarity Badges & Pebble Waveforms" sub="Tier badges and organic audio energy visualizer">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <span className="tier-badge legendary">Legendary</span>
            <span className="tier-badge epic">Epic</span>
            <span className="tier-badge rare">Rare</span>
            <span className="tier-badge common">Common</span>
            <Badge tone="ok">System Optimal</Badge>
            <Badge tone="warn">Queued</Badge>
            <Badge tone="err">Error</Badge>
          </div>

          <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
            <StatusDot status="online" label="Online" />
            <StatusDot status="busy" label="Capturing Stream" />
            <StatusDot status="offline" label="Offline" />
          </div>

          <div>
            <div className="label" style={{ marginBottom: 8 }}>Pebble Audio Waveform Visualizer</div>
            <PebbleWaveform count={24} activeCount={14} />
          </div>
        </div>
      </Panel>
    </div>
  );
}
