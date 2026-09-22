---
name: clippyme-intelligent-clipping
description: >
  Turns ClippyMe from a generic timestamp cutter into an agentic video-intelligence
  and editing system. Use this skill when the user wants to analyze a long-form
  video, discover high-value moments, create context-complete clips, generate
  platform-ready edits, learn an editing style from a reference video, or improve
  clip selection using feedback.
---

# ClippyMe Intelligent Clipping Skill

## Mission

ClippyMe should not behave like a timeline cutter.

It should behave like an **AI video editor + content strategist** that understands the
entire source video before deciding what deserves to become a clip.

The core principle:

> **Do not ask "where can I cut?" Ask "what is worth turning into a piece of content?"**

A clip is a content unit, not merely a time range.

The system should:
1. Understand the full source.
2. Build a searchable semantic representation of the video.
3. Discover candidate moments.
4. Score candidates for audience value and retention.
5. Expand candidates into context-complete clips.
6. Generate multiple possible hooks/angles.
7. Choose an edit strategy based on the source layout and content.
8. Render the clip programmatically.
9. Validate the result.
10. Learn from user approvals/rejections and published performance.

---

# 1. PRODUCT CONCEPT

## From generic clipping to a "Clip Brain"

The new architecture should have a central reasoning layer called **Clip Brain**.

Clip Brain receives:
- video
- audio
- transcript
- word-level timestamps
- speaker identities
- scene/shot boundaries
- face positions
- screen/UI regions
- visual objects
- audio intensity
- pauses/silences
- source metadata
- optional reference editing style
- optional brand configuration
- optional target platform

Clip Brain produces:

```json
{
  "source_understanding": {},
  "candidate_moments": [],
  "selected_clips": [],
  "edit_plan": {},
  "render_plan": {},
  "quality_report": {}
}
```

The reasoning layer must exist independently from the renderer.

Do NOT tightly couple "finding a good moment" with "cutting a timestamp."

---

# 2. PIPELINE

Implement the workflow as explicit stages:

```text
INGEST
  ↓
MEDIA ANALYSIS
  ↓
TRANSCRIPTION
  ↓
CONTEXT GRAPH
  ↓
MOMENT DISCOVERY
  ↓
CLIP SCORING
  ↓
CONTEXT EXPANSION
  ↓
HOOK / ANGLE GENERATION
  ↓
EDIT PLANNING
  ↓
AUTO REFRAME
  ↓
CAPTION DESIGN
  ↓
MOTION / B-ROLL / SFX
  ↓
RENDER
  ↓
QA
  ↓
USER REVIEW
  ↓
LEARN / FEEDBACK
```

Each stage should produce inspectable intermediate artifacts.

Never hide the reasoning entirely.

---

# 3. MEDIA INGESTION

When a new video is supplied, first inspect:

- duration
- resolution
- FPS
- orientation
- audio tracks
- sample rate
- codec
- number of speakers if detectable
- scene changes
- silence distribution
- speech density

Generate a project manifest:

```json
{
  "source": "...",
  "duration": 0,
  "fps": 0,
  "width": 0,
  "height": 0,
  "orientation": "landscape|portrait|square",
  "audio": {},
  "speakers": [],
  "scenes": []
}
```

Do not start clipping before the media manifest exists.

---

# 4. TRANSCRIPTION IS FOR INTELLIGENCE

Use a transcription engine capable of word-level timestamps.

Prefer:
- WhisperX
- faster-whisper
- another local/open-source timestamped ASR

Store:

```json
{
  "word": "example",
  "start": 12.42,
  "end": 12.91,
  "speaker": "speaker_1",
  "confidence": 0.97
}
```

The transcript is not merely for subtitles.

It is the primary semantic index of the video.

---

# 5. BUILD A CONTEXT GRAPH

Do not treat transcript sentences as independent chunks.

Create relationships between:

- claims
- stories
- examples
- questions
- answers
- arguments
- conclusions
- jokes
- emotional moments
- controversial statements
- personal experiences
- statistics
- demonstrations
- callbacks
- setup/payoff pairs

For each segment, identify:

```json
{
  "topic": "",
  "subtopic": "",
  "intent": "",
  "claim": "",
  "setup": "",
  "payoff": "",
  "emotion": "",
  "entities": [],
  "dependencies": [],
  "standalone": true
}
```

A strong clip should normally contain enough setup to understand its payoff.

---

# 6. MOMENT DISCOVERY

Find candidate moments using multiple signals.

## Semantic signals

Look for:
- surprising claims
- strong opinions
- useful advice
- contrarian ideas
- personal stories
- transformations
- mistakes
- lessons
- revelations
- numbers/statistics
- emotional vulnerability
- humor
- conflict
- disagreement
- "I learned..."
- "the biggest mistake..."
- "nobody tells you..."
- "the reason..."
- "here's how..."
- "you should..."
- "I was wrong..."
- "this changed..."
- "the problem is..."

Do not rely on trigger phrases alone.

The model must understand meaning.

## Retention signals

Prefer moments with:
- immediate curiosity
- unanswered questions
- escalating tension
- clear payoff
- information density
- emotional change
- pattern interruption
- strong first sentence
- strong final sentence

## Audio signals

Detect:
- emphasis
- volume spikes
- laughter
- applause
- rapid speech
- intentional pauses
- dramatic silence
- music changes

## Visual signals

Detect:
- speaker expression changes
- gestures
- demonstrations
- screen changes
- product/UI interaction
- visual reveals
- camera changes
- scene changes
- multiple speakers
- objects being shown

Use audio + visual + semantic signals together.

---

# 7. CLIP SCORING

Every candidate receives a transparent score.

Suggested default:

```text
Hook Strength             20%
Standalone Context        15%
Payoff / Completion       15%
Audience Value            15%
Curiosity / Tension       10%
Emotional Energy          10%
Novelty / Surprise         5%
Visual Editability         5%
Platform Fit               5%
```

Score 0-100.

Also classify the reason:

```json
{
  "score": 87,
  "primary_reason": "contrarian insight",
  "secondary_reasons": [
    "strong payoff",
    "high information density",
    "clear standalone context"
  ]
}
```

The score is not a truth claim. It is a ranking mechanism.

---

# 8. CONTEXT-COMPLETE CLIPPING

This is a critical feature.

Do NOT simply cut the exact sentence containing the interesting idea.

For each candidate, calculate:

- minimum setup
- main idea
- payoff
- optional emotional reaction
- natural ending

Then create:

```json
{
  "candidate_start": 312.4,
  "candidate_end": 349.8,
  "recommended_start": 306.9,
  "recommended_end": 354.2,
  "context_added_before": 5.5,
  "context_added_after": 4.4
}
```

Rules:

- Avoid starting mid-thought.
- Avoid ending before the payoff.
- Prefer natural sentence boundaries.
- Preserve pronoun/entity references.
- Include enough context to answer "what are they talking about?"
- Remove repetitive setup when possible.
- Do not destroy conversational rhythm.

A 45-second complete story is better than a 22-second confusing quote.

---

# 9. CLIP TYPES

Classify every selected clip.

Possible types:

- HOT TAKE
- STORY
- LESSON
- HOW-TO
- CONTRARIAN
- REACTION
- DEBATE
- FUNNY
- EMOTIONAL
- CASE STUDY
- PRODUCT DEMO
- NEWS/UPDATE
- LIST
- FAILURE/LESSON
- REVELATION
- Q&A
- BEFORE/AFTER

Use the type to choose the edit grammar.

---

# 10. MULTIPLE ANGLES FROM ONE MOMENT

One source moment can generate several clips.

For example:

```text
Original moment
 ├── curiosity angle
 ├── contrarian angle
 ├── educational angle
 ├── emotional angle
 └── money/business angle
```

Do not duplicate blindly.

Only create variants when the angle materially changes the value proposition.

Each variant should have:
- hook
- target audience
- title
- suggested opening
- caption emphasis
- platform suitability

---

# 11. HOOK ENGINE

The first 1-3 seconds matter.

If the source already begins with a strong sentence, preserve it.

If the source has a strong statement later in the clip, consider restructuring:

```text
COLD OPEN / HOOK
      ↓
ORIGINAL SETUP
      ↓
PAYOFF
```

Never fabricate a statement.

Generated hooks must be clearly marked as:
- source quote
- paraphrase
- editorial title

Prefer source-native hooks whenever possible.

---

# 12. SOURCE LAYOUT DETECTION

Automatically classify the video layout.

Support at minimum:

### A. Talking-head close-up
One person.

### B. Screen share + speaker
Screen and speaker occupy separate regions.

### C. Two-person podcast
Two speakers visible.

### D. Multi-speaker
More than two relevant faces.

### E. Presentation / slides
Slides are the dominant visual.

### F. Demonstration
Hands/product/UI are the dominant visual.

### G. Unknown
Fall back to conservative center crop.

---

# 13. AUTO-REFRAME

For 9:16:

- detect faces
- detect active speaker
- track the relevant subject
- smooth movement
- avoid abrupt crop jumps
- protect subtitles from faces and UI
- preserve important screen content

Use:

- MediaPipe where practical
- FFmpeg for media operations
- EMA or similar smoothing for tracking
- OpenCV when useful

For two-person conversations, intelligently alternate or split layouts.

Do not blindly center the frame.

---

# 14. CAPTION ENGINE

Captions should be treated as editorial design, not transcription output.

Requirements:

- word-level timing
- readable line lengths
- safe margins
- platform-aware positioning
- speaker-aware styling when useful
- emphasize high-impact words
- synchronize emphasis with speech
- avoid covering faces
- avoid covering important UI

Caption emphasis can be driven by:

- semantic importance
- emotional emphasis
- nouns
- numbers
- claims
- contrast words
- punchlines

Do not highlight every word.

---

# 15. EDITING GRAMMAR

Do not use one generic template for every clip.

Select an editing grammar based on content.

Possible grammars:

```text
minimal_talking_head
fast_paced_social
podcast_split
screen_demo
storytelling
educational
news
product_demo
high_energy
cinematic
```

Each grammar controls:

- cut frequency
- zoom frequency
- caption behavior
- overlay behavior
- transition behavior
- SFX
- B-roll opportunities
- CTA behavior

---

# 16. FRAME-BY-FRAME STYLE ANALYSIS

When a reference video is provided, reverse-engineer its editing grammar.

Inspect the reference frame by frame around:

- cuts
- camera movement
- zooms
- overlays
- captions
- graphics
- B-roll
- sound effects
- transitions
- CTA
- pacing

Create a style specification:

```json
{
  "cut_rhythm": {},
  "caption_style": {},
  "zoom_rules": {},
  "transition_rules": {},
  "overlay_rules": {},
  "sfx_rules": {},
  "color_system": {},
  "typography": {},
  "cta": {}
}
```

The goal is to learn the editing language, not blindly copy another creator's branded assets.

Save successful style specifications as reusable templates.

---

# 17. PROGRAMMATIC EDITING

Prefer code-driven rendering.

Recommended stack where compatible with the existing project:

- Remotion for composition
- FFmpeg for media operations
- React/TypeScript for visual components
- MediaPipe/OpenCV for tracking
- WhisperX/faster-whisper for transcription

Create reusable compositions instead of hard-coded one-off edits.

Example structure:

```text
.claude/
  skills/
    clippyme-intelligent-clipping/
      SKILL.md

src/
  clip-brain/
    ingest/
    transcription/
    context/
    discovery/
    scoring/
    hooks/
    planning/
    reframing/
    captions/
    qa/

  compositions/
    TalkingHead.tsx
    Podcast.tsx
    ScreenDemo.tsx
    Story.tsx
    Educational.tsx

  tracking/
  rendering/
  feedback/
```

Adapt to the existing ClippyMe architecture instead of replacing working code unnecessarily.

---

# 18. EDIT PLAN

Before rendering, create an explicit edit plan.

Example:

```json
{
  "clip": {
    "start": 306.9,
    "end": 354.2
  },
  "format": "9:16",
  "grammar": "educational",
  "reframe": {
    "mode": "active_speaker",
    "smooth": true
  },
  "captions": {
    "style": "brand_default",
    "highlight": "semantic"
  },
  "overlays": [
    {
      "type": "keyword_card",
      "at": 11.4,
      "duration": 2.8
    }
  ],
  "sfx": [],
  "cta": {
    "enabled": true,
    "type": "watch_full"
  }
}
```

The renderer should execute this plan.

---

# 19. B-ROLL AND OVERLAYS

First preference:
1. use actual source footage
2. use screenshots/product assets
3. use user-provided media
4. use licensed stock/media integrations
5. optionally generate AI B-roll if configured

Never hallucinate factual visual evidence.

If B-roll is unavailable, use motion graphics, typography, crops, screenshots, or source footage rather than inventing misleading footage.

---

# 20. PLATFORM ADAPTATION

Generate platform-specific versions where requested.

Support:

- TikTok
- Instagram Reels
- YouTube Shorts
- LinkedIn
- Threads

Platform adaptation can modify:

- duration
- title
- caption
- CTA
- safe zones
- pacing
- metadata

The underlying content should remain the same unless the user requests a platform-specific editorial variant.

---

# 21. HUMAN-IN-THE-LOOP

Do not force full autonomy.

At important checkpoints, offer:

```text
I found 12 strong moments.

Top 5:
1. ...
2. ...
3. ...
4. ...
5. ...

Approve all / select / reject / regenerate
```

The user should be able to say:

- "make it punchier"
- "less zoom"
- "don't cut the sentence"
- "keep both speakers"
- "more captions"
- "use my style"
- "make it LinkedIn-friendly"

The system should update the edit plan, not start from scratch.

---

# 22. QUALITY ASSURANCE

Before returning a clip, automatically check:

### Content
- Does the clip make sense without the full video?
- Is the payoff present?
- Did the cut remove necessary context?
- Are statements accurately represented?

### Audio
- no clipping
- no accidental silence
- no awkward word cuts
- no severe background noise problems

### Video
- correct aspect ratio
- no black bars unless intentional
- face not cropped incorrectly
- no jittery tracking
- important UI not hidden

### Captions
- correct words
- correct timing
- no overflow
- no unsafe placement
- emphasis is sensible

### Editing
- no accidental duplicate frames
- no broken overlays
- transitions complete
- SFX aligned
- CTA not covering content

Return a QA report with the render.

---

# 23. FEEDBACK LOOP

Every user action is training data.

Track:

```json
{
  "clip_id": "",
  "approved": true,
  "edited_by_user": true,
  "changes": [],
  "reason": "",
  "published": true,
  "performance": {
    "views": 0,
    "retention": 0,
    "shares": 0,
    "comments": 0
  }
}
```

Learn preferences such as:

- preferred clip duration
- preferred pacing
- preferred caption density
- preferred zoom frequency
- preferred clip types
- preferred hooks
- preferred CTA
- preferred visual style

Do not silently change core behavior based on a single example.

Use accumulated evidence.

---

# 24. "CLIP INTELLIGENCE" UI

If the product UI supports it, show more than timestamps.

For every candidate:

```text
87/100  Strong candidate

TYPE
Contrarian / Lesson

WHY IT WORKS
• Strong opening claim
• Clear payoff
• High information density
• Standalone context

BEST ANGLE
"Most people are approaching X completely wrong."

TIME
05:07 → 05:51

VISUAL
Talking head → active speaker crop

EDIT
Fast educational

VARIANTS
[Curiosity] [Contrarian] [How-to]
```

This turns ClippyMe from a cutter into an editorial assistant.

---

# 25. BATCH MODE

For long videos:

1. analyze transcript in chunks
2. build candidates in parallel
3. deduplicate overlapping moments
4. rank globally
5. select a diverse set

Avoid returning ten clips that all communicate the same idea.

Diversity should include:

- topic
- emotional tone
- clip type
- duration
- audience angle

---

# 26. DEDUPLICATION

Two candidates are duplicates if they have substantially overlapping:

- timestamps
- claims
- narrative purpose

Keep the stronger candidate.

If two versions have different hooks and audiences, retain both as variants rather than duplicate clips.

---

# 27. COMMAND BEHAVIOR

When the user says:

### "Clip this"
Do full semantic analysis before selecting the clip.

### "Find the best clips"
Return ranked candidates with reasons.

### "Make 5 clips"
Find at least 5 strong, diverse moments.

### "Make this viral"
Do not promise virality. Optimize hook, clarity, retention, pacing, and platform fit.

### "Use this editing style"
Analyze the reference video first, then generate a style specification.

### "Make it like this video"
Extract editing grammar rather than copying proprietary/branded assets.

### "Make it shorter"
Preserve the narrative core and payoff while removing redundancy.

### "Make it punchier"
Tighten setup, improve opening, reduce dead air, increase visual emphasis, and preserve meaning.

---

# 28. IMPORTANT PRODUCT DIFFERENTIATOR

Generic AI clippers mostly answer:

> "Which parts of this video look clip-worthy?"

ClippyMe should answer:

> "Which ideas inside this video deserve distribution, why, who are they for, what is the strongest angle, and how should they be edited?"

That is the product moat.

---

# 29. IMPLEMENTATION PRINCIPLES

1. Inspect the existing ClippyMe codebase before changing architecture.
2. Reuse working components.
3. Add capabilities incrementally.
4. Keep analysis artifacts inspectable and debuggable.
5. Make model decisions deterministic where possible.
6. Store timestamps as source-of-truth.
7. Never edit the original source file destructively.
8. Keep rendering reproducible.
9. Prefer local/open-source models when practical.
10. Make external APIs optional.
11. Never claim a clip will go viral.
12. Never fabricate source content.
13. Never remove context solely to make a clip shorter.
14. Optimize for actual audience value, not merely visual activity.

---

# 30. FIRST IMPLEMENTATION TASK

When this skill is installed in an existing ClippyMe repository:

### Step 1
Inspect the repository.

Identify:
- current clipping flow
- transcript implementation
- current clip selection logic
- render pipeline
- UI
- database/schema
- existing AI calls
- existing FFmpeg/Remotion code
- existing tests

### Step 2
Write a short architecture assessment before modifying code.

### Step 3
Implement the smallest vertical slice:

```text
video
→ transcript
→ semantic candidate discovery
→ scoring
→ context expansion
→ clip plan
→ existing renderer
→ output
```

### Step 4
Add frame/face analysis.

### Step 5
Add adaptive captions.

### Step 6
Add editing grammars.

### Step 7
Add reference-style analysis.

### Step 8
Add feedback learning.

Do not attempt the entire system in one massive refactor.

---

# SUCCESS CRITERIA

ClippyMe is successfully upgraded when a user can give it a long-form video and say:

> "Find me the best clips."

And the system can return not just timestamps, but:

- what the moment is about
- why it is strong
- what type of clip it is
- how much context it needs
- the best hook/angle
- the recommended duration
- the visual layout
- the editing grammar
- the caption strategy
- a rendered preview
- a confidence/quality report

The final experience should feel less like "AI found a timestamp" and more like:

> **"An editor watched my entire video, understood it, and prepared the best content from it."**
