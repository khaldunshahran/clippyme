import { test, expect, beforeEach } from 'vitest';
import {
  PRESET_KEYS,
  captureOpts,
  saveUserPreset,
  loadUserPresets,
  deleteUserPreset,
  setDefaultPreset,
  getDefaultPresetId,
  getDefaultPresetOpts,
  allPresets,
} from './presets.js';

beforeEach(() => {
  localStorage.clear();
});

test('PRESET_KEYS includes subtitle styling, banner, color grade, and duration', () => {
  expect(PRESET_KEYS).toContain('banner');
  expect(PRESET_KEYS).toContain('bannerPlatform');
  expect(PRESET_KEYS).toContain('bannerHandle');
  expect(PRESET_KEYS).toContain('subAlign');
  expect(PRESET_KEYS).toContain('subStroke');
  expect(PRESET_KEYS).toContain('subFontSize');
  expect(PRESET_KEYS).toContain('gradePreset');
  expect(PRESET_KEYS).toContain('durationMode');
});

test('captureOpts captures all customized subtitle and banner options', () => {
  const opts = {
    subtitles: true,
    subMode: 'karaoke',
    subPreset: 'hormozi_bold',
    subAlign: 'left',
    subFontSize: 42,
    subColor: '#FFFFFF',
    subStroke: '#000000',
    banner: false,
    bannerPlatform: 'kick',
    bannerHandle: 'teststreamer',
    gradePreset: 'warm_cinematic',
    url: 'https://youtube.com/watch?v=123', // url must not be captured in preset
  };

  const captured = captureOpts(opts);
  expect(captured.subtitles).toBe(true);
  expect(captured.subAlign).toBe('left');
  expect(captured.subFontSize).toBe(42);
  expect(captured.banner).toBe(false);
  expect(captured.bannerHandle).toBe('teststreamer');
  expect(captured.gradePreset).toBe('warm_cinematic');
  expect(captured.url).toBeUndefined();
});

test('saveUserPreset persists preset to localStorage and can be loaded or set as default', () => {
  const customOpts = {
    subtitles: true,
    subPreset: 'mrbeast',
    banner: false,
    gradePreset: 'vivid_pop',
  };

  const preset = saveUserPreset('My Viral Preset', customOpts);
  expect(preset.title).toBe('My Viral Preset');
  expect(preset.opts.subPreset).toBe('mrbeast');
  expect(preset.opts.banner).toBe(false);

  const list = loadUserPresets();
  expect(list.length).toBe(1);
  expect(list[0].id).toBe(preset.id);

  setDefaultPreset(preset.id);
  expect(getDefaultPresetId()).toBe(preset.id);

  const defaultOpts = getDefaultPresetOpts();
  expect(defaultOpts).toMatchObject({
    subPreset: 'mrbeast',
    banner: false,
    gradePreset: 'vivid_pop',
  });

  expect(allPresets().length).toBeGreaterThan(0);

  deleteUserPreset(preset.id);
  expect(loadUserPresets().length).toBe(0);
});
