import { beforeEach, describe, expect, test } from 'vitest';
import {
  collectionsKey, loadCollections, saveCollections, emptyStore,
  clipKeyFor, clipRefFromClip, isFavorite, toggleFavorite, favoriteList,
  createCollection, renameCollection, deleteCollection,
  addClipToCollection, removeClipFromCollection, collectionsContaining,
} from './collections';

const UID = 'user-123';
const refA = { key: 'job1:0', jobId: 'job1', index: 0, title: 'Clip 1', score: 80, duration: 30, thumb: null };
const refB = { key: 'job1:1', jobId: 'job1', index: 1, title: 'Clip 2', score: 60, duration: 20, thumb: null };

beforeEach(() => { window.localStorage.clear(); });

describe('collections store', () => {
  test('storage key is namespaced per user', () => {
    expect(collectionsKey('abc')).toBe('nugget_collections_abc');
    expect(collectionsKey(null)).toBe('nugget_collections_anon');
  });

  test('loads empty store when nothing saved; isolates users', () => {
    expect(loadCollections(UID)).toEqual(emptyStore());
    saveCollections(UID, toggleFavorite(emptyStore(), refA));
    expect(favoriteList(loadCollections(UID))).toHaveLength(1);
    expect(favoriteList(loadCollections('other'))).toHaveLength(0);
  });

  test('corrupt JSON degrades to an empty store', () => {
    window.localStorage.setItem(collectionsKey(UID), 'not-json{{{');
    expect(loadCollections(UID)).toEqual(emptyStore());
  });

  test('clip keys are stable per job+index', () => {
    expect(clipKeyFor('job1', 2)).toBe('job1:2');
  });

  test('clipRefFromClip snapshots renderable fields', () => {
    const clip = { video_title_for_youtube_short: 'Hot take', viral_score: 91, start: 4, end: 34 };
    const ref = clipRefFromClip('job9', clip, 3);
    expect(ref).toMatchObject({ key: 'job9:3', jobId: 'job9', index: 3, title: 'Hot take', score: 91, duration: 30 });
  });

  test('favorite toggle adds and removes', () => {
    let s = emptyStore();
    s = toggleFavorite(s, refA);
    expect(isFavorite(s, refA.key)).toBe(true);
    s = toggleFavorite(s, refA);
    expect(isFavorite(s, refA.key)).toBe(false);
  });

  test('favorites persist across loads', () => {
    saveCollections(UID, toggleFavorite(emptyStore(), refA));
    expect(isFavorite(loadCollections(UID), refA.key)).toBe(true);
  });

  test('create / rename / delete collections', () => {
    let s = emptyStore();
    const r1 = createCollection(s, '  Campaign A  ');
    expect(r1.id).toBeTruthy();
    s = r1.store;
    expect(s.collections[0].name).toBe('Campaign A');
    s = renameCollection(s, r1.id, 'Campaign B');
    expect(s.collections[0].name).toBe('Campaign B');
    // empty/blank names are rejected
    expect(createCollection(s, '   ').id).toBeNull();
    s = deleteCollection(s, r1.id);
    expect(s.collections).toHaveLength(0);
  });

  test('add / remove clips from collections', () => {
    let s = createCollection(emptyStore(), 'Picks').store;
    const id = s.collections[0].id;
    s = addClipToCollection(s, id, refA);
    s = addClipToCollection(s, id, refB);
    expect(Object.keys(s.collections[0].clips)).toHaveLength(2);
    expect(collectionsContaining(s, refA.key)).toEqual([id]);
    s = removeClipFromCollection(s, id, refA.key);
    expect(collectionsContaining(s, refA.key)).toEqual([]);
    expect(Object.keys(s.collections[0].clips)).toHaveLength(1);
  });

  test('collections persist across loads', () => {
    let s = createCollection(emptyStore(), 'Saved').store;
    s = addClipToCollection(s, s.collections[0].id, refA);
    saveCollections(UID, s);
    const reloaded = loadCollections(UID);
    expect(reloaded.collections[0].clips[refA.key].title).toBe('Clip 1');
  });
});
