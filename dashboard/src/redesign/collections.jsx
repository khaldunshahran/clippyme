// Nugget — Collections: favorites + custom clip collections.
//
// FavoritesView and CollectionsView render from the clipRef snapshots in
// lib/collections.js (title/score/duration), so they work even when the
// source job isn't loaded. Opening a clip hands its ref to the parent, which
// restores the job and jumps to the clip.
import { useEffect, useRef, useState } from 'react';
import { Icon, Btn, Badge, Panel } from './primitives';
import { Hero } from './chrome';
import {
  collectionsContaining, addClipToCollection, createCollection,
  removeClipFromCollection,
} from '../lib/collections';

/* ---------------- card controls (live clip cards) ---------------- */

/** Heart toggle for a clip card. Controlled: parent owns the store. */
export function FavoriteButton({ active, onToggle, title }) {
  return (
    <button
      type="button"
      className={'mini fav-btn' + (active ? ' on' : '')}
      title={active ? 'Remove from favorites' : 'Add to favorites'}
      aria-label={active ? `Remove ${title} from favorites` : `Add ${title} to favorites`}
      aria-pressed={active}
      onClick={(e) => { e.stopPropagation(); onToggle(); }}
    >
      <Icon n="heart" />
    </button>
  );
}

/**
 * "Add to collection" menu for a clip card. Popover with the user's
 * collections (checkbox behaviour) + inline create. Closes on outside click.
 */
export function CollectionMenu({ clipRef, store, onStoreChange, title }) {
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const wrapRef = useRef(null);
  const inCols = new Set(collectionsContaining(store, clipRef.key));
  const cols = store?.collections || [];

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => { if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open ]);

  const commitCreate = () => {
    const { store: next, id } = createCollection(store, name);
    if (id) {
      onStoreChange(addClipToCollection(next, id, clipRef));
      pushSaved();
    }
    setName('');
    setCreating(false);
  };
  const pushSaved = () => setOpen(false);

  return (
    <div className="colmenu-wrap" ref={wrapRef} style={{ position: 'relative' }}>
      <button
        type="button"
        className="mini"
        title="Add to collection"
        aria-label={`Add ${title} to a collection`}
        aria-expanded={open}
        onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}
      >
        <Icon n="folder-plus" />
      </button>
      {open && (
        <div className="colmenu fade-in" role="menu" aria-label="Collections">
          {cols.length === 0 && !creating && (
            <div className="colmenu-empty">No collections yet — create one below.</div>
          )}
          {cols.map((c) => {
            const has = inCols.has(c.id);
            return (
              <button
                key={c.id}
                type="button"
                role="menuitemcheckbox"
                aria-checked={has}
                className="colmenu-item"
                onClick={(e) => {
                  e.stopPropagation();
                  onStoreChange(
                    has ? removeClipFromCollection(store, c.id, clipRef.key)
                         : addClipToCollection(store, c.id, clipRef)
                  );
                }}
              >
                <Icon n={has ? 'check-square' : 'square'} />
                <span className="colmenu-name">{c.name}</span>
                <span className="colmenu-count">{Object.keys(c.clips || {}).length}</span>
              </button>
            );
          })}
          {creating ? (
            <div className="colmenu-create">
              <input
                className="input-field" autoFocus value={name} placeholder="Collection name"
                aria-label="New collection name"
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') commitCreate(); if (e.key === 'Escape') setCreating(false); }}
              />
              <Btn variant="primary" size="sm" onClick={commitCreate} disabled={!name.trim()}>Create</Btn>
            </div>
          ) : (
            <button type="button" className="colmenu-item colmenu-new" onClick={() => setCreating(true)}>
              <Icon n="plus" />
              <span>New collection</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------------- shared clip-ref card ---------------- */

function RefCard({ clipRef, onOpen, trailing, favActive, onToggleFav }) {
  return (
    <div className="card raised-sm ref-card">
      <div className="ref-thumb" aria-hidden="true">
        <Icon n="play" />
        <span className="ref-score">{clipRef.score > 0 ? clipRef.score : '—'}</span>
      </div>
      <div className="ref-body">
        <div className="ref-title" title={clipRef.title}>{clipRef.title}</div>
        <div className="ref-meta">
          {clipRef.duration != null && `${clipRef.duration}s`}
          {clipRef.jobId ? '' : ' · source unavailable'}
        </div>
      </div>
      <div className="ref-actions">
        {onToggleFav && <FavoriteButton active={favActive} onToggle={() => onToggleFav(clipRef)} title={clipRef.title} />}
        {clipRef.jobId ? (
          <Btn variant="ghost" size="sm" icon="arrow-left" onClick={() => onOpen(clipRef)}>
            Open
          </Btn>
        ) : (
          <span className="od">unavailable</span>
        )}
        {trailing}
      </div>
    </div>
  );
}

/* ---------------- favorites view ---------------- */

export function FavoritesView({ store, onOpenClip, onToggleFav }) {
  const favs = Object.values(store?.favorites || {}).sort((a, b) => (b.score || 0) - (a.score || 0));
  return (
    <div className="container fade-in">
      <div className="results-head" style={{ marginBottom: 18 }}>
        <h2>Favorites</h2>
        <Badge tone="out">{favs.length} clip{favs.length === 1 ? '' : 's'}</Badge>
      </div>
      {favs.length === 0 ? (
        <div className="empty">
          <div className="ei"><Icon n="heart" /></div>
          <h3>No favorites yet</h3>
          <p>Tap the heart on any clip card and it will wait for you here.</p>
        </div>
      ) : (
        <div className="ref-grid">
          {favs.map((r) => (
            <RefCard key={r.key} clipRef={r} onOpen={onOpenClip} favActive onToggleFav={onToggleFav} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------------- collections view ---------------- */

function CollectionCard({ collection, onOpenClip, onRename, onDelete }) {
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(collection.name);
  const clips = Object.values(collection.clips || {});
  const commit = () => {
    if (name.trim() && name.trim() !== collection.name) onRename(collection.id, name.trim());
    setRenaming(false);
  };
  return (
    <Panel className="collection-card">
      <div className="collection-head">
        <span className="collection-ic" aria-hidden="true"><Icon n="folder" /></span>
        {renaming ? (
          <input
            className="input-field" autoFocus value={name} aria-label="Collection name"
            onChange={(e) => setName(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') setRenaming(false); }}
            style={{ flex: 1 }}
          />
        ) : (
          <h4 className="collection-name" title={collection.name}>{collection.name}</h4>
        )}
        <Badge tone="out">{clips.length}</Badge>
        <button type="button" className="mini" title="Rename" aria-label={`Rename ${collection.name}`}
          onClick={() => { setName(collection.name); setRenaming(true); }}>
          <Icon n="pencil" />
        </button>
        <button type="button" className="mini" title="Delete collection" aria-label={`Delete ${collection.name}`}
          onClick={() => { if (window.confirm(`Delete “${collection.name}”? The clips themselves are kept.`)) onDelete(collection.id); }}>
          <Icon n="trash-2" />
        </button>
      </div>
      {clips.length === 0 ? (
        <div className="od">Empty — add clips from any clip card menu.</div>
      ) : (
        <div className="ref-grid">
          {clips.map((r) => (
            <RefCard key={r.key} clipRef={r} onOpen={onOpenClip} />
          ))}
        </div>
      )}
    </Panel>
  );
}

export function CollectionsView({ store, onOpenClip, onCreate, onRename, onDelete }) {
  const cols = store?.collections || [];
  const [name, setName] = useState('');
  return (
    <div className="container fade-in">
      <div className="results-head" style={{ marginBottom: 18 }}>
        <h2>Collections</h2>
        <Badge tone="out">{cols.length} collection{cols.length === 1 ? '' : 's'}</Badge>
      </div>
      <Panel className="collection-new" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            className="input-field" value={name} placeholder="Name a new collection…"
            aria-label="New collection name" style={{ flex: '1 1 220px' }}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && name.trim()) { onCreate(name.trim()); setName(''); } }}
          />
          <Btn variant="primary" size="sm" icon="plus" disabled={!name.trim()}
            onClick={() => { onCreate(name.trim()); setName(''); }}>
            Create collection
          </Btn>
        </div>
      </Panel>
      {cols.length === 0 ? (
        <div className="empty">
          <div className="ei"><Icon n="folder" /></div>
          <h3>No collections yet</h3>
          <p>Group clips by campaign, client, vibe — whatever you need. Free for everyone.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: 16 }}>
          {cols.map((c) => (
            <CollectionCard key={c.id} collection={c} onOpenClip={onOpenClip} onRename={onRename} onDelete={onDelete} />
          ))}
        </div>
      )}
    </div>
  );
}
