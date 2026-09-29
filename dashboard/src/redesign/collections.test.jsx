import { describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FavoritesView, CollectionsView, FavoriteButton, CollectionMenu } from './collections';
import { emptyStore, toggleFavorite, createCollection, addClipToCollection } from '../lib/collections';

const refA = { key: 'job1:0', jobId: 'job1', index: 0, title: 'Clip 1', score: 80, duration: 30, thumb: null };

describe('FavoriteButton', () => {
  test('toggles with honest labels', () => {
    const onToggle = vi.fn();
    const { rerender } = render(<FavoriteButton active={false} onToggle={onToggle} title="Clip 1" />);
    fireEvent.click(screen.getByRole('button', { name: 'Add Clip 1 to favorites' }));
    expect(onToggle).toHaveBeenCalledTimes(1);
    rerender(<FavoriteButton active onToggle={onToggle} title="Clip 1" />);
    expect(screen.getByRole('button', { name: 'Remove Clip 1 from favorites' }).getAttribute('aria-pressed')).toBe('true');
  });
});

describe('FavoritesView', () => {
  test('empty state guides the user', () => {
    render(<FavoritesView store={emptyStore()} onOpenClip={() => {}} onToggleFav={() => {}} />);
    expect(screen.getByText('No favorites yet')).toBeTruthy();
  });

  test('lists favorited clips and opens them', () => {
    const store = toggleFavorite(emptyStore(), refA);
    const onOpenClip = vi.fn();
    render(<FavoritesView store={store} onOpenClip={onOpenClip} onToggleFav={() => {}} />);
    expect(screen.getByText('Clip 1')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Open' }));
    expect(onOpenClip).toHaveBeenCalledWith(refA);
  });
});

describe('CollectionsView', () => {
  test('empty state + create flow', () => {
    const onCreate = vi.fn();
    render(<CollectionsView store={emptyStore()} onOpenClip={() => {}} onCreate={onCreate} onRename={() => {}} onDelete={() => {}} />);
    expect(screen.getByText('No collections yet')).toBeTruthy();
    fireEvent.change(screen.getByLabelText('New collection name'), { target: { value: 'Launch' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create collection' }));
    expect(onCreate).toHaveBeenCalledWith('Launch');
  });

  test('rename and delete with confirmation', () => {
    // one collection with one clip
    let s = createCollection(emptyStore(), 'Old').store;
    const id = s.collections[0].id;
    s = addClipToCollection(s, id, refA);
    const onRename = vi.fn();
    const onDelete = vi.fn();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    render(<CollectionsView store={s} onOpenClip={() => {}} onCreate={() => {}} onRename={onRename} onDelete={onDelete} />);
    expect(screen.getByText('Old')).toBeTruthy();
    expect(screen.getByText('Clip 1')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Rename Old' }));
    const input = screen.getByLabelText('Collection name');
    fireEvent.change(input, { target: { value: 'New' } });
    fireEvent.blur(input);
    expect(onRename).toHaveBeenCalledWith(id, 'New');
    fireEvent.click(screen.getByRole('button', { name: 'Delete Old' }));
    expect(onDelete).toHaveBeenCalledWith(id);
    confirm.mockRestore();
  });
});

describe('CollectionMenu', () => {
  test('adds a clip to a collection from the card menu', () => {
    let s = createCollection(emptyStore(), 'Picks').store;
    const id = s.collections[0].id;
    const onStoreChange = vi.fn((next) => { s = next; });
    const { rerender } = render(
      <CollectionMenu clipRef={refA} store={s} onStoreChange={onStoreChange} title="Clip 1" />
    );
    fireEvent.click(screen.getByRole('button', { name: 'Add Clip 1 to a collection' }));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: /Picks/ }));
    expect(onStoreChange).toHaveBeenCalledTimes(1);
    expect(s.collections[0].clips[refA.key].title).toBe('Clip 1');
    // menu stays open for multi-toggle; rerender with the saved store shows it checked
    rerender(<CollectionMenu clipRef={refA} store={s} onStoreChange={onStoreChange} title="Clip 1" />);
    expect(screen.getByRole('menuitemcheckbox', { name: /Picks/ }).getAttribute('aria-checked')).toBe('true');
    expect(id).toBeTruthy();
  });
});
