import { describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { deriveProjects, projectOpenJob, ProjectsView } from './projects';

const H = [
  { jobId: 'j1', status: 'complete', timestamp: 1000, source: 'https://youtube.com/watch?v=abc', sourceType: 'url', clipCount: 5, title: 'Video A' },
  { jobId: 'j2', status: 'complete', timestamp: 2000, source: 'https://youtube.com/watch?v=abc&t=10s', sourceType: 'url', clipCount: 3, title: 'Video A' },
  { jobId: 'j3', status: 'error', timestamp: 3000, source: 'talk.mp4', sourceType: 'file', clipCount: 0, title: 'Talk' },
];

describe('deriveProjects', () => {
  test('groups re-runs of the same source into one project', () => {
    const projects = deriveProjects(H);
    expect(projects).toHaveLength(2);
    const a = projects.find((p) => p.id.includes('youtube.com'));
    expect(a.jobCount).toBe(2);
    expect(a.clipCount).toBe(8);
    expect(a.title).toBe('Video A');
  });

  test('sorts newest first and rolls status up', () => {
    const projects = deriveProjects(H);
    expect(projects[0].source).toBe('talk.mp4');
    expect(projects[0].status).toBe('error');
  });

  test('empty history → empty projects', () => {
    expect(deriveProjects([])).toEqual([]);
    expect(deriveProjects(null)).toEqual([]);
  });
});

describe('projectOpenJob', () => {
  test('prefers the newest completed job', () => {
    const [p] = deriveProjects(H.filter((h) => h.jobId !== 'j3'));
    expect(projectOpenJob(p).jobId).toBe('j2');
  });
});

describe('ProjectsView', () => {
  test('renders project cards with counts and opens on click', () => {
    const onSelect = vi.fn();
    render(<ProjectsView projects={deriveProjects(H)} onSelect={onSelect} />);
    expect(screen.getByText('Video A')).toBeTruthy();
    expect(screen.getByText(/8 clips/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Open project Video A/ }));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  test('honest empty state mentions the backend when unreachable', () => {
    render(<ProjectsView projects={[]} backendReachable={false} />);
    expect(screen.getByText(/backend is unreachable/i)).toBeTruthy();
  });

  test('honest empty state guides to Create when backend state unknown', () => {
    render(<ProjectsView projects={[]} />);
    expect(screen.getByText(/Head to Create/i)).toBeTruthy();
  });
});
