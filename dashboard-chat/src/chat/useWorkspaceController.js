import { useCallback, useEffect, useMemo, useState } from 'react';
import { createProjectController } from './projectController.js';
import { onSessionReset } from '../api/sessionReset.js';
import {
  getClipProject,
  saveClipProject,
  composeProject,
  backfillClipProjects,
  listClipProjectVersions,
  editClipAI,
  uploadAudio,
} from '../api/realApi.js';

/**
 * Task 17 — the ONE project authority for the chat editing workspace.
 *
 * Lifted out of EditorView (Task 16) so the workspace panel, the manual
 * transcript/controls, main-chat proposals, undo/redo and export all share
 * exactly one controller per (jobId, clipIndex):
 *
 *  - creates exactly one createProjectController per identity (useMemo),
 *  - subscribes the snapshot, then load() + listVersions() on mount,
 *  - dispose() on identity change / unmount (terminal — late responses
 *    can never touch the next clip's state),
 *  - resetSession() on session reset, plus a resetKey so the transcript
 *    panel remounts and drops selection/editing state.
 *
 * Pass { disabled: true } to skip controller creation (EditorView's
 * Task-17 mode receives the controller as a prop instead).
 * With no jobId/clipIndex it returns a null controller and does nothing.
 */
export function useWorkspaceController({ jobId, clipIndex, disabled }) {
  const [snap, setSnap] = useState(null);
  const [toast, setToast] = useState(null);
  // Bumped on session reset so the TranscriptPanel remounts and drops its
  // selection / editing state along with the cleared project.
  const [resetKey, setResetKey] = useState(0);

  // Function-property timer (no useRef) so the hook stays usable under the
  // minimal coordinator-probe react harness.
  const pushToast = useCallback((msg) => {
    setToast(msg);
    clearTimeout(pushToast._t);
    pushToast._t = setTimeout(() => setToast(null), 2600);
    console.log('[editor]', msg);
  }, []);

  const enabled = !disabled && jobId != null && clipIndex != null;

  const controller = useMemo(
    () => {
      if (!enabled) return null;
      return createProjectController({
        jobId,
        clipIndex,
        api: {
          getClipProject,
          saveClipProject,
          composeProject,
          backfillClipProjects,
          listClipProjectVersions,
          editClipAI,
          uploadAudio,
        },
        onToast: pushToast,
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [enabled, jobId, clipIndex]
  );

  useEffect(() => {
    if (!controller) {
      setSnap(null);
      return undefined;
    }
    const unsubscribe = controller.subscribe(setSnap);
    controller.load().catch(() => {});
    controller.listVersions();
    return () => {
      unsubscribe();
      controller.dispose();
    };
  }, [controller]);

  // Session reset (sign-out or signed-in user change): drop the undo/redo
  // history and the cached project; the host remounts the transcript panel
  // via resetKey so no state leaks across users.
  useEffect(() => {
    if (!controller) return undefined;
    return onSessionReset(() => {
      controller.resetSession();
      setResetKey((k) => k + 1);
    });
  }, [controller]);

  return { controller, snap, resetKey, toast, pushToast };
}
