/**
 * Task 17 — chat proposal plumbing (React-free).
 *
 * Ordinary main-chat edits never mutate the project directly. They go
 * through two explicit steps:
 *
 *  1. requestChatProposal: calls the EXISTING edit-ai route (mode=patch)
 *     and returns a proposal bound to the captured {jobId, clipIndex}
 *     identity and the patch's base version. It throws on an unusable
 *     patch and NEVER saves — proposal generation is not saved success.
 *  2. applyChatProposal: applies the proposal through the Task 16 project
 *     controller with optimistic concurrency (expectedVersion = the
 *     captured base version). It refuses with code STALE_SELECTION unless
 *     the live controller's identity still matches the captured one.
 *
 * Both are pure with respect to React; the API surface is injected so
 * tests can run them against the real controller with a fake backend.
 */

/**
 * Ask the backend for a validated patch (mode=patch — returned, not saved).
 * Resolves { identity, baseVersion, explanation, patch, project }.
 * Throws when the response has no usable project (no version to bind to).
 */
export async function requestChatProposal({ editAiPatchFn, jobId, clipIndex, instruction }) {
  if (typeof editAiPatchFn !== 'function') {
    throw new Error('requestChatProposal: editAiPatchFn is required');
  }
  const identity = { jobId, clipIndex };
  const r = await editAiPatchFn(jobId, clipIndex, instruction);
  const project = r?.project ?? null;
  if (!project || typeof project !== 'object' || project.version == null) {
    throw new Error(
      'The AI edit came back unusable (no project version) — nothing was changed. Try rephrasing the edit.'
    );
  }
  return {
    identity,
    baseVersion: project.version,
    explanation: r?.explanation || 'Patch ready.',
    patch: r?.patch ?? null,
    project,
  };
}

/**
 * Apply a proposal through the shared project controller.
 *
 * Identity gate first: the controller's (jobId, clipIndex) must match the
 * identity captured when the proposal was generated, otherwise the apply
 * is refused with code STALE_SELECTION and ZERO api calls are made
 * (a wrong-target apply must be impossible, not just unlikely).
 *
 * Otherwise the proposal goes through controller.applyEdit with
 * expectedVersion = the captured base version: a newer manual save makes
 * the server answer 409 and the project is never overwritten.
 */
export async function applyChatProposal({ controller, identity, baseVersion, project, label }) {
  if (
    !controller ||
    !identity ||
    controller.jobId !== identity.jobId ||
    controller.clipIndex !== identity.clipIndex
  ) {
    const e = new Error(
      'The selection changed since this proposal was made — nothing was applied.'
    );
    e.code = 'STALE_SELECTION';
    throw e;
  }
  if (!project || typeof project !== 'object') {
    throw new Error('Nothing to apply: the proposal has no project.');
  }
  return controller.applyEdit(
    label || 'chat edit',
    () => project,
    { expectedVersion: baseVersion }
  );
}
