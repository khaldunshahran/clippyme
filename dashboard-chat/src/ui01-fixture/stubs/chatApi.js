/**
 * UI-01 offline fixture — stubbed ../api/chatApi.js. No network.
 * Covers every name imported anywhere in the app (many importers never
 * render in the fixture, but the modules must resolve).
 */
export const editAiPatch = async () => null;
export const scheduleClipPost = async () => ({});
export const getZernioAccounts = async () => ({ accounts: [] });
export const zernioTargets = () => [];
export const getJobClips = async () => [];
export const normClip = (c) => c;
export const fmtDur = (s) => `${s}s`;
export const fmtEta = (s) => `${s}s`;
export const duplicateClip = async () => null;
export const editAiTrim = async () => null;
export const exportClipXml = async () => null;
export const extractUrl = (t) => t;
export const getHistory = async () => [];
export const getProgress = async () => ({});
export const publishClip = async () => null;
export const steerJob = async () => null;
export const submitJob = async () => ({});
export const upscaleClip = async () => null;
export const validateUrl = async () => ({ ok: true });
