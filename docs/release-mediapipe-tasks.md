# Provisioned MediaPipe Tasks compatibility

MediaPipe wheels without `solutions` may use local official models through
`CLIPPYME_FACE_DETECTOR_MODEL` and `CLIPPYME_FACE_LANDMARKER_MODEL`.
Unconfigured installations retain their existing fallback. No model is
downloaded by a running worker. Missing configured files fail clearly.

The adapter uses IMAGE mode for independently cropped faces, exposes the
existing normalized detection/landmark contracts, and stores no images.
Configured model content hashes and the adapter version bind framing-plan
identity; changing either model invalidates cached plans. Paths themselves
do not affect identity. Provision and verify models before starting workers,
keep them immutable for the process lifetime, and restart workers after a
model/configuration change so lazy detector instances cannot retain old bytes.

Release validation used official model version 1:

- BlazeFace full range float16 detector: SHA256
  `3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b`.
- Face landmarker float16 task: SHA256
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

Real inference confirms runtime compatibility. It does not establish that
mouth motion identifies the correct speaker in moving crowds; that remains
a real-footage editorial evaluation requirement.
