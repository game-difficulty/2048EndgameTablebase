# Live picture-in-picture acceptance

Open the normal live root or `/rooms/ai-classic`; no query parameter is required.
Use the room-header Mini player button. Automatic mode prefers Document PiP;
the video fallback uses a local muted 960x540 canvas stream. No video is uploaded
and no second WebSocket is created. The old `?pip=1` URL remains compatible.
Diagnostics and test labels were removed for the production launch.

## Device procedure

1. Open the normal URL, then click Mini player in the room header. Try Automatic,
   Room window and Video fallback independently.
2. Observe changing boards in the floating window. Switch to another tab,
   then another app. Remain away for at least four minutes.
3. Confirm the floating board continues changing, including a new game.
4. In Document PiP, resize the window and switch main/equal layouts: the stage
   keeps its 16:9 ratio and fits the available space. Close it and verify the
   original content node, selected AI, and layout are restored.
5. Briefly disconnect networking while PiP is open, then restore it; check
   reconnect and the next board. Close PiP using its system close control.
6. Confirm the video source/track is released; without PiP the usual three-minute
   background timeout resumes. Repeat start/close and navigate away mid-start.

Target matrix: desktop Chrome/Edge; Android Chrome, Samsung Internet and Baidu.
Test background tabs and switching apps separately. Lock-screen playback is
not promised. Missing video PiP/canvas capture is reported, not replaced with
another kind of window. Android emulation is not an Android device test.

## Developer inspection

Use browser tooling to inspect video playback and media track cleanup when needed.
Increasing decoded frames is useful evidence, but a human must still check the
system floating window on each device. Some PiP environments keep the opener
document visible after switching tabs; that does not simulate mobile freezing.

## Automated checks

`node --test tests/livePipPolicy.test.js tests/liveLayout.test.js`

For browser smoke tests use synthetic snapshot messages through Playwright's
WebSocket routing. Native PiP should actually open; do not mock its API.
Disable support separately to verify the explanatory failure state. Compare
video playback counters over time and sample decoded video pixels.
Application timeout checks may inject document.hidden=true, but explicitly
label that simulation: it does not reproduce mobile OS throttling.

The production launch updates only the live frontend. Physical mobile device
behavior still needs device-specific checks; desktop emulation is not certification.

## 2026-09-18 smoke results

- Windows Chromium 151: real video PiP request succeeded (headed and headless).
- Static-build headed test, synthetic snapshots on socket heartbeat, injected
  document.hidden=true: 190 seconds, 19 hidden board updates, 209 decoded video
  frames, video time 189 seconds, maximum observed timer gap 1015ms. No three-minute
  disconnect. This is an application-policy test, not mobile throttling evidence.
- Decoded 480x480 video had nonblank, multicolor pixels.
- Closing PiP stopped tracks; unmounting the app closed PiP, ended the track and
  cleared srcObject. Unsupported API check displayed an error.
- Android Chrome/Samsung Internet/Baidu physical-device tests remain pending.
