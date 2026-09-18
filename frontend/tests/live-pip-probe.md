# Experimental live picture-in-picture

Open `/live/?pip=1` locally (or `/?pip=1` on the live subdomain).
The normal URL has no experimental UI. Nothing is published or recorded: a
480x480 canvas feeds a local muted video MediaStream. No extra WebSocket is
created. This prototype redraws complete boards, without move/pop animations.

## Device procedure

1. Open the experimental URL, then click the square Experimental PiP icon to
   the right of Copy board. Record any API error.
2. Observe changing boards in the floating window. Switch to another tab,
   then another app. Remain away for at least four minutes.
3. Confirm the floating board continues changing, including a new game.
4. Return and expand PiP diagnostics. Record browser/version, lastUpdate,
   hiddenUpdates, maxHiddenTickGapMs, videoFrames and videoTime.
5. Briefly disconnect networking while PiP is open, then restore it; check
   reconnect and the next board. Close PiP using its system close control.
6. Confirm the video source/track is released; without PiP the usual three-minute
   background timeout resumes. Repeat start/close and navigate away mid-start.

Target matrix: desktop Chrome/Edge; Android Chrome, Samsung Internet and Baidu.
Test background tabs and switching apps separately. Lock-screen playback is
not promised. Missing video PiP/canvas capture is reported, not replaced with
another kind of window. Android emulation is not an Android device test.

## What counters mean

`draws` counts canvas renders, not necessarily video frames presented by the OS.
`videoFrames` comes from getVideoPlaybackQuality, or null if unavailable.
`hiddenUpdates` counts distinct board revisions received while document.hidden
is true; some PiP environments keep the document visible even after switching
tabs. Increasing decoded video frames is useful evidence, but a human must
still check the system floating window on each device.

## Automated checks

`node --test tests/livePipPolicy.test.js tests/liveLayout.test.js`

For browser smoke tests use synthetic snapshot messages through Playwright's
WebSocket routing. Native PiP should actually open; do not mock its API.
Disable support separately to verify the explanatory failure state. Compare
video playback counters over time and sample decoded video pixels.
Application timeout checks may inject document.hidden=true, but explicitly
label that simulation: it does not reproduce mobile OS throttling.

No server changes, production deployment or Android device certification are
part of this prototype. First establish device viability before animations.

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
