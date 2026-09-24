# Rendering compatibility

Both entry documents load `/compat/render-compat.js?v=2` before the app.
The classic ES5 detector tests actual cascade-layer application, color-mix,
OKLCH, aspect-ratio and registered properties. Modern engines do not request
compatibility CSS. Production builds traverse the main entry's Vite manifest,
including lazy pages and dialogs, then generate one fingerprinted stylesheet
from all reachable main-site CSS. This covers Tailwind utilities and Vue scoped
styles for game, trainer, tester, battle, minigames, leaderboards, replay,
settings, help, announcements, more and admin pages. The separate live entry
retains its existing structural fallback.
The smaller hand-maintained stylesheet is used in development and if the
generated asset cannot be loaded.

Diagnostics: `window.__RENDER_COMPAT__` records the reasons; the root
`data-css-compat` attribute is modern/loading/ready/failed. There is no user
override yet. A stylesheet network error does not hide the page indefinitely.

The build flattens cascade layers, supplies defaults for registered properties,
and lowers static CSS syntax. Runtime `color-mix()` using theme variables uses
the closest plain theme variable in compatibility mode. The small structural
stylesheet provides board and palette geometry where CSS syntax lowering is
not sufficient. The browser still needs the app's module runtime, CSS
variables, Grid and Flexbox.

Validation:
- `node --test tests/renderCompat.test.js tests/liveLayout.test.js`
- `npm run build`
- In browser automation, override a capability before navigation, then remove
  CSSLayerBlockRules to emulate ignored Tailwind layers. Check the trainer's
  palette, tester, settings theme swatches and help at wide and narrow widths.
- Verify the modern path never requests the generated stylesheet.
- Repeat these checks on the affected Baidu browser before production rollout.

Deployment must include `dist/compat/` along with both HTML entries. Deploy
the fingerprinted asset before `index.html`; the generated URL is inserted into
that HTML by the Vite build. Update the loader URL versions only when the
hand-maintained loader or fallback changes.

## Live board geometry (2026-09-23)

The three-player content previously used `min(100cqw,100cqh)` for both board
dimensions. Engines that ignore container units dropped those declarations,
exposing the older `width:100%;height:100%` rule inside a rectangular flex slot.
With container units and layers disabled, the main board reproduced at roughly
340 x 204 screen pixels. The existing compatibility stylesheet alone could not
preserve its shape.

AiRunBoard now measures its unscaled layout slot and gives a centered wrapper
the same explicit pixel width and height: min(available width, available height).
Tile fonts use that same side length. The overlay stays inside the square above
the board's isolated stacking context. RoomStage uses percentage padding to keep
16:9 without aspect-ratio support. No container units are needed in either layout.

roomSurfaceSize observes the element's actual owning window and rebinds after
Document PiP adoption/return. Its resize fallback reads on the next animation
frame, after the room's zoom and width listeners; it also supports layout switches
without ResizeObserver. It does not measure transformed screen rectangles.

Validation: 33 relevant unit tests and production build passed. In Chromium,
explicitly removed container-unit declarations, container-type, aspect-ratio,
layer blocks and registered properties while enabling the existing compatibility
stylesheet; checked 1440 x 1000, 390 x 844 and 844 x 390, both layouts, selected
player changes, and square tiles fitting their slots. Repeated with ResizeObserver
unavailable. Modern Document PiP passed window resizing, both layouts, selection
and return to a resized opener. This is feature-removal emulation, not a claim of
testing the user's exact Baidu browser build.

Published live-entry/assets only and repeated compatibility emulation against
production. The stream remained paused with its original batch. Rollback entry:
`/opt/2048tables/backups/square-fix-20260923T115321Z/live-index.html`.

Reference: https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/length#container_query_length_units
