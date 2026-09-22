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
