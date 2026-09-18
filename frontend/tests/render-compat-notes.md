# Rendering compatibility, phase one

Both entry documents load `/compat/render-compat.js?v=1` before the app.
The classic ES5 detector tests actual cascade-layer application, color-mix,
OKLCH, aspect-ratio and registered properties. Modern engines do not request
the fallback CSS. Other engines receive an unlayered, scoped stylesheet.

Diagnostics: `window.__RENDER_COMPAT__` records the reasons; the root
`data-css-compat` attribute is modern/loading/ready/failed. There is no user
override yet. A stylesheet network error does not hide the page indefinitely.

Scope: shared BaseBoard, basic navigation/home grid, account/login overlays,
forms and live chat. This is not a universal Tailwind transpiler, nor a JS
polyfill. Browsers still need the app's module runtime, CSS variables, Grid
and Flexbox. Other pages and advanced decorative effects need later audits.

Validation:
- `node --test tests/renderCompat.test.js tests/liveLayout.test.js`
- `npm run build`
- In browser automation, override color capability detection before navigation,
  then remove CSSLayerBlockRules to emulate ignored Tailwind layers. Check board
  containing blocks, tile alignment, navigation and opening login dialogs.
- Verify the modern path never requests render-compat.css.
- This simulation is not a substitute for an older Baidu browser device test.

Deployment must include `dist/compat/` along with both HTML entries. Copy new
static files before the HTML, and change the version in the two entries and
the loader's stylesheet URL whenever these compatibility assets change.
