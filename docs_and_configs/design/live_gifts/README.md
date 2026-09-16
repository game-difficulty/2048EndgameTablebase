# Live Gift Artwork

`references/` contains the original user-provided folder, moved from
`C:/Users/Administrator/Desktop/pic`. Do not serve this directory publicly.

## Delivery

- Static catalog/chat art: `frontend/src/live/GiftIcon.vue` and
  `frontend/public/live-gifts/*.webp` (without `-motion`).
- Supporter animations: button, tea and moai use optimized animated WebP;
  whale uses a CSS spout; chicken alternates the two provided poses; serious
  uses an SVG chicken with animated legs and body based on the reference pose.
- Ceremony art: `GiftAnimation.vue`, four 512 tiles merge twice to 2048;
  a crown lands on 32768; a FINAL PHASE stamp lands on 1024; two black 32768
  tiles merge to a purple 65536 with a commemorative border and laurels.
- All other gifts have a static icon and a simple banner only.
- Rendering policy: `giftArtwork.js`; queue and combo lifecycle: `giftEvents.js`.
- Pricing and localized names are server-owned. Coffee, fireworks, merge and brilliant
  are retired: not listed or purchasable; existing receipts remain readable/idempotent.

## Reference Mapping

| Gift ID | Reference |
| --- | --- |
| knowledge | 知识增加.png |
| meaning | 何意味.png |
| whale | 鲸鱼.png |
| button | 按钮.gif (按钮.json retained as source only) |
| moai | 什.gif |
| tea | 茶.png / 茶.gif |
| chicken | 幽默唤鸡.jpg / 幽默唤鸡2.jpg |
| serious | 严肃唤鸡.jpg |

WebP derivatives use a maximum dimension of 160px, quality 85 for posters;
animated derivatives use 15 fps, quality 80 and loop only while mounted. Their
sources are loaded on demand, never as animated gift-picker thumbnails. Current
Only the referenced assets are loaded by the page, excluding the original references.

## Artwork Refinement

- Two/four use explicit central SVG baselines within their tile rectangles.
- Heart, bouquet and dealer glove have distinct highlight, wrapping and seam details.
- 666 uses raised gold lettering; BUG! uses a broken terminal badge; klbm ?? uses a
  question bubble. These are independently drawn SVGs, not recolors of one shape.
- `knowledge-cutout.webp` and `moai-cutout.webp` are transparent static assets,
  edited with the built-in image generator and resized to 160px lossless WebP.
  Prompts: preserve the original art; remove only the external white background
  (knowledge: only the four exterior corners, keep the inner white rays).
- The original animated Moai keeps its frames and timing; an inline SVG white-matte
  filter removes the white background during rendering in both themes.
- Gift quantity reuses BattleNumberInput's themed stepper. The hover panel omits
  the redundant price footer; catalog prices, bulk-button tooltips and expensive
  gift confirmation still show the cost before purchase.

## Verification

`frontend/tests/liveGifts.test.js` checks reference posters contain no ANIM chunk,
supporter gating, ceremony serialization and chat combo idempotency. Python gift
tests cover exact prices, atomic charging, old receipt recovery and shared chat.
