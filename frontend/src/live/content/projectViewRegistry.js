import StreamProjectView from './StreamProjectView.vue';

const renderers = Object.freeze({
  '2048-board|2048-board-v1': StreamProjectView,
  '2048-board|2048-board-v2': StreamProjectView,
  'cargo-transport|cargo-transport-v1': StreamProjectView,
  'polyomino-board|polyomino-board-v1': StreamProjectView,
});

export function projectViewRenderer(view) {
  if (!view) return null;
  return renderers[`${view.view_kind}|${view.view_protocol}`] || null;
}
