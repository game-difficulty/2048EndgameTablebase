# Frontend

This directory contains the Vue 3 + Vite frontend for the desktop application.

## Scripts

```bash
npm install
npm run dev
npm run build
npm run preview
```

## Development flow

- `npm run dev` starts the Vite development server.
- `npm run build` writes the production bundle to `frontend/dist`.
- `backend_server.py` serves `frontend/dist` when the built assets exist, and otherwise can point to the Vite dev server during development.

## Source layout

- `src/components/`: shared app-level views and layout pieces
- `src/features/`: feature-specific pages, composables, and modules
- `src/locales/`: UI translations

## Styling

The frontend uses Vue 3, Vite, and Tailwind CSS v4.