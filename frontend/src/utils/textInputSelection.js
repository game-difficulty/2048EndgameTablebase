export const selectTextInputContentsOnFocus = (event) => {
  const target = event?.currentTarget;
  if (typeof target?.select !== 'function') return;

  window.requestAnimationFrame(() => {
    if (document.activeElement === target) {
      target.select();
    }
  });
};
