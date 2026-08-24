export const selectTextInputContents = (event) => {
  const target = event?.currentTarget;
  if (typeof target?.select === 'function') {
    target.select();
  }
};
