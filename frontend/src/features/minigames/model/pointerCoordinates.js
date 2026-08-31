export function clientPointToElementSpace(element, clientX, clientY) {
  const rect = element?.getBoundingClientRect?.();
  if (!rect) {
    return { x: 0, y: 0 };
  }

  const layoutWidth = Number(element.offsetWidth) || Number(element.clientWidth) || Number(rect.width) || 1;
  const layoutHeight = Number(element.offsetHeight) || Number(element.clientHeight) || Number(rect.height) || 1;
  const scaleX = Number(rect.width) > 0 ? layoutWidth / Number(rect.width) : 1;
  const scaleY = Number(rect.height) > 0 ? layoutHeight / Number(rect.height) : 1;

  return {
    x: (Number(clientX) - Number(rect.left)) * scaleX,
    y: (Number(clientY) - Number(rect.top)) * scaleY,
  };
}
