import { ref, onMounted, onBeforeUnmount } from 'vue';

export function useLiveTheme() {
  const read = () => typeof document === 'undefined' || document.documentElement.dataset.theme !== 'light';
  const dark = ref(read());
  let observer;
  onMounted(() => {
    dark.value = read();
    observer = new MutationObserver(() => { dark.value = read(); });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  });
  onBeforeUnmount(() => observer?.disconnect());
  return dark;
}
