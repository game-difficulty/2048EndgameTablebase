<template>
  <div
    ref="rootRef"
    class="ui-popover-select relative min-w-0"
    v-bind="attrs"
  >
    <button
      ref="triggerRef"
      type="button"
      :id="triggerId"
      :disabled="disabled"
      :aria-controls="listboxId"
      :aria-expanded="open ? 'true' : 'false'"
      aria-haspopup="listbox"
      :aria-label="ariaLabel || displayLabel"
      :class="[
        'ui-popover-select-trigger group flex w-full items-center gap-3 text-left outline-none transition-all disabled:cursor-not-allowed disabled:opacity-45',
        triggerClass,
        open ? 'border-accent/70 shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_14%,transparent),0_14px_30px_color-mix(in_srgb,var(--accent)_12%,transparent)]' : ''
      ]"
      @click="toggleMenu"
      @keydown="handleTriggerKeydown"
    >
      <span
        :class="[
          'min-w-0 flex-1 truncate',
          selectedOption ? '' : 'opacity-60',
          labelClass
        ]"
        :title="displayLabel"
      >
        {{ displayLabel }}
      </span>
      <span
        :class="[
          'ui-popover-select-chevron flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-transparent text-text-secondary transition-all duration-150',
          open
            ? 'bg-accent/14 text-accent'
            : 'group-hover:border-border-main/60 group-hover:bg-bg-main/55 group-hover:text-text-main'
        ]"
        aria-hidden="true"
      >
        <svg
          class="h-3.5 w-3.5 transition-transform duration-180"
          :class="open ? 'translate-y-[1px] rotate-180' : ''"
          viewBox="0 0 16 16"
          fill="none"
        >
          <path
            d="M3.5 6.25L8 10.75L12.5 6.25"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </span>
    </button>

    <transition name="ui-popover-select-menu">
      <div
        v-if="open"
        ref="menuRef"
        :class="[
          'ui-popover-select-menu absolute top-full z-[180] mt-2 overflow-hidden rounded-[20px] border border-border-main/75 bg-bg-card/96 shadow-[0_22px_56px_rgba(0,0,0,0.2)] backdrop-blur-xl',
          align === 'right' ? 'right-0' : 'left-0',
          menuClass
        ]"
        :style="menuStyle"
      >
        <div
          ref="listboxRef"
          :id="listboxId"
          role="listbox"
          tabindex="-1"
          :aria-labelledby="triggerId"
          class="ui-popover-select-list max-h-[280px] overflow-y-auto p-1.5"
          @keydown="handleListboxKeydown"
        >
          <button
            v-for="(option, index) in normalizedOptions"
            :key="option.key"
            :ref="(el) => setOptionRef(el, index)"
            type="button"
            role="option"
            :aria-selected="option.value === modelValue ? 'true' : 'false'"
            :disabled="option.disabled"
            :title="option.title || option.label"
            :class="[
              'ui-popover-select-option flex w-full items-center gap-3 rounded-[14px] border px-3 py-1.5 text-left transition-all duration-150',
              optionClass,
              option.disabled
                ? 'cursor-not-allowed border-transparent text-text-secondary/45 opacity-65'
                : option.value === modelValue
                  ? 'border-accent/28 bg-[linear-gradient(135deg,color-mix(in_srgb,var(--accent)_16%,transparent),color-mix(in_srgb,var(--accent)_8%,transparent))] text-prominent shadow-sm'
                  : activeIndex === index
                    ? 'border-border-main/65 bg-bg-main/82 text-text-main shadow-sm'
                    : 'border-transparent bg-transparent text-text-main hover:border-border-main/55 hover:bg-bg-main/70'
            ]"
            @click="selectOption(index)"
            @mouseenter="setActiveIndex(index)"
            @focus="setActiveIndex(index)"
          >
            <span
              :class="[
                'min-w-0 flex-1 truncate',
                option.value === modelValue ? 'font-black' : ''
              ]"
            >
              {{ option.label }}
            </span>
            <span
              v-if="option.value === modelValue"
              class="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/14 text-accent"
              aria-hidden="true"
            >
              <svg class="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none">
                <path
                  d="M3.75 8.25L6.6 11.1L12.25 5.45"
                  stroke="currentColor"
                  stroke-width="1.8"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </span>
          </button>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import {
  computed,
  nextTick,
  onMounted,
  onBeforeUnmount,
  ref,
  useAttrs,
  watch,
} from 'vue';

defineOptions({
  inheritAttrs: false,
});

const props = defineProps({
  modelValue: {
    type: [String, Number, Boolean],
    default: '',
  },
  options: {
    type: Array,
    default: () => [],
  },
  placeholder: {
    type: String,
    default: '',
  },
  ariaLabel: {
    type: String,
    default: '',
  },
  disabled: {
    type: Boolean,
    default: false,
  },
  align: {
    type: String,
    default: 'left',
  },
  triggerClass: {
    type: String,
    default: '',
  },
  menuClass: {
    type: String,
    default: '',
  },
  optionClass: {
    type: String,
    default: '',
  },
  labelClass: {
    type: String,
    default: '',
  },
  matchTriggerWidth: {
    type: Boolean,
    default: true,
  },
});

const emit = defineEmits(['update:modelValue', 'change', 'open', 'close']);

const attrs = useAttrs();
const rootRef = ref(null);
const triggerRef = ref(null);
const menuRef = ref(null);
const listboxRef = ref(null);
const open = ref(false);
const activeIndex = ref(-1);
const optionRefs = ref([]);
const triggerId = `ui-select-trigger-${Math.random().toString(36).slice(2, 9)}`;
const listboxId = `ui-select-listbox-${Math.random().toString(36).slice(2, 9)}`;

let typeaheadBuffer = '';
let typeaheadTimer = null;

const normalizedOptions = computed(() =>
  (props.options || []).map((option, index) => {
    if (option && typeof option === 'object' && !Array.isArray(option)) {
      return {
        key: option.key ?? `${String(option.value ?? '')}-${index}`,
        value: option.value,
        label: option.label ?? String(option.value ?? ''),
        disabled: Boolean(option.disabled),
        title: option.title ?? '',
      };
    }

    return {
      key: `${String(option ?? '')}-${index}`,
      value: option,
      label: String(option ?? ''),
      disabled: false,
      title: '',
    };
  })
);

const selectedIndex = computed(() =>
  normalizedOptions.value.findIndex((option) => option.value === props.modelValue)
);

const selectedOption = computed(() =>
  selectedIndex.value >= 0 ? normalizedOptions.value[selectedIndex.value] : null
);

const displayLabel = computed(() => {
  if (selectedOption.value) {
    return selectedOption.value.label;
  }
  return props.placeholder || '';
});

const menuStyle = computed(() =>
  props.matchTriggerWidth ? { minWidth: '100%' } : undefined
);

const clearTypeahead = () => {
  typeaheadBuffer = '';
  if (typeaheadTimer !== null) {
    window.clearTimeout(typeaheadTimer);
    typeaheadTimer = null;
  }
};

const getFirstEnabledIndex = () =>
  normalizedOptions.value.findIndex((option) => !option.disabled);

const getLastEnabledIndex = () => {
  for (let index = normalizedOptions.value.length - 1; index >= 0; index -= 1) {
    if (!normalizedOptions.value[index].disabled) {
      return index;
    }
  }
  return -1;
};

const findEnabledIndex = (startIndex, direction) => {
  if (!normalizedOptions.value.length) {
    return -1;
  }

  let index = startIndex;
  for (let count = 0; count < normalizedOptions.value.length; count += 1) {
    index += direction;
    if (index < 0) {
      index = normalizedOptions.value.length - 1;
    } else if (index >= normalizedOptions.value.length) {
      index = 0;
    }

    if (!normalizedOptions.value[index].disabled) {
      return index;
    }
  }

  return -1;
};

const scrollActiveOptionIntoView = () => {
  nextTick(() => {
    optionRefs.value[activeIndex.value]?.scrollIntoView({
      block: 'nearest',
    });
  });
};

const setActiveIndex = (index) => {
  const option = normalizedOptions.value[index];
  if (!option || option.disabled) {
    return;
  }
  activeIndex.value = index;
  scrollActiveOptionIntoView();
};

const syncActiveIndex = (preferredIndex = null) => {
  if (!normalizedOptions.value.length) {
    activeIndex.value = -1;
    return;
  }

  if (
    preferredIndex !== null &&
    preferredIndex >= 0 &&
    preferredIndex < normalizedOptions.value.length &&
    !normalizedOptions.value[preferredIndex].disabled
  ) {
    activeIndex.value = preferredIndex;
    scrollActiveOptionIntoView();
    return;
  }

  if (
    selectedIndex.value >= 0 &&
    !normalizedOptions.value[selectedIndex.value].disabled
  ) {
    activeIndex.value = selectedIndex.value;
    scrollActiveOptionIntoView();
    return;
  }

  activeIndex.value = getFirstEnabledIndex();
  scrollActiveOptionIntoView();
};

const focusListbox = () => {
  nextTick(() => {
    listboxRef.value?.focus();
  });
};

const openMenu = (preferredIndex = null) => {
  if (props.disabled || !normalizedOptions.value.length) {
    return;
  }

  if (!open.value) {
    open.value = true;
    emit('open');
  }

  syncActiveIndex(preferredIndex);
  focusListbox();
};

const closeMenu = ({ restoreFocus = false } = {}) => {
  if (!open.value) {
    return;
  }

  open.value = false;
  emit('close');
  clearTypeahead();

  if (restoreFocus) {
    nextTick(() => {
      triggerRef.value?.focus();
    });
  }
};

const toggleMenu = () => {
  if (open.value) {
    closeMenu();
    return;
  }
  openMenu();
};

const selectOption = (index) => {
  const option = normalizedOptions.value[index];
  if (!option || option.disabled) {
    return;
  }

  if (option.value !== props.modelValue) {
    emit('update:modelValue', option.value);
    emit('change', option.value);
  }

  closeMenu({ restoreFocus: true });
};

const selectActiveOption = () => {
  if (activeIndex.value < 0) {
    return;
  }
  selectOption(activeIndex.value);
};

const moveActiveIndex = (direction) => {
  const currentIndex = activeIndex.value >= 0 ? activeIndex.value : selectedIndex.value;
  const nextIndex = findEnabledIndex(currentIndex >= 0 ? currentIndex : -direction, direction);
  if (nextIndex >= 0) {
    setActiveIndex(nextIndex);
  }
};

const handleTypeahead = (key) => {
  const printableKey = key.trim().toLowerCase();
  if (!printableKey) {
    return;
  }

  typeaheadBuffer += printableKey;
  if (typeaheadTimer !== null) {
    window.clearTimeout(typeaheadTimer);
  }
  typeaheadTimer = window.setTimeout(() => {
    clearTypeahead();
  }, 450);

  const startIndex = activeIndex.value >= 0 ? activeIndex.value : selectedIndex.value;
  const searchOrder = normalizedOptions.value
    .map((_, index) => index)
    .sort((left, right) => {
      const leftDistance =
        (left - startIndex - 1 + normalizedOptions.value.length) %
        normalizedOptions.value.length;
      const rightDistance =
        (right - startIndex - 1 + normalizedOptions.value.length) %
        normalizedOptions.value.length;
      return leftDistance - rightDistance;
    });

  const matchIndex = searchOrder.find((index) => {
    const option = normalizedOptions.value[index];
    return !option.disabled && option.label.toLowerCase().startsWith(typeaheadBuffer);
  });

  if (matchIndex >= 0) {
    if (!open.value) {
      openMenu(matchIndex);
      return;
    }
    setActiveIndex(matchIndex);
  }
};

const isPrintableKey = (event) =>
  event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey;

const handleTriggerKeydown = (event) => {
  if (props.disabled) {
    return;
  }

  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault();
      if (!open.value) {
        openMenu(selectedIndex.value >= 0 ? selectedIndex.value : getFirstEnabledIndex());
      } else {
        moveActiveIndex(1);
      }
      break;
    case 'ArrowUp':
      event.preventDefault();
      if (!open.value) {
        openMenu(selectedIndex.value >= 0 ? selectedIndex.value : getLastEnabledIndex());
      } else {
        moveActiveIndex(-1);
      }
      break;
    case 'Enter':
    case ' ':
      event.preventDefault();
      toggleMenu();
      break;
    case 'Home':
      event.preventDefault();
      openMenu(getFirstEnabledIndex());
      break;
    case 'End':
      event.preventDefault();
      openMenu(getLastEnabledIndex());
      break;
    default:
      if (isPrintableKey(event)) {
        event.preventDefault();
        handleTypeahead(event.key);
      }
      break;
  }
};

const handleListboxKeydown = (event) => {
  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault();
      moveActiveIndex(1);
      break;
    case 'ArrowUp':
      event.preventDefault();
      moveActiveIndex(-1);
      break;
    case 'Home':
      event.preventDefault();
      setActiveIndex(getFirstEnabledIndex());
      break;
    case 'End':
      event.preventDefault();
      setActiveIndex(getLastEnabledIndex());
      break;
    case 'Enter':
    case ' ':
      event.preventDefault();
      selectActiveOption();
      break;
    case 'Escape':
      event.preventDefault();
      closeMenu({ restoreFocus: true });
      break;
    case 'Tab':
      closeMenu();
      break;
    default:
      if (isPrintableKey(event)) {
        event.preventDefault();
        handleTypeahead(event.key);
      }
      break;
  }
};

const setOptionRef = (element, index) => {
  if (!element) {
    optionRefs.value[index] = undefined;
    return;
  }
  optionRefs.value[index] = element;
};

const handlePointerDownOutside = (event) => {
  if (!open.value) {
    return;
  }

  const root = rootRef.value;
  if (root instanceof HTMLElement && !root.contains(event.target)) {
    closeMenu();
  }
};

const handleViewportChange = () => {
  if (open.value) {
    closeMenu();
  }
};

watch(
  () => props.modelValue,
  () => {
    if (open.value) {
      syncActiveIndex();
    }
  }
);

watch(
  () => props.options,
  () => {
    optionRefs.value = [];
    if (open.value) {
      syncActiveIndex();
    }
  },
  { deep: true }
);

watch(
  () => props.disabled,
  (disabled) => {
    if (disabled) {
      closeMenu();
    }
  }
);

onMounted(() => {
  document.addEventListener('pointerdown', handlePointerDownOutside);
  window.addEventListener('resize', handleViewportChange);
});

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', handlePointerDownOutside);
  window.removeEventListener('resize', handleViewportChange);
  clearTypeahead();
});
</script>

<style scoped>
.ui-popover-select-menu {
  width: max-content;
  max-width: min(30rem, calc(100vw - 2rem));
}

.ui-popover-select-list {
  scrollbar-gutter: stable;
}

.ui-popover-select-menu-enter-active,
.ui-popover-select-menu-leave-active {
  transition: opacity 0.16s ease, transform 0.16s ease;
}

.ui-popover-select-menu-enter-from,
.ui-popover-select-menu-leave-to {
  opacity: 0;
  transform: translateY(-4px) scale(0.985);
}
</style>
