import { readonly, ref } from 'vue';

export const KEYBOARD_OWNERS = Object.freeze({
  PRIMARY: 'primary',
  TRAINER: 'trainer',
});

const VALID_KEYBOARD_OWNERS = new Set(Object.values(KEYBOARD_OWNERS));
const keyboardOwnerState = ref(KEYBOARD_OWNERS.PRIMARY);
const splitKeyboardModeState = ref(false);

export const keyboardOwner = readonly(keyboardOwnerState);
export const splitKeyboardMode = readonly(splitKeyboardModeState);

export const normalizeKeyboardOwner = (owner) => (
  VALID_KEYBOARD_OWNERS.has(owner) ? owner : KEYBOARD_OWNERS.PRIMARY
);

export const setKeyboardOwner = (owner) => {
  keyboardOwnerState.value = normalizeKeyboardOwner(owner);
};

export const setSplitKeyboardMode = (active) => {
  splitKeyboardModeState.value = Boolean(active);
};

export const keyboardInputAllowed = (owner) => (
  !splitKeyboardModeState.value
  || keyboardOwnerState.value === normalizeKeyboardOwner(owner)
);

export const resetKeyboardOwnership = () => {
  keyboardOwnerState.value = KEYBOARD_OWNERS.PRIMARY;
  splitKeyboardModeState.value = false;
};
