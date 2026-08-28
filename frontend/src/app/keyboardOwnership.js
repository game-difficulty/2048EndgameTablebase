import { readonly, ref } from 'vue';

export const KEYBOARD_OWNERS = Object.freeze({
  PRIMARY: 'primary',
  TRAINER: 'trainer',
});

const keyboardOwnerState = ref(KEYBOARD_OWNERS.PRIMARY);
const splitKeyboardModeState = ref(false);

export const keyboardOwner = readonly(keyboardOwnerState);
export const splitKeyboardMode = readonly(splitKeyboardModeState);

export const setKeyboardOwner = (owner) => {
  keyboardOwnerState.value = owner === KEYBOARD_OWNERS.TRAINER
    ? KEYBOARD_OWNERS.TRAINER
    : KEYBOARD_OWNERS.PRIMARY;
};

export const setSplitKeyboardMode = (active) => {
  splitKeyboardModeState.value = Boolean(active);
};

export const keyboardInputAllowed = (owner) => (
  !splitKeyboardModeState.value || keyboardOwnerState.value === owner
);

export const resetKeyboardOwnership = () => {
  keyboardOwnerState.value = KEYBOARD_OWNERS.PRIMARY;
  splitKeyboardModeState.value = false;
};
