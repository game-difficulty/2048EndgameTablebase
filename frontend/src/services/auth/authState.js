import { computed, ref } from 'vue';

import { authClient } from './authClient';
import { emitAuthChanged } from './authEvents';

const ready = ref(false);
const user = ref(null);
const dialogOpen = ref(false);
const dialogMode = ref('login');

const isAuthenticated = computed(() => !!user.value);

export function useAuthState() {
  const refreshAuth = async () => {
    const wasReady = ready.value;
    const previousUserId = user.value?.id ?? null;
    try {
      const result = await authClient.me();
      user.value = result?.authenticated ? result.user : null;
      return user.value;
    } catch (error) {
      user.value = null;
      return null;
    } finally {
      ready.value = true;
      const nextUserId = user.value?.id ?? null;
      if (wasReady && previousUserId !== nextUserId) {
        emitAuthChanged();
      }
    }
  };

  const openAuthDialog = (mode = 'login') => {
    dialogMode.value = mode === 'register' ? 'register' : 'login';
    dialogOpen.value = true;
  };

  const closeAuthDialog = () => {
    dialogOpen.value = false;
  };

  const requireAuth = () => {
    if (user.value) {
      return true;
    }
    openAuthDialog('login');
    return false;
  };

  const logout = async () => {
    try {
      await authClient.logout();
    } finally {
      user.value = null;
      emitAuthChanged();
    }
  };

  return {
    ready,
    user,
    isAuthenticated,
    dialogOpen,
    dialogMode,
    refreshAuth,
    openAuthDialog,
    closeAuthDialog,
    requireAuth,
    logout,
  };
}
