import { computed, ref } from 'vue';

import { authClient } from './authClient';
import { emitAuthChanged } from './authEvents';
import { clearDeviceSession, DEVICE_SESSION_STORAGE_KEY } from './sessionTokenStore';
import {
  clearGuestSession,
  GUEST_SESSION_STORAGE_KEY,
} from './guestSessionStore';

const ready = ref(false);
const user = ref(null);
const guest = ref(null);
const dialogOpen = ref(false);
const dialogMode = ref('login');
let authGeneration = 0;
let latestRefreshId = 0;
let storageListenerInstalled = false;

const isAuthenticated = computed(() => !!user.value);
const isGuest = computed(() => !user.value && !!guest.value);
const currentActor = computed(() => {
  if (user.value) {
    return {
      kind: 'user',
      actor_key: `u:${user.value.id}`,
      user_id: user.value.id,
      guest_id: null,
      display_name: user.value.display_name || user.value.email || '',
    };
  }
  if (guest.value) {
    return {
      kind: 'guest',
      actor_key: `g:${guest.value.guest_id}`,
      user_id: null,
      guest_id: guest.value.guest_id,
      display_name: guest.value.display_name || '',
    };
  }
  return null;
});

const applyTokenBalance = (tokenBalance) => {
  if (!user.value || !tokenBalance) {
    return;
  }
  user.value = {
    ...user.value,
    token_balance: tokenBalance,
  };
};

if (typeof window !== 'undefined') {
  window.addEventListener('token-balance-updated', (event) => {
    applyTokenBalance(event?.detail || null);
  });
  window.addEventListener('guest-session-invalidated', () => {
    if (!user.value) guest.value = null;
  });
}

export function useAuthState() {
  const refreshAuth = async () => {
    const previousUserId = user.value?.id ?? null;
    const requestId = latestRefreshId + 1;
    latestRefreshId = requestId;
    const generationAtStart = authGeneration;
    try {
      const result = await authClient.me();
      if (requestId !== latestRefreshId || generationAtStart !== authGeneration) {
        return user.value;
      }
      if (result?.authenticated) {
        user.value = result.user;
        guest.value = null;
      } else {
        clearDeviceSession();
        user.value = null;
        guest.value = result?.guest || null;
        if (!guest.value) clearGuestSession();
      }
      return user.value;
    } catch (error) {
      if (requestId !== latestRefreshId || generationAtStart !== authGeneration) {
        return user.value;
      }
      if (error?.status === 401) {
        clearDeviceSession();
        user.value = null;
        return null;
      }
      return user.value;
    } finally {
      if (requestId !== latestRefreshId || generationAtStart !== authGeneration) {
        return;
      }
      ready.value = true;
      const nextUserId = user.value?.id ?? null;
      if (previousUserId !== nextUserId) {
        emitAuthChanged();
      }
    }
  };

  if (typeof window !== 'undefined' && !storageListenerInstalled) {
    storageListenerInstalled = true;
    window.addEventListener('storage', (event) => {
      if (![DEVICE_SESSION_STORAGE_KEY, GUEST_SESSION_STORAGE_KEY].includes(event.key)) return;
      authGeneration += 1;
      latestRefreshId += 1;
      void refreshAuth();
    });
  }

  const openAuthDialog = (mode = 'login') => {
    dialogMode.value = ['login', 'register', 'forgot', 'reset'].includes(mode) ? mode : 'login';
    dialogOpen.value = true;
  };

  const closeAuthDialog = () => {
    dialogOpen.value = false;
  };

  const setAuthenticatedUser = (authenticatedUser) => {
    const previousUserId = user.value?.id ?? null;
    authGeneration += 1;
    latestRefreshId += 1;
    user.value = authenticatedUser || null;
    if (authenticatedUser) guest.value = null;
    ready.value = true;
    const nextUserId = user.value?.id ?? null;
    if (previousUserId !== nextUserId) {
      emitAuthChanged();
    }
  };

  const requireAuth = () => {
    if (user.value) {
      return true;
    }
    openAuthDialog('login');
    return false;
  };

  const ensureGuestSession = async () => {
    if (user.value) return currentActor.value;
    if (guest.value) return currentActor.value;
    const result = await authClient.createGuestSession();
    guest.value = result?.guest || null;
    if (!guest.value) {
      throw new Error('Unable to create a guest session.');
    }
    authGeneration += 1;
    latestRefreshId += 1;
    emitAuthChanged();
    return currentActor.value;
  };

  const logout = async () => {
    try {
      await authClient.logout();
    } finally {
      authGeneration += 1;
      latestRefreshId += 1;
      clearDeviceSession();
      user.value = null;
      emitAuthChanged();
    }
  };

  return {
    ready,
    user,
    guest,
    isAuthenticated,
    isGuest,
    currentActor,
    dialogOpen,
    dialogMode,
    refreshAuth,
    openAuthDialog,
    closeAuthDialog,
    setAuthenticatedUser,
    requireAuth,
    ensureGuestSession,
    logout,
    applyTokenBalance,
  };
}
