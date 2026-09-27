function normalizedId(value) {
  if (value === null || value === undefined || value === '') return null;
  return String(value);
}

export function isProfileOwner(profile, viewer) {
  if (!profile?.player) return false;
  if (profile.is_owner === true) return true;
  if (!viewer) return false;
  const profileId = normalizedId(profile.player.id);
  const viewerId = normalizedId(viewer.id);
  return profileId !== null && viewerId !== null && profileId === viewerId;
}
