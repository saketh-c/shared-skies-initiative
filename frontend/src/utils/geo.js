/**
 * Find nearest census tract to given coordinates
 * Returns the tract and its GEOID
 */
export function findNearestTract(lat, lon, tracts) {
  if (!tracts || tracts.length === 0) return null;

  let nearest = tracts[0];
  let minDistance = calculateDistance(lat, lon, nearest.lat, nearest.lon);

  for (let i = 1; i < tracts.length; i++) {
    const tract = tracts[i];
    const distance = calculateDistance(lat, lon, tract.lat, tract.lon);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = tract;
    }
  }

  return nearest;
}

/**
 * Calculate distance between two coordinate pairs using Haversine formula
 * Returns distance in miles
 */
export function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 3959; // Earth radius in miles
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}
