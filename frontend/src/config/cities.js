export const CITY_CONFIG = {
  dallas: {
    center: [32.7767, -96.797],
    zoom: 11,
    name: "Dallas County",
    state: "TX",
    fips: "48113"
  },
  austin: {
    center: [30.2672, -97.7431],
    zoom: 11,
    name: "Travis County",
    state: "TX",
    fips: "48453"
  },
  houston: {
    center: [29.7604, -95.3698],
    zoom: 11,
    name: "Harris County",
    state: "TX",
    fips: "48201"
  },
  san_antonio: {
    center: [29.4241, -98.4936],
    zoom: 11,
    name: "Bexar County",
    state: "TX",
    fips: "48029"
  }
};

export const CITY_LIST = Object.entries(CITY_CONFIG).map(([id, config]) => ({
  id,
  ...config
}));
