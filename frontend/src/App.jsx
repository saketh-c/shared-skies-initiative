import { useEffect, useState, useCallback, useRef } from "react";
import MapView from "./components/MapView.jsx";
import SidePanel from "./components/SidePanel.jsx";
import SearchBar from "./components/SearchBar.jsx";
import { findNearestTract } from "./utils/geo.js";

const REFRESH_MS = 30 * 60 * 1000; // 30 min

export default function App() {
  const [predictions, setPredictions] = useState(null);
  const [geojson, setGeojson] = useState(null);
  const [selectedTract, setSelectedTract] = useState(null);
  const [localWeather, setLocalWeather] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const mapRef = useRef(null);

  const fetchPredictions = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/texas/predictions");
      if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
      const data = await res.json();
      setPredictions(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      console.error("Failed to fetch predictions:", e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchGeojson = useCallback(async () => {
    try {
      const res = await fetch("/api/texas/tracts/geojson");
      if (!res.ok) throw new Error(`GeoJSON error ${res.status}`);
      const data = await res.json();
      if (data.features?.length > 0 && data.features[0].geometry) {
        setGeojson(data);
      }
    } catch (e) {
      console.warn("GeoJSON not available, using markers:", e.message);
    }
  }, []);

  useEffect(() => {
    fetchPredictions();
    fetchGeojson();
    const timer = setInterval(fetchPredictions, REFRESH_MS);
    return () => clearInterval(timer);
  }, [fetchPredictions, fetchGeojson]);

  const handleTractSelect = useCallback(async (geoid) => {
    if (!predictions) return;

    // Use the EXACT SAME data from bulk predictions — matches the tooltip perfectly
    const tract = predictions.tracts?.find((t) => t.geoid === geoid);
    if (!tract) return;

    // Set tract data immediately with bulk prediction values (same as tooltip)
    setSelectedTract({ ...tract });
    setLocalWeather(null);

    // Fetch real-time local weather for the WEATHER WIDGET ONLY (not PM2.5)
    setWeatherLoading(true);
    try {
      const res = await fetch(`/api/tract/${geoid}`);
      if (res.ok) {
        const data = await res.json();
        // ONLY update weather display — NEVER touch PM2.5 value
        setLocalWeather(data.weather);
      }
    } catch (e) {
      console.error("Failed to fetch local weather:", e);
    } finally {
      setWeatherLoading(false);
    }
  }, [predictions]);

  const handleDeselect = useCallback(() => {
    setSelectedTract(null);
    setLocalWeather(null);
  }, []);

  const handleSearch = useCallback((coords) => {
    if (!predictions?.tracts) return;
    const nearestTract = findNearestTract(coords.lat, coords.lon, predictions.tracts);
    if (nearestTract) {
      handleTractSelect(nearestTract.geoid);
      if (mapRef.current) {
        mapRef.current.zoom(coords.lat, coords.lon);
      }
    }
  }, [predictions, handleTractSelect]);

  return (
    <div className="app-layout">
      <SidePanel
        predictions={predictions}
        selectedTract={selectedTract}
        localWeather={localWeather}
        onDeselect={handleDeselect}
        loading={loading}
        weatherLoading={weatherLoading}
        error={error}
        lastUpdated={lastUpdated}
        statewide={true}
      />
      <div className="map-wrapper">
        <SearchBar onSearch={handleSearch} loading={loading} />
        <MapView
          ref={mapRef}
          geojson={geojson}
          predictions={predictions?.tracts ?? []}
          onTractSelect={handleTractSelect}
          selectedGeoid={selectedTract?.geoid ?? null}
          statewide={true}
        />
      </div>
    </div>
  );
}
