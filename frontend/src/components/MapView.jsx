import { useMemo, useRef, useEffect, useState, forwardRef, useCallback } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap, Circle } from "react-leaflet";
import { BREAKPOINTS } from "../utils/aqi.js";

const CARTO_LIGHT_NOLABELS =
  "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png";
const CARTO_LIGHT_LABELS =
  "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png";
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>';

const TEXAS_CENTER = [31.5, -99.0];
const TEXAS_ZOOM = 6;

function normGeoid(val) {
  if (!val) return "";
  return String(val).padStart(11, "0");
}

// Fly to a target — only triggers when the target reference changes
function FlyToHandler({ target }) {
  const map = useMap();
  const lastTargetRef = useRef(null);
  useEffect(() => {
    if (!target || isNaN(target.lat) || isNaN(target.lon)) return;
    // Only fly if target actually changed (compare values, not reference)
    const last = lastTargetRef.current;
    if (last && last.lat === target.lat && last.lon === target.lon) return;
    lastTargetRef.current = { lat: target.lat, lon: target.lon };
    map.flyTo([target.lat, target.lon], 13, { duration: 1.8, easeLinearity: 0.2 });
  }, [target, map]);
  return null;
}

// Map background click handler — uses ref to track polygon clicks synchronously
function BackgroundClickHandler({ onBackgroundClick, justClickedRef }) {
  const map = useMap();
  useEffect(() => {
    const handleMapClick = () => {
      // If this click came from a polygon, the polygon's click handler set this ref to true
      if (justClickedRef.current) {
        justClickedRef.current = false;
        return;
      }
      // True background click — deselect
      onBackgroundClick?.();
    };
    map.on("click", handleMapClick);
    return () => map.off("click", handleMapClick);
  }, [map, onBackgroundClick, justClickedRef]);
  return null;
}

function SmoothWheelZoom() {
  const map = useMap();
  useEffect(() => {
    map.scrollWheelZoom.disable();

    let accDelta = 0;
    let lastMousePoint = null;
    let lastMouseLatLng = null;
    let timer = null;

    const onWheel = (e) => {
      e.preventDefault();
      accDelta += e.deltaY < 0 ? 0.5 : -0.5;
      lastMousePoint = map.mouseEventToContainerPoint(e);
      lastMouseLatLng = map.containerPointToLatLng(lastMousePoint);

      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        const newZoom = Math.max(1, Math.min(18, map.getZoom() + accDelta));
        const mouseNewPx = map.project(lastMouseLatLng, newZoom);
        const newCenterPx = mouseNewPx
          .subtract(lastMousePoint)
          .add(map.getSize().divideBy(2));
        const newCenter = map.unproject(newCenterPx, newZoom);

        map.flyTo(newCenter, newZoom, {
          animate: true,
          duration: 0.6,
          easeLinearity: 0.15,
        });

        accDelta = 0;
        timer = null;
      }, 40);
    };

    map.getContainer().addEventListener("wheel", onWheel, { passive: false });
    return () => {
      map.getContainer().removeEventListener("wheel", onWheel);
      if (timer) clearTimeout(timer);
    };
  }, [map]);
  return null;
}

const MapViewContent = forwardRef(
  ({ geojson, predictions, onTractSelect, onBackgroundClick, selectedGeoid, searchMarker }, _ref) => {
    const [mapKey, setMapKey] = useState(0);
    const justClickedRef = useRef(false);
    const prevPredLen = useRef(0);

    // Use refs for callbacks so polygon click handlers always see the LATEST functions
    // (not stale closures from when the GeoJSON layer was first created)
    const onTractSelectRef = useRef(onTractSelect);
    const selectedGeoidRef = useRef(selectedGeoid);
    const predMapRef = useRef({});

    useEffect(() => { onTractSelectRef.current = onTractSelect; }, [onTractSelect]);
    useEffect(() => { selectedGeoidRef.current = selectedGeoid; }, [selectedGeoid]);

    useEffect(() => {
      if (predictions.length > 0 && prevPredLen.current === 0) {
        setMapKey((k) => k + 1);
      }
      prevPredLen.current = predictions.length;
    }, [predictions.length]);

    const predMap = useMemo(() => {
      const m = {};
      predictions.forEach((p) => {
        m[normGeoid(p.geoid)] = p;
      });
      return m;
    }, [predictions]);

    useEffect(() => { predMapRef.current = predMap; }, [predMap]);

    const styleFeature = useCallback((feature) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      const pred = predMap[geoid];
      const isSelected = geoid === selectedGeoid;
      return {
        fillColor: pred ? pred.color : "#2d3436",
        fillOpacity: isSelected ? 0.95 : 0.80,
        color: isSelected ? "#ffffff" : (pred ? pred.color : "rgba(0,0,0,0.08)"),
        weight: isSelected ? 2.5 : 1.0,
        lineCap: "round",
        lineJoin: "round",
      };
    }, [predMap, selectedGeoid]);

    // onEachFeature is only called ONCE per feature when GeoJSON is created.
    // We use refs to access latest state inside the handlers.
    const onEachFeature = useCallback((feature, layer) => {
      const geoid = normGeoid(feature.properties?.GEOID);

      // CRITICAL: prevent click events from bubbling to the map's click handler
      layer.options.bubblingMouseEvents = false;

      // Build tooltip from current predictions
      const pred = predMapRef.current[geoid];
      const name = feature.properties?.NAME ?? geoid;
      const county = pred?.county ? pred.county.replace(/ County$/i, "") : "";
      const displayName = name.startsWith("Census Tract") ? name : `Census Tract ${name}`;

      if (pred) {
        layer.bindTooltip(
          `<div style="font-weight:600;margin-bottom:3px">${displayName}</div>
           <div style="font-size:16px;font-weight:800;color:${pred.color}">${pred.pm25} <span style="font-size:10px;font-weight:400;opacity:0.7">µg/m³</span></div>
           <div style="opacity:0.65;margin-top:2px">${pred.category}</div>
           ${county ? `<div style="opacity:0.45;margin-top:2px;font-size:10px">${county} County</div>` : ""}`,
          { sticky: true, className: "tract-tooltip" }
        );
      }

      layer.on({
        click: () => {
          // Mark synchronously so the map background handler knows
          justClickedRef.current = true;
          // Use the LATEST callback via ref, not a stale closure
          onTractSelectRef.current?.(geoid);
        },
        mouseover: (e) => {
          const isSelected = geoid === selectedGeoidRef.current;
          e.target.setStyle({
            fillOpacity: 0.92,
            weight: isSelected ? 2.5 : 1.5,
            color: isSelected ? "#ffffff" : "rgba(255,255,255,0.6)",
          });
          e.target.bringToFront();
        },
        mouseout: (e) => {
          // Recompute style with the LATEST predictions and selectedGeoid via refs
          const currentPred = predMapRef.current[geoid];
          const isSelected = geoid === selectedGeoidRef.current;
          e.target.setStyle({
            fillColor: currentPred ? currentPred.color : "#2d3436",
            fillOpacity: isSelected ? 0.95 : 0.80,
            color: isSelected ? "#ffffff" : (currentPred ? currentPred.color : "rgba(0,0,0,0.08)"),
            weight: isSelected ? 2.5 : 1.0,
          });
        },
      });
    }, []); // Empty deps — handlers use refs for latest state

    const hasPolygons = geojson?.features?.length > 0 && geojson.features[0]?.geometry;

    return (
      <MapContainer
        center={TEXAS_CENTER}
        zoom={TEXAS_ZOOM}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        zoomSnap={0}
        preferCanvas={true}
        maxBounds={[[25.5, -106.6], [36.5, -93.5]]}
        maxZoom={13}
        minZoom={4}
      >
        <TileLayer url={CARTO_LIGHT_NOLABELS} attribution={ATTRIBUTION} zIndex={1} keepBuffer={8} />

        <FlyToHandler target={searchMarker} />
        <SmoothWheelZoom />
        <BackgroundClickHandler
          onBackgroundClick={onBackgroundClick}
          justClickedRef={justClickedRef}
        />

        {hasPolygons && (
          <GeoJSON
            key={`geojson-${mapKey}`}
            data={geojson}
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        )}

        {searchMarker && (
          <Circle
            center={[searchMarker.lat, searchMarker.lon]}
            radius={120}
            interactive={false}
            bubblingMouseEvents={false}
            className="search-marker-circle"
            pathOptions={{ color: "#fff", weight: 2, fillColor: "#0077b6", fillOpacity: 1 }}
          />
        )}

        <TileLayer url={CARTO_LIGHT_LABELS} zIndex={650} pane="shadowPane" keepBuffer={8} />

        <div className="map-legend">
          <div className="legend-title">PM2.5 µg/m³</div>
          {BREAKPOINTS.map((b) => (
            <div className="legend-row" key={b.category}>
              <div className="legend-swatch" style={{ background: b.color }} />
              <span>{b.label}</span>
            </div>
          ))}
        </div>
      </MapContainer>
    );
  }
);

MapViewContent.displayName = "MapViewContent";

export default forwardRef((props, ref) => <MapViewContent {...props} />);
