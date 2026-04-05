import { useMemo, useRef, useEffect, useState, forwardRef, useImperativeHandle } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import { BREAKPOINTS } from "../utils/aqi.js";
import L from "leaflet";

const CARTO_DARK_NOLABELS =
  "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png";
const CARTO_LABELS =
  "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png";
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>';

const TEXAS_CENTER = [31.5, -99.0];
const TEXAS_ZOOM = 6;

function normGeoid(val) {
  if (!val) return "";
  return String(val).padStart(11, "0");
}

// ── Zoom controller ───────────────────────────────────────────────────────
function MapZoomController({ mapRef }) {
  const map = useMap();
  useImperativeHandle(mapRef, () => ({
    zoom: (lat, lon) => {
      map.flyTo([lat, lon], 13, { duration: 1.5, easeLinearity: 0.25 });
    },
  }));
  return null;
}

const MapViewContent = forwardRef(
  ({ geojson, predictions, onTractSelect, selectedGeoid, statewide }, mapRef) => {
    const [mapKey, setMapKey] = useState(0);
    const prevPredLen = useRef(0);

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

    // ── GeoJSON polygon styling ────────────────────────────────────────────
    const styleFeature = (feature) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      const pred = predMap[geoid];
      const isSelected = geoid === selectedGeoid;
      return {
        fillColor: pred ? pred.color : "#2d3436",
        fillOpacity: isSelected ? 0.95 : 0.80,
        // USE FILL COLOR as border to eliminate black gaps between tracts
        color: isSelected ? "#ffffff" : (pred ? pred.color : "rgba(0,0,0,0.08)"),
        weight: isSelected ? 2.5 : 1.0,
        lineCap: "round",
        lineJoin: "round",
      };
    };

    const onEachFeature = (feature, layer) => {
      const geoid = normGeoid(feature.properties?.GEOID);
      const pred = predMap[geoid];
      const name = feature.properties?.NAME ?? geoid;
      // Fix: county from backend already has "County" suffix sometimes
      const county = pred?.county ? pred.county.replace(/ County$/i, "") : "";

      // NAME field is like "Census Tract 9501" or just "9501" — normalize
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
        click: () => onTractSelect(geoid),
        mouseover: (e) => {
          e.target.setStyle({
            fillOpacity: 0.9,
            weight: geoid === selectedGeoid ? 2.5 : 1.5,
            color: geoid === selectedGeoid ? "#ffffff" : "rgba(255,255,255,0.6)",
          });
          e.target.bringToFront();
        },
        mouseout: (e) => {
          e.target.setStyle(styleFeature(feature));
        },
      });
    };

    const hasPolygons = geojson?.features?.length > 0 && geojson.features[0]?.geometry;

    return (
      <MapContainer
        center={TEXAS_CENTER}
        zoom={TEXAS_ZOOM}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        preferCanvas={true}
      >
        {/* Layer 1: Dark base map without labels */}
        <TileLayer url={CARTO_DARK_NOLABELS} attribution={ATTRIBUTION} zIndex={1} />

        <MapZoomController mapRef={mapRef} />

        {/* Census tract polygons */}
        {hasPolygons && (
          <GeoJSON
            key={`geojson-${mapKey}`}
            data={geojson}
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        )}

        {/* Layer 4: City labels on TOP */}
        <TileLayer url={CARTO_LABELS} zIndex={650} pane="shadowPane" />

        {/* Legend */}
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

export default forwardRef((props, ref) => <MapViewContent {...props} mapRef={ref} />);
