import React from 'react';
import { CITY_LIST } from '../config/cities';
import './CitySelector.css';

export default function CitySelector({ selectedCity, onCityChange }) {
  return (
    <div className="city-selector">
      <label htmlFor="city-select">Select Region:</label>
      <div className="city-buttons">
        {CITY_LIST.map(city => (
          <button
            key={city.id}
            className={`city-btn ${selectedCity === city.id ? 'active' : ''}`}
            onClick={() => onCityChange(city.id)}
            title={city.name}
          >
            {city.id.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
          </button>
        ))}
      </div>
    </div>
  );
}
