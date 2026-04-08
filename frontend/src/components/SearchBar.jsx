import React, { useState, useRef, useEffect } from 'react';
import './SearchBar.css';

export default function SearchBar({ onSearch, loading }) {
  const [searchInput, setSearchInput] = useState('');
  const [searchType, setSearchType] = useState('address');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [error, setError] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const searchRef = useRef(null);
  const timeoutRef = useRef(null);

  // Fetch suggestions from Nominatim
  useEffect(() => {
    if (searchType !== 'address' || !searchInput.trim() || searchInput.length < 3) {
      setSuggestions([]);
      return;
    }

    // Clear previous timeout
    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    // Debounce the search
    timeoutRef.current = setTimeout(async () => {
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
            searchInput + ', Texas'
          )}&limit=5`
        );
        const results = await response.json();
        setSuggestions(results);
        setShowSuggestions(true);
      } catch (err) {
        console.error('Autocomplete error:', err);
        setSuggestions([]);
      }
    }, 300);

    return () => clearTimeout(timeoutRef.current);
  }, [searchInput, searchType]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const geocodeAddress = async (address, lat, lon) => {
    try {
      setIsSearching(true);
      setError('');

      // Validate Texas bounds
      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError('Address is outside Texas. Please enter a Texas address.');
        setIsSearching(false);
        return;
      }

      onSearch({ lat, lon, address });
      setSuggestions([]);
      setShowSuggestions(false);
    } catch (err) {
      setError('Could not geocode address. Please try again.');
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    geocodeAddress(
      suggestion.display_name,
      parseFloat(suggestion.lat),
      parseFloat(suggestion.lon)
    );
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setError('');

    if (searchType === 'address') {
      if (!searchInput.trim()) {
        setError('Please enter an address');
        return;
      }
      if (suggestions.length > 0) {
        handleSuggestionClick(suggestions[0]);
      } else {
        // No suggestions loaded yet — do a fresh Nominatim lookup
        setIsSearching(true);
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
              searchInput + ', Texas'
            )}&limit=1`
          );
          const results = await response.json();
          if (results.length > 0) {
            const r = results[0];
            geocodeAddress(r.display_name, parseFloat(r.lat), parseFloat(r.lon));
          } else {
            setError('Address not found. Try a more specific Texas address.');
          }
        } catch (err) {
          setError('Search failed. Please try again.');
        } finally {
          setIsSearching(false);
        }
      }
    } else {
      // Coordinates mode
      const coords = searchInput.trim().split(',');
      if (coords.length !== 2) {
        setError('Please enter coordinates as: latitude, longitude');
        return;
      }

      const lat = parseFloat(coords[0].trim());
      const lon = parseFloat(coords[1].trim());

      if (isNaN(lat) || isNaN(lon)) {
        setError('Please enter valid numbers for latitude and longitude');
        return;
      }

      if (lat < 25.8 || lat > 36.5 || lon < -106.6 || lon > -93.5) {
        setError('Coordinates must be within Texas bounds');
        return;
      }

      onSearch({ lat, lon });
      setError('');
    }
  };

  return (
    <div className="search-bar" ref={searchRef}>
      <form onSubmit={handleSearch}>
        <div className="search-tabs">
          <button
            type="button"
            className={`search-tab ${searchType === 'address' ? 'active' : ''}`}
            onClick={() => {
              setSearchType('address');
              setSearchInput('');
              setSuggestions([]);
              setError('');
            }}
          >
            📍 Address
          </button>
          <button
            type="button"
            className={`search-tab ${searchType === 'coordinates' ? 'active' : ''}`}
            onClick={() => {
              setSearchType('coordinates');
              setSearchInput('');
              setSuggestions([]);
              setError('');
            }}
          >
            🧭 Coordinates
          </button>
        </div>

        <div className="search-input-group">
          <div className="search-input-wrapper">
            <input
              type="text"
              placeholder={
                searchType === 'address'
                  ? 'Enter address (e.g., Austin, TX)'
                  : 'Latitude, Longitude (e.g., 30.2672, -97.7431)'
              }
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onFocus={() => searchType === 'address' && setShowSuggestions(true)}
              required
            />

            {/* Autocomplete Dropdown */}
            {searchType === 'address' && showSuggestions && suggestions.length > 0 && (
              <div className="suggestions-dropdown">
                {suggestions.map((suggestion, idx) => (
                  <div
                    key={idx}
                    className="suggestion-item"
                    onClick={() => handleSuggestionClick(suggestion)}
                  >
                    <div className="suggestion-name">{suggestion.name || suggestion.display_name.split(',')[0]}</div>
                    <div className="suggestion-address">{suggestion.display_name}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button type="submit" disabled={loading || isSearching}>
            {isSearching ? 'Searching...' : loading ? 'Loading...' : 'Search'}
          </button>
        </div>

        {error && <div className="search-error">{error}</div>}
      </form>

      <div className="search-hint">
        {searchType === 'coordinates' ? 'Enter latitude and longitude separated by comma' : ''}
      </div>
    </div>
  );
}
