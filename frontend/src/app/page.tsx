'use client'

import React, { useState, useEffect, useRef } from 'react';
import Image from "next/image";

interface RouteResponse {
  path: string[];
  imageUrl: string;
}

export default function Home() {
  const [startRoom, setStartRoom] = useState('');
  const [endRoom, setEndRoom] = useState('');
  const [routeImage, setRouteImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [overlayPositioned, setOverlayPositioned] = useState(false);
  const blueprintRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLImageElement>(null);

  // Available rooms for each floor
  const floor1Rooms = [
    '1001', '1002', '1003', '1004', '1006', '1007', '1008', '1009', 
    '1011', '1012', '1012B', '1802', '1902'
  ];
  
  const floor2Rooms = [
    '2001', '2002', '2003', '2004', '2006', '2007', '2008', '2009',
    '2011', '2012', '2013', '2014', '2016', '2017', '2018', '2019',
    '2022', '2023', '2024', '2026', '2027', '2028', '2029', '2031',
    '2032', '2033'
  ];

  const landmarks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Stair', 'Elevator'];

  const allRooms = [...floor1Rooms, ...floor2Rooms, ...landmarks];

  // Handle overlay positioning when both images are loaded
  useEffect(() => {
    if (routeImage && blueprintRef.current && overlayRef.current) {
      const blueprint = blueprintRef.current;
      const overlay = overlayRef.current;
      
      const positionOverlay = () => {
        if (blueprint && overlay) {
          overlay.style.width = blueprint.offsetWidth + 'px';
          overlay.style.height = blueprint.offsetHeight + 'px';
          setOverlayPositioned(true);
        }
      };

      // Position overlay when both images are loaded
      if (blueprint.complete && overlay.complete) {
        positionOverlay();
      } else {
        blueprint.onload = positionOverlay;
        overlay.onload = positionOverlay;
      }
    }
  }, [routeImage]);

  const getRoute = async () => {
    if (!startRoom || !endRoom) {
      setError('Please select both start and end rooms');
      return;
    }

    setLoading(true);
    setError(null);
    setRouteImage(null);
    setOverlayPositioned(false);

    try {
      console.log(`Fetching route from ${startRoom} to ${endRoom}`);
      const response = await fetch(`http://localhost:5000/${startRoom}/${endRoom}`);
      
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Response error:', errorText);
        throw new Error(`Failed to get route: ${response.status} ${response.statusText}`);
      }

      // Get the image blob
      const blob = await response.blob();
      console.log('Blob received:', blob.size, 'bytes');
      const imageUrl = URL.createObjectURL(blob);
      setRouteImage(imageUrl);
      console.log('Route image set successfully');
    } catch (err) {
      console.error('Error fetching route:', err);
      setError(`Failed to get route: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const clearRoute = () => {
    setRouteImage(null);
    setError(null);
    setOverlayPositioned(false);
    if (routeImage) {
      URL.revokeObjectURL(routeImage);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-gray-900 mb-3">
            UniMap
          </h1>
          <p className="text-xl text-gray-600 font-medium">
            ERC Building Navigation System
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Find the shortest path between any two rooms in the ERC building
          </p>
        </div>

        {/* Navigation Controls */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8 border border-gray-200">
          <h2 className="text-3xl font-semibold mb-6 text-gray-800 flex items-center">
            Find Your Route
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            {/* Start Room Selection */}
            <div className="space-y-3">
              <label htmlFor="startRoom" className="block text-lg font-medium text-gray-700">
                Start Room
              </label>
              <select
                id="startRoom"
                value={startRoom}
                onChange={(e) => setStartRoom(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-700 text-lg transition-colors"
              >
                <option value="">Select start room</option>
                <optgroup label="Floor 1 Rooms" className="font-semibold">
                  {floor1Rooms.map(room => (
                    <option key={room} value={room} className="py-1">{room}</option>
                  ))}
                </optgroup>
                <optgroup label="Floor 2 Rooms" className="font-semibold">
                  {floor2Rooms.map(room => (
                    <option key={room} value={room} className="py-1">{room}</option>
                  ))}
                </optgroup>
                <optgroup label="Landmarks" className="font-semibold">
                  {landmarks.map(landmark => (
                    <option key={landmark} value={landmark} className="py-1">{landmark}</option>
                  ))}
                </optgroup>
              </select>
            </div>

            {/* End Room Selection */}
            <div className="space-y-3">
              <label htmlFor="endRoom" className="block text-lg font-medium text-gray-700">
                End Room
              </label>
              <select
                id="endRoom"
                value={endRoom}
                onChange={(e) => setEndRoom(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-700 text-lg transition-colors"
              >
                <option value="">Select end room</option>
                <optgroup label="Floor 1 Rooms" className="font-semibold">
                  {floor1Rooms.map(room => (
                    <option key={room} value={room} className="py-1">{room}</option>
                  ))}
                </optgroup>
                <optgroup label="Floor 2 Rooms" className="font-semibold">
                  {floor2Rooms.map(room => (
                    <option key={room} value={room} className="py-1">{room}</option>
                  ))}
                </optgroup>
                <optgroup label="Landmarks" className="font-semibold">
                  {landmarks.map(landmark => (
                    <option key={landmark} value={landmark} className="py-1">{landmark}</option>
                  ))}
                </optgroup>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 flex-wrap">
            <button
              onClick={getRoute}
              disabled={loading || !startRoom || !endRoom}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed font-semibold text-lg transition-all duration-200 shadow-md hover:shadow-lg flex items-center gap-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Finding Route...
                </>
              ) : (
                <>
                  Find Route
                </>
              )}
            </button>
            <button
              onClick={clearRoute}
              className="px-8 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 font-semibold text-lg transition-all duration-200 shadow-md hover:shadow-lg flex items-center gap-2"
            >
              Clear
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 text-red-700 rounded-lg flex items-center gap-2">
              <span className="text-xl">⚠️</span>
              <span className="font-medium">{error}</span>
            </div>
          )}
        </div>

        {/* Route Display */}
        {routeImage && (
          <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
            <h3 className="text-2xl font-semibold mb-6 text-gray-800 flex items-center gap-2">
              🛣️ Route from {startRoom} to {endRoom}
            </h3>
            
            {/* Combined Blueprint and Route Display */}
            <div className="relative">
              {/* Original Blueprint as Background */}
              <div className="mb-4">
                <h4 className="text-lg font-medium text-gray-700 mb-3">Blueprint with Route Overlay</h4>
                <div className="relative inline-block">
                  <img
                    ref={blueprintRef}
                    src="/blueprint.png"
                    alt="Original blueprint"
                    className="max-w-full h-auto border-2 border-gray-300 rounded-lg shadow-md"
                    style={{ maxWidth: '800px' }}
                  />
                  <img
                    ref={overlayRef}
                    src={routeImage}
                    alt="Route overlay"
                    className="absolute top-0 opacity-80"
                    style={{ 
                      pointerEvents: 'none',
                      width: overlayPositioned ? '80%' : 'auto',
                      height: overlayPositioned ? '100%' : 'auto',
                      objectFit: 'contain',
                      left: '-200px'
                    }}
                  />
                </div>
              </div>
              
              {/* Separate Route Graph */}
              <div>
                <h4 className="text-lg font-medium text-gray-700 mb-3">Generated Route Graph</h4>
                <div className="flex justify-center">
                  <img
                    src={routeImage}
                    alt="Navigation route"
                    className="max-w-full h-auto border-2 border-gray-300 rounded-lg shadow-md"
                    style={{ maxWidth: '800px' }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
