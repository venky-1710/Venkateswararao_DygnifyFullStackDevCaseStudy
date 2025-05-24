import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, Area, ComposedChart,
  ScatterChart, Scatter, ZAxis 
} from 'recharts';
import { Calendar, MapPin, Download, TrendingUp, Thermometer, CloudRain, Sun } from 'lucide-react';

const ClimateHazardAnalyzer = () => {
  // Add script loading state
  const [leafletScriptLoaded, setLeafletScriptLoaded] = useState(false);
  // Add new state for WebSocket and analysis status
  const [ws, setWs] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState({ lat: 40.7128, lon: -74.0060, name: 'New York, NY' });
  const [hazardType, setHazardType] = useState('heatwave');
  const [timeRange, setTimeRange] = useState({ start: '2000', end: '2023' });
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mapInitialized, setMapInitialized] = useState(false);
  const [mapLoading, setMapLoading] = useState(true);
  const mapRef = useRef(null);
  const leafletMapRef = useRef(null);
  const mapContainerRef = useRef(null);

  // Mock API data for demonstration
  const mockData = {
    heatwave: {
      trends: [
        { year: 2000, frequency: 12, intensity: 3.2, duration: 4.1 },
        { year: 2005, frequency: 15, intensity: 3.5, duration: 4.3 },
        { year: 2010, frequency: 18, intensity: 3.8, duration: 4.7 },
        { year: 2015, frequency: 22, intensity: 4.1, duration: 5.2 },
        { year: 2020, frequency: 28, intensity: 4.6, duration: 5.8 },
        { year: 2023, frequency: 31, intensity: 4.9, duration: 6.1 }
      ],
      summary: {
        percentageIncrease: 158,
        averageIncrease: 1.58,
        significance: 0.001,
        trend: 'Increasing'
      }
    },
    drought: {
      trends: [
        { year: 2000, frequency: 8, intensity: 2.1, duration: 12.3 },
        { year: 2005, frequency: 9, intensity: 2.3, duration: 13.1 },
        { year: 2010, frequency: 11, intensity: 2.6, duration: 14.2 },
        { year: 2015, frequency: 13, intensity: 2.8, duration: 15.8 },
        { year: 2020, frequency: 16, intensity: 3.2, duration: 17.4 },
        { year: 2023, frequency: 18, intensity: 3.5, duration: 18.9 }
      ],
      summary: {
        percentageIncrease: 125,
        averageIncrease: 1.25,
        significance: 0.003,
        trend: 'Increasing'
      }
    },
    heavyRain: {
      trends: [
        { year: 2000, frequency: 24, intensity: 1.8, duration: 2.1 },
        { year: 2005, frequency: 26, intensity: 1.9, duration: 2.2 },
        { year: 2010, frequency: 29, intensity: 2.1, duration: 2.4 },
        { year: 2015, frequency: 32, intensity: 2.3, duration: 2.6 },
        { year: 2020, frequency: 35, intensity: 2.5, duration: 2.8 },
        { year: 2023, frequency: 38, intensity: 2.7, duration: 3.0 }
      ],
      summary: {
        percentageIncrease: 58,
        averageIncrease: 0.58,
        significance: 0.02,
        trend: 'Increasing'
      }
    }
  };

  // Initialize Leaflet map
  useEffect(() => {
    if (!mapRef.current || leafletMapRef.current) return;

    const loadMap = async () => {
      setMapLoading(true);
      try {
        // Load Leaflet CSS
        if (!document.querySelector('link[href*="leaflet.css"]')) {
          const link = document.createElement('link');
          link.rel = 'stylesheet';
          link.href = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css';
          document.head.appendChild(link);
        }

        // Load Leaflet JS
        if (!window.L) {
          await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
          });
        }

        // Initialize map
        const map = window.L.map(mapRef.current).setView([selectedRegion.lat, selectedRegion.lon], 8);
        
        window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        const marker = window.L.marker([selectedRegion.lat, selectedRegion.lon]).addTo(map);
        
        map.on('click', (e) => {
          const { lat, lng } = e.latlng;
          marker.setLatLng([lat, lng]);
          setSelectedRegion({
            lat: parseFloat(lat.toFixed(4)),
            lon: parseFloat(lng.toFixed(4)),
            name: `${lat.toFixed(4)}, ${lng.toFixed(4)}`
          });
        });

        leafletMapRef.current = { map, marker };
        setMapInitialized(true);
      } catch (error) {
        console.error('Error initializing map:', error);
      } finally {
        setMapLoading(false);
      }
    };

    loadMap();

    // Cleanup function
    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.map.remove();
        leafletMapRef.current = null;
        setMapInitialized(false);
      }
    };
  }, [selectedRegion.lat, selectedRegion.lon]);

  // Update map marker when region changes
  useEffect(() => {
    if (leafletMapRef.current && mapInitialized) {
      const { map, marker } = leafletMapRef.current;
      map.setView([selectedRegion.lat, selectedRegion.lon], 8);
      marker.setLatLng([selectedRegion.lat, selectedRegion.lon]);
    }
  }, [selectedRegion, mapInitialized]);

  // Modify analyzeClimateData function
  const analyzeClimateData = async () => {
    if (!ws) {
      setAnalysisStatus({ type: 'error', message: 'WebSocket not connected' });
      return;
    }

    setLoading(true);
    setAnalysisData(null);
    setAnalysisStatus({ type: 'starting', message: 'Starting analysis...' });

    const request = {
      region: selectedRegion,
      hazard_type: hazardType,
      time_range: {
        start_year: parseInt(timeRange.start),
        end_year: parseInt(timeRange.end)
      }
    };

    ws.send(JSON.stringify(request));
  };

  // Add WebSocket message handler
  const handleWebSocketMessage = useCallback((data) => {
    if (data.status === 'complete') {
      setAnalysisData(data.data);
      setLoading(false);
      setAnalysisStatus(null);
    } else if (data.status === 'error') {
      setLoading(false);
      setAnalysisStatus({ type: 'error', message: data.message });
    } else {
      setAnalysisStatus({ type: data.status, message: data.message });
    }
  }, []);

  // Update WebSocket effect to use the handler
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(process.env.REACT_APP_WS_URL);
      
      ws.onopen = () => {
        console.log('WebSocket Connected');
        setWs(ws);
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error processing message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setAnalysisStatus({ type: 'error', message: 'Connection error' });
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnected(false);
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      return ws;
    };

    const ws = connectWebSocket();
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [handleWebSocketMessage]);

  // Add status message display
  const renderStatusMessage = () => {
    if (!analysisStatus) return null;

    const statusColors = {
      starting: 'bg-blue-100 text-blue-800',
      fetching: 'bg-purple-100 text-purple-800',
      analyzing: 'bg-indigo-100 text-indigo-800',
      calculating: 'bg-green-100 text-green-800',
      error: 'bg-red-100 text-red-800'
    };

    return (
      <div className={`p-4 rounded-lg mb-4 ${statusColors[analysisStatus.type]}`}>
        <p className="text-sm font-medium">{analysisStatus.message}</p>
      </div>
    );
  };

  const exportData = (format) => {
    if (!analysisData) return;
    
    const data = analysisData.trends;
    
    if (format === 'csv') {
      const csvContent = [
        ['Year', 'Frequency', 'Intensity', 'Duration'],
        ...data.map(d => [d.year, d.frequency, d.intensity, d.duration])
      ].map(row => row.join(',')).join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `climate-analysis-${hazardType}-${Date.now()}.csv`;
      a.click();
    } else if (format === 'pdf') {
      // Simulate PDF export
      alert('PDF export would be implemented with a library like jsPDF');
    }
  };

  const predefinedRegions = [
    { name: 'New York, NY', lat: 40.7128, lon: -74.0060 },
    { name: 'Los Angeles, CA', lat: 34.0522, lon: -118.2437 },
    { name: 'Chicago, IL', lat: 41.8781, lon: -87.6298 },
    { name: 'Miami, FL', lat: 25.7617, lon: -80.1918 },
    { name: 'Phoenix, AZ', lat: 33.4484, lon: -112.0740 }
  ];

  const hazardTypes = [
    { id: 'heatwave', name: 'Heatwave', icon: Thermometer, color: '#ef4444' },
    { id: 'drought', name: 'Drought', icon: Sun, color: '#f59e0b' },
    { id: 'heavy_rain', name: 'Heavy Rainfall', icon: CloudRain, color: '#3b82f6' }
  ];

  const currentHazard = hazardTypes.find(h => h.id === hazardType);

  const chartConfig = {
    lineChart: {
      stroke: "#3b82f6",
      strokeWidth: 3,
      activeDot: { r: 8, strokeWidth: 2 },
      dot: { r: 4, strokeWidth: 2 },
    },
    barChart: {
      radius: [4, 4, 0, 0],
      barGap: 4,
    },
    colors: {
      primary: "#3b82f6",
      secondary: "#10b981",
      danger: "#ef4444",
      warning: "#f59e0b",
      seasonal: {
        winter: "#60a5fa",
        spring: "#34d399",
        summer: "#f59e0b",
        fall: "#7c3aed"
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <TrendingUp className="text-blue-600" />
            Climate Hazard Analysis Dashboard
          </h1>
          <p className="text-gray-600 mt-2">Analyze historical climate patterns and visualize trends in extreme weather events</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Control Panel */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Region Selection */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <MapPin className="text-blue-600" />
                Select Region
              </h3>
              
              <div className="space-y-4">
                <select 
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  value={`${selectedRegion.lat},${selectedRegion.lon}`}
                  onChange={(e) => {
                    const [lat, lon] = e.target.value.split(',');
                    const region = predefinedRegions.find(r => r.lat == lat && r.lon == lon);
                    setSelectedRegion(region || { lat: parseFloat(lat), lon: parseFloat(lon), name: `${lat}, ${lon}` });
                  }}
                >
                  {predefinedRegions.map(region => (
                    <option key={region.name} value={`${region.lat},${region.lon}`}>
                      {region.name}
                    </option>
                  ))}
                </select>
                
                <div className="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    placeholder="Latitude"
                    className="p-2 border border-gray-300 rounded-lg text-sm"
                    value={selectedRegion.lat}
                    onChange={(e) => setSelectedRegion({...selectedRegion, lat: parseFloat(e.target.value) || 0})}
                    step="0.0001"
                  />
                  <input
                    type="number"
                    placeholder="Longitude"
                    className="p-2 border border-gray-300 rounded-lg text-sm"
                    value={selectedRegion.lon}
                    onChange={(e) => setSelectedRegion({...selectedRegion, lon: parseFloat(e.target.value) || 0})}
                    step="0.0001"
                  />
                </div>
              </div>
              
              {/* Interactive Map */}
              <div className="mt-4 h-64 border border-gray-300 rounded-lg overflow-hidden relative">
                <div ref={mapRef} className="w-full h-full"></div>
                {mapLoading && (
                  <div className="absolute inset-0 bg-gray-100 bg-opacity-75 flex items-center justify-center">
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                      <span className="text-sm text-gray-600">Loading map...</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Hazard Type Selection */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Hazard Type</h3>
              <div className="space-y-2">
                {hazardTypes.map(type => {
                  const Icon = type.icon;
                  return (
                    <button
                      key={type.id}
                      onClick={() => setHazardType(type.id)}
                      className={`w-full p-3 rounded-lg border-2 transition-all flex items-center gap-3 ${
                        hazardType === type.id 
                          ? 'border-blue-500 bg-blue-50 text-blue-700' 
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <Icon size={20} style={{ color: type.color }} />
                      <span className="font-medium">{type.name}</span>
                      {hazardType === type.id && <span className="ml-auto text-sm">(Selected)</span>}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Time Range */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Calendar className="text-blue-600" />
                Time Range
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Start Year</label>
                  <input
                    type="number"
                    min="1980"
                    max="2023"
                    value={timeRange.start}
                    onChange={(e) => setTimeRange({...timeRange, start: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">End Year</label>
                  <input
                    type="number"
                    min="1980"
                    max="2023"
                    value={timeRange.end}
                    onChange={(e) => setTimeRange({...timeRange, end: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded-lg"
                  />
                </div>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={analyzeClimateData}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-700 hover:to-blue-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Analyzing...
                </div>
              ) : (
                'Analyze Climate Data'
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            
            {analysisStatus && renderStatusMessage()}

            {analysisData && (
              <>
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Frequency Increase</p>
                        <p className="text-2xl font-bold text-green-600">
                          {analysisData.summary.percentage_increase ? 
                            `+${analysisData.summary.percentage_increase}%` : 
                            'N/A'}
                        </p>
                      </div>
                      <TrendingUp className="text-green-600" size={24} />
                    </div>
                    <p className="text-xs text-gray-500 mt-2">Since {timeRange.start}</p>
                  </div>
                  
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Annual Events</p>
                        <p className="text-2xl font-bold text-blue-600">
                          {analysisData.summary.avg_events_per_year ? 
                            analysisData.summary.avg_events_per_year : 
                            'N/A'}
                        </p>
                      </div>
                      <Calendar className="text-blue-600" size={24} />
                    </div>
                    <p className="text-xs text-gray-500 mt-2">Average per year</p>
                  </div>
                  
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Annual Increase</p>
                        <p className="text-2xl font-bold text-orange-600">
                          {analysisData.summary.average_increase ? 
                            `+${analysisData.summary.average_increase}` : 
                            'N/A'}
                        </p>
                      </div>
                      <TrendingUp className="text-orange-600" size={24} />
                    </div>
                    <p className="text-xs text-gray-500 mt-2">Events per year</p>
                  </div>
                </div>

                {/* Key Insight */}
                <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl shadow-lg p-6 text-white">
                  <h3 className="text-xl font-bold mb-2">Key Insight</h3>
                  <p className="text-blue-100">
                    {currentHazard.name} events have increased by <strong>{analysisData.summary.percentageIncrease}%</strong> in {selectedRegion.name} since {timeRange.start}. 
                    The trend is statistically significant ({analysisData.summary.significance}), indicating a clear pattern of increasing climate hazard frequency in this region.
                  </p>
                </div>

                {/* Enhanced Visualizations */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Frequency Trend */}
                  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
                    <h3 className="text-lg font-semibold mb-4">Event Frequency</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={analysisData.trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="year" 
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <YAxis 
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            borderRadius: '8px',
                            border: 'none',
                            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="frequency" 
                          stroke={chartConfig.colors.primary}
                          strokeWidth={chartConfig.lineChart.strokeWidth}
                          dot={chartConfig.lineChart.dot}
                          activeDot={chartConfig.lineChart.activeDot}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Intensity Distribution */}
                  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
                    <h3 className="text-lg font-semibold mb-4">Event Intensity</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={analysisData.trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="year"
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <YAxis 
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            borderRadius: '8px',
                            border: 'none',
                            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                          }}
                        />
                        <Bar 
                          dataKey="intensity" 
                          fill={chartConfig.colors.secondary}
                          radius={chartConfig.barChart.radius}
                          barSize={20}
                        />
                        <Bar 
                          dataKey="max_intensity" 
                          fill={chartConfig.colors.danger}
                          radius={chartConfig.barChart.radius}
                          barSize={20}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Seasonal Distribution */}
                  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
                    <h3 className="text-lg font-semibold mb-4">Seasonal Pattern</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={analysisData.trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="year"
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <YAxis 
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            borderRadius: '8px',
                            border: 'none',
                            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                          }}
                          formatter={(value, name) => [value, name.charAt(0).toUpperCase() + name.slice(1)]}
                        />
                        <Legend />
                        <Bar 
                          name="Winter" 
                          dataKey="winter" 
                          stackId="season" 
                          fill={chartConfig.colors.seasonal.winter}
                          radius={[4, 4, 0, 0]}
                        />
                        <Bar 
                          name="Spring"
                          dataKey="spring" 
                          stackId="season" 
                          fill={chartConfig.colors.seasonal.spring}
                          radius={[4, 4, 0, 0]}
                        />
                        <Bar 
                          name="Summer"
                          dataKey="summer" 
                          stackId="season" 
                          fill={chartConfig.colors.seasonal.summer}
                          radius={[4, 4, 0, 0]}
                        />
                        <Bar 
                          name="Fall"
                          dataKey="fall" 
                          stackId="season" 
                          fill={chartConfig.colors.seasonal.fall}
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Duration Analysis */}
                  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
                    <h3 className="text-lg font-semibold mb-4">Event Duration</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <ComposedChart data={analysisData.trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="year"
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                        />
                        <YAxis 
                          yAxisId="left"
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                          label={{ value: 'Total Duration (days)', angle: -90, position: 'insideLeft' }}
                        />
                        <YAxis 
                          yAxisId="right" 
                          orientation="right"
                          tick={{ fill: '#6b7280' }}
                          axisLine={{ stroke: '#e5e7eb' }}
                          label={{ value: 'Max Duration (days)', angle: 90, position: 'insideRight' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            borderRadius: '8px',
                            border: 'none',
                            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                          }}
                        />
                        <Legend />
                        <Area 
                          name="Total Duration"
                          type="monotone" 
                          dataKey="total_duration" 
                          fill="#dbeafe"
                          fillOpacity={0.6}
                          stroke={chartConfig.colors.primary}
                          yAxisId="left"
                        />
                        <Line 
                          name="Max Duration"
                          type="monotone" 
                          dataKey="max_duration" 
                          stroke={chartConfig.colors.danger}
                          strokeWidth={3}
                          yAxisId="right"
                          dot={{ r: 4, strokeWidth: 2 }}
                          activeDot={{ r: 8, strokeWidth: 2 }}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Correlation Analysis */}
                <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
                  <h3 className="text-lg font-semibold mb-4">Multi-Metric Analysis</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey="intensity" 
                        name="Intensity"
                        tick={{ fill: '#6b7280' }}
                        axisLine={{ stroke: '#e5e7eb' }}
                      />
                      <YAxis 
                        dataKey="duration" 
                        name="Duration"
                        tick={{ fill: '#6b7280' }}
                        axisLine={{ stroke: '#e5e7eb' }}
                      />
                      <ZAxis dataKey="frequency" range={[50, 400]} name="Frequency" />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        contentStyle={{ 
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          borderRadius: '8px',
                          border: 'none',
                          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                        }}
                      />
                      <Scatter 
                        data={analysisData.trends} 
                        fill={currentHazard.color}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClimateHazardAnalyzer;
