import React, { useState, useEffect } from 'react';
import './index.css';

interface Parameter {
  name: string;
  current_value: any;
  type: string;
  description: string;
  min_value?: number;
  max_value?: number;
  allowed_values?: string[];
  validation_rules?: any;
}

interface ApiResponse {
  success: boolean;
  message: string;
  data?: any;
  timestamp: string;
}

function App() {
  const [parameters, setParameters] = useState<Record<string, Parameter>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const API_BASE = 'http://localhost:8080';

  // Check API health
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const data = await response.json();
      setApiStatus(data.success ? 'online' : 'offline');
    } catch (error) {
      setApiStatus('offline');
    }
  };

  // Load parameters on component mount
  useEffect(() => {
    if (apiStatus === 'online') {
      loadParameters();
    }
  }, [apiStatus]);

  const loadParameters = async () => {
    try {
      console.log("Attempting to load parameters...");
      setLoading(true);
      setError(null);
      
      console.log(`Fetching from: ${API_BASE}/config/parameters`);
      const response = await fetch(`${API_BASE}/config/parameters`);
      console.log("Fetch response received:", response);

      if (!response.ok) {
        console.error("Fetch response not OK:", response.status);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: ApiResponse = await response.json();
      console.log("Parsed JSON data:", data);

      if (data.success) {
        console.log("Setting parameters state.");
        setParameters(data.data);
        setSuccessMessage('Configuration loaded successfully');
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        console.error("API returned success=false:", data.message);
        throw new Error(data.message);
      }
    } catch (error) {
      console.error("Caught an error in loadParameters:", error);
      setError(`Failed to load parameters: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      console.log("Finished loading parameters, setting loading to false.");
      setLoading(false);
    }
  };

  const updateParameter = async (paramName: string, value: any) => {
    try {
      const response = await fetch(`${API_BASE}/config/parameter`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          key: paramName,
          value: value,
          validate: true
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      if (data.success) {
        // Update local state
        setParameters(prev => ({
          ...prev,
          [paramName]: {
            ...prev[paramName],
            current_value: value
          }
        }));
        setSuccessMessage(`Updated ${paramName} successfully`);
        setTimeout(() => setSuccessMessage(null), 2000);
      } else {
        throw new Error(data.message);
      }
    } catch (error) {
      setError(`Failed to update ${paramName}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setError(null), 3000);
    }
  };

  const saveConfiguration = async () => {
    try {
      setSaving(true);
      setError(null);

      const response = await fetch(`${API_BASE}/config/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ versioned: true }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      if (data.success) {
        setSuccessMessage('Configuration saved successfully');
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        throw new Error(data.message);
      }
    } catch (error) {
      setError(`Failed to save configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setError(null), 3000);
    } finally {
      setSaving(false);
    }
  };

  const resetConfiguration = async () => {
    if (window.confirm('Are you sure you want to reset all parameters to defaults? This action cannot be undone.')) {
      try {
        // Reload the original configuration
        await loadParameters();
        setSuccessMessage('Configuration reset to defaults');
        setTimeout(() => setSuccessMessage(null), 3000);
      } catch (error) {
        setError('Failed to reset configuration');
        setTimeout(() => setError(null), 3000);
      }
    }
  };

  const handleSliderChange = (paramName: string, value: string) => {
    const numValue = parseFloat(value);
    updateParameter(paramName, numValue);
  };

  const handleInputChange = (paramName: string, value: string) => {
    const param = parameters[paramName];
    if (!param) return;

    let processedValue: any = value;
    
    if (param.type === 'int') {
      processedValue = parseInt(value);
    } else if (param.type === 'float') {
      processedValue = parseFloat(value);
    } else if (param.type === 'bool') {
      processedValue = value === 'true';
    }

    updateParameter(paramName, processedValue);
  };

  const handleToggleChange = (paramName: string) => {
    const currentValue = parameters[paramName]?.current_value;
    updateParameter(paramName, !currentValue);
  };

  const renderParameterControl = (paramName: string, param: Parameter) => {
    const { current_value, type, min_value, max_value, allowed_values, description } = param;

    if (type === 'bool') {
      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 capitalize">
            {paramName.replace(/_/g, ' ')}
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">Enabled</label>
              <button
                type="button"
                onClick={() => handleToggleChange(paramName)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  current_value ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 rounded-full bg-white shadow-lg transform transition-transform duration-200 ease-in-out ${
                    current_value ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            <div className="text-sm text-gray-500">
              Status: <span className="font-medium">{current_value ? 'Enabled' : 'Disabled'}</span>
            </div>
            <p className="text-xs text-gray-500">{description}</p>
          </div>
        </div>
      );
    }

    if (allowed_values && allowed_values.length > 0) {
      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 capitalize">
            {paramName.replace(/_/g, ' ')}
          </h3>
          <div className="space-y-4">
            <select 
              value={current_value || ''}
              onChange={(e) => handleInputChange(paramName, e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {allowed_values.map(value => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
            <div className="text-sm text-gray-500">
              Selected: <span className="font-medium capitalize">{current_value}</span>
            </div>
            <p className="text-xs text-gray-500">{description}</p>
          </div>
        </div>
      );
    }

    if (type === 'int' || type === 'float') {
      const step = type === 'int' ? 1 : 0.001;
      const min = min_value ?? 0;
      const max = max_value ?? 100;

      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 capitalize">
            {paramName.replace(/_/g, ' ')}
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-gray-700">Current Value</label>
              <input
                type="number"
                value={current_value || ''}
                onChange={(e) => handleInputChange(paramName, e.target.value)}
                step={step}
                min={min}
                max={max}
                className="w-20 px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={current_value || min}
              onChange={(e) => handleSliderChange(paramName, e.target.value)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>{min}</span>
              <span>{max}</span>
            </div>
            <p className="text-xs text-gray-500">{description}</p>
          </div>
        </div>
      );
    }

    return null;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600">Loading parameters...</p>
        </div>
      </div>
    );
  }

  const filteredParameters = Object.entries(parameters).filter(([name, param]) => {
    // Only show commonly tuned parameters
    const displayParams = [
      'grid_step_factor', 'risk_per_trade', 'session_filter_enabled', 
      'tp_type', 'momentum_weight', 'rsi_period', 'max_levels',
      'volatility_filter_threshold', 'ma_fast_period', 'ma_slow_period'
    ];
    return displayParams.includes(name);
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">FX AI-Quant Parameter Tuner</h1>
              <p className="mt-1 text-gray-600">Real-time configuration management for trading parameters</p>
            </div>
            <div className="flex space-x-3">
              <button 
                onClick={resetConfiguration}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
              >
                Reset Changes
              </button>
              <button 
                onClick={saveConfiguration}
                disabled={saving}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Configuration'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error/Success Messages */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {successMessage && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-green-800">{successMessage}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* API Status */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4">
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
          apiStatus === 'online' ? 'bg-green-100 text-green-800' :
          apiStatus === 'offline' ? 'bg-red-100 text-red-800' :
          'bg-yellow-100 text-yellow-800'
        }`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${
            apiStatus === 'online' ? 'bg-green-400' :
            apiStatus === 'offline' ? 'bg-red-400' :
            'bg-yellow-400'
          }`}></div>
          API Status: {apiStatus === 'online' ? 'Connected' : apiStatus === 'offline' ? 'Disconnected' : 'Checking...'}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {apiStatus === 'offline' ? (
          <div className="text-center py-12">
            <p className="text-lg text-gray-600 mb-4">Backend API is not available</p>
            <button 
              onClick={checkApiHealth}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Retry Connection
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredParameters.map(([name, param]) => (
              <div key={name}>
                {renderParameterControl(name, param)}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-white border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center text-sm text-gray-500">
            <div>{filteredParameters.length} parameters loaded • API Status: {apiStatus}</div>
            <div>Last updated: {new Date().toLocaleTimeString()}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 