import React, { useState, useEffect } from 'react';
import { Parameter } from '../api/config';

interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  parameter: Parameter;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export const ParamSlider: React.FC<ParamSliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  parameter,
  onChange,
  disabled = false,
}) => {
  const [localValue, setLocalValue] = useState(value);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value);
    setLocalValue(newValue);
    onChange(newValue);
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value);
    if (!isNaN(newValue)) {
      setLocalValue(newValue);
    }
  };

  const handleInputBlur = () => {
    setIsEditing(false);
    // Clamp value to min/max bounds
    const clampedValue = Math.max(min, Math.min(max, localValue));
    setLocalValue(clampedValue);
    onChange(clampedValue);
  };

  const handleInputKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleInputBlur();
    }
  };

  const percentage = ((localValue - min) / (max - min)) * 100;

  return (
    <div className="space-y-3 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium text-gray-700">
          {label}
        </label>
        <div className="flex items-center space-x-2">
          <input
            type="number"
            value={localValue}
            min={min}
            max={max}
            step={step}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            onKeyPress={handleInputKeyPress}
            onFocus={() => setIsEditing(true)}
            disabled={disabled}
            className="w-20 px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <span className="text-xs text-gray-500">
            ({min} - {max})
          </span>
        </div>
      </div>

      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleSliderChange}
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`
          }}
        />
        <div 
          className="absolute top-0 h-2 bg-blue-500 rounded-lg pointer-events-none"
          style={{ width: `${percentage}%` }}
        />
      </div>

      {parameter.description && (
        <p className="text-xs text-gray-500">{parameter.description}</p>
      )}

      <div className="flex justify-between text-xs text-gray-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .slider:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}; 