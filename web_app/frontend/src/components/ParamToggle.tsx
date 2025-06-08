import React from 'react';
import { Parameter } from '../api/config';

interface ParamToggleProps {
  label: string;
  value: boolean;
  parameter: Parameter;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}

export const ParamToggle: React.FC<ParamToggleProps> = ({
  label,
  value,
  parameter,
  onChange,
  disabled = false,
}) => {
  const handleToggle = () => {
    if (!disabled) {
      onChange(!value);
    }
  };

  return (
    <div className="space-y-3 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-700">
          {label}
        </label>
        <button
          type="button"
          onClick={handleToggle}
          disabled={disabled}
          className={`
            relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            ${value ? 'bg-blue-600' : 'bg-gray-200'}
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <span
            className={`
              inline-block h-4 w-4 rounded-full bg-white shadow-lg transform transition-transform duration-200 ease-in-out
              ${value ? 'translate-x-6' : 'translate-x-1'}
            `}
          />
        </button>
      </div>

      <div className="flex items-center justify-between text-sm">
        <span className={`${value ? 'text-green-600 font-medium' : 'text-gray-500'}`}>
          {value ? 'Enabled' : 'Disabled'}
        </span>
        <span className="text-xs text-gray-400">
          {value ? 'ON' : 'OFF'}
        </span>
      </div>

      {parameter.description && (
        <p className="text-xs text-gray-500">{parameter.description}</p>
      )}
    </div>
  );
};
