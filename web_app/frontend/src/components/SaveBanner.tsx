import React, { useEffect, useState } from 'react';

interface SaveBannerProps {
  show: boolean;
  message: string;
  type?: 'success' | 'error' | 'warning';
  onClose?: () => void;
  autoHide?: boolean;
  duration?: number;
}

export const SaveBanner: React.FC<SaveBannerProps> = ({
  show,
  message,
  type = 'success',
  onClose,
  autoHide = true,
  duration = 3000,
}) => {
  const [isVisible, setIsVisible] = useState(show);

  useEffect(() => {
    if (show) {
      setIsVisible(true);
      
      if (autoHide) {
        const timer = setTimeout(() => {
          setIsVisible(false);
          onClose?.();
        }, duration);
        
        return () => clearTimeout(timer);
      }
    } else {
      setIsVisible(false);
    }
  }, [show, autoHide, duration, onClose]);

  if (!isVisible) return null;

  const getTypeStyles = () => {
    switch (type) {
      case 'success':
        return 'bg-green-500 text-white border-green-600';
      case 'error':
        return 'bg-red-500 text-white border-red-600';
      case 'warning':
        return 'bg-yellow-500 text-white border-yellow-600';
      default:
        return 'bg-green-500 text-white border-green-600';
    }
  };

  return (
    <div
      className={`
        fixed top-4 right-4 z-50 min-w-80 max-w-md px-4 py-3 rounded-lg shadow-lg border
        ${getTypeStyles()}
      `}
    >
      <div className="flex items-center space-x-3">
        <span className="text-sm font-medium flex-1">{message}</span>
        {onClose && (
          <button
            onClick={onClose}
            className="ml-2 text-white hover:text-gray-200 focus:outline-none"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}; 