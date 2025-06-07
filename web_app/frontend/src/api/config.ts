import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Parameter {
  current_value: any;
  type: string;
  description: string;
  min_value?: number;
  max_value?: number;
  allowed_values?: string[];
  validation_rules?: any;
}

export interface ParameterUpdate {
  key: string;
  value: any;
  validate?: boolean;
}

export interface Configuration {
  config: Record<string, any>;
  metadata: Record<string, any>;
  last_modified: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  timestamp: string;
}

export interface ConfigVersion {
  timestamp: string;
  filename: string;
  size: number;
  description?: string;
}

export class ConfigAPI {
  // Get current configuration
  static async getConfiguration(): Promise<Configuration> {
    const response = await api.get<Configuration>('/config');
    return response.data;
  }

  // Get all parameters with metadata
  static async getAllParameters(): Promise<Record<string, Parameter>> {
    const response = await api.get<ApiResponse<Record<string, Parameter>>>('/config/parameters');
    return response.data.data || {};
  }

  // Get specific parameter
  static async getParameter(paramName: string): Promise<{ name: string; value: any; info: any }> {
    const response = await api.get<ApiResponse<{ name: string; value: any; info: any }>>(
      `/config/parameter/${paramName}`
    );
    return response.data.data!;
  }

  // Update single parameter
  static async updateParameter(update: ParameterUpdate): Promise<ApiResponse> {
    const response = await api.post<ApiResponse>('/config/parameter', update);
    return response.data;
  }

  // Update multiple parameters
  static async updateParametersBulk(parameters: Record<string, any>, validate: boolean = true): Promise<ApiResponse> {
    const response = await api.post<ApiResponse>('/config/parameters/bulk', {
      parameters,
      validate,
    });
    return response.data;
  }

  // Save configuration
  static async saveConfiguration(versioned: boolean = true): Promise<ApiResponse> {
    const response = await api.post<ApiResponse>(`/config/save?versioned=${versioned}`);
    return response.data;
  }

  // Get configuration versions
  static async getConfigurationVersions(): Promise<ConfigVersion[]> {
    const response = await api.get<ApiResponse<{ versions: ConfigVersion[]; count: number }>>(
      '/config/versions'
    );
    return response.data.data?.versions || [];
  }

  // Restore configuration version
  static async restoreConfiguration(timestamp: string): Promise<ApiResponse> {
    const response = await api.post<ApiResponse>(`/config/restore/${timestamp}`);
    return response.data;
  }

  // Validate configuration
  static async validateConfiguration(): Promise<ApiResponse> {
    const response = await api.get<ApiResponse>('/config/validate');
    return response.data;
  }

  // Get parameter suggestions
  static async getParameterSuggestions(paramName: string): Promise<ApiResponse> {
    const response = await api.get<ApiResponse>(`/config/suggest/${paramName}`);
    return response.data;
  }

  // Get parameter schema
  static async getParameterSchema(): Promise<ApiResponse> {
    const response = await api.get<ApiResponse>('/config/schema');
    return response.data;
  }

  // Health check
  static async healthCheck(): Promise<ApiResponse> {
    const response = await api.get<ApiResponse>('/health');
    return response.data;
  }
}

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    
    if (error.response?.status === 404) {
      throw new Error('Resource not found');
    } else if (error.response?.status === 400) {
      throw new Error(error.response.data?.detail || 'Bad request');
    } else if (error.response?.status === 500) {
      throw new Error('Server error. Please try again later.');
    }
    
    throw new Error(error.message || 'An error occurred');
  }
);

export default api; 