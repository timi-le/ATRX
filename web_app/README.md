# FX AI-Quant Parameter Tuner Web Application

A beautiful React-based web interface for tuning FX AI-Quant trading system parameters in real-time.

## 🚀 Features

- **Real-time Parameter Tuning**: Adjust trading parameters with immediate visual feedback
- **Beautiful UI**: Built with Tailwind CSS and shadcn/ui components
- **Live Configuration Management**: Load, edit, validate, and save YAML configurations
- **Parameter Validation**: Built-in validation with min/max ranges and type checking
- **Version History**: Track configuration changes with automatic versioning
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠 Technology Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **shadcn/ui** for component library
- **Axios** for API communication
- **React Query** for data management

### Backend
- **FastAPI** for RESTful API
- **Pydantic** for data validation
- **YAML** configuration management
- **Uvicorn** ASGI server

## 📁 Project Structure

```
web_app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ParamSlider.tsx
│   │   │   ├── ParamToggle.tsx
│   │   │   ├── ParamDropdown.tsx
│   │   │   └── SaveBanner.tsx
│   │   ├── pages/
│   │   │   └── ParameterTuner.tsx
│   │   ├── api/
│   │   │   └── config.ts   # API client
│   │   ├── lib/
│   │   │   └── utils.ts    # Utility functions
│   │   ├── App.tsx         # Main application
│   │   ├── index.tsx       # Application entry point
│   │   └── index.css       # Global styles
│   ├── package.json        # Frontend dependencies
│   ├── tailwind.config.js  # Tailwind configuration
│   └── postcss.config.js   # PostCSS configuration
└── README.md               # This file
```

## 🔧 Setup Instructions

### Prerequisites

- **Node.js** 16+ and npm
- **Python** 3.8+ and pip
- **Git** for cloning the repository

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Fx_Quant_System/web_app
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py --reload
```

The backend will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from web_app root)
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at: `http://localhost:3000`

## 🎯 Usage Guide

### Starting the Application

1. **Start the Backend**:
   ```bash
   cd backend
   python main.py --reload
   ```

2. **Start the Frontend**:
   ```bash
   cd frontend
   npm start
   ```

3. **Open your browser** to `http://localhost:3000`

### Parameter Tuning Interface

#### Slider Components
- **Numeric Parameters**: Use sliders for continuous values like `grid_step_factor`, `risk_per_trade`
- **Real-time Updates**: Values update immediately as you drag the slider
- **Manual Input**: Click the numeric input to enter precise values
- **Range Validation**: Values are automatically clamped to valid ranges

#### Toggle Components
- **Boolean Parameters**: Use switches for on/off settings like `session_filter_enabled`
- **Visual Feedback**: Green indicates enabled, gray indicates disabled
- **One-click Toggle**: Click anywhere on the switch to toggle

#### Dropdown Components
- **Enum Parameters**: Use dropdowns for choice-based parameters like `tp_type`, `sl_type`
- **Available Options**: Shows all valid choices
- **Search Functionality**: Type to filter options (if implemented)

### Saving Changes

1. **Make Parameter Changes**: Adjust any parameters using the interface
2. **Review Changes**: Pending changes are highlighted in the header
3. **Save Configuration**: Click "Save Configuration" button
4. **Confirmation**: Success banner appears confirming the save

### Version Management

- **Automatic Versioning**: Each save creates a timestamped backup
- **Version History**: Access previous configurations
- **Rollback Capability**: Restore any previous version

## 🔄 API Endpoints

### Configuration Management
- `GET /config` - Get current configuration
- `GET /config/parameters` - Get all parameters with metadata
- `POST /config/parameter` - Update single parameter
- `POST /config/parameters/bulk` - Update multiple parameters
- `POST /config/save` - Save configuration to file

### Version Management
- `GET /config/versions` - List all configuration versions
- `POST /config/restore/{timestamp}` - Restore specific version

### Validation & Health
- `GET /config/validate` - Validate current configuration
- `GET /health` - API health check
- `GET /config/schema` - Get parameter schema

## 🧪 Development

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production

```bash
# Build frontend
cd frontend
npm run build

# The build files will be in frontend/build/
```

### Environment Variables

Create `.env` files for configuration:

#### Backend (.env)
```env
API_HOST=127.0.0.1
API_PORT=8000
CONFIG_PATH=../../config/live_config.yaml
LOG_LEVEL=INFO
```

#### Frontend (.env)
```env
REACT_APP_API_URL=http://localhost:8000
```

## 🎨 Customization

### Styling
- **Tailwind CSS**: Modify `tailwind.config.js` for theme customization
- **Component Styles**: Edit individual component files
- **Global Styles**: Update `src/index.css`

### Adding New Parameters
1. Update the backend validation rules in `tools/parameter_tuner.py`
2. The frontend will automatically detect and render new parameters
3. Choose appropriate component type based on parameter type

### Custom Components
- Create new components in `src/components/`
- Follow the existing pattern for parameter components
- Use TypeScript interfaces for type safety

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start
- Check Python version (3.8+ required)
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Ensure the config file path is correct

#### Frontend Won't Start
- Check Node.js version (16+ required)
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

#### API Connection Issues
- Verify backend is running on port 8000
- Check CORS settings in backend
- Confirm proxy setting in `package.json`

#### Parameter Changes Not Saving
- Check backend logs for validation errors
- Verify write permissions to config directory
- Ensure parameter values are within valid ranges

### Debug Mode

Enable debug logging:

```bash
# Backend
export LOG_LEVEL=DEBUG
python main.py --reload

# Frontend
npm start
# Open browser dev tools for console logs
```

## 📈 Performance

### Optimization Tips
- Parameters update in real-time for immediate feedback
- Bulk updates reduce API calls when saving multiple changes
- Lazy loading for large parameter sets
- Debounced input validation

### Monitoring
- Backend provides health check endpoint
- Frontend shows connection status
- Parameter validation feedback
- Save operation status indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-parameter-type`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is part of the FX AI-Quant Trading System. See the main project license for details.

## 🆘 Support

For issues and questions:
1. Check this README for common solutions
2. Review the API documentation at `http://localhost:8000/docs`
3. Open an issue in the repository
4. Contact the development team

---

**Happy Parameter Tuning! 🎛️📈**
