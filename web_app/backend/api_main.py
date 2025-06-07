#!/usr/bin/env python3
"""
FastAPI Backend for Parameter Tuning Web Application

This backend provides RESTful API endpoints for managing trading system parameters
through a web interface. It integrates with the existing ParameterTuner class
to provide real-time configuration management.

Features:
- Load current configuration as JSON
- Update parameters with validation
- Version management and history
- Parameter suggestions and optimization
- Real-time configuration validation
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tools.parameter_tuner import ParameterTuner, ParameterValidationError
except ImportError:
    # Fallback if import fails
    class ParameterTuner:
        def __init__(self, *args, **kwargs):
            pass
    class ParameterValidationError(Exception):
        pass

# FastAPI app initialization
app = FastAPI(
    title="FX AI-Quant Parameter Tuner API",
    description="RESTful API for managing trading system parameters",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3002"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parameter tuner
tuner = ParameterTuner()

# Pydantic models for request/response validation
class ParameterUpdate(BaseModel):
    """Model for parameter update requests"""
    key: str = Field(..., description="Parameter key (supports dot notation)")
    value: Any = Field(..., description="New parameter value")
    validate: bool = Field(True, description="Whether to validate the parameter")

class BulkParameterUpdate(BaseModel):
    """Model for bulk parameter updates"""
    parameters: Dict[str, Any] = Field(..., description="Dictionary of parameter updates")
    validate: bool = Field(True, description="Whether to validate parameters")

class ConfigurationResponse(BaseModel):
    """Model for configuration responses"""
    config: Dict[str, Any] = Field(..., description="Current configuration")
    metadata: Dict[str, Any] = Field(..., description="Configuration metadata")
    last_modified: str = Field(..., description="Last modification timestamp")

class ParameterInfo(BaseModel):
    """Model for parameter information"""
    name: str
    type: str
    description: str
    current_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None

class ApiResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# API Endpoints

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FX AI-Quant Parameter Tuner API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if tuner is working
        config = tuner.config
        return ApiResponse(
            success=True,
            message="API is healthy",
            data={
                "status": "operational",
                "config_loaded": bool(config),
                "parameters_count": len(tuner.VALIDATION_RULES)
            }
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"Health check failed: {str(e)}"
        )

@app.get("/config", response_model=ConfigurationResponse)
async def get_configuration():
    """
    Get the current configuration as JSON
    
    Returns:
        Current configuration with metadata
    """
    try:
        config = tuner.config.copy()
        
        # Add runtime metadata
        metadata = config.get("metadata", {})
        metadata.update({
            "api_version": "1.0.0",
            "loaded_at": datetime.now().isoformat(),
            "parameter_count": sum(len(section) for section in config.values() if isinstance(section, dict))
        })
        
        return ConfigurationResponse(
            config=config,
            metadata=metadata,
            last_modified=metadata.get("last_modified", datetime.now().isoformat())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")

@app.get("/config/parameters")
async def get_all_parameters():
    """
    Get all parameters with their current values, types, and validation rules
    
    Returns:
        Dictionary of all parameters with metadata
    """
    try:
        parameters = {}
        
        for param_name, rules in tuner.VALIDATION_RULES.items():
            try:
                current_value = tuner.get_parameter(param_name)
            except:
                # Try to find parameter in nested structure
                current_value = None
                for section_name, section in tuner.config.items():
                    if isinstance(section, dict) and param_name in section:
                        current_value = section[param_name]
                        break
            
            # Convert validation rules to serializable format
            serializable_rules = {}
            for key, value in rules.items():
                if key == "type":
                    # Convert type object to string name
                    if hasattr(value, '__name__'):
                        serializable_rules[key] = value.__name__
                    else:
                        serializable_rules[key] = str(value)
                else:
                    serializable_rules[key] = value
            
            # Get type name safely
            type_name = "unknown"
            if "type" in rules:
                if hasattr(rules["type"], '__name__'):
                    type_name = rules["type"].__name__
                else:
                    type_name = str(rules["type"])
            
            parameters[param_name] = {
                "current_value": current_value,
                "type": type_name,
                "description": rules.get("description", ""),
                "min_value": rules.get("min"),
                "max_value": rules.get("max"),
                "allowed_values": rules.get("allowed_values"),
                "validation_rules": serializable_rules
            }
        
        return ApiResponse(
            success=True,
            message="Parameters retrieved successfully",
            data=parameters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get parameters: {str(e)}")

@app.get("/config/parameter/{param_name}")
async def get_parameter(param_name: str):
    """
    Get a specific parameter value and information
    
    Args:
        param_name: Name of the parameter to retrieve
        
    Returns:
        Parameter information and current value
    """
    try:
        value = tuner.get_parameter(param_name)
        info = tuner.get_parameter_info(param_name)
        
        return ApiResponse(
            success=True,
            message=f"Parameter {param_name} retrieved successfully",
            data={
                "name": param_name,
                "value": value,
                "info": info
            }
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Parameter not found: {str(e)}")

@app.post("/config/parameter")
async def update_parameter(update: ParameterUpdate):
    """
    Update a single parameter
    
    Args:
        update: Parameter update request
        
    Returns:
        Success confirmation and updated value
    """
    try:
        success = tuner.set_parameter(update.key, update.value, validate=update.validate)
        
        if success:
            return ApiResponse(
                success=True,
                message=f"Parameter {update.key} updated successfully",
                data={
                    "key": update.key,
                    "old_value": tuner.get_parameter(update.key) if success else None,
                    "new_value": update.value
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Parameter update failed")
            
    except ParameterValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@app.post("/config/parameters/bulk")
async def update_parameters_bulk(update: BulkParameterUpdate):
    """
    Update multiple parameters at once
    
    Args:
        update: Bulk parameter update request
        
    Returns:
        Success confirmation and update summary
    """
    try:
        results = {}
        errors = {}
        
        for key, value in update.parameters.items():
            try:
                success = tuner.set_parameter(key, value, validate=update.validate)
                results[key] = {
                    "success": success,
                    "new_value": value
                }
            except Exception as e:
                errors[key] = str(e)
        
        return ApiResponse(
            success=len(errors) == 0,
            message=f"Bulk update completed. {len(results)} successful, {len(errors)} failed",
            data={
                "successful_updates": results,
                "errors": errors,
                "summary": {
                    "total_updates": len(update.parameters),
                    "successful": len(results),
                    "failed": len(errors)
                }
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk update failed: {str(e)}")

@app.post("/config/save")
async def save_configuration(background_tasks: BackgroundTasks, versioned: bool = True):
    """
    Save the current configuration to file
    
    Args:
        versioned: Whether to create a versioned backup
        
    Returns:
        Save confirmation and file path
    """
    try:
        version_file = tuner.save_config(versioned=versioned)
        
        return ApiResponse(
            success=True,
            message="Configuration saved successfully",
            data={
                "version_file": str(version_file) if versioned else None,
                "config_file": str(tuner.config_path),
                "versioned": versioned,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.get("/config/versions")
async def get_configuration_versions():
    """
    Get list of all configuration versions
    
    Returns:
        List of available configuration versions
    """
    try:
        versions = tuner.list_versions()
        
        return ApiResponse(
            success=True,
            message="Configuration versions retrieved successfully",
            data={
                "versions": versions,
                "count": len(versions)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get versions: {str(e)}")

@app.post("/config/restore/{timestamp}")
async def restore_configuration(timestamp: str):
    """
    Restore configuration from a specific version
    
    Args:
        timestamp: Timestamp of the version to restore
        
    Returns:
        Restore confirmation
    """
    try:
        success = tuner.restore_version(timestamp)
        
        if success:
            return ApiResponse(
                success=True,
                message=f"Configuration restored from version {timestamp}",
                data={
                    "restored_timestamp": timestamp,
                    "current_config": tuner.config
                }
            )
        else:
            raise HTTPException(status_code=404, detail=f"Version {timestamp} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@app.get("/config/validate")
async def validate_configuration():
    """
    Validate the current configuration
    
    Returns:
        Validation results and any errors
    """
    try:
        validation_results = tuner.validate_all_parameters()
        
        return ApiResponse(
            success=validation_results["valid"],
            message="Configuration validation completed",
            data=validation_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/config/suggest/{param_name}")
async def get_parameter_suggestions(param_name: str):
    """
    Get optimization suggestions for a specific parameter
    
    Args:
        param_name: Parameter to get suggestions for
        
    Returns:
        Optimization suggestions and recommendations
    """
    try:
        suggestions = tuner.suggest_optimization(param_name)
        
        return ApiResponse(
            success=True,
            message=f"Suggestions for {param_name} generated successfully",
            data=suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Suggestions failed: {str(e)}")

@app.get("/config/schema")
async def get_parameter_schema():
    """
    Get the parameter validation schema
    
    Returns:
        Complete parameter schema with validation rules
    """
    try:
        return ApiResponse(
            success=True,
            message="Parameter schema retrieved successfully",
            data={
                "validation_rules": tuner.VALIDATION_RULES,
                "schema_version": "1.0.0",
                "parameters_count": len(tuner.VALIDATION_RULES)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ApiResponse(
            success=False,
            message="Endpoint not found"
        ).dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            success=False,
            message="Internal server error"
        ).dict()
    )

# Development server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter Tuner API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting Parameter Tuner API server on http://{args.host}:{args.port}")
    print("API Documentation: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    ) 