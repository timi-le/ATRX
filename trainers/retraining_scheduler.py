"""
Retraining Scheduler for FX AI-Quant Trading System.

This module provides automated retraining capabilities including:
- Scheduled retraining based on time intervals
- Performance-based retraining triggers
- Data drift detection and response
- Model versioning and rollback capabilities
- Out-of-sample validation before deployment
"""

import asyncio
import json
import schedule
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import structlog
from scipy import stats
import joblib

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from trainers.train_model import ProductionTrainingPipeline, TrainingResult
from trainers.cv_utils import sharpe_ratio_score, information_ratio_score
from core.config.settings import SystemConfig

logger = structlog.get_logger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    timestamp: datetime
    model_path: str
    onnx_path: Optional[str]
    performance_score: float
    validation_score: float
    training_data_hash: str
    config_hash: str
    is_active: bool = False


@dataclass
class RetrainingTrigger:
    """Retraining trigger event."""
    trigger_type: str  # 'scheduled', 'performance', 'drift', 'manual'
    timestamp: datetime
    reason: str
    metrics: Dict[str, Any]
    triggered_by: str


@dataclass
class DriftDetectionResult:
    """Data drift detection result."""
    drift_detected: bool
    drift_score: float
    drift_method: str
    threshold: float
    feature_drifts: Dict[str, float]
    timestamp: datetime


class DataDriftDetector:
    """
    Data drift detection using statistical tests.
    
    Implements multiple drift detection methods including:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Chi-square test
    """
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.1):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset for comparison
            threshold: Drift threshold for triggering alerts
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.logger = logger.bind(component="DataDriftDetector")
    
    def calculate_psi(
        self, 
        reference: np.ndarray, 
        current: np.ndarray, 
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        cur_props = np.where(cur_props == 0, 0.0001, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi
    
    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    def chi2_test(
        self, 
        reference: np.ndarray, 
        current: np.ndarray, 
        bins: int = 10
    ) -> Tuple[float, float]:
        """
        Perform Chi-square test.
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins
            
        Returns:
            Tuple of (statistic, p_value)
        """
        # Create bins
        bin_edges = np.histogram_bin_edges(
            np.concatenate([reference, current]), bins=bins
        )
        
        # Calculate observed frequencies
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Perform chi-square test
        statistic, p_value = stats.chisquare(cur_counts, ref_counts)
        
        return statistic, p_value
    
    def detect_drift(
        self, 
        current_data: np.ndarray, 
        methods: List[str] = ["psi", "ks_test", "chi2_test"]
    ) -> DriftDetectionResult:
        """
        Detect data drift using multiple methods.
        
        Args:
            current_data: Current data to compare against reference
            methods: List of drift detection methods to use
            
        Returns:
            DriftDetectionResult object
        """
        drift_scores = {}
        feature_drifts = {}
        
        # Handle multi-dimensional data
        if len(current_data.shape) > 1:
            # Flatten sequences for drift detection
            if len(current_data.shape) == 3:  # (samples, sequence, features)
                current_flat = current_data.reshape(-1, current_data.shape[-1])
                reference_flat = self.reference_data.reshape(-1, self.reference_data.shape[-1])
            else:
                current_flat = current_data
                reference_flat = self.reference_data
            
            # Check drift for each feature
            for feature_idx in range(current_flat.shape[1]):
                ref_feature = reference_flat[:, feature_idx]
                cur_feature = current_flat[:, feature_idx]
                
                feature_drift_scores = {}
                
                for method in methods:
                    if method == "psi":
                        score = self.calculate_psi(ref_feature, cur_feature)
                        feature_drift_scores[method] = score
                    elif method == "ks_test":
                        statistic, p_value = self.ks_test(ref_feature, cur_feature)
                        feature_drift_scores[method] = statistic
                    elif method == "chi2_test":
                        statistic, p_value = self.chi2_test(ref_feature, cur_feature)
                        feature_drift_scores[method] = statistic
                
                feature_drifts[f"feature_{feature_idx}"] = feature_drift_scores
                
            # Calculate overall drift score (max across features and methods)
            all_scores = []
            for feature_scores in feature_drifts.values():
                all_scores.extend(feature_scores.values())
            
            overall_drift_score = max(all_scores) if all_scores else 0.0
            
        else:
            # 1D data
            for method in methods:
                if method == "psi":
                    score = self.calculate_psi(self.reference_data, current_data)
                    drift_scores[method] = score
                elif method == "ks_test":
                    statistic, p_value = self.ks_test(self.reference_data, current_data)
                    drift_scores[method] = statistic
                elif method == "chi2_test":
                    statistic, p_value = self.chi2_test(self.reference_data, current_data)
                    drift_scores[method] = statistic
            
            overall_drift_score = max(drift_scores.values()) if drift_scores else 0.0
            feature_drifts = {"overall": drift_scores}
        
        # Determine if drift is detected
        drift_detected = overall_drift_score > self.threshold
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            drift_method=methods[0] if methods else "unknown",
            threshold=self.threshold,
            feature_drifts=feature_drifts,
            timestamp=datetime.now()
        )
        
        self.logger.info(
            "Drift detection completed",
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            threshold=self.threshold
        )
        
        return result


class ModelVersionManager:
    """
    Manages model versions and deployment.
    
    Handles model versioning, rollback capabilities, and deployment validation.
    """
    
    def __init__(self, models_dir: Path, max_versions: int = 5):
        """
        Initialize model version manager.
        
        Args:
            models_dir: Directory to store model versions
            max_versions: Maximum number of versions to keep
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        self.versions_file = self.models_dir / "versions.json"
        self.logger = logger.bind(component="ModelVersionManager")
        
        # Load existing versions
        self.versions = self._load_versions()
    
    def _load_versions(self) -> List[ModelVersion]:
        """Load model versions from file."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                
                versions = []
                for item in data:
                    # Convert timestamp string back to datetime
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    versions.append(ModelVersion(**item))
                
                return versions
            except Exception as e:
                self.logger.error("Failed to load versions", error=str(e))
                return []
        return []
    
    def _save_versions(self) -> None:
        """Save model versions to file."""
        try:
            data = []
            for version in self.versions:
                version_dict = asdict(version)
                # Convert datetime to string for JSON serialization
                version_dict['timestamp'] = version.timestamp.isoformat()
                data.append(version_dict)
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error("Failed to save versions", error=str(e))
    
    def add_version(
        self, 
        model_path: str, 
        performance_score: float,
        validation_score: float,
        training_data_hash: str,
        config_hash: str,
        onnx_path: Optional[str] = None
    ) -> ModelVersion:
        """
        Add a new model version.
        
        Args:
            model_path: Path to the model file
            performance_score: Model performance score
            validation_score: Out-of-sample validation score
            training_data_hash: Hash of training data
            config_hash: Hash of model configuration
            onnx_path: Path to ONNX model (optional)
            
        Returns:
            ModelVersion object
        """
        # Generate version string
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create new version
        new_version = ModelVersion(
            version=version,
            timestamp=datetime.now(),
            model_path=model_path,
            onnx_path=onnx_path,
            performance_score=performance_score,
            validation_score=validation_score,
            training_data_hash=training_data_hash,
            config_hash=config_hash,
            is_active=False
        )
        
        # Add to versions list
        self.versions.append(new_version)
        
        # Sort by timestamp (newest first)
        self.versions.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Remove old versions if exceeding max_versions
        if len(self.versions) > self.max_versions:
            old_versions = self.versions[self.max_versions:]
            self.versions = self.versions[:self.max_versions]
            
            # Clean up old model files
            for old_version in old_versions:
                try:
                    if Path(old_version.model_path).exists():
                        Path(old_version.model_path).unlink()
                    if old_version.onnx_path and Path(old_version.onnx_path).exists():
                        Path(old_version.onnx_path).unlink()
                except Exception as e:
                    self.logger.warning(
                        "Failed to delete old model file",
                        path=old_version.model_path,
                        error=str(e)
                    )
        
        # Save versions
        self._save_versions()
        
        self.logger.info(
            "New model version added",
            version=version,
            performance_score=performance_score,
            validation_score=validation_score
        )
        
        return new_version
    
    def activate_version(self, version: str) -> bool:
        """
        Activate a specific model version.
        
        Args:
            version: Version string to activate
            
        Returns:
            True if successful
        """
        # Deactivate all versions
        for v in self.versions:
            v.is_active = False
        
        # Activate specified version
        for v in self.versions:
            if v.version == version:
                v.is_active = True
                self._save_versions()
                
                self.logger.info("Model version activated", version=version)
                return True
        
        self.logger.error("Version not found", version=version)
        return False
    
    def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        for version in self.versions:
            if version.is_active:
                return version
        return None
    
    def get_best_version(self, metric: str = "validation_score") -> Optional[ModelVersion]:
        """
        Get the best model version based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best ModelVersion or None
        """
        if not self.versions:
            return None
        
        if metric == "validation_score":
            return max(self.versions, key=lambda x: x.validation_score)
        elif metric == "performance_score":
            return max(self.versions, key=lambda x: x.performance_score)
        else:
            return self.versions[0]  # Most recent
    
    def rollback_to_previous(self) -> bool:
        """
        Rollback to the previous best version.
        
        Returns:
            True if successful
        """
        current_active = self.get_active_version()
        
        # Find the best version that's not currently active
        best_version = None
        best_score = float('-inf')
        
        for version in self.versions:
            if not version.is_active and version.validation_score > best_score:
                best_version = version
                best_score = version.validation_score
        
        if best_version:
            return self.activate_version(best_version.version)
        
        return False


class RetrainingScheduler:
    """
    Automated retraining scheduler for production ML models.
    
    Handles scheduled retraining, performance monitoring, drift detection,
    and automatic model deployment with validation.
    """
    
    def __init__(
        self, 
        pipeline: ProductionTrainingPipeline,
        config_path: str = "config/model_config.yaml",
        system_config: Optional[SystemConfig] = None
    ):
        """
        Initialize retraining scheduler.
        
        Args:
            pipeline: Training pipeline instance
            config_path: Path to model configuration
            system_config: System configuration
        """
        self.pipeline = pipeline
        self.config_path = Path(config_path)
        self.system_config = system_config or SystemConfig()
        self.logger = logger.bind(component="RetrainingScheduler")
        
        # Load retraining configuration
        self.retraining_config = self.pipeline.model_config.get("retraining", {})
        
        # Initialize components
        self.version_manager = ModelVersionManager(
            models_dir=self.pipeline.output_dirs["models"],
            max_versions=self.retraining_config.get("keep_n_versions", 5)
        )
        
        self.drift_detector = None  # Will be initialized with reference data
        self.reference_data = None
        
        # Tracking
        self.last_training_time = None
        self.performance_history = []
        self.trigger_history = []
        
        # Callbacks
        self.retraining_callbacks: List[Callable] = []
        
        self.logger.info("Retraining scheduler initialized")
    
    def set_reference_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            X: Reference feature data
            y: Reference target data
        """
        self.reference_data = (X, y)
        
        # Initialize drift detector
        drift_threshold = self.retraining_config.get("data_drift_threshold", 0.1)
        self.drift_detector = DataDriftDetector(X, threshold=drift_threshold)
        
        self.logger.info(
            "Reference data set for drift detection",
            X_shape=X.shape,
            y_shape=y.shape,
            drift_threshold=drift_threshold
        )
    
    def add_retraining_callback(self, callback: Callable) -> None:
        """Add a callback function to be called after retraining."""
        self.retraining_callbacks.append(callback)
    
    def check_performance_degradation(
        self, 
        current_score: float, 
        baseline_score: float
    ) -> bool:
        """
        Check if model performance has degraded significantly.
        
        Args:
            current_score: Current model performance score
            baseline_score: Baseline performance score
            
        Returns:
            True if degradation detected
        """
        threshold = self.retraining_config.get("performance_threshold", 0.05)
        degradation = baseline_score - current_score
        
        return degradation > threshold
    
    def check_data_drift(self, current_data: np.ndarray) -> DriftDetectionResult:
        """
        Check for data drift in current data.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            DriftDetectionResult
        """
        if self.drift_detector is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        drift_methods = self.retraining_config.get("drift_methods", ["psi", "ks_test"])
        return self.drift_detector.detect_drift(current_data, methods=drift_methods)
    
    def should_retrain(
        self, 
        current_score: Optional[float] = None,
        current_data: Optional[np.ndarray] = None,
        force_check: bool = False
    ) -> Tuple[bool, List[RetrainingTrigger]]:
        """
        Determine if retraining should be triggered.
        
        Args:
            current_score: Current model performance score
            current_data: Current data for drift detection
            force_check: Force checking even if not scheduled
            
        Returns:
            Tuple of (should_retrain, list_of_triggers)
        """
        triggers = []
        
        # Check if retraining is enabled
        if not self.retraining_config.get("enabled", True):
            return False, triggers
        
        # Check minimum time between retraining
        min_days = self.retraining_config.get("min_days_between_retraining", 1)
        if (self.last_training_time and 
            datetime.now() - self.last_training_time < timedelta(days=min_days) and
            not force_check):
            return False, triggers
        
        # Check scheduled retraining
        frequency = self.retraining_config.get("frequency", "daily")
        if self._is_scheduled_time(frequency) or force_check:
            triggers.append(RetrainingTrigger(
                trigger_type="scheduled",
                timestamp=datetime.now(),
                reason=f"Scheduled retraining ({frequency})",
                metrics={},
                triggered_by="scheduler"
            ))
        
        # Check performance degradation
        if current_score is not None:
            active_version = self.version_manager.get_active_version()
            if active_version:
                baseline_score = active_version.validation_score
                if self.check_performance_degradation(current_score, baseline_score):
                    triggers.append(RetrainingTrigger(
                        trigger_type="performance",
                        timestamp=datetime.now(),
                        reason="Performance degradation detected",
                        metrics={
                            "current_score": current_score,
                            "baseline_score": baseline_score,
                            "degradation": baseline_score - current_score
                        },
                        triggered_by="performance_monitor"
                    ))
        
        # Check data drift
        if current_data is not None and self.drift_detector is not None:
            drift_result = self.check_data_drift(current_data)
            if drift_result.drift_detected:
                triggers.append(RetrainingTrigger(
                    trigger_type="drift",
                    timestamp=datetime.now(),
                    reason="Data drift detected",
                    metrics={
                        "drift_score": drift_result.drift_score,
                        "threshold": drift_result.threshold,
                        "method": drift_result.drift_method
                    },
                    triggered_by="drift_detector"
                ))
        
        should_retrain = len(triggers) > 0
        
        if should_retrain:
            self.trigger_history.extend(triggers)
            self.logger.info(
                "Retraining triggered",
                n_triggers=len(triggers),
                trigger_types=[t.trigger_type for t in triggers]
            )
        
        return should_retrain, triggers
    
    def _is_scheduled_time(self, frequency: str) -> bool:
        """Check if current time matches scheduled retraining time."""
        now = datetime.now()
        scheduled_time = self.retraining_config.get("time", "02:00")
        
        try:
            scheduled_hour, scheduled_minute = map(int, scheduled_time.split(":"))
        except:
            scheduled_hour, scheduled_minute = 2, 0
        
        if frequency == "hourly":
            return True  # Always retrain hourly if enabled
        elif frequency == "daily":
            return (now.hour == scheduled_hour and 
                   now.minute >= scheduled_minute and 
                   now.minute < scheduled_minute + 5)  # 5-minute window
        elif frequency == "weekly":
            return (now.weekday() == 0 and  # Monday
                   now.hour == scheduled_hour and
                   now.minute >= scheduled_minute and 
                   now.minute < scheduled_minute + 5)
        elif frequency == "monthly":
            return (now.day == 1 and  # First day of month
                   now.hour == scheduled_hour and
                   now.minute >= scheduled_minute and 
                   now.minute < scheduled_minute + 5)
        
        return False
    
    async def retrain_models(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        models_to_train: Optional[List[str]] = None,
        triggers: Optional[List[RetrainingTrigger]] = None
    ) -> List[TrainingResult]:
        """
        Perform model retraining.
        
        Args:
            X: Training data features
            y: Training data targets
            models_to_train: List of models to retrain
            triggers: List of triggers that caused retraining
            
        Returns:
            List of training results
        """
        self.logger.info("Starting model retraining")
        start_time = datetime.now()
        
        try:
            # Run training pipeline
            results = self.pipeline.run_full_training_pipeline(
                X=X,
                y=y,
                models_to_train=models_to_train,
                optimize_hyperparameters=True
            )
            
            # Validate and deploy new models
            deployed_models = []
            for result in results:
                if self._validate_new_model(result):
                    # Add to version manager
                    version = self.version_manager.add_version(
                        model_path=result.model_path,
                        performance_score=result.best_score,
                        validation_score=result.metrics.get("sharpe_ratio", result.best_score),
                        training_data_hash=self._hash_data(X, y) if X is not None else "unknown",
                        config_hash=self._hash_config(),
                        onnx_path=result.onnx_path
                    )
                    
                    # Activate new version
                    self.version_manager.activate_version(version.version)
                    deployed_models.append(result)
                    
                    self.logger.info(
                        "New model deployed",
                        model_name=result.model_name,
                        version=version.version,
                        score=result.best_score
                    )
                else:
                    self.logger.warning(
                        "New model failed validation",
                        model_name=result.model_name,
                        score=result.best_score
                    )
            
            # Update tracking
            self.last_training_time = start_time
            self.performance_history.append({
                "timestamp": start_time,
                "results": [asdict(r) for r in results],
                "triggers": [asdict(t) for t in triggers] if triggers else [],
                "deployed_models": len(deployed_models)
            })
            
            # Call callbacks
            for callback in self.retraining_callbacks:
                try:
                    await callback(results, deployed_models)
                except Exception as e:
                    self.logger.error("Callback failed", error=str(e))
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                "Model retraining completed",
                training_time=training_time,
                n_models_trained=len(results),
                n_models_deployed=len(deployed_models)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Model retraining failed", error=str(e), exc_info=True)
            return []
    
    def _validate_new_model(self, result: TrainingResult) -> bool:
        """
        Validate new model before deployment.
        
        Args:
            result: Training result to validate
            
        Returns:
            True if model passes validation
        """
        min_score = self.retraining_config.get("min_validation_score", 0.6)
        
        # Check minimum score
        validation_score = result.metrics.get("sharpe_ratio", result.best_score)
        if validation_score < min_score:
            return False
        
        # Compare with current active model
        active_version = self.version_manager.get_active_version()
        if active_version:
            # New model should be better than current
            if validation_score <= active_version.validation_score:
                return False
        
        return True
    
    def _hash_data(self, X: np.ndarray, y: np.ndarray) -> str:
        """Generate hash for training data."""
        import hashlib
        
        # Create hash from data shapes and sample values
        data_str = f"{X.shape}_{y.shape}_{np.mean(X)}_{np.std(X)}_{np.mean(y)}_{np.std(y)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _hash_config(self) -> str:
        """Generate hash for model configuration."""
        import hashlib
        
        config_str = json.dumps(self.pipeline.model_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def start_monitoring(self, check_interval: int = 3600) -> None:
        """
        Start monitoring loop for automatic retraining.
        
        Args:
            check_interval: Check interval in seconds
        """
        self.logger.info("Starting retraining monitoring", check_interval=check_interval)
        
        def check_and_retrain():
            """Check if retraining is needed and trigger if necessary."""
            try:
                should_retrain, triggers = self.should_retrain()
                
                if should_retrain:
                    self.logger.info("Triggering automatic retraining")
                    # Run retraining in background
                    asyncio.create_task(self.retrain_models(triggers=triggers))
                    
            except Exception as e:
                self.logger.error("Monitoring check failed", error=str(e))
        
        # Schedule periodic checks
        schedule.every(check_interval).seconds.do(check_and_retrain)
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of retraining scheduler."""
        active_version = self.version_manager.get_active_version()
        
        return {
            "retraining_enabled": self.retraining_config.get("enabled", True),
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "active_model_version": active_version.version if active_version else None,
            "total_versions": len(self.version_manager.versions),
            "total_triggers": len(self.trigger_history),
            "recent_triggers": [asdict(t) for t in self.trigger_history[-5:]],
            "drift_detector_ready": self.drift_detector is not None,
            "reference_data_set": self.reference_data is not None
        } 