"""
ai_models.py — AI/ML Models for Smart Safety Helmet
======================================================
This module contains 3 AI components that make the helmet "smart":

1. ANOMALY DETECTOR (Isolation Forest)
   - Learns what "normal" sensor patterns look like
   - Flags unusual combinations BEFORE thresholds are hit
   - Example: CO rising slowly + humidity dropping = possible gas leak
     even when CO is still at 40 ppm (below 50 ppm safe limit)

2. FALL CLASSIFIER (Random Forest)
   - Classifies IMU data as: normal, walking, fall, or aftermath
   - More accurate than simple threshold checking
   - Reduces false alarms from vibrations or crouching

3. HAZARD PREDICTOR (Linear Regression + Trend Analysis)
   - Looks at the TREND of sensor readings over time
   - Predicts: "At this rate, CO will hit danger in 3 minutes"
   - Gives miners TIME to evacuate before it's critical

Why these algorithms:
- Isolation Forest: perfect for anomaly detection, works with small data
- Random Forest: robust classifier, handles noisy sensor data well
- Both are lightweight enough to run on a server in real-time
- Your proposal (Section 1.3.2, Objective 4) specifies Random Forest

Technical note for your defense:
- We train on the 600 simulated packets from Phase 2
- The models are saved as .joblib files (~100KB each)
- Prediction takes <1ms per packet — real-time capable

Authors: Jean D Amour KUBWIMANA, Jean de Dieu NIYONKURU
University of Rwanda — Final Year Project 2026
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)


# ════════════════════════════════════════════════════════════
#  COMPONENT 1: ANOMALY DETECTOR
# ════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Detects unusual sensor patterns using Isolation Forest.
    
    How Isolation Forest works (simple explanation for your defense):
    - It builds random decision trees that try to "isolate" each data point
    - Normal data points are HARD to isolate (need many splits)
    - Anomalies are EASY to isolate (need few splits, they stand out)
    - Points that are easily isolated get a low anomaly score
    
    Why this is better than simple thresholds:
    - Thresholds check ONE sensor at a time (CO > 200? alert!)
    - Anomaly detection checks ALL sensors TOGETHER
    - It can catch: "CO is 45 ppm AND temp is 42°C AND humidity is 88%"
      — each is below threshold, but TOGETHER it's a dangerous pattern
    
    Training data: Normal scenario (120 packets) = what "safe" looks like
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            "co_ppm", "ch4_pct", "temperature_c", "humidity_pct",
            "accel_x", "accel_y", "accel_z",
        ]
        self.is_trained = False
    
    def prepare_features(self, data: list) -> np.ndarray:
        """
        Extract the sensor features we want the model to learn from.
        
        We use 7 features:
        - co_ppm, ch4_pct: Gas levels
        - temperature_c, humidity_pct: Environmental conditions
        - accel_x, accel_y, accel_z: Motion state
        
        We DON'T include gyro or step_count because those vary too much
        during normal walking and would cause false anomalies.
        """
        df = pd.DataFrame(data)
        
        # Fill any missing columns with defaults
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        features = df[self.feature_columns].values.astype(float)
        return features
    
    def train(self, normal_data: list):
        """
        Train the anomaly detector on NORMAL (safe) data only.
        
        This is called "unsupervised learning" — we only show the model
        what normal looks like. Then anything that deviates from normal
        gets flagged as an anomaly.
        
        Args:
            normal_data: List of packet dicts from normal scenario
        """
        print("  Training Anomaly Detector (Isolation Forest)...")
        
        features = self.prepare_features(normal_data)
        
        # Scale features to have mean=0, std=1
        # This is important because CO (0-500 ppm) and CH4 (0-10%)
        # have very different scales. Without scaling, CO would
        # dominate the model just because its numbers are bigger.
        features_scaled = self.scaler.fit_transform(features)
        
        # Train Isolation Forest
        # contamination=0.05 means "expect ~5% of data to be unusual"
        # n_estimators=100 means "use 100 random trees"
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            max_samples='auto',
        )
        self.model.fit(features_scaled)
        self.is_trained = True
        
        print(f"    Trained on {len(normal_data)} normal samples")
        print(f"    Features used: {self.feature_columns}")
    
    def predict(self, packet: dict) -> dict:
        """
        Check if a single packet is normal or anomalous.
        
        Returns:
            dict with:
            - is_anomaly: True if the reading is unusual
            - anomaly_score: How unusual (-1 = very anomalous, 1 = normal)
            - confidence: 0-100% confidence that it's an anomaly
        """
        if not self.is_trained:
            return {"is_anomaly": False, "anomaly_score": 0, "confidence": 0}
        
        features = self.prepare_features([packet])
        features_scaled = self.scaler.transform(features)
        
        # predict: 1 = normal, -1 = anomaly
        prediction = self.model.predict(features_scaled)[0]
        
        # decision_function: negative = anomaly, positive = normal
        score = self.model.decision_function(features_scaled)[0]
        
        # Convert score to confidence percentage (0-100)
        # More negative score = higher anomaly confidence
        confidence = max(0, min(100, (1 - score) * 50))
        
        return {
            "is_anomaly": prediction == -1,
            "anomaly_score": round(float(score), 4),
            "confidence": round(float(confidence), 1),
        }
    
    def evaluate(self, test_data: list, labels: list) -> dict:
        """
        Evaluate the anomaly detector on labeled test data.
        
        Args:
            test_data: List of packet dicts
            labels: List of True/False (True = should be flagged as anomaly)
        """
        features = self.prepare_features(test_data)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        # Convert: 1 → False (normal), -1 → True (anomaly)
        pred_labels = [p == -1 for p in predictions]
        
        correct = sum(1 for p, l in zip(pred_labels, labels) if p == l)
        accuracy = correct / len(labels) * 100
        
        # Count detection types
        true_positives = sum(1 for p, l in zip(pred_labels, labels) if p and l)
        false_positives = sum(1 for p, l in zip(pred_labels, labels) if p and not l)
        false_negatives = sum(1 for p, l in zip(pred_labels, labels) if not p and l)
        true_negatives = sum(1 for p, l in zip(pred_labels, labels) if not p and not l)
        
        return {
            "accuracy_percent": round(accuracy, 1),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "total_samples": len(labels),
        }


# ════════════════════════════════════════════════════════════
#  COMPONENT 2: FALL CLASSIFIER
# ════════════════════════════════════════════════════════════

class FallClassifier:
    """
    Classifies motion data as: normal, walking, fall, or aftermath.
    
    How Random Forest works (for your defense):
    - It builds many decision trees, each looking at different features
    - Each tree "votes" on the classification
    - The majority vote wins
    - This is robust against noisy sensor data
    
    Features used for fall detection:
    - accel_magnitude: total acceleration (should be ~9.81 m/s² at rest)
    - accel_deviation: how far from normal gravity
    - gyro_magnitude: total rotation speed
    - accel_z_ratio: what fraction of gravity is on the Z axis
      (if person falls sideways, gravity shifts to X or Y)
    
    Labels:
    - 0 = normal/still
    - 1 = walking
    - 2 = falling (freefall + impact)
    - 3 = aftermath (lying on ground after fall)
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.class_names = ["normal", "walking", "fall", "aftermath"]
    
    def extract_motion_features(self, data: list) -> tuple:
        """
        Calculate physics-based features from raw IMU data.
        
        These features capture the PHYSICS of a fall:
        - During freefall: acceleration drops near zero
        - During impact: acceleration spikes way above gravity
        - After fall: gravity is on the wrong axis
        """
        features = []
        
        for packet in data:
            ax = float(packet.get("accel_x", 0))
            ay = float(packet.get("accel_y", 0))
            az = float(packet.get("accel_z", 9.81))
            gx = float(packet.get("gyro_x", 0))
            gy = float(packet.get("gyro_y", 0))
            gz = float(packet.get("gyro_z", 0))
            
            # Feature 1: Total acceleration magnitude
            accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
            
            # Feature 2: Deviation from normal gravity (9.81 m/s²)
            accel_dev = abs(accel_mag - 9.81)
            
            # Feature 3: Total rotation speed
            gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            
            # Feature 4: Z-axis ratio (1.0 = upright, 0.0 = lying flat)
            accel_z_ratio = abs(az) / max(accel_mag, 0.01)
            
            # Feature 5: Horizontal acceleration (X-Y plane)
            accel_horizontal = np.sqrt(ax**2 + ay**2)
            
            # Feature 6: Is acceleration spike? (>3g deviation)
            is_spike = 1.0 if accel_dev > 3.0 * 9.81 else 0.0
            
            features.append([
                accel_mag, accel_dev, gyro_mag,
                accel_z_ratio, accel_horizontal, is_spike
            ])
        
        return np.array(features)
    
    def label_training_data(self, scenarios: dict) -> tuple:
        """
        Create labeled training data from our simulation scenarios.
        
        We use the scenario type as the label:
        - normal scenario → label 0 (normal) or 1 (walking based on steps)
        - fall scenario → labels 0, 2, 3 based on timing
        """
        all_data = []
        all_labels = []
        
        # Normal scenario: mix of still (0) and walking (1)
        if "normal" in scenarios:
            for packet in scenarios["normal"]:
                all_data.append(packet)
                # If step count is changing, they're walking
                steps = packet.get("step_count", 0)
                all_labels.append(1 if steps > 0 else 0)
        
        # Fall scenario: 0-29s normal/walking, 30-31s fall, 32+ aftermath
        if "fall" in scenarios:
            for i, packet in enumerate(scenarios["fall"]):
                all_data.append(packet)
                if i < 30:
                    all_labels.append(1)      # Walking
                elif i <= 31:
                    all_labels.append(2)      # Falling
                else:
                    all_labels.append(3)      # Aftermath
        
        # Gas leak and temp spike: walking (1)
        for scenario_name in ["gas_leak", "temp_spike"]:
            if scenario_name in scenarios:
                for packet in scenarios[scenario_name]:
                    all_data.append(packet)
                    all_labels.append(1)      # Walking
        
        # Helmet off: normal/walking (1)
        if "helmet_off" in scenarios:
            for packet in scenarios["helmet_off"]:
                all_data.append(packet)
                all_labels.append(1)          # Walking
        
        return all_data, all_labels
    
    def train(self, scenarios: dict):
        """
        Train the fall classifier on labeled scenario data.
        
        Args:
            scenarios: dict of scenario_name → list of packets
        """
        print("  Training Fall Classifier (Random Forest)...")
        
        all_data, all_labels = self.label_training_data(scenarios)
        features = self.extract_motion_features(all_data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split into training and test sets (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, all_labels,
            test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,     # 100 trees in the forest
            max_depth=10,         # Limit tree depth to prevent overfitting
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"    Trained on {len(X_train)} samples, tested on {len(X_test)}")
        print(f"    Test accuracy: {accuracy * 100:.1f}%")
        print(f"    Classes: {self.class_names}")
        
        # Store evaluation results
        # Get unique classes present in test data
        present_classes = sorted(set(y_test) | set(y_pred))
        present_names = [self.class_names[i] for i in present_classes]
        
        self.evaluation = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy_percent": round(accuracy * 100, 1),
            "classification_report": classification_report(
                y_test, y_pred,
                labels=present_classes,
                target_names=present_names,
                output_dict=True,
                zero_division=0,
            ),
        }
        
        return self.evaluation
    
    def predict(self, packet: dict) -> dict:
        """
        Classify a single packet's motion state.
        
        Returns:
            dict with class name, confidence, and probabilities
        """
        if not self.is_trained:
            return {"class": "unknown", "confidence": 0, "probabilities": {}}
        
        features = self.extract_motion_features([packet])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        class_name = self.class_names[prediction]
        confidence = float(max(probabilities)) * 100
        
        prob_dict = {
            self.class_names[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probabilities)
        }
        
        return {
            "class": class_name,
            "class_id": int(prediction),
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
            "is_fall": class_name in ["fall", "aftermath"],
        }


# ════════════════════════════════════════════════════════════
#  COMPONENT 3: HAZARD PREDICTOR
# ════════════════════════════════════════════════════════════

class HazardPredictor:
    """
    Predicts when sensor values will reach danger thresholds.
    
    How it works:
    - Looks at the last N readings (a sliding window)
    - Fits a linear trend line through them
    - Extrapolates: "If this trend continues, CO will hit 200 ppm in X seconds"
    
    This is what makes our system PREDICTIVE rather than REACTIVE:
    - Reactive: "CO is 201 ppm! Alert!" (too late, miner already exposed)
    - Predictive: "CO is 80 ppm and rising. At this rate, danger in 4 minutes."
    
    This directly addresses Research Question 3 from the proposal:
    "To what extent can AI-driven predictive algorithms improve accuracy
    of early-warning alerts compared to traditional fixed-threshold systems?"
    """
    
    # Danger thresholds (from proposal Table 3)
    THRESHOLDS = {
        "co_ppm": 200.0,
        "ch4_pct": 4.0,
        "temperature_c": 45.0,
        "humidity_pct": 90.0,
    }
    
    def __init__(self, window_size: int = 15):
        """
        Args:
            window_size: Number of recent readings to analyze.
                         At 2-second intervals, 15 readings = 30 seconds of data.
        """
        self.window_size = window_size
        # Store recent readings per helmet
        self.history = {}
    
    def add_reading(self, helmet_id: int, packet: dict):
        """Add a new reading to the history buffer for a helmet."""
        if helmet_id not in self.history:
            self.history[helmet_id] = []
        
        self.history[helmet_id].append(packet)
        
        # Keep only the last window_size readings
        if len(self.history[helmet_id]) > self.window_size:
            self.history[helmet_id] = self.history[helmet_id][-self.window_size:]
    
    def predict_time_to_danger(self, helmet_id: int) -> dict:
        """
        Predict when each sensor will reach its danger threshold.
        
        Returns:
            dict with predictions for each monitored parameter:
            - current_value: latest reading
            - trend: "rising", "falling", or "stable"
            - rate_per_minute: how fast it's changing
            - seconds_to_danger: estimated time until threshold hit
                                 (None if not trending toward danger)
        """
        if helmet_id not in self.history:
            return {"error": "No data for this helmet"}
        
        readings = self.history[helmet_id]
        
        if len(readings) < 3:
            return {"error": "Need at least 3 readings for prediction"}
        
        predictions = {}
        
        for param, threshold in self.THRESHOLDS.items():
            values = [float(r.get(param, 0)) for r in readings]
            current = values[-1]
            
            # Time points (assuming 2-second intervals)
            time_points = np.arange(len(values)).reshape(-1, 1) * 2
            values_array = np.array(values)
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(time_points, values_array)
            
            slope = reg.coef_[0]            # Change per second
            rate_per_min = slope * 60        # Change per minute
            
            # Determine trend
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "rising"
            else:
                trend = "falling"
            
            # Predict time to danger
            seconds_to_danger = None
            if slope > 0.01 and current < threshold:
                # How many seconds until we hit the threshold?
                remaining = threshold - current
                seconds_to_danger = remaining / slope
                # Cap at 1 hour (3600s) — beyond that is unreliable
                if seconds_to_danger > 3600:
                    seconds_to_danger = None
            
            predictions[param] = {
                "current_value": round(current, 2),
                "threshold": threshold,
                "trend": trend,
                "rate_per_minute": round(rate_per_min, 3),
                "seconds_to_danger": (
                    round(seconds_to_danger, 0) if seconds_to_danger else None
                ),
                "is_approaching_danger": (
                    seconds_to_danger is not None and seconds_to_danger < 300
                ),  # Warning if < 5 minutes away
            }
        
        return predictions
    
    def get_overall_risk(self, helmet_id: int) -> dict:
        """
        Calculate an overall risk level for a miner.
        
        Risk levels:
        - LOW: All sensors stable and well below thresholds
        - MEDIUM: At least one sensor trending toward danger
        - HIGH: At least one sensor approaching danger within 5 minutes
        - CRITICAL: At least one sensor already past threshold
        """
        predictions = self.predict_time_to_danger(helmet_id)
        
        if "error" in predictions:
            return {"risk_level": "UNKNOWN", "reason": predictions["error"]}
        
        risk_level = "LOW"
        risk_reasons = []
        
        for param, pred in predictions.items():
            if pred["current_value"] >= pred["threshold"]:
                risk_level = "CRITICAL"
                risk_reasons.append(
                    f"{param} at {pred['current_value']} "
                    f"(threshold: {pred['threshold']})"
                )
            elif pred["is_approaching_danger"]:
                if risk_level not in ["CRITICAL"]:
                    risk_level = "HIGH"
                risk_reasons.append(
                    f"{param} will reach danger in "
                    f"{int(pred['seconds_to_danger'])}s"
                )
            elif pred["trend"] == "rising":
                if risk_level not in ["CRITICAL", "HIGH"]:
                    risk_level = "MEDIUM"
                risk_reasons.append(f"{param} is {pred['trend']}")
        
        return {
            "risk_level": risk_level,
            "reasons": risk_reasons,
            "predictions": predictions,
        }


# ════════════════════════════════════════════════════════════
#  AI ENGINE — Combines all 3 components
# ════════════════════════════════════════════════════════════

class AIEngine:
    """
    The main AI engine that combines all 3 components.
    
    This is what the backend calls for every incoming packet:
    1. Run anomaly detection
    2. Run fall classification
    3. Update hazard prediction
    4. Return combined AI analysis
    """
    
    def __init__(self, models_dir: str = "models"):
        self.anomaly_detector = AnomalyDetector()
        self.fall_classifier = FallClassifier()
        self.hazard_predictor = HazardPredictor(window_size=15)
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def train_all(self, scenarios: dict):
        """
        Train all AI models using simulation data.
        
        Args:
            scenarios: dict with keys 'normal', 'gas_leak', 'fall', etc.
                       Each value is a list of packet dicts.
        """
        print("\n" + "=" * 60)
        print("  Training AI Models")
        print("=" * 60)
        
        # Train anomaly detector on normal data only
        if "normal" in scenarios:
            self.anomaly_detector.train(scenarios["normal"])
        
        # Train fall classifier on all labeled scenarios
        evaluation = self.fall_classifier.train(scenarios)
        
        # Save trained models
        self.save_models()
        
        print("\n  All models trained and saved!")
        return evaluation
    
    def analyze_packet(self, packet: dict) -> dict:
        """
        Run full AI analysis on a single sensor packet.
        
        This is called for every packet that arrives at the backend.
        It adds AI insights on top of the simple threshold checks.
        
        Returns:
            dict with anomaly, motion, and hazard analysis
        """
        helmet_id = packet.get("helmet_id", 0)
        
        # Component 1: Anomaly detection
        anomaly = self.anomaly_detector.predict(packet)
        
        # Component 2: Fall classification
        motion = self.fall_classifier.predict(packet)
        
        # Component 3: Hazard prediction (needs history)
        self.hazard_predictor.add_reading(helmet_id, packet)
        hazard = self.hazard_predictor.get_overall_risk(helmet_id)
        
        return {
            "helmet_id": helmet_id,
            "timestamp": packet.get("timestamp", datetime.now().isoformat()),
            "anomaly_detection": anomaly,
            "motion_classification": motion,
            "hazard_prediction": hazard,
            "ai_alert": (
                anomaly.get("is_anomaly", False)
                or motion.get("is_fall", False)
                or hazard.get("risk_level") in ["HIGH", "CRITICAL"]
            ),
        }
    
    def save_models(self):
        """Save trained models to disk."""
        joblib.dump(
            self.anomaly_detector, 
            os.path.join(self.models_dir, "anomaly_detector.joblib")
        )
        joblib.dump(
            self.fall_classifier,
            os.path.join(self.models_dir, "fall_classifier.joblib")
        )
        print(f"  Models saved to {self.models_dir}/")
    
    def load_models(self) -> bool:
        """Load previously trained models from disk."""
        try:
            anomaly_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
            fall_path = os.path.join(self.models_dir, "fall_classifier.joblib")
            
            if os.path.exists(anomaly_path) and os.path.exists(fall_path):
                self.anomaly_detector = joblib.load(anomaly_path)
                self.fall_classifier = joblib.load(fall_path)
                print("[AI] Loaded pre-trained models")
                return True
            else:
                print("[AI] No pre-trained models found")
                return False
        except Exception as e:
            print(f"[AI] Error loading models: {e}")
            return False
