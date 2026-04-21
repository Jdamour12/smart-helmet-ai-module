"""
train_models.py — Train and Evaluate AI Models
=================================================
This script loads the simulation data from Phase 2, trains all
3 AI models, evaluates their performance, and saves the results.

Run this ONCE after generating simulation data.
The trained models are saved as .joblib files and loaded by the
backend server automatically.

Usage:
    python train_models.py

Output:
    models/anomaly_detector.joblib  — Trained anomaly detection model
    models/fall_classifier.joblib   — Trained fall classification model
    output/ai_evaluation.json       — Performance metrics

Authors: Jean D Amour KUBWIMANA, Jean de Dieu NIYONKURU
"""

import json
import os
import sys
from ai_models import AIEngine

# Add simulator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulator'))


def load_scenarios() -> dict:
    """Load all scenario data from the simulator output."""
    sim_output = os.path.join(
        os.path.dirname(__file__), '..', 'simulator', 'output'
    )
    
    if not os.path.exists(sim_output):
        print("[ERROR] Simulator output not found!")
        print("        Run the simulator first:")
        print("        cd ../simulator && python sensor_simulator.py")
        sys.exit(1)
    
    scenarios = {}
    scenario_files = {
        "normal": "scenario_normal.json",
        "gas_leak": "scenario_gas_leak.json",
        "fall": "scenario_fall.json",
        "helmet_off": "scenario_helmet_off.json",
        "temp_spike": "scenario_temp_spike.json",
    }
    
    for name, filename in scenario_files.items():
        filepath = os.path.join(sim_output, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                scenarios[name] = json.load(f)
            print(f"  Loaded {name}: {len(scenarios[name])} packets")
        else:
            print(f"  [SKIP] {filename} not found")
    
    return scenarios


def test_predictions(engine: AIEngine, scenarios: dict):
    """
    Test the AI models on sample packets from each scenario.
    Shows what the AI output looks like for different situations.
    """
    print("\n" + "=" * 60)
    print("  Testing AI Predictions")
    print("=" * 60)
    
    test_cases = [
        ("Normal conditions", scenarios["normal"][60]),
        ("Early gas leak (t=80s)", scenarios["gas_leak"][80]),
        ("Dangerous gas (t=150s)", scenarios["gas_leak"][150]),
        ("Just before fall (t=29s)", scenarios["fall"][29]),
        ("During fall (t=31s)", scenarios["fall"][31]),
        ("After fall (t=40s)", scenarios["fall"][40]),
        ("Helmet removed (t=50s)", scenarios["helmet_off"][50]),
        ("High temperature (t=120s)", scenarios["temp_spike"][120]),
    ]
    
    for name, packet in test_cases:
        result = engine.analyze_packet(packet)
        
        anomaly = result["anomaly_detection"]
        motion = result["motion_classification"]
        hazard = result["hazard_prediction"]
        
        print(f"\n  📊 {name}")
        print(f"     Anomaly:  {'⚠ YES' if anomaly['is_anomaly'] else '✓ Normal'} "
              f"(confidence: {anomaly['confidence']}%)")
        print(f"     Motion:   {motion['class']} "
              f"(confidence: {motion['confidence']}%)")
        print(f"     Risk:     {hazard['risk_level']}")
        if hazard.get("reasons"):
            for reason in hazard["reasons"]:
                print(f"               → {reason}")
        print(f"     AI Alert: {'🔴 YES' if result['ai_alert'] else '🟢 No'}")


def evaluate_anomaly_detector(engine: AIEngine, scenarios: dict) -> dict:
    """
    Evaluate anomaly detection accuracy.
    
    Normal scenario = should NOT be flagged (label: False)
    Gas leak (after t=60) = SHOULD be flagged (label: True)
    Temp spike (after t=40) = SHOULD be flagged (label: True)
    """
    print("\n" + "=" * 60)
    print("  Evaluating Anomaly Detector")
    print("=" * 60)
    
    test_data = []
    labels = []
    
    # Normal data → should NOT trigger anomaly
    for packet in scenarios.get("normal", []):
        test_data.append(packet)
        labels.append(False)
    
    # Gas leak data → should trigger anomaly after onset
    for i, packet in enumerate(scenarios.get("gas_leak", [])):
        test_data.append(packet)
        labels.append(i > 80)  # Gas becomes clearly abnormal after t=80
    
    # Temp spike → should trigger after onset
    for i, packet in enumerate(scenarios.get("temp_spike", [])):
        test_data.append(packet)
        labels.append(i > 60)  # Temp clearly abnormal after t=60
    
    results = engine.anomaly_detector.evaluate(test_data, labels)
    
    print(f"  Total samples:     {results['total_samples']}")
    print(f"  Accuracy:          {results['accuracy_percent']}%")
    print(f"  True positives:    {results['true_positives']} (correctly detected anomalies)")
    print(f"  False positives:   {results['false_positives']} (false alarms)")
    print(f"  False negatives:   {results['false_negatives']} (missed anomalies)")
    print(f"  True negatives:    {results['true_negatives']} (correctly identified normal)")
    
    return results


def main():
    print("=" * 60)
    print("  AI Model Training — Smart Safety Helmet")
    print("  Phase 6: AI/ML Module")
    print("=" * 60)
    
    # Step 1: Load simulation data
    print("\n📂 Loading simulation data...")
    scenarios = load_scenarios()
    
    if not scenarios:
        print("[ERROR] No scenario data found!")
        return
    
    # Step 2: Create and train AI engine
    engine = AIEngine(models_dir="models")
    evaluation = engine.train_all(scenarios)
    
    # Step 3: Evaluate anomaly detector
    anomaly_eval = evaluate_anomaly_detector(engine, scenarios)
    
    # Step 4: Test predictions on sample data
    test_predictions(engine, scenarios)
    
    # Step 5: Save evaluation results
    os.makedirs("output", exist_ok=True)
    results = {
        "fall_classifier": evaluation,
        "anomaly_detector": anomaly_eval,
        "models_saved": ["models/anomaly_detector.joblib", "models/fall_classifier.joblib"],
    }
    
    with open("output/ai_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Evaluation saved → output/ai_evaluation.json")
    
    print("\n" + "=" * 60)
    print("  Phase 6 Complete — AI Models Trained!")
    print("=" * 60)


if __name__ == "__main__":
    main()
