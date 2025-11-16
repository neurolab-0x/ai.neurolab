# NLP Recommendations Integration Guide

## Overview
The NLP-based recommendation system has been successfully integrated into the NeuroLab EEG analysis pipeline. It provides personalized, context-aware recommendations based on mental state analysis.

## Key Features

### 1. **Intelligent Recommendation Generation**
- Context-aware suggestions based on stress, focus, and relaxation levels
- Priority-based recommendations (high, medium, low)
- Considers cognitive load and mental fatigue
- Adapts to state transitions and session stability

### 2. **Comprehensive Knowledge Base**
- Stress management techniques (mild, moderate, severe)
- Focus enhancement strategies
- Relaxation optimization
- Cognitive load management
- Mental fatigue recovery
- General wellness tips

### 3. **Detailed Reporting**
- Session summaries with state distributions
- Wellness scoring (0-100 with ratings)
- Actionable insights based on patterns
- Cognitive metrics analysis
- Confidence indicators

## API Endpoints

### 1. Analyze EEG Data (Enhanced)
```http
POST /analyze
Content-Type: application/json

{
  "alpha": 8.5,
  "beta": 15.2,
  "theta": 6.3,
  "delta": 3.1,
  "gamma": 2.8,
  "subject_id": "user123",
  "session_id": "session_001"
}
```

**Response includes:**
- Mental state classification
- Confidence scores
- State durations and percentages
- Cognitive metrics
- **NLP-based recommendations**
- Temporal analysis

### 2. Generate Detailed Report (New)
```http
POST /detailed-report?save_report=false
Content-Type: application/json

{
  "alpha": 8.5,
  "beta": 15.2,
  "theta": 6.3,
  "delta": 3.1,
  "gamma": 2.8,
  "subject_id": "user123",
  "session_id": "session_001"
}
```

**Response includes:**
- Complete analysis results
- Wellness score with rating
- Detailed insights
- Comprehensive recommendations
- Session summary
- Optional: saved report filepath

### 3. Get Recommendations (New)
```http
POST /recommendations?max_recommendations=5
Content-Type: application/json

{
  "state_durations": {
    "0": 25,
    "1": 35,
    "2": 40
  },
  "total_duration": 100,
  "confidence": 82.5,
  "cognitive_metrics": {
    "cognitive_load": 2.1,
    "mental_fatigue": 0.55
  },
  "state_transitions": 15
}
```

## Python Usage

### Basic Usage
```python
from utils.nlp_recommendations import NLPRecommendationEngine

engine = NLPRecommendationEngine()

recommendations = engine.generate_recommendations(
    state_durations={0: 20, 1: 30, 2: 50},
    total_duration=100,
    confidence=85.0,
    cognitive_metrics={'cognitive_load': 2.5, 'mental_fatigue': 0.6},
    state_transitions=12
)

for rec in recommendations:
    print(rec)
```

### Generate Detailed Report
```python
report = engine.generate_detailed_report(
    state_durations={0: 25, 1: 35, 2: 40},
    total_duration=100,
    confidence=82.5,
    cognitive_metrics={'cognitive_load': 2.1, 'mental_fatigue': 0.55},
    state_transitions=15
)

print(f"Wellness Score: {report['wellness_score']['description']}")
print(f"Insights: {report['insights']}")
print(f"Recommendations: {report['recommendations']}")
```

### Using ML Processor
```python
from utils.ml_processor import MLProcessor

processor = MLProcessor()

# Analyze EEG data
result = processor.process_eeg_data({
    'alpha': 8.5,
    'beta': 15.2,
    'theta': 6.3,
    'delta': 3.1,
    'gamma': 2.8
})

# Get recommendations from result
recommendations = result['recommendations']

# Or generate detailed report
detailed_report = processor.generate_detailed_report(
    data={'alpha': 8.5, 'beta': 15.2, 'theta': 6.3, 'delta': 3.1, 'gamma': 2.8},
    save_report=True  # Saves to reports/ directory
)
```

## Recommendation Categories

1. **Stress Management** (🧠)
   - Breathing techniques
   - Progressive muscle relaxation
   - Binaural beats
   - Meditation practices

2. **Focus Enhancement** (🎯)
   - Pomodoro technique
   - Distraction elimination
   - Single-tasking strategies
   - Concentration music

3. **Relaxation Optimization** (🌿)
   - Visualization techniques
   - Creative thinking activities
   - Mindfulness practices

4. **Cognitive Load Management** (🎯)
   - Task chunking
   - Memory aids
   - Break strategies
   - 20-20-20 rule

5. **Mental Fatigue Recovery** (💤)
   - Power naps
   - Yoga nidra
   - Hydration and nutrition
   - Task switching

6. **General Wellness** (💚)
   - Sleep hygiene
   - Exercise routines
   - Hydration reminders
   - Posture tips

## Wellness Scoring

The system calculates a wellness score (0-100) based on:
- Stress levels (negative impact)
- Focus quality (positive impact)
- Relaxation balance
- State stability
- Cognitive load
- Mental fatigue

**Ratings:**
- 80-100: 🌟 Excellent
- 60-79: 👍 Good
- 40-59: ⚠️ Fair
- 0-39: 🔴 Needs Attention

## Testing

Run the integration test:
```bash
python test_nlp_integration.py
```

This will test:
- Basic recommendation generation
- Detailed report generation
- ML Processor integration

## Files Modified

1. **utils/nlp_recommendations.py** - New comprehensive NLP recommendation engine
2. **utils/ml_processor.py** - Integrated NLP engine into processing pipeline
3. **main.py** - Added new endpoints for detailed reports and recommendations

## Migration Notes

The old `utils/recommendations.py` is still available but deprecated. The new system provides:
- More detailed and personalized recommendations
- Context-aware suggestion generation
- Comprehensive reporting capabilities
- Better cognitive metrics integration

All existing endpoints continue to work with enhanced recommendations automatically.
