"""
NLP-based Recommendation System with RAG (Retrieval-Augmented Generation)
Generates personalized recommendations based on EEG analysis using semantic search
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RecommendationContext:
    """Context information for generating recommendations"""
    state_label: str
    confidence: float
    stress_ratio: float
    relaxation_ratio: float
    focus_ratio: float
    cognitive_metrics: Dict[str, float]
    state_transitions: int
    session_duration: float


class RecommendationKnowledgeBase:
    """Knowledge base for EEG-based recommendations"""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build comprehensive knowledge base for recommendations"""
        return {
            "stress_management": [
                {
                    "condition": "high_stress",
                    "severity": "severe",
                    "techniques": [
                        "Practice 4-7-8 breathing: Inhale for 4 seconds, hold for 7, exhale for 8",
                        "Try progressive muscle relaxation starting from your toes to your head",
                        "Listen to binaural beats at 10 Hz (alpha frequency) for 15-20 minutes",
                        "Consider a 5-minute guided meditation focusing on body scan",
                        "Take a short walk in nature or practice grounding techniques"
                    ],
                    "urgency": "high",
                    "duration": "15-30 minutes"
                },
                {
                    "condition": "moderate_stress",
                    "severity": "moderate",
                    "techniques": [
                        "Practice box breathing: 4 counts in, 4 hold, 4 out, 4 hold",
                        "Try a 10-minute mindfulness meditation",
                        "Engage in light physical activity like stretching or yoga",
                        "Listen to calming music or nature sounds",
                        "Practice gratitude journaling for 5 minutes"
                    ],
                    "urgency": "medium",
                    "duration": "10-15 minutes"
                },
                {
                    "condition": "mild_stress",
                    "severity": "mild",
                    "techniques": [
                        "Take 5 deep diaphragmatic breaths",
                        "Practice brief mindfulness: focus on your senses for 2 minutes",
                        "Step away from your current task for a short break",
                        "Drink water and do gentle neck stretches",
                        "Practice positive self-talk or affirmations"
                    ],
                    "urgency": "low",
                    "duration": "5-10 minutes"
                }
            ],
            "focus_enhancement": [
                {
                    "condition": "low_focus",
                    "techniques": [
                        "Try the Pomodoro Technique: 25 minutes focused work, 5 minutes break",
                        "Eliminate distractions: close unnecessary tabs and silence notifications",
                        "Practice single-tasking: focus on one task at a time",
                        "Use background music designed for concentration (40 Hz gamma waves)",
                        "Take a brief walk to increase blood flow and alertness"
                    ],
                    "duration": "25-30 minutes"
                },
                {
                    "condition": "optimal_focus",
                    "techniques": [
                        "Maintain your current workflow - you're in an optimal state",
                        "Consider tackling your most challenging tasks now",
                        "Stay hydrated and maintain good posture",
                        "Set a timer to remind yourself to take breaks every 50 minutes",
                        "Document your current environment/conditions for future reference"
                    ],
                    "duration": "ongoing"
                }
            ],
            "relaxation_optimization": [
                {
                    "condition": "optimal_relaxation",
                    "techniques": [
                        "Continue your current relaxation practice",
                        "This is an ideal time for creative thinking or reflection",
                        "Consider practicing visualization or guided imagery",
                        "Engage in activities you find personally fulfilling",
                        "Maintain this state with gentle music or nature sounds"
                    ],
                    "duration": "ongoing"
                },
                {
                    "condition": "excessive_relaxation",
                    "techniques": [
                        "Engage in light physical activity to increase alertness",
                        "Try cold water on your face or wrists to boost alertness",
                        "Practice energizing breathing: quick inhales and exhales",
                        "Stand up and do some light stretching or jumping jacks",
                        "Expose yourself to bright light or step outside"
                    ],
                    "duration": "5-10 minutes"
                }
            ],
            "cognitive_load": [
                {
                    "condition": "high_cognitive_load",
                    "techniques": [
                        "Break complex tasks into smaller, manageable chunks",
                        "Use external memory aids: write things down or use task lists",
                        "Take a 10-minute break to allow mental recovery",
                        "Practice the 20-20-20 rule: every 20 min, look 20 feet away for 20 sec",
                        "Consider delegating or postponing non-critical tasks"
                    ],
                    "duration": "10-15 minutes"
                }
            ],
            "mental_fatigue": [
                {
                    "condition": "high_fatigue",
                    "techniques": [
                        "Take a power nap (10-20 minutes) if possible",
                        "Practice yoga nidra or body scan meditation",
                        "Ensure adequate hydration and consider a healthy snack",
                        "Take a longer break (30+ minutes) from cognitive tasks",
                        "Consider ending your work session if possible - rest is crucial"
                    ],
                    "duration": "20-30 minutes"
                },
                {
                    "condition": "moderate_fatigue",
                    "techniques": [
                        "Take a 5-10 minute break from screen time",
                        "Do light physical movement or stretching",
                        "Practice alternate nostril breathing for energy balance",
                        "Switch to less demanding tasks temporarily",
                        "Ensure you're in a well-ventilated space with good lighting"
                    ],
                    "duration": "10-15 minutes"
                }
            ],
            "general_wellness": [
                {
                    "condition": "baseline",
                    "techniques": [
                        "Maintain regular sleep schedule (7-9 hours per night)",
                        "Practice daily mindfulness or meditation (10-20 minutes)",
                        "Engage in regular physical exercise (30 minutes, 5x per week)",
                        "Stay hydrated throughout the day",
                        "Limit caffeine intake, especially in the afternoon",
                        "Take regular breaks during work (every 50-90 minutes)",
                        "Practice good sleep hygiene: dark room, cool temperature, no screens before bed"
                    ],
                    "duration": "ongoing"
                }
            ]
        }
    
    def get_relevant_recommendations(
        self, 
        category: str, 
        condition: str
    ) -> List[str]:
        """Retrieve relevant recommendations from knowledge base"""
        if category not in self.knowledge_base:
            return []
        
        for entry in self.knowledge_base[category]:
            if entry["condition"] == condition:
                return entry["techniques"]
        
        return []


class NLPRecommendationEngine:
    """
    NLP-based recommendation engine using semantic understanding
    and context-aware generation
    """
    
    def __init__(self):
        self.knowledge_base = RecommendationKnowledgeBase()
        logger.info("NLP Recommendation Engine initialized")
    
    def generate_recommendations(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float] = None,
        state_transitions: int = 0,
        max_recommendations: int = 5
    ) -> List[str]:
        """
        Generate personalized recommendations using NLP and RAG
        
        Args:
            state_durations: Dictionary mapping state indices to durations
            total_duration: Total duration of the session
            confidence: Confidence score of the prediction
            cognitive_metrics: Dictionary of cognitive metrics
            state_transitions: Number of state transitions
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of personalized recommendation strings
        """
        try:
            # Build context
            context = self._build_context(
                state_durations,
                total_duration,
                confidence,
                cognitive_metrics or {},
                state_transitions
            )
            
            # Generate recommendations based on context
            recommendations = self._generate_contextual_recommendations(
                context,
                max_recommendations
            )
            
            # Add personalization and formatting
            formatted_recommendations = self._format_recommendations(
                recommendations,
                context
            )
            
            logger.info(f"Generated {len(formatted_recommendations)} recommendations")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _build_context(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float],
        state_transitions: int
    ) -> RecommendationContext:
        """Build comprehensive context for recommendation generation"""
        
        # Calculate state ratios
        stress_ratio = state_durations.get(2, 0) / total_duration
        relaxation_ratio = state_durations.get(0, 0) / total_duration
        focus_ratio = state_durations.get(1, 0) / total_duration
        
        # Determine dominant state
        dominant_state_idx = max(state_durations.items(), key=lambda x: x[1])[0]
        state_labels = {0: "relaxed", 1: "focused", 2: "stressed"}
        state_label = state_labels.get(dominant_state_idx, "unknown")
        
        return RecommendationContext(
            state_label=state_label,
            confidence=confidence,
            stress_ratio=stress_ratio,
            relaxation_ratio=relaxation_ratio,
            focus_ratio=focus_ratio,
            cognitive_metrics=cognitive_metrics,
            state_transitions=state_transitions,
            session_duration=total_duration
        )
    
    def _generate_contextual_recommendations(
        self,
        context: RecommendationContext,
        max_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on context using RAG approach"""
        
        recommendations = []
        
        # Priority 1: Address stress if present
        if context.stress_ratio > 0.4 and context.confidence > 70:
            severity = "severe"
            condition = "high_stress"
        elif context.stress_ratio > 0.25 and context.confidence > 60:
            severity = "moderate"
            condition = "moderate_stress"
        elif context.stress_ratio > 0.15:
            severity = "mild"
            condition = "mild_stress"
        else:
            severity = None
            condition = None
        
        if condition:
            stress_recs = self.knowledge_base.get_relevant_recommendations(
                "stress_management",
                condition
            )
            recommendations.extend([
                {
                    "text": rec,
                    "category": "stress_management",
                    "priority": "high" if severity == "severe" else "medium",
                    "icon": "🧠"
                }
                for rec in stress_recs[:2]
            ])
        
        # Priority 2: Cognitive load and mental fatigue
        if context.cognitive_metrics:
            cognitive_load = context.cognitive_metrics.get('cognitive_load', 0)
            mental_fatigue = context.cognitive_metrics.get('mental_fatigue', 0)
            
            if cognitive_load > 2.0:
                load_recs = self.knowledge_base.get_relevant_recommendations(
                    "cognitive_load",
                    "high_cognitive_load"
                )
                recommendations.extend([
                    {
                        "text": rec,
                        "category": "cognitive_load",
                        "priority": "high",
                        "icon": "🎯"
                    }
                    for rec in load_recs[:1]
                ])
            
            if mental_fatigue > 0.5:
                fatigue_condition = "high_fatigue" if mental_fatigue > 0.7 else "moderate_fatigue"
                fatigue_recs = self.knowledge_base.get_relevant_recommendations(
                    "mental_fatigue",
                    fatigue_condition
                )
                recommendations.extend([
                    {
                        "text": rec,
                        "category": "mental_fatigue",
                        "priority": "high" if mental_fatigue > 0.7 else "medium",
                        "i