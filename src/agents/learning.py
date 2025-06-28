#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Learning and Adaptation System
Advanced learning capabilities for AI agents
"""

import json
import time
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of learning"""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    IMITATION = "imitation"
    TRANSFER = "transfer"
    META = "meta"
    CONTINUAL = "continual"


class FeedbackType(Enum):
    """Types of feedback"""
    REWARD = "reward"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    DEMONSTRATION = "demonstration"
    CRITIQUE = "critique"
    RATING = "rating"


@dataclass
class LearningExperience:
    """Individual learning experience"""
    id: str = field(default_factory=lambda: str(time.time()))
    state: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Learning performance metrics"""
    total_experiences: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    average_reward: float = 0.0
    learning_rate: float = 0.01
    adaptation_score: float = 0.0
    confidence_level: float = 0.5
    last_updated: float = field(default_factory=time.time)


class BaseLearningSystem(ABC):
    """Base class for learning systems"""
    
    def __init__(
        self,
        learning_type: LearningType,
        learning_rate: float = 0.01,
        memory_size: int = 10000,
        enable_transfer: bool = True
    ):
        """
        Initialize learning system
        
        Args:
            learning_type: Type of learning
            learning_rate: Learning rate
            memory_size: Maximum number of experiences to store
            enable_transfer: Whether to enable transfer learning
        """
        self.learning_type = learning_type
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.enable_transfer = enable_transfer
        
        # Experience storage
        self.experiences: List[LearningExperience] = []
        self.knowledge_base: Dict[str, Any] = {}
        
        # Learning state
        self.metrics = LearningMetrics(learning_rate=learning_rate)
        self.is_learning_enabled = True
        
        # Adaptation rules
        self.adaptation_rules: Dict[str, Callable] = {}
        self.pattern_recognition: Dict[str, Any] = {}
        
        logger.info(f"Learning system initialized: {learning_type.value}")
    
    @abstractmethod
    def learn_from_experience(self, experience: LearningExperience) -> None:
        """Learn from a single experience"""
        pass
    
    @abstractmethod
    def predict_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict best action for given state"""
        pass
    
    @abstractmethod
    def evaluate_performance(self) -> Dict[str, float]:
        """Evaluate learning performance"""
        pass
    
    def add_experience(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
        reward: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new learning experience"""
        
        experience = LearningExperience(
            state=state,
            action=action,
            result=result,
            feedback=feedback or {},
            reward=reward,
            context=context or {}
        )
        
        # Add to experience buffer
        self.experiences.append(experience)
        
        # Maintain memory size limit
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
        
        # Update metrics
        self.metrics.total_experiences += 1
        if result.get("success", False):
            self.metrics.successful_actions += 1
        else:
            self.metrics.failed_actions += 1
        
        # Update average reward
        total_reward = sum(exp.reward for exp in self.experiences)
        self.metrics.average_reward = total_reward / len(self.experiences)
        
        # Learn from experience if enabled
        if self.is_learning_enabled:
            self.learn_from_experience(experience)
        
        logger.debug(f"Added learning experience with reward {reward}")
    
    def get_similar_experiences(
        self,
        state: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[LearningExperience]:
        """Get experiences similar to current state"""
        
        similar_experiences = []
        
        for experience in self.experiences:
            similarity = self._calculate_state_similarity(state, experience.state)
            if similarity >= similarity_threshold:
                similar_experiences.append((experience, similarity))
        
        # Sort by similarity and return top results
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similar_experiences[:max_results]]
    
    def adapt_behavior(self, feedback: Dict[str, Any]) -> None:
        """Adapt behavior based on feedback"""
        
        feedback_type = FeedbackType(feedback.get("type", "reward"))
        
        if feedback_type == FeedbackType.REWARD:
            self._adapt_from_reward(feedback)
        elif feedback_type == FeedbackType.CORRECTION:
            self._adapt_from_correction(feedback)
        elif feedback_type == FeedbackType.PREFERENCE:
            self._adapt_from_preference(feedback)
        elif feedback_type == FeedbackType.CRITIQUE:
            self._adapt_from_critique(feedback)
        
        # Update adaptation score
        self._update_adaptation_score()
        
        logger.info(f"Adapted behavior from {feedback_type.value} feedback")
    
    def transfer_knowledge(self, source_domain: str, target_domain: str) -> bool:
        """Transfer knowledge between domains"""
        
        if not self.enable_transfer:
            return False
        
        try:
            # Extract relevant knowledge from source domain
            source_knowledge = self.knowledge_base.get(source_domain, {})
            
            if not source_knowledge:
                logger.warning(f"No knowledge found for source domain: {source_domain}")
                return False
            
            # Adapt knowledge for target domain
            adapted_knowledge = self._adapt_knowledge(source_knowledge, target_domain)
            
            # Store in target domain
            if target_domain not in self.knowledge_base:
                self.knowledge_base[target_domain] = {}
            
            self.knowledge_base[target_domain].update(adapted_knowledge)
            
            logger.info(f"Transferred knowledge from {source_domain} to {target_domain}")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {str(e)}")
            return False
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learning progress"""
        
        if not self.experiences:
            return {"message": "No learning experiences available"}
        
        # Analyze learning trends
        recent_experiences = self.experiences[-100:]  # Last 100 experiences
        recent_rewards = [exp.reward for exp in recent_experiences]
        
        # Calculate trends
        reward_trend = "improving" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "stable"
        
        # Identify patterns
        patterns = self._identify_patterns()
        
        # Success rate analysis
        success_rate = self.metrics.successful_actions / max(self.metrics.total_experiences, 1)
        
        return {
            "total_experiences": self.metrics.total_experiences,
            "success_rate": success_rate,
            "average_reward": self.metrics.average_reward,
            "reward_trend": reward_trend,
            "confidence_level": self.metrics.confidence_level,
            "adaptation_score": self.metrics.adaptation_score,
            "identified_patterns": patterns,
            "knowledge_domains": list(self.knowledge_base.keys()),
            "learning_enabled": self.is_learning_enabled
        }
    
    def _calculate_state_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Calculate similarity between two states"""
        
        # Simple similarity based on common keys and values
        common_keys = set(state1.keys()) & set(state2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        
        for key in common_keys:
            val1, val2 = state1[key], state2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1)
                similarity = 1 - abs(val1 - val2) / max_val
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple overlap)
                words1 = set(val1.lower().split())
                words2 = set(val2.lower().split())
                if words1 or words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                else:
                    similarity = 1.0 if val1 == val2 else 0.0
            else:
                # Exact match for other types
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarity_scores.append(similarity)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _adapt_from_reward(self, feedback: Dict[str, Any]) -> None:
        """Adapt from reward feedback"""
        reward_value = feedback.get("value", 0.0)
        
        # Adjust learning rate based on reward
        if reward_value > 0:
            self.learning_rate = min(self.learning_rate * 1.1, 0.1)
        else:
            self.learning_rate = max(self.learning_rate * 0.9, 0.001)
        
        self.metrics.learning_rate = self.learning_rate
    
    def _adapt_from_correction(self, feedback: Dict[str, Any]) -> None:
        """Adapt from correction feedback"""
        correction = feedback.get("correction", {})
        
        # Store correction as adaptation rule
        rule_id = f"correction_{len(self.adaptation_rules)}"
        self.adaptation_rules[rule_id] = lambda state: correction
    
    def _adapt_from_preference(self, feedback: Dict[str, Any]) -> None:
        """Adapt from preference feedback"""
        preferred_action = feedback.get("preferred_action", {})
        rejected_action = feedback.get("rejected_action", {})
        
        # Update preference patterns
        if "preferences" not in self.pattern_recognition:
            self.pattern_recognition["preferences"] = {"preferred": [], "rejected": []}
        
        self.pattern_recognition["preferences"]["preferred"].append(preferred_action)
        self.pattern_recognition["preferences"]["rejected"].append(rejected_action)
    
    def _adapt_from_critique(self, feedback: Dict[str, Any]) -> None:
        """Adapt from critique feedback"""
        critique = feedback.get("critique", "")
        suggestions = feedback.get("suggestions", [])
        
        # Store critique for future reference
        if "critiques" not in self.knowledge_base:
            self.knowledge_base["critiques"] = []
        
        self.knowledge_base["critiques"].append({
            "critique": critique,
            "suggestions": suggestions,
            "timestamp": time.time()
        })
    
    def _adapt_knowledge(self, source_knowledge: Dict[str, Any], target_domain: str) -> Dict[str, Any]:
        """Adapt knowledge for transfer to target domain"""
        
        # Simple adaptation - in practice, this would be more sophisticated
        adapted_knowledge = {}
        
        for key, value in source_knowledge.items():
            # Add domain prefix to avoid conflicts
            adapted_key = f"transferred_{key}"
            adapted_knowledge[adapted_key] = value
        
        return adapted_knowledge
    
    def _update_adaptation_score(self) -> None:
        """Update adaptation score based on recent performance"""
        
        if len(self.experiences) < 10:
            return
        
        # Calculate adaptation based on improvement over time
        recent_rewards = [exp.reward for exp in self.experiences[-10:]]
        older_rewards = [exp.reward for exp in self.experiences[-20:-10]] if len(self.experiences) >= 20 else []
        
        if older_rewards:
            recent_avg = sum(recent_rewards) / len(recent_rewards)
            older_avg = sum(older_rewards) / len(older_rewards)
            
            improvement = (recent_avg - older_avg) / max(abs(older_avg), 1)
            self.metrics.adaptation_score = max(0, min(1, 0.5 + improvement))
        
        self.metrics.last_updated = time.time()
    
    def _identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in experiences"""
        
        patterns = []
        
        if len(self.experiences) < 5:
            return patterns
        
        # Pattern 1: State-action success correlation
        state_action_success = {}
        
        for exp in self.experiences:
            state_key = str(sorted(exp.state.items()))
            action_key = str(sorted(exp.action.items()))
            key = f"{state_key}_{action_key}"
            
            if key not in state_action_success:
                state_action_success[key] = {"successes": 0, "total": 0}
            
            state_action_success[key]["total"] += 1
            if exp.result.get("success", False):
                state_action_success[key]["successes"] += 1
        
        # Find high-success patterns
        for key, stats in state_action_success.items():
            if stats["total"] >= 3 and stats["successes"] / stats["total"] > 0.8:
                patterns.append({
                    "type": "high_success_pattern",
                    "pattern": key,
                    "success_rate": stats["successes"] / stats["total"],
                    "occurrences": stats["total"]
                })
        
        return patterns


class ReinforcementLearningSystem(BaseLearningSystem):
    """Reinforcement learning implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(learning_type=LearningType.REINFORCEMENT, **kwargs)
        
        # Q-learning parameters
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.9    # Discount factor
    
    def learn_from_experience(self, experience: LearningExperience) -> None:
        """Learn using Q-learning algorithm"""
        
        state_key = self._state_to_key(experience.state)
        action_key = self._action_to_key(experience.action)
        
        # Initialize Q-table entries if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Q-learning update
        current_q = self.q_table[state_key][action_key]
        
        # Estimate future reward (simplified)
        future_reward = 0.0
        if experience.result.get("next_state"):
            next_state_key = self._state_to_key(experience.result["next_state"])
            if next_state_key in self.q_table:
                future_reward = max(self.q_table[next_state_key].values(), default=0.0)
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (
            experience.reward + self.gamma * future_reward - current_q
        )
        
        self.q_table[state_key][action_key] = new_q
        
        logger.debug(f"Updated Q-value for {state_key}_{action_key}: {current_q} -> {new_q}")
    
    def predict_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict best action using epsilon-greedy policy"""
        
        state_key = self._state_to_key(state)
        
        # Exploration vs exploitation
        if np.random.random() < self.epsilon or state_key not in self.q_table:
            # Explore: random action
            return self._generate_random_action(state)
        else:
            # Exploit: best known action
            best_action_key = max(self.q_table[state_key], key=self.q_table[state_key].get)
            return self._key_to_action(best_action_key)
    
    def evaluate_performance(self) -> Dict[str, float]:
        """Evaluate Q-learning performance"""
        
        if not self.experiences:
            return {"q_table_size": 0, "average_q_value": 0.0}
        
        # Calculate Q-table statistics
        total_q_values = []
        for state_actions in self.q_table.values():
            total_q_values.extend(state_actions.values())
        
        avg_q_value = sum(total_q_values) / len(total_q_values) if total_q_values else 0.0
        
        return {
            "q_table_size": len(self.q_table),
            "total_state_action_pairs": sum(len(actions) for actions in self.q_table.values()),
            "average_q_value": avg_q_value,
            "exploration_rate": self.epsilon,
            "discount_factor": self.gamma
        }
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key"""
        return json.dumps(state, sort_keys=True, default=str)
    
    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """Convert action to string key"""
        return json.dumps(action, sort_keys=True, default=str)
    
    def _key_to_action(self, key: str) -> Dict[str, Any]:
        """Convert string key back to action"""
        try:
            return json.loads(key)
        except json.JSONDecodeError:
            return {"action": "default"}
    
    def _generate_random_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random action for exploration"""
        # Simple random action generation
        return {
            "type": "explore",
            "value": np.random.random(),
            "timestamp": time.time()
        }


class ImitationLearningSystem(BaseLearningSystem):
    """Imitation learning implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(learning_type=LearningType.IMITATION, **kwargs)
        
        # Demonstration storage
        self.demonstrations: List[LearningExperience] = []
        self.behavior_model: Dict[str, Any] = {}
    
    def learn_from_experience(self, experience: LearningExperience) -> None:
        """Learn from demonstration"""
        
        if experience.feedback.get("type") == "demonstration":
            self.demonstrations.append(experience)
            self._update_behavior_model()
    
    def predict_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict action by imitating demonstrations"""
        
        # Find most similar demonstration
        best_demo = None
        best_similarity = 0.0
        
        for demo in self.demonstrations:
            similarity = self._calculate_state_similarity(state, demo.state)
            if similarity > best_similarity:
                best_similarity = similarity
                best_demo = demo
        
        if best_demo and best_similarity > 0.5:
            return best_demo.action
        else:
            return {"action": "no_demonstration_found"}
    
    def evaluate_performance(self) -> Dict[str, float]:
        """Evaluate imitation learning performance"""
        
        return {
            "demonstrations_count": len(self.demonstrations),
            "behavior_model_size": len(self.behavior_model),
            "average_demonstration_quality": self._calculate_demo_quality()
        }
    
    def _update_behavior_model(self) -> None:
        """Update behavior model from demonstrations"""
        
        # Group demonstrations by similar states
        state_groups = {}
        
        for demo in self.demonstrations:
            state_key = self._state_to_key(demo.state)
            if state_key not in state_groups:
                state_groups[state_key] = []
            state_groups[state_key].append(demo)
        
        # Create behavior model
        for state_key, demos in state_groups.items():
            if len(demos) >= 2:  # Need multiple demonstrations for reliability
                # Find most common action
                action_counts = {}
                for demo in demos:
                    action_key = self._action_to_key(demo.action)
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1
                
                most_common_action = max(action_counts, key=action_counts.get)
                self.behavior_model[state_key] = {
                    "action": most_common_action,
                    "confidence": action_counts[most_common_action] / len(demos),
                    "demonstrations": len(demos)
                }
    
    def _calculate_demo_quality(self) -> float:
        """Calculate average quality of demonstrations"""
        
        if not self.demonstrations:
            return 0.0
        
        total_quality = sum(demo.reward for demo in self.demonstrations)
        return total_quality / len(self.demonstrations)
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key"""
        return json.dumps(state, sort_keys=True, default=str)
    
    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """Convert action to string key"""
        return json.dumps(action, sort_keys=True, default=str)
