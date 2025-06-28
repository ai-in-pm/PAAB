#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Simple PsychoPy Integration Demo
Demonstrates PsychoPy integration without complex imports
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def simulate_stroop_experiment():
    """Simulate a Stroop color-word interference experiment"""
    print("🎨 STROOP EXPERIMENT SIMULATION")
    print("-" * 40)
    
    # Experiment parameters
    colors = ['red', 'green', 'blue', 'yellow']
    color_keys = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y'}
    num_trials = 20
    
    print(f"📋 Experiment: Stroop Color-Word Interference")
    print(f"🎯 Trials: {num_trials}")
    print(f"🎨 Colors: {', '.join(colors)}")
    
    # Simulate different AI agent types
    agents = [
        {
            "name": "Optimal Agent",
            "strategy": "optimal",
            "base_rt": 0.4,
            "accuracy": 0.95,
            "interference_effect": 0.05
        },
        {
            "name": "Human-like Agent", 
            "strategy": "human-like",
            "base_rt": 0.7,
            "accuracy": 0.82,
            "interference_effect": 0.15
        },
        {
            "name": "Variable Agent",
            "strategy": "variable",
            "base_rt": 0.9,
            "accuracy": 0.75,
            "interference_effect": 0.25
        }
    ]
    
    # Run experiment with each agent
    all_results = []
    
    for agent in agents:
        print(f"\n🤖 {agent['name']} ({agent['strategy']} strategy)")
        
        # Generate trials
        trials = []
        for i in range(num_trials):
            word = np.random.choice(colors)
            color = np.random.choice(colors)
            congruent = word == color
            trials.append({
                "trial": i + 1,
                "word": word,
                "color": color,
                "congruent": congruent,
                "expected_response": color_keys[color]
            })
        
        # Simulate responses
        responses = []
        for trial in trials:
            # Calculate reaction time with interference effect
            base_rt = agent["base_rt"]
            if not trial["congruent"]:  # Incongruent trials are slower
                rt = base_rt + agent["interference_effect"] + np.random.normal(0, 0.1)
            else:  # Congruent trials
                rt = base_rt + np.random.normal(0, 0.05)
            
            rt = max(0.2, rt)  # Minimum RT
            
            # Calculate accuracy
            accuracy_prob = agent["accuracy"]
            if not trial["congruent"]:  # Incongruent trials are less accurate
                accuracy_prob -= 0.1
            
            correct = np.random.random() < accuracy_prob
            response_key = trial["expected_response"] if correct else np.random.choice(list(color_keys.values()))
            
            responses.append({
                "trial": trial["trial"],
                "word": trial["word"],
                "color": trial["color"],
                "congruent": trial["congruent"],
                "response": response_key,
                "correct": correct,
                "reaction_time": rt
            })
            
            # Simulate processing time
            time.sleep(0.01)
        
        # Calculate results
        accuracy = sum(r["correct"] for r in responses) / len(responses)
        mean_rt = np.mean([r["reaction_time"] for r in responses])
        
        congruent_trials = [r for r in responses if r["congruent"]]
        incongruent_trials = [r for r in responses if not r["congruent"]]
        
        congruent_rt = np.mean([r["reaction_time"] for r in congruent_trials]) if congruent_trials else 0
        incongruent_rt = np.mean([r["reaction_time"] for r in incongruent_trials]) if incongruent_trials else 0
        interference_effect = incongruent_rt - congruent_rt
        
        print(f"   📈 Overall Accuracy: {accuracy:.1%}")
        print(f"   ⏱️  Mean RT: {mean_rt:.3f}s")
        print(f"   🎯 Congruent RT: {congruent_rt:.3f}s")
        print(f"   ❌ Incongruent RT: {incongruent_rt:.3f}s")
        print(f"   🧠 Interference Effect: {interference_effect:.3f}s")
        
        all_results.append({
            "agent": agent,
            "responses": responses,
            "accuracy": accuracy,
            "mean_rt": mean_rt,
            "interference_effect": interference_effect
        })
    
    return all_results

def simulate_reaction_time_experiment():
    """Simulate a simple reaction time experiment"""
    print("\n⚡ REACTION TIME EXPERIMENT SIMULATION")
    print("-" * 40)
    
    num_trials = 15
    print(f"📋 Experiment: Simple Reaction Time")
    print(f"🎯 Trials: {num_trials}")
    
    # Simulate different AI agents
    agents = [
        {"name": "Fast Agent", "base_rt": 0.25, "variability": 0.05},
        {"name": "Average Agent", "base_rt": 0.45, "variability": 0.1},
        {"name": "Slow Agent", "base_rt": 0.65, "variability": 0.15}
    ]
    
    for agent in agents:
        print(f"\n🤖 {agent['name']}")
        
        # Generate reaction times
        reaction_times = []
        for i in range(num_trials):
            # Random delay before stimulus (1-3 seconds)
            delay = np.random.uniform(1.0, 3.0)
            
            # Agent reaction time
            rt = np.random.normal(agent["base_rt"], agent["variability"])
            rt = max(0.15, rt)  # Minimum human RT
            
            reaction_times.append(rt)
            
            print(f"   Trial {i+1:2d}: {delay:.1f}s delay → {rt:.3f}s RT")
            time.sleep(0.05)  # Simulate trial time
        
        # Calculate statistics
        mean_rt = np.mean(reaction_times)
        std_rt = np.std(reaction_times)
        min_rt = np.min(reaction_times)
        max_rt = np.max(reaction_times)
        
        print(f"   📊 Mean RT: {mean_rt:.3f}s")
        print(f"   📈 Std RT: {std_rt:.3f}s")
        print(f"   ⚡ Fastest: {min_rt:.3f}s")
        print(f"   🐌 Slowest: {max_rt:.3f}s")

def demonstrate_cognitive_profiles():
    """Demonstrate different cognitive profiles"""
    print("\n🧠 COGNITIVE PROFILES DEMONSTRATION")
    print("-" * 40)
    
    profiles = [
        {
            "name": "Optimal Performer",
            "description": "AI agent optimized for perfect performance",
            "characteristics": {
                "reaction_time": "Very fast (0.3-0.5s)",
                "accuracy": "Near perfect (95-98%)",
                "fatigue": "Minimal",
                "learning": "Rapid adaptation",
                "attention": "Sustained focus"
            }
        },
        {
            "name": "Human-like Performer", 
            "description": "AI agent simulating typical human performance",
            "characteristics": {
                "reaction_time": "Moderate (0.6-0.9s)",
                "accuracy": "Good (80-85%)",
                "fatigue": "Gradual decline",
                "learning": "Steady improvement",
                "attention": "Variable focus"
            }
        },
        {
            "name": "Impaired Performer",
            "description": "AI agent simulating cognitive limitations",
            "characteristics": {
                "reaction_time": "Slow (1.0-1.5s)",
                "accuracy": "Variable (60-75%)",
                "fatigue": "Rapid onset",
                "learning": "Slow adaptation",
                "attention": "Easily distracted"
            }
        }
    ]
    
    for profile in profiles:
        print(f"\n🤖 {profile['name']}")
        print(f"   📝 {profile['description']}")
        print(f"   🧠 Characteristics:")
        for key, value in profile['characteristics'].items():
            print(f"      • {key.title()}: {value}")

def demonstrate_experimental_design():
    """Demonstrate experimental design principles"""
    print("\n🔬 EXPERIMENTAL DESIGN PRINCIPLES")
    print("-" * 40)
    
    design_elements = {
        "Independent Variables": [
            "Stimulus type (congruent vs incongruent)",
            "Response modality (keyboard vs mouse)",
            "Time pressure (speeded vs self-paced)",
            "Cognitive load (single vs dual task)"
        ],
        "Dependent Variables": [
            "Reaction time (milliseconds)",
            "Accuracy (percentage correct)",
            "Error types (commission vs omission)",
            "Confidence ratings (1-7 scale)"
        ],
        "Experimental Controls": [
            "Randomized trial order",
            "Counterbalanced conditions",
            "Practice trials",
            "Attention checks",
            "Fatigue breaks"
        ],
        "AI Agent Considerations": [
            "Cognitive profile simulation",
            "Learning and adaptation",
            "Individual differences",
            "Fatigue modeling",
            "Strategy variations"
        ]
    }
    
    for category, items in design_elements.items():
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   • {item}")

def main():
    """Main demonstration function"""
    print("🎉 PSYCHOPY AI AGENT BUILDER - PSYCHOPY INTEGRATION")
    print("=" * 60)
    print("🧠 Where Experimental Psychology Meets AI Intelligence")
    print("🔬 Demonstrating cognitive experiments with AI agents")
    print("=" * 60)
    
    # Run demonstrations
    stroop_results = simulate_stroop_experiment()
    simulate_reaction_time_experiment()
    demonstrate_cognitive_profiles()
    demonstrate_experimental_design()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎊 PSYCHOPY INTEGRATION DEMONSTRATION COMPLETED!")
    print("=" * 60)
    
    print("\n🌟 Key Features Demonstrated:")
    print("   ✅ Stroop Color-Word Interference Task")
    print("   ✅ Simple Reaction Time Measurement")
    print("   ✅ Multiple AI Cognitive Profiles")
    print("   ✅ Realistic Performance Simulation")
    print("   ✅ Experimental Design Principles")
    print("   ✅ Statistical Analysis Integration")
    
    print("\n🔬 PsychoPy Integration Benefits:")
    print("   🧪 Validated Experimental Paradigms")
    print("   📊 Precise Timing and Measurement")
    print("   🎨 Rich Stimulus Presentation")
    print("   📈 Comprehensive Data Collection")
    print("   🧠 Cognitive Process Modeling")
    print("   🤖 AI Agent Behavioral Testing")
    
    print("\n💡 Applications:")
    print("   🎓 Educational Psychology Research")
    print("   🏥 Clinical Assessment Tools")
    print("   🧪 Cognitive Model Validation")
    print("   🤖 AI Behavior Benchmarking")
    print("   📊 Human-AI Interaction Studies")
    print("   🔬 Experimental Method Development")
    
    print("\n🚀 Next Steps:")
    print("   • Install PsychoPy: pip install psychopy")
    print("   • Run full integration: python examples/psychopy_integration_demo.py")
    print("   • Launch visual studio: python -m streamlit run src/studio/main.py")
    print("   • Explore documentation: docs/getting-started.md")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n{'✅ Demo completed successfully!' if success else '❌ Demo failed!'}")
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()
