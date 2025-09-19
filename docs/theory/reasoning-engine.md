# Reasoning Engine

The reasoning engine is the cognitive heart of Cognito Simulation Engine, providing sophisticated symbolic reasoning capabilities that go beyond traditional machine learning approaches.

## Overview

The reasoning engine implements multiple inference strategies to enable human-like problem-solving, logical deduction, and creative hypothesis generation. Unlike neural network approaches, our symbolic reasoning system provides:

- **Interpretable reasoning chains** - Every inference step is traceable
- **Formal logical foundations** - Based on established logical systems
- **Uncertainty handling** - Confidence propagation through reasoning
- **Multiple inference modes** - Forward chaining, backward chaining, and abduction

## Core Components

### InferenceEngine Class

The main orchestrator that coordinates different reasoning strategies:

```python
from cognito_sim_engine import InferenceEngine, Goal, Fact

# Create inference engine with configuration
engine = InferenceEngine(
    depth_limit=10,           # Maximum reasoning depth
    confidence_threshold=0.5, # Minimum confidence for conclusions
    timeout=5.0,             # Maximum reasoning time (seconds)
    strategy="mixed"         # Reasoning strategy selection
)

# Add domain knowledge
facts = [
    Fact("human", ["socrates"], confidence=1.0),
    Fact("mortal", ["humans"], confidence=0.95)
]

# Add reasoning rules
from cognito_sim_engine import Rule
mortality_rule = Rule(
    conditions=[Fact("human", ["?x"])],
    conclusion=Fact("mortal", ["?x"]),
    confidence=0.9,
    name="mortality_rule"
)

engine.reasoner.add_rule(mortality_rule)

# Perform inference
goal = Goal("Prove mortality", target_facts=[Fact("mortal", ["socrates"])])
result = engine.infer(goal, facts)
```

### SymbolicReasoner

The core reasoning component that implements logical inference:

```python
from cognito_sim_engine import SymbolicReasoner

reasoner = SymbolicReasoner()

# Knowledge base management
fact_id = reasoner.add_fact(Fact("bird", ["tweety"]))
rule_id = reasoner.add_rule(Rule(
    conditions=[Fact("bird", ["?x"])],
    conclusion=Fact("can_fly", ["?x"]),
    name="birds_fly"
))

# Forward chaining inference
result = reasoner.forward_chaining(max_iterations=10)
print(f"Derived {len(result.derived_facts)} new facts")
```

## Reasoning Strategies

### 1. Forward Chaining (Data-Driven)

Forward chaining starts with known facts and applies rules to derive new conclusions:

```python
# Example: Scientific discovery simulation
initial_facts = [
    Fact("organism", ["bacterium_x"], confidence=1.0),
    Fact("thrives_in", ["bacterium_x", "high_temperature"], confidence=0.9),
    Fact("produces", ["bacterium_x", "enzyme_y"], confidence=0.8)
]

discovery_rules = [
    Rule(
        conditions=[
            Fact("organism", ["?o"]), 
            Fact("thrives_in", ["?o", "high_temperature"])
        ],
        conclusion=Fact("thermophile", ["?o"]),
        confidence=0.85,
        name="thermophile_classification"
    ),
    Rule(
        conditions=[
            Fact("thermophile", ["?o"]),
            Fact("produces", ["?o", "?e"])
        ],
        conclusion=Fact("thermostable_enzyme", ["?e"]),
        confidence=0.8,
        name="enzyme_stability_inference"
    )
]

# Add to reasoner
for fact in initial_facts:
    reasoner.add_fact(fact)
for rule in discovery_rules:
    reasoner.add_rule(rule)

# Perform forward chaining
result = reasoner.forward_chaining(max_iterations=5)

print("🔬 Scientific discoveries:")
for fact in result.derived_facts:
    print(f"  • {fact.predicate}({', '.join(fact.arguments)}) [{fact.confidence:.2f}]")
```

**Output:**

```plaintext
🔬 Scientific discoveries:
  • thermophile(bacterium_x) [0.85]
  • thermostable_enzyme(enzyme_y) [0.68]
```

### 2. Backward Chaining (Goal-Driven)

Backward chaining starts with a goal and works backwards to find supporting evidence:

```python
# Example: Diagnostic reasoning
diagnostic_rules = [
    Rule(
        conditions=[Fact("fever", ["?p"]), Fact("cough", ["?p"])],
        conclusion=Fact("respiratory_infection", ["?p"]),
        confidence=0.7,
        name="respiratory_diagnosis"
    ),
    Rule(
        conditions=[Fact("respiratory_infection", ["?p"]), Fact("bacteria_positive", ["?p"])],
        conclusion=Fact("bacterial_pneumonia", ["?p"]),
        confidence=0.8,
        name="pneumonia_diagnosis"
    )
]

# Available evidence
evidence = [
    Fact("fever", ["patient_1"], confidence=0.9),
    Fact("cough", ["patient_1"], confidence=0.8),
    Fact("bacteria_positive", ["patient_1"], confidence=0.75)
]

# Goal: Diagnose bacterial pneumonia
diagnosis_goal = Fact("bacterial_pneumonia", ["patient_1"])

# Perform backward chaining
result = reasoner.backward_chaining(diagnosis_goal)

if result.success:
    print("🏥 Diagnosis confirmed:")
    print(f"  Confidence: {result.confidence:.2f}")
    print("  Reasoning chain:")
    for i, step in enumerate(result.proof_steps):
        print(f"    {i+1}. {step}")
```

### 3. Abductive Reasoning (Hypothesis Generation)

Abductive reasoning generates explanatory hypotheses for observed phenomena:

```python
# Example: Fault diagnosis in complex systems
class AbductiveReasoner:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.hypothesis_rules = []
    
    def add_hypothesis_rule(self, observation_pattern, hypothesis, confidence):
        """Add a rule for generating hypotheses"""
        rule = Rule(
            conditions=[observation_pattern],
            conclusion=hypothesis,
            confidence=confidence,
            name=f"hypothesis_{len(self.hypothesis_rules)}"
        )
        self.hypothesis_rules.append(rule)
    
    def generate_hypotheses(self, observations):
        """Generate explanatory hypotheses for observations"""
        hypotheses = []
        
        for obs in observations:
            for rule in self.hypothesis_rules:
                can_apply, bindings = rule.can_apply([obs])
                if can_apply:
                    hypothesis = rule.apply(bindings)
                    hypothesis.confidence *= obs.confidence
                    hypotheses.append(hypothesis)
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

# System fault diagnosis
abductive = AbductiveReasoner(reasoner)

# Add hypothesis generation rules
abductive.add_hypothesis_rule(
    Fact("system_slow", ["?s"]),
    Fact("memory_leak", ["?s"]),
    confidence=0.6
)

abductive.add_hypothesis_rule(
    Fact("system_slow", ["?s"]),
    Fact("cpu_overload", ["?s"]),
    confidence=0.7
)

abductive.add_hypothesis_rule(
    Fact("frequent_crashes", ["?s"]),
    Fact("memory_corruption", ["?s"]),
    confidence=0.8
)

# Observed symptoms
symptoms = [
    Fact("system_slow", ["server_1"], confidence=0.9),
    Fact("frequent_crashes", ["server_1"], confidence=0.7)
]

# Generate hypotheses
hypotheses = abductive.generate_hypotheses(symptoms)

print("🔍 Diagnostic hypotheses:")
for h in hypotheses:
    print(f"  • {h.predicate}({', '.join(h.arguments)}) [{h.confidence:.2f}]")
```

## Advanced Reasoning Features

### Uncertainty Propagation

The reasoning engine handles uncertainty through confidence values:

```python
# Uncertainty in rule chaining
uncertain_facts = [
    Fact("weather", ["cloudy"], confidence=0.8),
    Fact("season", ["winter"], confidence=0.9)
]

weather_rules = [
    Rule(
        conditions=[Fact("weather", ["cloudy"]), Fact("season", ["winter"])],
        conclusion=Fact("likely_snow", ["today"]),
        confidence=0.7,
        name="snow_prediction"
    )
]

# The final confidence is: 0.8 * 0.9 * 0.7 = 0.504
result = reasoner.forward_chaining_with_uncertainty(uncertain_facts, weather_rules)
```

### Meta-Reasoning

Reasoning about reasoning strategies and their effectiveness:

```python
class MetaReasoner:
    def __init__(self):
        self.strategy_performance = {
            "forward_chaining": {"success_rate": 0.8, "avg_time": 0.1},
            "backward_chaining": {"success_rate": 0.7, "avg_time": 0.2},
            "abductive": {"success_rate": 0.6, "avg_time": 0.3}
        }
    
    def select_strategy(self, goal_type, time_constraint):
        """Meta-reasoning for strategy selection"""
        if time_constraint < 0.15:
            return "forward_chaining"  # Fastest option
        elif goal_type == "diagnostic":
            return "backward_chaining"  # Best for goal-driven tasks
        elif goal_type == "explanatory":
            return "abductive"  # Best for hypothesis generation
        else:
            # Choose based on success rate
            best_strategy = max(
                self.strategy_performance.items(),
                key=lambda x: x[1]["success_rate"]
            )
            return best_strategy[0]
    
    def update_performance(self, strategy, success, execution_time):
        """Learn from reasoning experience"""
        perf = self.strategy_performance[strategy]
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        perf["success_rate"] = (1 - alpha) * perf["success_rate"] + alpha * (1.0 if success else 0.0)
        
        # Update average time
        perf["avg_time"] = (1 - alpha) * perf["avg_time"] + alpha * execution_time

meta_reasoner = MetaReasoner()
```

### Constraint Satisfaction

Solving problems with multiple constraints:

```python
class ConstraintSolver:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.constraints = []
    
    def add_constraint(self, constraint_rule):
        """Add a constraint to the problem"""
        self.constraints.append(constraint_rule)
    
    def solve(self, variables, domains):
        """Solve constraint satisfaction problem"""
        solution = {}
        
        def is_consistent(assignment):
            """Check if current assignment satisfies all constraints"""
            for constraint in self.constraints:
                if not self.check_constraint(constraint, assignment):
                    return False
            return True
        
        def backtrack(assignment, remaining_vars):
            if not remaining_vars:
                return assignment if is_consistent(assignment) else None
            
            var = remaining_vars[0]
            for value in domains[var]:
                assignment[var] = value
                if is_consistent(assignment):
                    result = backtrack(assignment, remaining_vars[1:])
                    if result is not None:
                        return result
                del assignment[var]
            
            return None
        
        return backtrack({}, list(variables))
    
    def check_constraint(self, constraint_rule, assignment):
        """Check if assignment satisfies constraint"""
        # Convert assignment to facts
        facts = [
            Fact("assigned", [var, str(val)]) 
            for var, val in assignment.items()
        ]
        
        # Check if constraint rule is satisfied
        can_apply, bindings = constraint_rule.can_apply(facts)
        return not can_apply  # Constraint violated if rule can apply

# Example: Scheduling problem
scheduler = ConstraintSolver(reasoner)

# Add constraints
scheduler.add_constraint(Rule(
    conditions=[
        Fact("assigned", ["task1", "?t1"]),
        Fact("assigned", ["task2", "?t2"])
    ],
    conclusion=Fact("conflict", ["?t1", "?t2"]),
    name="no_concurrent_tasks"
))

variables = ["task1", "task2", "task3"]
domains = {
    "task1": [1, 2, 3],
    "task2": [2, 3, 4], 
    "task3": [1, 3, 4]
}

solution = scheduler.solve(variables, domains)
print(f"📅 Scheduling solution: {solution}")
```

## Performance Optimization

### Reasoning Caching

Cache frequently used inference results:

```python
class CachedReasoner:
    def __init__(self, base_reasoner):
        self.base_reasoner = base_reasoner
        self.inference_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cached_forward_chaining(self, facts, max_iterations=10):
        """Forward chaining with result caching"""
        
        # Create cache key from facts
        cache_key = self._create_cache_key(facts)
        
        if cache_key in self.inference_cache:
            self.cache_hits += 1
            return self.inference_cache[cache_key]
        
        # Perform inference
        result = self.base_reasoner.forward_chaining(max_iterations)
        
        # Cache result
        self.inference_cache[cache_key] = result
        self.cache_misses += 1
        
        return result
    
    def _create_cache_key(self, facts):
        """Create hashable cache key from facts"""
        fact_strings = [f"{f.predicate}({','.join(f.arguments)})" for f in facts]
        return tuple(sorted(fact_strings))
    
    def get_cache_stats(self):
        """Get caching performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total, 1)
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate
        }

cached_reasoner = CachedReasoner(reasoner)
```

### Parallel Reasoning

Leverage multiple CPU cores for complex reasoning:

```python
import concurrent.futures
from typing import List

class ParallelReasoner:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
    
    def parallel_forward_chaining(self, fact_sets: List[List[Fact]], rules: List[Rule]):
        """Perform forward chaining on multiple fact sets in parallel"""
        
        def process_fact_set(facts):
            local_reasoner = SymbolicReasoner()
            
            # Add rules
            for rule in rules:
                local_reasoner.add_rule(rule)
            
            # Add facts
            for fact in facts:
                local_reasoner.add_fact(fact)
            
            # Perform inference
            return local_reasoner.forward_chaining()
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_fact_set, facts) for facts in fact_sets]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results
    
    def parallel_hypothesis_generation(self, observations: List[Fact], hypothesis_generators: List[Rule]):
        """Generate hypotheses in parallel for different observations"""
        
        def generate_for_observation(obs):
            hypotheses = []
            for rule in hypothesis_generators:
                can_apply, bindings = rule.can_apply([obs])
                if can_apply:
                    hypothesis = rule.apply(bindings)
                    hypotheses.append(hypothesis)
            return hypotheses
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(generate_for_observation, obs) for obs in observations]
            all_hypotheses = []
            for future in concurrent.futures.as_completed(futures):
                all_hypotheses.extend(future.result())
        
        return all_hypotheses

parallel_reasoner = ParallelReasoner(num_workers=4)
```

## Integration with Cognitive Agents

### Agent-Specific Reasoning

Different agent types use reasoning differently:

```python
# CognitiveAgent: Balanced reasoning for general problem-solving
cognitive_agent = CognitiveAgent("general_agent")
cognitive_agent.inference_engine.configure(
    strategy="adaptive",
    depth_limit=8,
    confidence_threshold=0.6
)

# ReasoningAgent: Deep logical analysis
reasoning_agent = ReasoningAgent("logic_agent")
reasoning_agent.inference_engine.configure(
    strategy="exhaustive",
    depth_limit=15,
    confidence_threshold=0.8,
    enable_proof_generation=True
)

# LearningAgent: Fast, adaptive reasoning
learning_agent = LearningAgent("adaptive_agent")
learning_agent.inference_engine.configure(
    strategy="heuristic",
    depth_limit=5,
    confidence_threshold=0.5,
    enable_learning=True
)

# MetaCognitiveAgent: Strategic reasoning about reasoning
meta_agent = MetaCognitiveAgent("meta_agent")
meta_agent.inference_engine.configure(
    strategy="meta_adaptive",
    depth_limit=12,
    confidence_threshold=0.7,
    enable_strategy_learning=True
)
```

### Reasoning in Context

Integrate reasoning with memory and goals:

```python
def contextual_reasoning(agent, current_goal):
    """Perform reasoning in context of agent's memory and goals"""
    
    # Retrieve relevant memories
    relevant_memories = agent.memory_manager.search_memories(
        current_goal.description, 
        limit=10
    )
    
    # Convert memories to facts
    context_facts = []
    for memory in relevant_memories:
        # Extract facts from memory content
        extracted_facts = extract_facts_from_text(memory.content)
        context_facts.extend(extracted_facts)
    
    # Add goal-specific facts
    goal_facts = current_goal.target_facts
    
    # Combine all facts
    all_facts = context_facts + goal_facts
    
    # Perform contextual reasoning
    result = agent.inference_engine.infer(current_goal, all_facts)
    
    # Store reasoning results in memory
    if result.success:
        reasoning_memory = MemoryItem(
            content=f"Successfully reasoned about {current_goal.description}",
            memory_type=MemoryType.EPISODIC,
            context={
                "goal": current_goal.description,
                "confidence": result.confidence,
                "steps": len(result.reasoning_steps)
            }
        )
        agent.memory_manager.store_memory(reasoning_memory)
    
    return result

def extract_facts_from_text(text):
    """Extract structured facts from natural language text"""
    # Simplified fact extraction (in practice, use NLP)
    facts = []
    
    # Pattern matching for common fact patterns
    import re
    
    # "X is Y" pattern
    is_pattern = r"(\w+) is (\w+)"
    matches = re.findall(is_pattern, text.lower())
    for match in matches:
        facts.append(Fact("is", [match[0], match[1]], confidence=0.7))
    
    # "X has Y" pattern  
    has_pattern = r"(\w+) has (\w+)"
    matches = re.findall(has_pattern, text.lower())
    for match in matches:
        facts.append(Fact("has", [match[0], match[1]], confidence=0.7))
    
    return facts
```

## Research Applications

### Cognitive Science Research

Model human reasoning patterns:

```python
# Study reasoning biases
def confirmation_bias_study():
    """Model confirmation bias in human reasoning"""
    
    biased_reasoner = SymbolicReasoner()
    
    # Add bias toward confirming existing beliefs
    def biased_rule_selection(available_rules, current_beliefs):
        """Select rules that confirm existing beliefs"""
        confirming_rules = []
        for rule in available_rules:
            if any(belief.predicate == rule.conclusion.predicate for belief in current_beliefs):
                confirming_rules.append(rule)
        
        return confirming_rules if confirming_rules else available_rules
    
    biased_reasoner.rule_selection_strategy = biased_rule_selection
    
    return biased_reasoner

# Study reasoning development
def developmental_reasoning():
    """Model how reasoning capabilities develop"""
    
    child_reasoner = SymbolicReasoner()
    child_reasoner.depth_limit = 3  # Limited reasoning depth
    child_reasoner.confidence_threshold = 0.8  # High confidence requirement
    
    adult_reasoner = SymbolicReasoner()
    adult_reasoner.depth_limit = 10  # Deeper reasoning
    adult_reasoner.confidence_threshold = 0.6  # More uncertainty tolerance
    
    return child_reasoner, adult_reasoner
```

### AI Safety Research

Test reasoning safety properties:

```python
def test_reasoning_safety():
    """Test safety properties of reasoning systems"""
    
    safety_tests = [
        {
            "name": "Contradiction Detection",
            "test": lambda r: test_contradiction_detection(r),
            "description": "Detect and handle logical contradictions"
        },
        {
            "name": "Inference Bounds",
            "test": lambda r: test_inference_bounds(r),
            "description": "Respect computational limits"
        },
        {
            "name": "Confidence Calibration", 
            "test": lambda r: test_confidence_calibration(r),
            "description": "Properly calibrated confidence estimates"
        }
    ]
    
    reasoner = SymbolicReasoner()
    
    results = {}
    for test in safety_tests:
        try:
            result = test["test"](reasoner)
            results[test["name"]] = {
                "passed": result,
                "description": test["description"]
            }
        except Exception as e:
            results[test["name"]] = {
                "passed": False,
                "error": str(e),
                "description": test["description"]
            }
    
    return results

def test_contradiction_detection(reasoner):
    """Test if reasoner can detect contradictions"""
    
    # Add contradictory facts
    reasoner.add_fact(Fact("mortal", ["socrates"], confidence=1.0))
    reasoner.add_fact(Fact("immortal", ["socrates"], confidence=1.0))
    
    # Add contradiction detection rule
    contradiction_rule = Rule(
        conditions=[Fact("mortal", ["?x"]), Fact("immortal", ["?x"])],
        conclusion=Fact("contradiction", ["?x"]),
        confidence=1.0,
        name="contradiction_detector"
    )
    reasoner.add_rule(contradiction_rule)
    
    # Should detect contradiction
    result = reasoner.forward_chaining()
    
    contradictions = [f for f in result.derived_facts if f.predicate == "contradiction"]
    return len(contradictions) > 0

# Additional safety tests...
```

## Best Practices

### 1. Rule Design

- **Specific conditions**: Avoid overly general rules
- **Appropriate confidence**: Calibrate confidence values carefully
- **Clear semantics**: Use meaningful predicate and argument names
- **Modular rules**: Break complex logic into smaller rules

### 2. Performance Optimization

- **Limit search depth**: Set reasonable depth limits
- **Cache results**: Use caching for repeated inferences
- **Prune search space**: Remove irrelevant facts and rules
- **Profile performance**: Monitor reasoning time and memory usage

### 3. Uncertainty Management

- **Confidence propagation**: Understand how confidence combines
- **Threshold setting**: Choose appropriate confidence thresholds
- **Uncertainty sources**: Account for all sources of uncertainty
- **Calibration**: Validate confidence estimates against ground truth

### 4. Integration Guidelines

- **Memory integration**: Connect reasoning with episodic and semantic memory
- **Goal alignment**: Ensure reasoning serves agent goals
- **Context awareness**: Use situational context in reasoning
- **Learning integration**: Update rules based on experience

---

The reasoning engine provides the logical foundation for intelligent behavior in Cognito Simulation Engine. By combining multiple inference strategies with uncertainty handling and performance optimization, it enables sophisticated cognitive modeling for AGI research and applications.

**Next**: Explore how reasoning integrates with [Agent Design](agent-design.md) patterns, or learn about [Memory Systems](memory-systems.md) that provide the knowledge base for reasoning.
