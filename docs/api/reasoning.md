# Reasoning API Reference

The Reasoning API provides sophisticated inference and symbolic reasoning capabilities for cognitive agents, including forward chaining, backward chaining, and advanced reasoning strategies.

## InferenceEngine

The core reasoning engine that performs logical inference and deduction.

```python
class InferenceEngine:
    """
    Core inference engine for logical reasoning.
    
    Supports multiple reasoning strategies including forward chaining,
    backward chaining, and hybrid approaches with uncertainty handling.
    """
    
    def __init__(self, config: Optional[ReasoningConfig] = None):
        """
        Initialize the inference engine.
        
        Args:
            config: Reasoning configuration object
        """
```

### Primary Methods

#### infer

```python
def infer(
    self,
    facts: List[Fact],
    rules: List[Rule],
    goal: Optional[Goal] = None,
    strategy: str = "mixed",
    max_depth: int = 10,
    timeout: float = 5.0
) -> InferenceResult:
    """
    Perform logical inference.
    
    Args:
        facts: Known facts
        rules: Inference rules
        goal: Target goal (for backward chaining)
        strategy: Reasoning strategy ("forward", "backward", "mixed")
        max_depth: Maximum inference depth
        timeout: Timeout in seconds
        
    Returns:
        InferenceResult: Results of inference process
    """
```

**Example Usage:**

```python
from cognito_sim_engine import InferenceEngine, Fact, Rule, Goal

# Create inference engine
engine = InferenceEngine()

# Define facts
facts = [
    Fact("neural_network", ["model"], confidence=1.0),
    Fact("has_layers", ["model", "hidden_layers"], confidence=0.9),
    Fact("has_weights", ["model", "learnable_weights"], confidence=1.0),
    Fact("training_data", ["large_dataset"], confidence=0.8)
]

# Define rules
rules = [
    Rule(
        conditions=[
            Fact("neural_network", ["?x"]),
            Fact("has_layers", ["?x", "hidden_layers"]),
            Fact("training_data", ["large_dataset"])
        ],
        conclusion=Fact("can_learn_complex_patterns", ["?x"]),
        confidence=0.85
    ),
    Rule(
        conditions=[
            Fact("can_learn_complex_patterns", ["?x"]),
            Fact("has_weights", ["?x", "learnable_weights"])
        ],
        conclusion=Fact("suitable_for_ml", ["?x"]),
        confidence=0.9
    )
]

# Perform forward chaining inference
result = engine.infer(
    facts=facts,
    rules=rules,
    strategy="forward",
    max_depth=5
)

print(f"Inference successful: {result.success}")
print(f"New facts derived: {len(result.derived_facts)}")
for fact in result.derived_facts:
    print(f"  {fact.predicate}({', '.join(fact.arguments)}) - confidence: {fact.confidence:.2f}")
```

#### forward_chain

```python
def forward_chain(
    self,
    facts: List[Fact],
    rules: List[Rule],
    max_iterations: int = 100
) -> List[Fact]:
    """
    Perform forward chaining inference.
    
    Args:
        facts: Initial facts
        rules: Inference rules
        max_iterations: Maximum iterations to prevent infinite loops
        
    Returns:
        List[Fact]: All derived facts
    """
```

#### backward_chain

```python
def backward_chain(
    self,
    goal: Goal,
    facts: List[Fact],
    rules: List[Rule],
    max_depth: int = 10
) -> bool:
    """
    Perform backward chaining to prove goal.
    
    Args:
        goal: Goal to prove
        facts: Known facts
        rules: Inference rules
        max_depth: Maximum recursion depth
        
    Returns:
        bool: True if goal can be proven
    """
```

## SymbolicReasoner

Advanced symbolic reasoning with logic programming capabilities.

```python
class SymbolicReasoner:
    """
    Symbolic reasoner with advanced logical capabilities.
    
    Supports first-order logic, unification, and complex reasoning patterns
    with uncertainty and non-monotonic reasoning support.
    """
    
    def __init__(
        self,
        depth_limit: int = 10,
        breadth_limit: int = 100,
        confidence_propagation: bool = True,
        contradiction_detection: bool = True
    ):
        """
        Initialize symbolic reasoner.
        
        Args:
            depth_limit: Maximum reasoning depth
            breadth_limit: Maximum breadth of search
            confidence_propagation: Enable confidence propagation
            contradiction_detection: Detect logical contradictions
        """
```

### Advanced Reasoning Methods

#### reason_with_uncertainty

```python
def reason_with_uncertainty(
    self,
    premises: List[Premise],
    conclusion: Conclusion,
    uncertainty_model: str = "bayesian"
) -> ReasoningResult:
    """
    Perform reasoning under uncertainty.
    
    Args:
        premises: Uncertain premises
        conclusion: Target conclusion
        uncertainty_model: Model for uncertainty ("bayesian", "fuzzy", "dempster_shafer")
        
    Returns:
        ReasoningResult: Reasoning result with uncertainty measures
    """
```

**Example Usage:**

```python
from cognito_sim_engine import SymbolicReasoner, Premise, Conclusion

# Create reasoner with uncertainty handling
reasoner = SymbolicReasoner(
    depth_limit=15,
    confidence_propagation=True,
    contradiction_detection=True
)

# Define uncertain premises
premises = [
    Premise("research_shows", ["neural_networks", "effective_for_nlp"], confidence=0.85),
    Premise("transformers_are", ["neural_networks"], confidence=0.95),
    Premise("current_model_is", ["transformer"], confidence=0.9),
    Premise("task_is", ["nlp_task"], confidence=1.0)
]

# Define conclusion to evaluate
conclusion = Conclusion("model_will_be_effective", ["current_model", "task"], confidence=None)

# Perform uncertain reasoning
result = reasoner.reason_with_uncertainty(
    premises=premises,
    conclusion=conclusion,
    uncertainty_model="bayesian"
)

print(f"Reasoning conclusion: {result.conclusion_supported}")
print(f"Confidence in conclusion: {result.confidence:.3f}")
print(f"Reasoning chain length: {len(result.reasoning_chain)}")
```

#### detect_contradictions

```python
def detect_contradictions(
    self,
    knowledge_base: List[Fact],
    new_fact: Fact
) -> List[Contradiction]:
    """
    Detect logical contradictions.
    
    Args:
        knowledge_base: Existing knowledge
        new_fact: New fact to check
        
    Returns:
        List[Contradiction]: Detected contradictions
    """
```

#### resolve_contradictions

```python
def resolve_contradictions(
    self,
    contradictions: List[Contradiction],
    resolution_strategy: str = "confidence_based"
) -> ResolutionResult:
    """
    Resolve logical contradictions.
    
    Args:
        contradictions: Contradictions to resolve
        resolution_strategy: Strategy for resolution
        
    Returns:
        ResolutionResult: Result of contradiction resolution
    """
```

## Goal-Directed Reasoning

Reasoning specifically directed toward achieving goals.

```python
class GoalDirectedReasoner:
    """
    Reasoning engine specialized for goal achievement.
    
    Combines planning, means-ends analysis, and logical inference
    to support goal-directed behavior.
    """
    
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.planning_strategies = PlanningStrategies()
        
    def reason_toward_goal(
        self,
        goal: Goal,
        current_state: State,
        available_actions: List[Action],
        constraints: List[Constraint] = None
    ) -> GoalReasoningResult:
        """
        Reason about how to achieve a goal.
        
        Args:
            goal: Target goal
            current_state: Current state
            available_actions: Available actions
            constraints: Constraints on actions
            
        Returns:
            GoalReasoningResult: Reasoning result with action plan
        """
```

**Example Usage:**

```python
from cognito_sim_engine import GoalDirectedReasoner, Goal, State, Action

# Create goal-directed reasoner
goal_reasoner = GoalDirectedReasoner(inference_engine)

# Define research goal
research_goal = Goal(
    goal_id="understand_transformers",
    description="Understand transformer architecture and applications",
    success_criteria=[
        "know_attention_mechanism",
        "understand_positional_encoding",
        "can_implement_basic_transformer"
    ],
    priority=0.9
)

# Current state of knowledge
current_state = State({
    "knowledge_level": 0.3,
    "available_resources": ["research_papers", "tutorials", "code_examples"],
    "time_available": 20,  # hours
    "current_understanding": ["basic_neural_networks", "backpropagation"]
})

# Available learning actions
available_actions = [
    Action("read_paper", duration=3, learning_gain=0.2, prerequisites=["basic_neural_networks"]),
    Action("watch_tutorial", duration=1, learning_gain=0.1, prerequisites=[]),
    Action("implement_code", duration=4, learning_gain=0.3, prerequisites=["basic_understanding"]),
    Action("discuss_with_expert", duration=2, learning_gain=0.25, prerequisites=["some_knowledge"])
]

# Reason about goal achievement
reasoning_result = goal_reasoner.reason_toward_goal(
    goal=research_goal,
    current_state=current_state,
    available_actions=available_actions
)

print(f"Goal achievable: {reasoning_result.goal_achievable}")
print(f"Estimated time to completion: {reasoning_result.estimated_time}")
print("Recommended action sequence:")
for i, action in enumerate(reasoning_result.action_sequence):
    print(f"  {i+1}. {action.name} (duration: {action.duration}h)")
```

## Analogical Reasoning

Reasoning by analogy with previous experiences.

```python
class AnalogicalReasoner:
    """
    Reasoning by analogy and similarity.
    
    Finds analogous situations and maps solutions from similar contexts
    to current problems.
    """
    
    def find_analogies(
        self,
        current_situation: Situation,
        memory_manager: MemoryManager,
        similarity_threshold: float = 0.7
    ) -> List[Analogy]:
        """
        Find analogous situations in memory.
        
        Args:
            current_situation: Current problem situation
            memory_manager: Access to episodic memory
            similarity_threshold: Minimum similarity score
            
        Returns:
            List[Analogy]: Found analogies with mappings
        """
        
    def reason_by_analogy(
        self,
        analogy: Analogy,
        current_problem: Problem
    ) -> AnalogicalSolution:
        """
        Generate solution based on analogy.
        
        Args:
            analogy: Analogous situation
            current_problem: Current problem to solve
            
        Returns:
            AnalogicalSolution: Solution mapped from analogy
        """
```

**Example Usage:**

```python
from cognito_sim_engine import AnalogicalReasoner, Situation, Problem

# Create analogical reasoner
analogical_reasoner = AnalogicalReasoner()

# Current research problem
current_situation = Situation(
    context="optimizing_neural_network",
    problem_type="hyperparameter_tuning",
    constraints=["limited_compute", "time_pressure"],
    resources=["small_dataset", "basic_hardware"],
    goal="maximize_accuracy"
)

current_problem = Problem(
    description="Need to find optimal learning rate for transformer training",
    domain="machine_learning",
    complexity=0.7,
    urgency=0.8
)

# Find analogies in memory
analogies = analogical_reasoner.find_analogies(
    current_situation=current_situation,
    memory_manager=agent.memory_manager,
    similarity_threshold=0.6
)

print(f"Found {len(analogies)} analogous situations")

# Use best analogy for reasoning
if analogies:
    best_analogy = analogies[0]  # Highest similarity
    solution = analogical_reasoner.reason_by_analogy(
        analogy=best_analogy,
        current_problem=current_problem
    )
    
    print(f"Analogical solution: {solution.description}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Based on: {best_analogy.source_situation.description}")
```

## Causal Reasoning

Reasoning about cause and effect relationships.

```python
class CausalReasoner:
    """
    Causal reasoning and intervention analysis.
    
    Models causal relationships and reasons about interventions
    and counterfactual scenarios.
    """
    
    def infer_causal_structure(
        self,
        observations: List[Observation],
        prior_knowledge: List[CausalRelation] = None
    ) -> CausalGraph:
        """
        Infer causal structure from observations.
        
        Args:
            observations: Observed data points
            prior_knowledge: Known causal relations
            
        Returns:
            CausalGraph: Inferred causal structure
        """
        
    def reason_about_interventions(
        self,
        causal_graph: CausalGraph,
        intervention: Intervention,
        target_outcome: str
    ) -> InterventionResult:
        """
        Reason about effects of interventions.
        
        Args:
            causal_graph: Causal model
            intervention: Proposed intervention
            target_outcome: Desired outcome
            
        Returns:
            InterventionResult: Predicted intervention effects
        """
```

**Example Usage:**

```python
from cognito_sim_engine import CausalReasoner, Observation, Intervention

# Create causal reasoner
causal_reasoner = CausalReasoner()

# Research productivity observations
observations = [
    Observation("sleep_hours", 7, {"productivity_score": 8.5}),
    Observation("sleep_hours", 5, {"productivity_score": 6.2}),
    Observation("caffeine_intake", 200, {"productivity_score": 7.8}),
    Observation("exercise_minutes", 30, {"productivity_score": 8.1}),
    Observation("interruptions", 15, {"productivity_score": 5.5})
]

# Infer causal structure
causal_graph = causal_reasoner.infer_causal_structure(observations)

# Analyze intervention to improve productivity
intervention = Intervention(
    target_variable="sleep_hours",
    intervention_value=8,
    duration="1_week"
)

intervention_result = causal_reasoner.reason_about_interventions(
    causal_graph=causal_graph,
    intervention=intervention,
    target_outcome="productivity_score"
)

print(f"Predicted productivity improvement: {intervention_result.effect_size:.2f}")
print(f"Confidence in prediction: {intervention_result.confidence:.2f}")
```

## Meta-Reasoning

Reasoning about reasoning itself.

```python
class MetaReasoner:
    """
    Meta-level reasoning about reasoning strategies.
    
    Monitors reasoning performance and adapts strategies
    based on problem characteristics and past performance.
    """
    
    def select_reasoning_strategy(
        self,
        problem: Problem,
        available_strategies: List[str],
        performance_history: Dict[str, float]
    ) -> str:
        """
        Select optimal reasoning strategy for problem.
        
        Args:
            problem: Problem to solve
            available_strategies: Available reasoning strategies
            performance_history: Past strategy performance
            
        Returns:
            str: Selected strategy name
        """
        
    def monitor_reasoning_performance(
        self,
        reasoning_session: ReasoningSession
    ) -> PerformanceMetrics:
        """
        Monitor performance of reasoning session.
        
        Args:
            reasoning_session: Active reasoning session
            
        Returns:
            PerformanceMetrics: Performance measurements
        """
        
    def adapt_reasoning_parameters(
        self,
        current_performance: PerformanceMetrics,
        target_performance: PerformanceMetrics
    ) -> ParameterAdjustments:
        """
        Adapt reasoning parameters based on performance.
        
        Args:
            current_performance: Current performance metrics
            target_performance: Target performance levels
            
        Returns:
            ParameterAdjustments: Recommended parameter changes
        """
```

**Example Usage:**

```python
from cognito_sim_engine import MetaReasoner, Problem, ReasoningSession

# Create meta-reasoner
meta_reasoner = MetaReasoner()

# Define problem characteristics
problem = Problem(
    domain="machine_learning",
    complexity=0.8,
    uncertainty=0.6,
    time_pressure=0.7,
    resources_available=0.5
)

# Available reasoning strategies
strategies = ["analytical", "analogical", "creative", "systematic"]

# Performance history (strategy -> success rate)
performance_history = {
    "analytical": 0.75,
    "analogical": 0.65,
    "creative": 0.55,
    "systematic": 0.85
}

# Select best strategy
selected_strategy = meta_reasoner.select_reasoning_strategy(
    problem=problem,
    available_strategies=strategies,
    performance_history=performance_history
)

print(f"Selected reasoning strategy: {selected_strategy}")

# Monitor reasoning session
reasoning_session = ReasoningSession(strategy=selected_strategy)
performance = meta_reasoner.monitor_reasoning_performance(reasoning_session)

print(f"Reasoning efficiency: {performance.efficiency:.2f}")
print(f"Solution quality: {performance.solution_quality:.2f}")

# Adapt parameters if needed
if performance.efficiency < 0.7:
    adjustments = meta_reasoner.adapt_reasoning_parameters(
        current_performance=performance,
        target_performance=PerformanceMetrics(efficiency=0.8, solution_quality=0.8)
    )
    print(f"Parameter adjustments: {adjustments}")
```

## Configuration and Data Types

### ReasoningConfig

```python
@dataclass
class ReasoningConfig:
    # Basic inference settings
    max_inference_depth: int = 10
    inference_timeout: float = 5.0
    confidence_threshold: float = 0.6
    
    # Strategy settings
    default_strategy: str = "mixed"
    enable_uncertainty_reasoning: bool = True
    enable_contradiction_detection: bool = True
    enable_analogical_reasoning: bool = True
    
    # Performance settings
    parallel_reasoning: bool = False
    max_reasoning_threads: int = 2
    reasoning_cache_enabled: bool = True
    cache_size_limit: int = 1000
    
    # Advanced features
    meta_reasoning_enabled: bool = True
    causal_reasoning_enabled: bool = True
    goal_directed_reasoning: bool = True
    learning_from_reasoning: bool = True
```

### Core Data Types

```python
@dataclass
class Fact:
    predicate: str
    arguments: List[str]
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: Optional[datetime] = None

@dataclass
class Rule:
    rule_id: str
    conditions: List[Fact]
    conclusion: Fact
    confidence: float = 1.0
    bidirectional: bool = False

@dataclass
class InferenceResult:
    success: bool
    derived_facts: List[Fact]
    reasoning_chain: List[ReasoningStep]
    confidence: float
    execution_time: float
    strategy_used: str

@dataclass
class ReasoningStep:
    step_id: str
    rule_applied: Rule
    input_facts: List[Fact]
    output_fact: Fact
    confidence: float
```

## Integration Patterns

### Memory-Guided Reasoning

```python
def memory_guided_reasoning(agent, query, goal=None):
    """
    Use memory to guide reasoning process.
    """
    
    # Retrieve relevant facts from memory
    relevant_memories = agent.memory_manager.retrieve(
        query=query,
        memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC]
    )
    
    # Extract facts from memories
    facts = []
    for memory in relevant_memories:
        facts.extend(extract_reasoning_facts(memory))
    
    # Get relevant rules from semantic memory
    rules = agent.memory_manager.semantic_memory.get_inference_rules(
        domain=extract_domain(query)
    )
    
    # Perform reasoning
    reasoning_result = agent.reasoning_engine.infer(
        facts=facts,
        rules=rules,
        goal=goal,
        strategy="mixed"
    )
    
    # Store reasoning episode in memory
    reasoning_episode = create_reasoning_episode(
        query=query,
        reasoning_result=reasoning_result,
        agent_id=agent.agent_id
    )
    
    agent.memory_manager.store_episode(reasoning_episode)
    
    return reasoning_result
```

### Collaborative Reasoning

```python
def collaborative_reasoning(agents, shared_problem):
    """
    Multiple agents collaborate on reasoning.
    """
    
    # Distribute reasoning tasks
    reasoning_tasks = decompose_reasoning_problem(shared_problem)
    
    # Each agent works on sub-problems
    partial_results = {}
    for agent, task in zip(agents, reasoning_tasks):
        result = agent.reasoning_engine.infer(
            facts=task.facts,
            rules=task.rules,
            goal=task.subgoal
        )
        partial_results[agent.agent_id] = result
    
    # Integrate results
    integrated_result = integrate_reasoning_results(
        partial_results,
        shared_problem
    )
    
    # Share insights with all agents
    for agent in agents:
        agent.memory_manager.store_collaborative_insight(
            insight=integrated_result,
            collaborators=[a.agent_id for a in agents if a != agent]
        )
    
    return integrated_result
```

---

The Reasoning API provides sophisticated logical inference and reasoning capabilities that enable cognitive agents to think, plan, and solve problems effectively. Use these components to create agents with advanced reasoning abilities.

**Related APIs:**

- [Memory API](memory.md) - Memory-guided reasoning integration
- [Agents API](agents.md) - Agent reasoning capabilities  
- [Engine API](engine.md) - Reasoning system orchestration
