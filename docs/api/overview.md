# API Overview

Welcome to the Cognito Simulation Engine API documentation. This section provides comprehensive technical reference for all classes, methods, and interfaces in the simulation engine.

## Core Architecture

The Cognito Simulation Engine is built with a modular architecture consisting of several key components:

### Primary Modules

- **[Engine](engine.md)** - Core simulation engine and cognitive processing
- **[Memory](memory.md)** - Memory systems (working, episodic, semantic, long-term)
- **[Reasoning](reasoning.md)** - Inference engine and symbolic reasoning
- **[Agents](agents.md)** - Agent architectures and behaviors
- **[Environment](environment.md)** - Simulation environments and dynamics

### Integration Components

- **[CLI](cli.md)** - Command-line interface utilities
- **[Utils](utils.md)** - Utility functions and helpers
- **[Types](types.md)** - Type definitions and data structures

## Quick Start Example

```python
from cognito_sim_engine import CognitiveEngine, CognitiveAgent, Environment

# Create simulation environment
env = Environment("research_lab", environment_type="collaborative")

# Create cognitive agent
agent = CognitiveAgent(
    agent_id="researcher_001",
    personality_traits={
        "openness": 0.8,
        "conscientiousness": 0.9,
        "extraversion": 0.6
    }
)

# Create simulation engine
engine = CognitiveEngine()
engine.add_environment(env)
env.add_agent(agent)

# Run simulation
results = engine.run_simulation(duration=3600)
```

## Core Classes Reference

### CognitiveEngine

The main simulation engine that orchestrates all cognitive processes.

```python
class CognitiveEngine:
    def __init__(self, config: Optional[Dict] = None)
    def add_environment(self, environment: Environment) -> None
    def run_simulation(self, duration: int, **kwargs) -> SimulationResults
    def step(self) -> StepResults
    def get_state(self) -> EngineState
```

**Key Methods:**

- `run_simulation()` - Execute simulation for specified duration
- `step()` - Execute single simulation step
- `add_environment()` - Add environment to simulation
- `get_metrics()` - Retrieve simulation metrics

### CognitiveAgent

Base class for all cognitive agents with memory, reasoning, and goal management.

```python
class CognitiveAgent:
    def __init__(self, agent_id: str, **kwargs)
    def perceive(self, environment: Environment) -> Perception
    def reason(self, perception: Perception) -> ReasoningResult
    def act(self, reasoning_result: ReasoningResult) -> Action
    def learn(self, experience: Experience) -> None
```

**Key Properties:**

- `memory_manager` - Access to agent's memory systems
- `reasoning_engine` - Inference and reasoning capabilities
- `goal_manager` - Goal setting and achievement tracking
- `personality_traits` - Personality configuration

### Memory Systems

#### WorkingMemory

Short-term memory for active information processing.

```python
class WorkingMemory:
    def __init__(self, capacity: int = 7, decay_rate: float = 0.1)
    def store(self, item: MemoryItem, activation: float = 1.0) -> None
    def retrieve(self, query: str, limit: int = None) -> List[MemoryItem]
    def update_activations(self) -> None
```

#### EpisodicMemory

Memory for personal experiences and events.

```python
class EpisodicMemory:
    def __init__(self, capacity: int = 10000)
    def store_episode(self, episode: Episode) -> None
    def retrieve_episodes(self, query: str, **kwargs) -> List[Episode]
    def find_similar_episodes(self, target: Episode, threshold: float = 0.7) -> List[Episode]
```

#### SemanticMemory

Memory for general knowledge and concepts.

```python
class SemanticMemory:
    def __init__(self)
    def store_concept(self, concept: Concept) -> None
    def retrieve_concepts(self, query: str, **kwargs) -> List[Concept]
    def get_related_concepts(self, concept_id: str, relation_type: str = None) -> List[Concept]
```

### Reasoning Components

#### InferenceEngine

Core reasoning and inference capabilities.

```python
class InferenceEngine:
    def __init__(self, config: Optional[Dict] = None)
    def infer(self, facts: List[Fact], rules: List[Rule], goal: Optional[Goal] = None) -> InferenceResult
    def forward_chain(self, facts: List[Fact], rules: List[Rule]) -> List[Fact]
    def backward_chain(self, goal: Goal, facts: List[Fact], rules: List[Rule]) -> bool
```

#### SymbolicReasoner

Symbolic logic and rule-based reasoning.

```python
class SymbolicReasoner:
    def __init__(self, depth_limit: int = 10, breadth_limit: int = 100)
    def reason(self, premises: List[Premise], conclusion: Conclusion) -> ReasoningResult
    def validate_reasoning(self, reasoning_chain: List[ReasoningStep]) -> bool
```

### Environment System

#### Environment

Base environment class for all simulation environments.

```python
class Environment:
    def __init__(self, env_id: str, environment_type: str = "basic", **kwargs)
    def add_agent(self, agent: CognitiveAgent) -> None
    def remove_agent(self, agent_id: str) -> None
    def step(self) -> EnvironmentState
    def get_percepts_for_agent(self, agent_id: str) -> List[Percept]
```

**Specialized Environments:**

- `CollaborativeEnvironment` - Multi-agent collaboration
- `LearningEnvironment` - Educational simulations
- `CompetitiveEnvironment` - Competition scenarios

## Data Types and Structures

### Core Data Types

```python
# Basic simulation types
@dataclass
class Perception:
    agent_id: str
    timestamp: float
    sensory_data: Dict[str, Any]
    environmental_state: EnvironmentState

@dataclass
class Action:
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float

@dataclass
class Experience:
    agent_id: str
    perception: Perception
    action: Action
    outcome: Any
    reward: float
    timestamp: float

# Memory types
@dataclass
class MemoryItem:
    content: Any
    activation: float
    timestamp: float
    memory_type: MemoryType
    tags: List[str]

@dataclass
class Episode:
    episode_id: str
    agent_id: str
    timestamp: float
    events: List[Event]
    context: Dict[str, Any]
    emotional_state: EmotionalState

# Reasoning types
@dataclass
class Fact:
    predicate: str
    arguments: List[str]
    confidence: float
    source: str

@dataclass
class Rule:
    rule_id: str
    conditions: List[Fact]
    conclusion: Fact
    confidence: float

@dataclass
class Goal:
    goal_id: str
    description: str
    goal_type: GoalType
    priority: float
    success_criteria: List[str]
    deadline: Optional[datetime]
```

### Enumerations

```python
class AgentType(Enum):
    COGNITIVE = "cognitive"
    LEARNING = "learning"
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"

class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    LONG_TERM = "long_term"

class GoalType(Enum):
    ACHIEVEMENT = "achievement"
    MAINTENANCE = "maintenance"
    AVOIDANCE = "avoidance"

class EnvironmentType(Enum):
    BASIC = "basic"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    LEARNING = "learning"
```

## Configuration Objects

### Engine Configuration

```python
@dataclass
class EngineConfig:
    # Simulation parameters
    time_step: float = 1.0
    max_steps: int = 1000
    real_time_factor: float = 1.0
    
    # Processing parameters
    parallel_processing: bool = False
    max_threads: int = 4
    batch_size: int = 100
    
    # Memory management
    memory_cleanup_interval: int = 100
    max_memory_usage: int = 1000000
    
    # Logging and debugging
    log_level: str = "INFO"
    debug_mode: bool = False
    profile_performance: bool = False
```

### Agent Configuration

```python
@dataclass
class AgentConfig:
    # Agent parameters
    agent_type: AgentType = AgentType.COGNITIVE
    personality_traits: Dict[str, float] = None
    
    # Memory configuration
    working_memory_capacity: int = 7
    episodic_memory_capacity: int = 10000
    semantic_memory_enabled: bool = True
    
    # Reasoning configuration
    reasoning_depth: int = 5
    confidence_threshold: float = 0.6
    inference_timeout: float = 5.0
    
    # Learning configuration
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    adaptation_enabled: bool = True
```

## Error Handling

### Custom Exceptions

```python
class CognitoSimError(Exception):
    """Base exception for Cognito Simulation Engine"""
    pass

class AgentError(CognitoSimError):
    """Exceptions related to agent operations"""
    pass

class MemoryError(CognitoSimError):
    """Exceptions related to memory operations"""
    pass

class ReasoningError(CognitoSimError):
    """Exceptions related to reasoning operations"""
    pass

class EnvironmentError(CognitoSimError):
    """Exceptions related to environment operations"""
    pass

class SimulationError(CognitoSimError):
    """Exceptions related to simulation execution"""
    pass
```

### Error Handling Patterns

```python
try:
    # Agent operations
    agent = CognitiveAgent("agent_001")
    result = agent.reason(perception)
except AgentError as e:
    logger.error(f"Agent error: {e}")
    # Handle agent-specific errors

try:
    # Memory operations
    memory_item = agent.memory_manager.retrieve("query")
except MemoryError as e:
    logger.error(f"Memory error: {e}")
    # Handle memory-specific errors

try:
    # Simulation operations
    engine = CognitiveEngine()
    results = engine.run_simulation(duration=3600)
except SimulationError as e:
    logger.error(f"Simulation error: {e}")
    # Handle simulation-specific errors
```

## Performance Considerations

### Memory Management

```python
# Efficient memory usage patterns
agent_config = AgentConfig(
    working_memory_capacity=7,  # Realistic cognitive limits
    episodic_memory_capacity=5000,  # Reasonable for long simulations
    memory_cleanup_enabled=True  # Automatic cleanup
)

# Memory monitoring
memory_stats = agent.memory_manager.get_memory_statistics()
if memory_stats.usage_percentage > 0.8:
    agent.memory_manager.cleanup_old_memories()
```

### Reasoning Optimization

```python
# Efficient reasoning configuration
reasoning_config = {
    "max_depth": 10,  # Prevent infinite recursion
    "timeout": 5.0,   # Reasonable time limits
    "cache_enabled": True,  # Cache reasoning results
    "parallel_processing": False  # For thread safety
}

inference_engine = InferenceEngine(reasoning_config)
```

### Simulation Scaling

```python
# Large-scale simulation patterns
engine_config = EngineConfig(
    parallel_processing=True,
    max_threads=4,
    batch_size=50,
    memory_cleanup_interval=100
)

# Monitor performance
engine = CognitiveEngine(engine_config)
engine.enable_performance_monitoring()
results = engine.run_simulation(duration=3600)
performance_metrics = engine.get_performance_metrics()
```

## Integration Patterns

### Custom Agent Types

```python
class ResearchAgent(CognitiveAgent):
    def __init__(self, agent_id: str, research_domain: str, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.research_domain = research_domain
        self.research_history = []
        
    def conduct_research(self, research_question: str) -> ResearchResult:
        # Custom research behavior
        perception = self.perceive_research_context(research_question)
        reasoning_result = self.reason_about_research(perception)
        research_action = self.plan_research_action(reasoning_result)
        return self.execute_research(research_action)
```

### Custom Environments

```python
class AcademicEnvironment(Environment):
    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id, environment_type="academic", **kwargs)
        self.research_papers = []
        self.collaboration_network = CollaborationNetwork()
        
    def facilitate_research_collaboration(self, agents: List[CognitiveAgent]) -> None:
        # Custom collaboration logic
        for agent in agents:
            collaborators = self.find_potential_collaborators(agent)
            self.initiate_collaboration(agent, collaborators)
```

### Event Handling

```python
# Custom event handlers
def on_agent_learning(agent: CognitiveAgent, learning_event: LearningEvent):
    logger.info(f"Agent {agent.agent_id} learned: {learning_event.content}")
    
def on_goal_achieved(agent: CognitiveAgent, goal: Goal):
    logger.info(f"Agent {agent.agent_id} achieved goal: {goal.description}")

# Register event handlers
engine.register_event_handler("agent_learning", on_agent_learning)
engine.register_event_handler("goal_achieved", on_goal_achieved)
```

## Testing and Validation

### Unit Testing Patterns

```python
import unittest
from cognito_sim_engine import CognitiveAgent, WorkingMemory

class TestWorkingMemory(unittest.TestCase):
    def setUp(self):
        self.memory = WorkingMemory(capacity=7)
        
    def test_memory_storage(self):
        item = MemoryItem("test_content", activation=1.0)
        self.memory.store(item)
        self.assertEqual(len(self.memory.items), 1)
        
    def test_memory_retrieval(self):
        item = MemoryItem("test_content", activation=1.0)
        self.memory.store(item)
        retrieved = self.memory.retrieve("test")
        self.assertGreater(len(retrieved), 0)
```

### Integration Testing

```python
def test_agent_environment_interaction():
    # Create test environment
    env = Environment("test_env")
    
    # Create test agent
    agent = CognitiveAgent("test_agent")
    env.add_agent(agent)
    
    # Test interaction
    perception = agent.perceive(env)
    assert perception is not None
    
    reasoning_result = agent.reason(perception)
    assert reasoning_result.success
    
    action = agent.act(reasoning_result)
    assert action.action_type is not None
```

---

This API overview provides the foundation for understanding and using the Cognito Simulation Engine. For detailed documentation on specific components, explore the individual module references.

**Next Steps:**

- [Engine API](engine.md) - Core simulation engine
- [Agents API](agents.md) - Agent architectures and behaviors
- [Memory API](memory.md) - Memory systems
- [Examples](../examples/overview.md) - Complete usage examples
