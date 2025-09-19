# Engine API Reference

The `CognitiveEngine` is the core component that orchestrates all cognitive simulation processes, managing agents, environments, and simulation execution.

## CognitiveEngine

The main simulation engine class that coordinates cognitive processes across agents and environments.

### Class Definition

```python
class CognitiveEngine:
    """
    Core simulation engine for cognitive simulations.
    
    The CognitiveEngine manages the simulation loop, coordinates agent interactions,
    handles environment dynamics, and provides comprehensive simulation control.
    """
    
    def __init__(self, config: Optional[EngineConfig] = None, logger: Optional[Logger] = None):
        """
        Initialize the cognitive engine.
        
        Args:
            config: Engine configuration object
            logger: Custom logger instance
        """
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `EngineConfig` | `None` | Configuration object for engine settings |
| `logger` | `Logger` | `None` | Custom logger for simulation events |

### Core Methods

#### run_simulation

```python
def run_simulation(
    self,
    duration: int,
    real_time_factor: float = 1.0,
    checkpoint_interval: Optional[int] = None,
    event_callbacks: Optional[Dict[str, Callable]] = None
) -> SimulationResults:
    """
    Execute a complete simulation run.
    
    Args:
        duration: Simulation duration in time steps
        real_time_factor: Real-time scaling factor (1.0 = real-time)
        checkpoint_interval: Steps between automatic checkpoints
        event_callbacks: Dictionary of event callbacks
        
    Returns:
        SimulationResults: Comprehensive simulation results
        
    Raises:
        SimulationError: If simulation cannot be executed
        EnvironmentError: If environment is not properly configured
    """
```

**Example Usage:**

```python
from cognito_sim_engine import CognitiveEngine, EngineConfig

# Configure engine
config = EngineConfig(
    time_step=1.0,
    max_steps=3600,
    parallel_processing=True,
    debug_mode=False
)

# Create engine
engine = CognitiveEngine(config)

# Add environments and agents
engine.add_environment(research_environment)
research_environment.add_agent(researcher_agent)

# Run simulation with callbacks
callbacks = {
    "agent_action": lambda agent, action: print(f"Agent {agent.agent_id} performed {action}"),
    "environment_change": lambda env, change: print(f"Environment changed: {change}")
}

results = engine.run_simulation(
    duration=3600,
    real_time_factor=0.1,  # 10x faster than real-time
    checkpoint_interval=300,  # Checkpoint every 5 minutes
    event_callbacks=callbacks
)

print(f"Simulation completed: {results.total_steps} steps")
```

#### step

```python
def step(self) -> StepResults:
    """
    Execute a single simulation step.
    
    Returns:
        StepResults: Results of the simulation step
        
    Raises:
        SimulationError: If step cannot be executed
    """
```

**Example Usage:**

```python
# Manual step-by-step simulation
engine = CognitiveEngine()
engine.add_environment(environment)

for step_num in range(100):
    step_results = engine.step()
    
    # Process step results
    print(f"Step {step_num}: {len(step_results.agent_actions)} actions")
    
    # Check for termination conditions
    if step_results.simulation_complete:
        break
    
    # Custom logic between steps
    if step_num % 10 == 0:
        engine.save_checkpoint(f"step_{step_num}.pkl")
```

#### add_environment

```python
def add_environment(self, environment: Environment) -> None:
    """
    Add an environment to the simulation.
    
    Args:
        environment: Environment instance to add
        
    Raises:
        EnvironmentError: If environment cannot be added
    """
```

#### remove_environment

```python
def remove_environment(self, environment_id: str) -> None:
    """
    Remove an environment from the simulation.
    
    Args:
        environment_id: ID of environment to remove
        
    Raises:
        EnvironmentError: If environment not found
    """
```

### State Management

#### get_state

```python
def get_state(self) -> EngineState:
    """
    Get current engine state.
    
    Returns:
        EngineState: Complete engine state snapshot
    """
```

#### save_checkpoint

```python
def save_checkpoint(self, filepath: str) -> None:
    """
    Save simulation checkpoint.
    
    Args:
        filepath: Path to save checkpoint file
        
    Raises:
        IOError: If checkpoint cannot be saved
    """
```

#### load_checkpoint

```python
def load_checkpoint(self, filepath: str) -> None:
    """
    Load simulation from checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        
    Raises:
        IOError: If checkpoint cannot be loaded
        SimulationError: If checkpoint is incompatible
    """
```

**Example Usage:**

```python
# Save and load simulation state
engine = CognitiveEngine()

# Set up simulation
engine.add_environment(environment)

# Run for some time
engine.run_simulation(duration=1800)

# Save checkpoint
engine.save_checkpoint("simulation_midpoint.pkl")

# Continue simulation
engine.run_simulation(duration=1800)

# Later, restore from checkpoint
new_engine = CognitiveEngine()
new_engine.load_checkpoint("simulation_midpoint.pkl")

# Continue from saved point
new_engine.run_simulation(duration=3600)
```

### Event System

#### register_event_handler

```python
def register_event_handler(self, event_type: str, handler: Callable) -> None:
    """
    Register an event handler.
    
    Args:
        event_type: Type of event to handle
        handler: Callback function for the event
    """
```

#### unregister_event_handler

```python
def unregister_event_handler(self, event_type: str, handler: Callable) -> None:
    """
    Unregister an event handler.
    
    Args:
        event_type: Type of event to unregister
        handler: Handler function to remove
    """
```

**Event Types:**

- `agent_created` - New agent added to simulation
- `agent_action` - Agent performs an action
- `agent_learning` - Agent learns from experience
- `goal_achieved` - Agent achieves a goal
- `environment_change` - Environment state changes
- `simulation_start` - Simulation begins
- `simulation_end` - Simulation completes
- `step_complete` - Simulation step completes
- `error_occurred` - Error during simulation

**Example Usage:**

```python
def on_agent_action(agent, action, environment):
    """Handler for agent actions"""
    print(f"Agent {agent.agent_id} performed {action.action_type}")
    
    # Log important actions
    if action.action_type == "collaborate":
        logger.info(f"Collaboration initiated by {agent.agent_id}")

def on_goal_achieved(agent, goal):
    """Handler for goal achievement"""
    print(f"🎯 Agent {agent.agent_id} achieved goal: {goal.description}")
    
    # Update metrics
    metrics.increment("goals_achieved")

# Register handlers
engine.register_event_handler("agent_action", on_agent_action)
engine.register_event_handler("goal_achieved", on_goal_achieved)

# Run simulation with event handling
results = engine.run_simulation(duration=3600)
```

### Metrics and Monitoring

#### get_metrics

```python
def get_metrics(self) -> EngineMetrics:
    """
    Get simulation metrics.
    
    Returns:
        EngineMetrics: Current simulation metrics
    """
```

#### enable_profiling

```python
def enable_profiling(self, profile_memory: bool = True, profile_cpu: bool = True) -> None:
    """
    Enable performance profiling.
    
    Args:
        profile_memory: Enable memory profiling
        profile_cpu: Enable CPU profiling
    """
```

#### get_profiling_results

```python
def get_profiling_results(self) -> ProfilingResults:
    """
    Get profiling results.
    
    Returns:
        ProfilingResults: Performance profiling data
    """
```

**Example Usage:**

```python
# Enable profiling
engine = CognitiveEngine()
engine.enable_profiling(profile_memory=True, profile_cpu=True)

# Run simulation
results = engine.run_simulation(duration=3600)

# Get performance metrics
metrics = engine.get_metrics()
profiling_results = engine.get_profiling_results()

print(f"Total steps: {metrics.total_steps}")
print(f"Average step time: {metrics.average_step_time:.3f}s")
print(f"Memory usage: {profiling_results.peak_memory_mb:.2f} MB")
print(f"CPU usage: {profiling_results.average_cpu_percent:.1f}%")
```

### Parallel Processing

#### set_parallel_processing

```python
def set_parallel_processing(self, enabled: bool, max_workers: int = None) -> None:
    """
    Configure parallel processing.
    
    Args:
        enabled: Enable/disable parallel processing
        max_workers: Maximum number of worker threads
    """
```

**Example Usage:**

```python
# Configure for parallel processing
engine = CognitiveEngine()
engine.set_parallel_processing(enabled=True, max_workers=4)

# Add multiple environments
for i in range(4):
    env = Environment(f"env_{i}")
    engine.add_environment(env)
    
    # Add agents to each environment
    for j in range(10):
        agent = CognitiveAgent(f"agent_{i}_{j}")
        env.add_agent(agent)

# Run with parallel processing
results = engine.run_simulation(duration=3600)
```

## EngineConfig

Configuration object for customizing engine behavior.

```python
@dataclass
class EngineConfig:
    # Simulation timing
    time_step: float = 1.0
    max_steps: int = 10000
    real_time_factor: float = 1.0
    
    # Processing configuration
    parallel_processing: bool = False
    max_threads: int = 4
    batch_size: int = 100
    
    # Memory management
    memory_cleanup_interval: int = 1000
    max_memory_usage_mb: int = 1000
    garbage_collection_enabled: bool = True
    
    # Checkpointing
    auto_checkpoint: bool = False
    checkpoint_interval: int = 3600
    checkpoint_directory: str = "checkpoints/"
    
    # Logging and debugging
    log_level: str = "INFO"
    debug_mode: bool = False
    profile_performance: bool = False
    event_logging: bool = True
    
    # Error handling
    continue_on_error: bool = False
    max_errors: int = 100
    error_callback: Optional[Callable] = None
```

**Example Configuration:**

```python
# Production configuration
production_config = EngineConfig(
    time_step=0.1,
    max_steps=36000,  # 1 hour at 0.1s steps
    real_time_factor=10.0,  # 10x speed
    parallel_processing=True,
    max_threads=8,
    memory_cleanup_interval=1000,
    auto_checkpoint=True,
    checkpoint_interval=3600,
    log_level="INFO",
    debug_mode=False,
    continue_on_error=True
)

# Development configuration
dev_config = EngineConfig(
    time_step=1.0,
    max_steps=1000,
    real_time_factor=1.0,
    parallel_processing=False,
    debug_mode=True,
    profile_performance=True,
    log_level="DEBUG",
    event_logging=True
)

# Create engines with configurations
prod_engine = CognitiveEngine(production_config)
dev_engine = CognitiveEngine(dev_config)
```

## SimulationResults

Results object containing comprehensive simulation data.

```python
@dataclass
class SimulationResults:
    # Basic metrics
    total_steps: int
    total_duration: float
    start_time: datetime
    end_time: datetime
    
    # Performance metrics
    average_step_time: float
    peak_memory_usage: int
    cpu_utilization: float
    
    # Simulation data
    agent_histories: Dict[str, AgentHistory]
    environment_states: Dict[str, List[EnvironmentState]]
    events: List[SimulationEvent]
    
    # Goal and learning metrics
    goals_achieved: int
    total_learning_events: int
    collaboration_events: int
    
    # Errors and warnings
    errors: List[SimulationError]
    warnings: List[SimulationWarning]
    
    # Checkpoints created
    checkpoints: List[str]
    
    def get_summary(self) -> str:
        """Get a human-readable summary of results"""
        
    def export_to_csv(self, filepath: str) -> None:
        """Export results to CSV format"""
        
    def export_to_json(self, filepath: str) -> None:
        """Export results to JSON format"""
        
    def generate_report(self, template: str = "default") -> str:
        """Generate an HTML report of the simulation"""
```

**Example Usage:**

```python
# Run simulation and analyze results
results = engine.run_simulation(duration=3600)

# Print summary
print(results.get_summary())

# Export data for analysis
results.export_to_csv("simulation_data.csv")
results.export_to_json("simulation_results.json")

# Generate HTML report
html_report = results.generate_report(template="detailed")
with open("simulation_report.html", "w") as f:
    f.write(html_report)

# Analyze specific metrics
print(f"Goals achieved: {results.goals_achieved}")
print(f"Average step time: {results.average_step_time:.3f}s")
print(f"Peak memory usage: {results.peak_memory_usage / 1024 / 1024:.2f} MB")

# Access agent-specific data
for agent_id, history in results.agent_histories.items():
    print(f"Agent {agent_id}: {len(history.actions)} actions")
```

## Error Handling

### SimulationError

```python
class SimulationError(CognitoSimError):
    """Errors during simulation execution"""
    
    def __init__(self, message: str, step: int = None, agent_id: str = None):
        super().__init__(message)
        self.step = step
        self.agent_id = agent_id
```

### Common Error Scenarios

```python
try:
    # Simulation execution
    results = engine.run_simulation(duration=3600)
    
except SimulationError as e:
    if e.step is not None:
        print(f"Error at step {e.step}: {e}")
    if e.agent_id is not None:
        print(f"Error with agent {e.agent_id}: {e}")
    
    # Handle specific error types
    if "memory" in str(e).lower():
        # Memory-related error
        engine.cleanup_memory()
    elif "timeout" in str(e).lower():
        # Timeout error
        engine.extend_timeout()
        
except EnvironmentError as e:
    print(f"Environment error: {e}")
    # Fix environment issues
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and handle unexpected errors
```

## Performance Optimization

### Memory Management

```python
# Configure memory-efficient engine
config = EngineConfig(
    memory_cleanup_interval=500,  # Clean up frequently
    max_memory_usage_mb=500,      # Limit memory usage
    garbage_collection_enabled=True
)

engine = CognitiveEngine(config)

# Monitor memory usage
def memory_monitor(engine):
    metrics = engine.get_metrics()
    if metrics.memory_usage_mb > 400:
        engine.force_memory_cleanup()

# Register memory monitor
engine.register_event_handler("step_complete", lambda: memory_monitor(engine))
```

### CPU Optimization

```python
# Configure for CPU efficiency
config = EngineConfig(
    parallel_processing=True,
    max_threads=4,
    batch_size=50,  # Process agents in batches
    profile_performance=True
)

engine = CognitiveEngine(config)

# Monitor CPU usage
profiling_results = engine.get_profiling_results()
if profiling_results.average_cpu_percent > 80:
    # Reduce batch size or thread count
    engine.set_parallel_processing(enabled=True, max_workers=2)
```

## Advanced Usage Patterns

### Custom Engine Extensions

```python
class ResearchEngine(CognitiveEngine):
    """Specialized engine for research simulations"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.research_metrics = ResearchMetrics()
        self.collaboration_tracker = CollaborationTracker()
        
    def step(self):
        # Custom step logic for research scenarios
        step_results = super().step()
        
        # Track research-specific metrics
        self.track_research_progress(step_results)
        self.analyze_collaborations(step_results)
        
        return step_results
    
    def track_research_progress(self, step_results):
        for agent_id, actions in step_results.agent_actions.items():
            for action in actions:
                if action.action_type == "research":
                    self.research_metrics.record_research_action(agent_id, action)
    
    def get_research_summary(self):
        return self.research_metrics.generate_summary()

# Use custom engine
research_engine = ResearchEngine()
```

### Simulation Orchestration

```python
class SimulationOrchestrator:
    """Manages multiple related simulations"""
    
    def __init__(self):
        self.engines = {}
        self.shared_knowledge = SharedKnowledgeBase()
        
    def create_simulation(self, sim_id: str, config: EngineConfig):
        engine = CognitiveEngine(config)
        self.engines[sim_id] = engine
        return engine
    
    def run_parallel_simulations(self, duration: int):
        """Run multiple simulations in parallel"""
        
        from concurrent.futures import ThreadPoolExecutor
        
        def run_sim(sim_id, engine):
            return sim_id, engine.run_simulation(duration=duration)
        
        with ThreadPoolExecutor(max_workers=len(self.engines)) as executor:
            futures = [
                executor.submit(run_sim, sim_id, engine)
                for sim_id, engine in self.engines.items()
            ]
            
            results = {}
            for future in futures:
                sim_id, result = future.result()
                results[sim_id] = result
                
        return results
    
    def cross_simulation_analysis(self, results):
        """Analyze results across multiple simulations"""
        
        comparative_metrics = {}
        for sim_id, result in results.items():
            comparative_metrics[sim_id] = {
                "goals_achieved": result.goals_achieved,
                "learning_events": result.total_learning_events,
                "collaboration_events": result.collaboration_events,
                "efficiency": result.total_steps / result.total_duration
            }
        
        return comparative_metrics

# Example usage
orchestrator = SimulationOrchestrator()

# Create multiple simulations
for i in range(3):
    config = EngineConfig(parallel_processing=True)
    engine = orchestrator.create_simulation(f"sim_{i}", config)
    
    # Set up each simulation differently
    env = Environment(f"env_{i}")
    engine.add_environment(env)

# Run all simulations
results = orchestrator.run_parallel_simulations(duration=3600)

# Analyze comparative results
analysis = orchestrator.cross_simulation_analysis(results)
print(f"Simulation comparison: {analysis}")
```

---

The Engine API provides comprehensive control over cognitive simulations, from basic execution to advanced parallel processing and custom extensions. Use these capabilities to create sophisticated research studies and cognitive experiments.

**Related APIs:**

- [Agents API](agents.md) - Agent management and configuration
- [Environment API](environment.md) - Environment setup and dynamics  
- [Memory API](memory.md) - Memory system configuration
