# Engine Configuration

The `CognitiveEngine` is the heart of your simulation. This guide covers how to configure it for different research scenarios and performance requirements.

## Basic Configuration

### SimulationConfig Class

The `SimulationConfig` class controls all aspects of your simulation:

```python
from cognito_sim_engine import SimulationConfig

config = SimulationConfig(
    max_cycles=100,                    # Maximum simulation cycles
    working_memory_capacity=7,         # Working memory capacity (7±2)
    enable_learning=True,              # Enable learning mechanisms
    enable_metacognition=False,        # Enable metacognitive processes
    enable_metrics=True,               # Collect performance metrics
    step_delay=0.0,                   # Delay between cycles (seconds)
    random_seed=42                     # For reproducible results
)
```

## Advanced Configuration Options

### Memory System Configuration

Fine-tune the memory architecture:

```python
config = SimulationConfig(
    # Working Memory
    working_memory_capacity=5,         # Reduce for constrained cognition
    working_memory_decay=0.1,          # How fast items decay
    
    # Episodic Memory
    episodic_memory_capacity=1000,     # Max episodes to store
    episodic_consolidation_threshold=0.7,  # When to consolidate
    
    # Long-term Memory
    longterm_memory_capacity=10000,    # Max long-term items
    memory_consolidation_rate=0.05,    # Rate of consolidation
    
    # Memory interference
    enable_memory_interference=True,   # Realistic memory conflicts
    interference_threshold=0.8         # Similarity threshold
)
```

### Reasoning Configuration

Control the reasoning engine:

```python
config = SimulationConfig(
    # Inference limits
    max_inference_depth=10,            # Maximum reasoning depth
    reasoning_timeout=5.0,             # Max time per reasoning cycle
    
    # Goal management
    max_active_goals=5,                # Concurrent goals limit
    goal_priority_decay=0.02,          # How goal importance fades
    
    # Rule application
    max_rule_applications=50,          # Per reasoning cycle
    confidence_threshold=0.5,          # Minimum confidence for facts
    
    # Uncertainty handling
    enable_uncertainty_reasoning=True,  # Probabilistic reasoning
    uncertainty_propagation=True       # Propagate uncertainty
)
```

### Performance Optimization

For large-scale simulations:

```python
config = SimulationConfig(
    # Computational limits
    max_cycles=1000,                   # Prevent infinite loops
    cycle_timeout=10.0,                # Max time per cycle
    
    # Memory optimization
    memory_cleanup_interval=100,       # Clean up every N cycles
    garbage_collection_threshold=0.8,  # Memory usage threshold
    
    # Parallel processing
    enable_parallel_agents=False,      # Parallel agent processing
    max_worker_threads=4,              # Thread pool size
    
    # Caching
    enable_reasoning_cache=True,       # Cache inference results
    cache_size_limit=1000             # Max cached items
)
```

## Simulation Profiles

### Research & Development

For deep exploration and learning:

```python
research_config = SimulationConfig(
    max_cycles=500,
    working_memory_capacity=7,
    enable_learning=True,
    enable_metacognition=True,
    enable_metrics=True,
    step_delay=0.1,                   # Slow for observation
    max_inference_depth=15,           # Deep reasoning
    enable_uncertainty_reasoning=True,
    memory_consolidation_rate=0.03    # Slower consolidation
)
```

### Performance Testing

For speed and efficiency evaluation:

```python
performance_config = SimulationConfig(
    max_cycles=10000,
    working_memory_capacity=5,        # Constrained for speed
    enable_learning=False,            # Disable for consistency
    enable_metacognition=False,       # Reduce overhead
    enable_metrics=True,              # Track performance
    step_delay=0.0,                   # Maximum speed
    max_inference_depth=5,            # Shallow reasoning
    enable_reasoning_cache=True,      # Use caching
    memory_cleanup_interval=50        # Frequent cleanup
)
```

### Educational Demonstration

For teaching and demonstration:

```python
demo_config = SimulationConfig(
    max_cycles=50,
    working_memory_capacity=5,
    enable_learning=True,
    enable_metacognition=True,
    enable_metrics=True,
    step_delay=0.5,                   # Slow for observation
    max_inference_depth=8,
    verbose_logging=True,             # Detailed output
    enable_visualization=True,        # Visual feedback
    save_intermediate_states=True     # For step-by-step analysis
)
```

### Cognitive Science Research

For validating cognitive theories:

```python
cognitive_research_config = SimulationConfig(
    max_cycles=200,
    working_memory_capacity=7,        # Miller's 7±2
    enable_learning=True,
    enable_metacognition=True,
    enable_metrics=True,
    step_delay=0.0,
    
    # Realistic cognitive constraints
    working_memory_decay=0.15,        # Realistic decay
    episodic_consolidation_threshold=0.8,
    enable_memory_interference=True,
    
    # Psychological realism
    enable_cognitive_load_tracking=True,
    cognitive_load_limit=1.0,
    attention_focus_decay=0.1,
    
    # Data collection
    collect_detailed_metrics=True,
    save_memory_traces=True,
    track_reasoning_patterns=True
)
```

## Environment Integration

### Connecting Configuration to Environment

```python
from cognito_sim_engine import CognitiveEngine, CognitiveEnvironment

# Create environment
env = CognitiveEnvironment("Research Lab")

# Configure environment based on simulation config
if config.enable_visualization:
    env.enable_visual_display()

if config.enable_metrics:
    env.enable_metric_collection()

# Create engine
engine = CognitiveEngine(config, env)
```

### Dynamic Configuration Updates

Update configuration during simulation:

```python
# Start with basic config
engine = CognitiveEngine(basic_config, env)

# Update during simulation
def on_cycle_complete(cycle_number):
    if cycle_number == 50:
        # Increase learning rate after warm-up
        engine.config.learning_rate *= 1.5
    
    if cycle_number == 100:
        # Enable metacognition mid-simulation
        engine.config.enable_metacognition = True

engine.add_callback('cycle_complete', on_cycle_complete)
```

## Validation and Testing

### Configuration Validation

```python
def validate_config(config):
    """Validate simulation configuration."""
    assert config.max_cycles > 0, "Max cycles must be positive"
    assert 0 < config.working_memory_capacity <= 15, "WM capacity out of range"
    assert 0.0 <= config.learning_rate <= 1.0, "Learning rate out of range"
    
    # Warn about performance implications
    if config.max_cycles > 10000:
        print("⚠️  Large max_cycles may impact performance")
    
    if config.enable_metacognition and config.max_cycles > 1000:
        print("⚠️  Metacognition + large cycles = slow simulation")

# Validate before use
validate_config(config)
```

### A/B Testing Configurations

```python
def compare_configurations(config_a, config_b, test_scenario):
    """Compare two configurations on the same scenario."""
    
    results_a = run_simulation(config_a, test_scenario)
    results_b = run_simulation(config_b, test_scenario)
    
    comparison = {
        'performance_a': results_a.metrics.cycles_per_second,
        'performance_b': results_b.metrics.cycles_per_second,
        'learning_a': results_a.metrics.learning_progress,
        'learning_b': results_b.metrics.learning_progress,
        'memory_efficiency_a': results_a.metrics.memory_usage,
        'memory_efficiency_b': results_b.metrics.memory_usage
    }
    
    return comparison
```

## Configuration Recipes

### Minimal Configuration

For simple experiments:

```python
minimal_config = SimulationConfig(
    max_cycles=20,
    working_memory_capacity=3,
    enable_learning=False,
    enable_metacognition=False,
    enable_metrics=False
)
```

### Maximum Realism

For human-like cognitive simulation:

```python
realistic_config = SimulationConfig(
    max_cycles=300,
    working_memory_capacity=7,
    working_memory_decay=0.2,
    enable_learning=True,
    enable_metacognition=True,
    enable_memory_interference=True,
    episodic_consolidation_threshold=0.8,
    cognitive_load_limit=1.0,
    enable_uncertainty_reasoning=True,
    attention_focus_decay=0.15,
    goal_priority_decay=0.05
)
```

### High-Performance

For large-scale simulations:

```python
performance_config = SimulationConfig(
    max_cycles=50000,
    working_memory_capacity=5,
    enable_learning=False,
    enable_metacognition=False,
    enable_metrics=True,
    step_delay=0.0,
    max_inference_depth=3,
    enable_reasoning_cache=True,
    memory_cleanup_interval=100,
    enable_parallel_agents=True,
    max_worker_threads=8
)
```

## Best Practices

### 1. Start Simple

Begin with minimal configuration and add complexity gradually:

```python
# Start here
basic_config = SimulationConfig(max_cycles=10)

# Then add features
enhanced_config = SimulationConfig(
    max_cycles=10,
    enable_learning=True
)

# Finally, full configuration
full_config = SimulationConfig(
    max_cycles=100,
    enable_learning=True,
    enable_metacognition=True,
    enable_metrics=True
)
```

### 2. Match Configuration to Research Goals

- **Cognitive modeling**: High realism, enable all features
- **Performance testing**: Minimal features, high cycles
- **Algorithm development**: Medium complexity, detailed metrics
- **Education**: Slow speed, visualization enabled

### 3. Monitor Performance

```python
import time

start_time = time.time()
metrics = engine.run_simulation()
duration = time.time() - start_time

print(f"Simulation time: {duration:.2f}s")
print(f"Cycles per second: {metrics.total_cycles / duration:.1f}")
```

### 4. Use Configuration Templates

Create reusable configuration templates:

```python
class ConfigurationTemplates:
    @staticmethod
    def research():
        return SimulationConfig(
            max_cycles=500,
            enable_learning=True,
            enable_metacognition=True,
            enable_metrics=True
        )
    
    @staticmethod
    def demo():
        return SimulationConfig(
            max_cycles=30,
            step_delay=0.3,
            verbose_logging=True
        )
    
    @staticmethod
    def performance():
        return SimulationConfig(
            max_cycles=10000,
            enable_learning=False,
            enable_metacognition=False
        )

# Usage
config = ConfigurationTemplates.research()
```

## Troubleshooting

### Common Configuration Issues

**Slow Performance**:

- Reduce `max_inference_depth`
- Disable `enable_metacognition`
- Increase `memory_cleanup_interval`
- Set `step_delay=0.0`

**Memory Issues**:

- Reduce `working_memory_capacity`
- Lower `episodic_memory_capacity`
- Enable `memory_cleanup_interval`
- Disable detailed metrics

**Unrealistic Behavior**:

- Enable `enable_memory_interference`
- Set realistic `working_memory_decay`
- Add `cognitive_load_limit`
- Enable `uncertainty_reasoning`

### Configuration Debugging

```python
def debug_config(config):
    """Print configuration analysis."""
    print("🔧 Configuration Analysis:")
    print(f"   Cycles: {config.max_cycles}")
    print(f"   Memory: {config.working_memory_capacity}")
    print(f"   Learning: {config.enable_learning}")
    print(f"   Metacognition: {config.enable_metacognition}")
    
    # Estimate performance
    complexity = 1.0
    if config.enable_learning:
        complexity *= 1.5
    if config.enable_metacognition:
        complexity *= 2.0
    complexity *= config.max_inference_depth / 5.0
    
    print(f"   Estimated complexity: {complexity:.1f}x baseline")
    
    if complexity > 5.0:
        print("   ⚠️  High complexity - consider optimization")

debug_config(your_config)
```

This configuration guide should help you tune the Cognito Simulation Engine for your specific research needs and performance requirements.
