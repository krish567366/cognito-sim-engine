# Quick Start

Get up and running with Cognito Simulation Engine in minutes!

## 30-Second Demo

Try the built-in demonstration:

```bash
cognito-sim demo
```

This runs a pre-configured simulation showcasing multiple cognitive agents in an interactive environment.

## Your First Simulation

### 1. Basic Setup

```python
from cognito_sim_engine import (
    CognitiveEngine, 
    SimulationConfig, 
    CognitiveAgent, 
    CognitiveEnvironment
)

# Create simulation configuration
config = SimulationConfig(
    max_cycles=50,
    working_memory_capacity=7,
    enable_learning=True,
    enable_metacognition=True
)

# Create environment
environment = CognitiveEnvironment("Research Lab")

# Create cognitive engine
engine = CognitiveEngine(config, environment)
```

### 2. Add an Agent

```python
# Create a cognitive agent
agent = CognitiveAgent("alice", "Alice Explorer")

# Add agent to environment
environment.add_agent("alice")

# Give the agent some goals
from cognito_sim_engine import Goal, Fact

goal = Goal(
    description="Explore the research lab",
    priority=0.8,
    target_facts=[Fact("explored", ["research_lab"])]
)

agent.add_goal(goal)
```

### 3. Run the Simulation

```python
# Run the simulation
metrics = engine.run_simulation()

# View results
print(f"Simulation completed in {metrics.total_cycles} cycles")
print(f"Agent performed {agent.total_actions} actions")

# Get agent's final state
state = agent.get_cognitive_state()
print(f"Memory items: {state['memory']['total_items']}")
```

## Complete Example

Here's a full working example you can run:

```python
"""
Simple cognitive simulation example
"""
from cognito_sim_engine import *

def main():
    # Configuration
    config = SimulationConfig(
        max_cycles=20,
        working_memory_capacity=5,
        enable_learning=True,
        enable_metacognition=False,
        step_delay=0.1  # Slow down for observation
    )
    
    # Environment setup
    env = CognitiveEnvironment("Learning Lab")
    engine = CognitiveEngine(config, env)
    
    # Create agent with personality
    personality = AgentPersonality(
        curiosity=0.8,
        caution=0.3,
        sociability=0.6
    )
    
    agent = CognitiveAgent("learner", "Alex", personality=personality)
    env.add_agent("learner")
    
    # Set up agent's knowledge base
    facts = [
        Fact("in_location", ["learner", "lab"]),
        Fact("wants_to_learn", ["learner"]),
        Fact("has_curiosity", ["learner"])
    ]
    
    for fact in facts:
        agent.inference_engine.reasoner.add_fact(fact)
    
    # Add learning goal
    learning_goal = Goal(
        description="Learn about the environment",
        priority=0.9,
        target_facts=[Fact("learned_about", ["environment"])]
    )
    
    agent.add_goal(learning_goal)
    
    # Run simulation
    print("🧠 Starting cognitive simulation...")
    metrics = engine.run_simulation()
    
    # Results
    print(f"\n📊 Simulation Results:")
    print(f"   Cycles: {metrics.total_cycles}")
    print(f"   Actions: {agent.total_actions}")
    
    # Memory analysis
    memory_stats = agent.memory_manager.get_memory_statistics()
    print(f"\n🧠 Memory Analysis:")
    print(f"   Working Memory: {memory_stats['working_memory']['items']} items")
    print(f"   Episodic Memory: {memory_stats['episodic_memory']['episodes']} episodes")
    print(f"   Total Memories: {memory_stats['total_memories']}")
    
    # Goal progress
    print(f"\n🎯 Goals Status:")
    for goal in agent.current_goals:
        print(f"   {goal.description}: {goal.status.value}")

if __name__ == "__main__":
    main()
```

## CLI Quick Start

The command-line interface provides instant access to simulations:

```bash
# Create and run a simulation
cognito-sim run --agent-count 2 --cycles 30

# Create a custom agent
cognito-sim create-agent --name "researcher" --personality curious

# Analyze simulation results
cognito-sim analyze results.json

# Get help
cognito-sim --help
```

## Interactive Mode

Launch an interactive simulation session:

```bash
cognito-sim --interactive
```

This opens a rich console interface where you can:

- Monitor agent states in real-time
- Modify goals and parameters on-the-fly
- Visualize memory and reasoning processes
- Export results and metrics

## Key Concepts

### Cognitive Cycles

Every agent follows a **Perceive → Reason → Act** cycle:

1. **Perceive**: Gather information from environment and memory
2. **Reason**: Process information using symbolic reasoning
3. **Act**: Select and execute actions based on goals

### Memory Systems

Three types of memory work together:

- **Working Memory**: Immediate, limited-capacity processing (7±2 items)
- **Episodic Memory**: Personal experiences and events
- **Long-term Memory**: Consolidated knowledge and skills

### Goals & Reasoning

Agents use symbolic reasoning to:

- Maintain and prioritize goals
- Plan action sequences
- Learn from experience
- Adapt to changing environments

## Next Steps

Now that you've created your first simulation, explore:

- [Your First Advanced Simulation](first-simulation.md) - Multi-agent scenarios
- [Agent Types](../guide/creating-agents.md) - Specialized cognitive architectures
- [Environment Design](../guide/environment-setup.md) - Custom simulation worlds
- [Memory Deep Dive](../theory/memory-systems.md) - Advanced memory modeling

## Common Patterns

### Learning Agents

```python
from cognito_sim_engine import LearningAgent

learner = LearningAgent("student", "Ada")
learner.set_learning_rate(0.1)
learner.enable_skill_tracking()
```

### Multi-Agent Systems

```python
agents = [
    CognitiveAgent("explorer", "Explorer"),
    ReasoningAgent("logician", "Logician"),
    LearningAgent("student", "Student")
]

for agent in agents:
    environment.add_agent(agent.agent_id)
```

### Custom Environments

```python
env = CognitiveEnvironment("Custom World")
env.add_object("treasure", {"x": 10, "y": 5, "valuable": True})
env.set_boundary(20, 20)  # 20x20 grid
```

Happy simulating! 🧠✨
