"""
Demo Agent - Comprehensive demonstration of the Cognito Simulation Engine.

This example showcases a complete cognitive agent with memory, reasoning,
and learning capabilities interacting with a rich environment.
"""

import time
import random
from cognito_sim_engine import (
    CognitiveEngine, SimulationConfig, CognitiveAgent, ReasoningAgent,
    LearningAgent, MetaCognitiveAgent, CognitiveEnvironment,
    Goal, Fact, MemoryItem, MemoryType, AgentPersonality
)


def main():
    """Run the comprehensive demo agent simulation."""
    print("🧠 Cognito Simulation Engine - Demo Agent")
    print("=" * 50)
    
    # Create simulation configuration
    config = SimulationConfig(
        max_cycles=100,
        working_memory_capacity=7,
        enable_metacognition=True,
        enable_learning=True,
        enable_metrics=True,
        log_level="INFO"
    )
    
    # Create cognitive environment
    environment = create_demo_environment()
    
    # Create cognitive engine
    engine = CognitiveEngine(config=config, environment=environment)
    
    # Create different types of agents
    agents = create_demo_agents(environment)
    
    # Add goals to agents
    add_demo_goals(agents)
    
    # Add some initial knowledge
    add_initial_knowledge(agents)
    
    print(f"Created {len(agents)} agents in demo environment")
    print("Starting simulation...")
    
    # Run simulation with progress tracking
    run_simulation_with_tracking(engine, agents)
    
    # Display results
    display_demo_results(engine, agents)


def create_demo_environment():
    """Create a rich demo environment with interactive objects."""
    env = CognitiveEnvironment("Cognitive Research Laboratory")
    
    # Add research equipment
    objects = [
        {
            "name": "computer",
            "type": "equipment",
            "position": {"x": 2, "y": 3, "z": 1},
            "properties": {"state": "on", "data": "research_papers", "accessible": True},
            "description": "A powerful computer with research databases"
        },
        {
            "name": "whiteboard",
            "type": "tool",
            "position": {"x": 0, "y": 5, "z": 2},
            "properties": {"content": "cognitive_theories", "writable": True},
            "description": "A whiteboard with cognitive architecture diagrams"
        },
        {
            "name": "book_shelf",
            "type": "storage",
            "position": {"x": 5, "y": 1, "z": 0},
            "properties": {"books": ["AI_textbook", "cognitive_science", "neuroscience"], "searchable": True},
            "description": "A shelf filled with AI and cognitive science books"
        },
        {
            "name": "experiment_station",
            "type": "equipment",
            "position": {"x": 3, "y": 7, "z": 1},
            "properties": {"status": "ready", "experiment_type": "memory_test"},
            "description": "Station for conducting cognitive experiments"
        }
    ]
    
    # Add objects to environment
    for obj_data in objects:
        from cognito_sim_engine.environment import EnvironmentObject
        obj = EnvironmentObject(
            name=obj_data["name"],
            object_type=obj_data["type"],
            position=obj_data["position"],
            properties=obj_data["properties"],
            description=obj_data["description"],
            interactable=True
        )
        env.state.add_object(obj)
    
    return env


def create_demo_agents(environment):
    """Create a diverse set of demo agents."""
    agents = []
    
    # 1. Basic Cognitive Agent - The Explorer
    explorer_personality = AgentPersonality(
        curiosity=0.9,
        caution=0.3,
        persistence=0.6,
        creativity=0.7
    )
    
    explorer = CognitiveAgent(
        agent_id="explorer_01",
        name="Dr. Explorer",
        personality=explorer_personality,
        working_memory_capacity=7,
        enable_metacognition=True
    )
    environment.add_agent("explorer_01")
    agents.append(explorer)
    
    # 2. Reasoning Agent - The Logician
    logician = ReasoningAgent(
        agent_id="logician_01",
        name="Prof. Logic"
    )
    environment.add_agent("logician_01")
    agents.append(logician)
    
    # 3. Learning Agent - The Student
    student_personality = AgentPersonality(
        curiosity=0.8,
        persistence=0.8,
        analyticalness=0.6,
        creativity=0.5
    )
    
    student = LearningAgent(
        agent_id="student_01",
        name="Ada Learner",
        personality=student_personality
    )
    environment.add_agent("student_01")
    agents.append(student)
    
    # 4. MetaCognitive Agent - The Philosopher
    philosopher = MetaCognitiveAgent(
        agent_id="philosopher_01",
        name="Sage Thinker"
    )
    environment.add_agent("philosopher_01")
    agents.append(philosopher)
    
    return agents


def add_demo_goals(agents):
    """Add realistic research goals to the agents."""
    
    # Goals for Explorer
    explorer = agents[0]
    explorer_goals = [
        Goal(
            description="Map the laboratory environment",
            priority=0.7,
            target_facts=[Fact("mapped", ["laboratory"])],
            metadata={"type": "exploration", "difficulty": "easy"}
        ),
        Goal(
            description="Discover all available research tools",
            priority=0.8,
            target_facts=[Fact("discovered", ["research_tools"])],
            metadata={"type": "discovery", "difficulty": "medium"}
        )
    ]
    
    for goal in explorer_goals:
        explorer.add_goal(goal)
    
    # Goals for Logician
    logician = agents[1]
    logician_goals = [
        Goal(
            description="Analyze cognitive architecture theories",
            priority=0.9,
            target_facts=[Fact("analyzed", ["cognitive_theories"])],
            metadata={"type": "reasoning", "difficulty": "hard"}
        ),
        Goal(
            description="Develop logical framework for AI",
            priority=0.8,
            target_facts=[Fact("developed", ["logical_framework"])],
            metadata={"type": "creation", "difficulty": "hard"}
        )
    ]
    
    for goal in logician_goals:
        logician.add_goal(goal)
    
    # Goals for Student
    student = agents[2]
    student_goals = [
        Goal(
            description="Learn about memory systems",
            priority=0.8,
            target_facts=[Fact("learned", ["memory_systems"])],
            metadata={"type": "learning", "difficulty": "medium"}
        ),
        Goal(
            description="Master reasoning techniques",
            priority=0.7,
            target_facts=[Fact("mastered", ["reasoning_techniques"])],
            metadata={"type": "skill_acquisition", "difficulty": "medium"}
        )
    ]
    
    for goal in student_goals:
        student.add_goal(goal)
    
    # Goals for Philosopher
    philosopher = agents[3]
    philosopher_goals = [
        Goal(
            description="Reflect on the nature of cognition",
            priority=0.9,
            target_facts=[Fact("reflected", ["nature_of_cognition"])],
            metadata={"type": "metacognition", "difficulty": "very_hard"}
        ),
        Goal(
            description="Understand consciousness and AI",
            priority=0.8,
            target_facts=[Fact("understood", ["consciousness_ai"])],
            metadata={"type": "philosophical", "difficulty": "very_hard"}
        )
    ]
    
    for goal in philosopher_goals:
        philosopher.add_goal(goal)


def add_initial_knowledge(agents):
    """Add initial knowledge and facts to agents."""
    
    for agent in agents:
        # Basic environmental facts
        basic_facts = [
            Fact("location", [agent.agent_id, "laboratory"]),
            Fact("can_move", [agent.agent_id]),
            Fact("can_observe", [agent.agent_id]),
            Fact("can_interact", [agent.agent_id]),
            Fact("has_goals", [agent.agent_id])
        ]
        
        for fact in basic_facts:
            agent.inference_engine.reasoner.add_fact(fact)
        
        # Add initial memories
        initial_memories = [
            MemoryItem(
                content=f"I am {agent.name}, a cognitive agent in a research laboratory",
                memory_type=MemoryType.SEMANTIC,
                importance=0.9,
                metadata={"type": "self_knowledge"}
            ),
            MemoryItem(
                content="The laboratory contains research equipment and resources",
                memory_type=MemoryType.SEMANTIC,
                importance=0.7,
                metadata={"type": "environmental_knowledge"}
            ),
            MemoryItem(
                content="My purpose is to achieve my goals through reasoning and learning",
                memory_type=MemoryType.SEMANTIC,
                importance=0.8,
                metadata={"type": "purpose"}
            )
        ]
        
        for memory in initial_memories:
            agent.memory_manager.store_memory(memory)


def run_simulation_with_tracking(engine, agents):
    """Run simulation with detailed progress tracking."""
    
    # Add callbacks for tracking
    cycle_data = []
    
    def track_cycle(engine):
        data = {
            "cycle": engine.current_cycle,
            "active_goals": len(engine.current_goals),
            "agent_states": {}
        }
        
        for agent in agents:
            data["agent_states"][agent.name] = {
                "working_memory": len(agent.memory_manager.get_working_memory()),
                "total_actions": agent.total_actions,
                "reasoning_cycles": agent.total_reasoning_cycles,
                "active_goals": len([g for g in agent.current_goals if g.is_active()])
            }
        
        cycle_data.append(data)
        
        # Print progress every 10 cycles
        if engine.current_cycle % 10 == 0:
            print(f"Cycle {engine.current_cycle}: {len(engine.current_goals)} engine goals, "
                  f"{sum(len(a.current_goals) for a in agents)} total agent goals")
    
    engine.add_cycle_callback(track_cycle)
    
    # Set up agent callbacks
    for agent in agents:
        def action_callback(agent, action, success):
            if success and random.random() < 0.1:  # 10% chance of learning feedback
                feedback = {
                    "reward": random.uniform(0.1, 0.8),
                    "skill": action.action_type.value,
                    "performance": random.uniform(0.3, 0.9)
                }
                agent.learn(feedback)
        
        agent.action_callbacks.append(action_callback)
    
    # Run the simulation
    start_time = time.time()
    metrics = engine.run_simulation()
    end_time = time.time()
    
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    return metrics, cycle_data


def display_demo_results(engine, agents):
    """Display comprehensive demo results."""
    print("\n" + "=" * 60)
    print("🎯 DEMO SIMULATION RESULTS")
    print("=" * 60)
    
    # Engine metrics
    state_summary = engine.get_state_summary()
    print(f"\n📊 Engine Summary:")
    print(f"  • Total cycles: {state_summary['current_cycle']}")
    print(f"  • Simulation state: {state_summary['simulation_state']}")
    print(f"  • Final working memory items: {state_summary['working_memory_items']}")
    
    metrics = state_summary['metrics']
    print(f"  • Goals achieved: {metrics['goals_achieved']}")
    print(f"  • Goals failed: {metrics['goals_failed']}")
    print(f"  • Average reasoning time: {metrics['avg_reasoning_time']:.3f}s")
    print(f"  • Attention switches: {metrics['attention_switches']}")
    
    # Individual agent results
    print(f"\n🤖 Agent Results:")
    for i, agent in enumerate(agents):
        print(f"\n  Agent {i+1}: {agent.name} ({agent.__class__.__name__})")
        
        status = agent.get_status()
        print(f"    • Total actions: {status['metrics']['total_actions']}")
        print(f"    • Success rate: {agent.success_rate:.2f}")
        print(f"    • Reasoning cycles: {status['metrics']['total_reasoning_cycles']}")
        
        # Memory statistics
        memory_stats = agent.memory_manager.get_memory_statistics()
        print(f"    • Working memory usage: {memory_stats['working_memory']['usage']:.2f}")
        print(f"    • Total memories: {memory_stats['total_memories']}")
        
        # Goal analysis
        achieved_goals = [g for g in agent.current_goals if g.is_achieved()]
        active_goals = [g for g in agent.current_goals if g.is_active()]
        print(f"    • Goals achieved: {len(achieved_goals)}")
        print(f"    • Goals remaining: {len(active_goals)}")
        
        # Metacognitive insights (if available)
        if hasattr(agent, 'metacognitive_insights') and agent.metacognitive_insights:
            print(f"    • Metacognitive insights: {len(agent.metacognitive_insights)}")
            if agent.metacognitive_insights:
                latest_insight = agent.metacognitive_insights[-1]
                print(f"      Latest: {latest_insight[:60]}...")
        
        # Learning progress (if available)
        if hasattr(agent, 'skill_levels') and agent.skill_levels:
            print(f"    • Skills learned: {len(agent.skill_levels)}")
            for skill, level in agent.skill_levels.items():
                print(f"      {skill}: {level:.2f}")
    
    # Environment summary
    env_summary = engine.environment.get_environment_summary()
    print(f"\n🌍 Environment Summary:")
    print(f"  • Environment: {env_summary['name']}")
    print(f"  • Objects: {env_summary['objects']}")
    print(f"  • Total actions executed: {env_summary['total_actions']}")
    print(f"  • Total perceptions generated: {env_summary['total_perceptions']}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"This demonstration showcased:")
    print(f"  • Multi-agent cognitive simulation")
    print(f"  • Different agent architectures working together") 
    print(f"  • Memory formation and reasoning processes")
    print(f"  • Goal-directed behavior and learning")
    print(f"  • Environmental interaction and perception")
    print(f"  • Metacognitive reflection and self-awareness")


if __name__ == "__main__":
    main()
