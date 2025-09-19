# Creating Agents

This guide covers everything you need to know about creating and configuring cognitive agents in Cognito Simulation Engine.

## Quick Start: Creating Your First Agent

```python
from cognito_sim_engine import CognitiveAgent, Goal

# Create a basic cognitive agent
agent = CognitiveAgent(
    agent_id="my_first_agent",
    personality_traits={
        "openness": 0.7,
        "conscientiousness": 0.8,
        "extraversion": 0.6,
        "agreeableness": 0.9,
        "neuroticism": 0.3
    }
)

# Set a goal for the agent
goal = Goal(
    description="Learn about machine learning",
    priority=0.8,
    target_facts=["understand_ml_basics", "apply_ml_algorithms"]
)

agent.add_goal(goal)

print(f"Created agent: {agent.agent_id}")
print(f"Agent goals: {[g.description for g in agent.goals]}")
```

## Agent Types and Selection

### 1. CognitiveAgent - General Purpose

Best for: Balanced cognitive tasks, social interaction, general problem-solving

```python
from cognito_sim_engine import CognitiveAgent

# General-purpose cognitive agent
general_agent = CognitiveAgent(
    agent_id="general_assistant",
    personality_traits={
        "openness": 0.7,
        "conscientiousness": 0.6,
        "extraversion": 0.5,
        "agreeableness": 0.8,
        "neuroticism": 0.4
    },
    cognitive_config={
        "memory_capacity": 1000,
        "reasoning_depth": 8,
        "learning_rate": 0.1,
        "attention_span": 50
    }
)

# Configure capabilities
general_agent.enable_capabilities([
    "logical_reasoning",
    "social_interaction", 
    "learning_adaptation",
    "goal_management"
])
```

### 2. ReasoningAgent - Logical Analysis

Best for: Mathematical proofs, logical puzzles, systematic analysis

```python
from cognito_sim_engine import ReasoningAgent

# Specialized reasoning agent
logic_agent = ReasoningAgent(
    agent_id="logic_specialist",
    reasoning_config={
        "inference_strategy": "exhaustive",
        "proof_generation": True,
        "uncertainty_handling": True,
        "max_reasoning_depth": 20
    }
)

# Add domain-specific rules
from cognito_sim_engine import Rule, Fact

mathematical_rules = [
    Rule(
        conditions=[Fact("number", ["?x"]), Fact("even", ["?x"])],
        conclusion=Fact("divisible_by_two", ["?x"]),
        confidence=1.0,
        name="even_number_rule"
    ),
    Rule(
        conditions=[Fact("triangle", ["?t"]), Fact("sides_equal", ["?t", "3"])],
        conclusion=Fact("equilateral", ["?t"]),
        confidence=1.0,
        name="equilateral_triangle_rule"
    )
]

for rule in mathematical_rules:
    logic_agent.add_reasoning_rule(rule)
```

### 3. LearningAgent - Adaptive Intelligence

Best for: Dynamic environments, pattern recognition, skill acquisition

```python
from cognito_sim_engine import LearningAgent, LearningStrategy

# Adaptive learning agent
adaptive_agent = LearningAgent(
    agent_id="adaptive_learner",
    learning_config={
        "strategy": LearningStrategy.REINFORCEMENT,
        "exploration_rate": 0.2,
        "learning_rate": 0.15,
        "memory_consolidation": True,
        "transfer_learning": True
    }
)

# Configure learning objectives
adaptive_agent.set_learning_objectives([
    "optimize_task_performance",
    "minimize_error_rate",
    "adapt_to_changes",
    "generalize_knowledge"
])

# Set up reward function
def custom_reward_function(action, outcome, context):
    """Define how the agent learns from outcomes"""
    reward = 0.0
    
    if outcome.success:
        reward += 1.0
    
    # Efficiency bonus
    if outcome.execution_time < action.expected_time:
        reward += 0.5
    
    # Error penalty
    reward -= 0.2 * len(outcome.errors)
    
    # Context-specific adjustments
    if context.get("difficulty") == "high" and outcome.success:
        reward += 0.3
    
    return reward

adaptive_agent.set_reward_function(custom_reward_function)
```

### 4. MetaCognitiveAgent - Strategic Thinking

Best for: Planning, strategy selection, self-monitoring

```python
from cognito_sim_engine import MetaCognitiveAgent, MetaStrategy

# Meta-cognitive agent
meta_agent = MetaCognitiveAgent(
    agent_id="strategic_planner",
    meta_config={
        "self_monitoring": True,
        "strategy_selection": True,
        "cognitive_control": True,
        "reflection_depth": 5
    }
)

# Add meta-cognitive strategies
planning_strategy = MetaStrategy(
    name="task_planning",
    trigger_conditions=["new_complex_task", "multiple_goals"],
    meta_actions=[
        "decompose_task",
        "prioritize_subtasks", 
        "allocate_resources",
        "monitor_progress"
    ]
)

error_recovery_strategy = MetaStrategy(
    name="error_recovery",
    trigger_conditions=["task_failure", "unexpected_outcome"],
    meta_actions=[
        "analyze_failure_cause",
        "adjust_approach",
        "revise_expectations",
        "learn_from_mistake"
    ]
)

meta_agent.add_meta_strategy(planning_strategy)
meta_agent.add_meta_strategy(error_recovery_strategy)
```

## Personality Configuration

### Big Five Personality Traits

Configure agent personality using the scientifically validated Big Five model:

```python
def create_personality_profiles():
    """Create different personality archetypes"""
    
    # Creative researcher profile
    creative_profile = {
        "openness": 0.9,        # High creativity, curiosity
        "conscientiousness": 0.6, # Moderately organized
        "extraversion": 0.7,    # Social, energetic
        "agreeableness": 0.8,   # Cooperative, trusting
        "neuroticism": 0.4      # Some anxiety drives creativity
    }
    
    # Methodical analyst profile
    analytical_profile = {
        "openness": 0.6,        # Open but focused
        "conscientiousness": 0.95, # Extremely organized
        "extraversion": 0.3,    # More introverted
        "agreeableness": 0.7,   # Cooperative but assertive
        "neuroticism": 0.2      # Very stable
    }
    
    # Social facilitator profile
    social_profile = {
        "openness": 0.7,        # Open to others' ideas
        "conscientiousness": 0.7, # Well-organized
        "extraversion": 0.9,    # Highly social
        "agreeableness": 0.9,   # Very cooperative
        "neuroticism": 0.3      # Stable under social pressure
    }
    
    return {
        "creative": creative_profile,
        "analytical": analytical_profile,
        "social": social_profile
    }

# Use personality profiles
profiles = create_personality_profiles()

creative_agent = CognitiveAgent(
    "creative_researcher",
    personality_traits=profiles["creative"]
)

analytical_agent = CognitiveAgent(
    "methodical_analyst", 
    personality_traits=profiles["analytical"]
)

social_agent = CognitiveAgent(
    "team_facilitator",
    personality_traits=profiles["social"]
)
```

### Personality Effects on Behavior

Personality traits influence all aspects of agent behavior:

```python
def demonstrate_personality_effects():
    """Show how personality affects agent behavior"""
    
    # Create agents with different personalities
    cautious_agent = CognitiveAgent(
        "cautious",
        personality_traits={"neuroticism": 0.8, "conscientiousness": 0.9}
    )
    
    adventurous_agent = CognitiveAgent(
        "adventurous", 
        personality_traits={"openness": 0.9, "neuroticism": 0.2}
    )
    
    # Same task, different approaches
    risky_task = Task(
        description="Explore new research direction",
        risk_level=0.7,
        novelty=0.8,
        time_pressure=0.6
    )
    
    # Cautious agent approach
    cautious_plan = cautious_agent.plan_approach(risky_task)
    print("🛡️ Cautious Agent Plan:")
    print(f"  Strategy: {cautious_plan.strategy}")  # Likely: "systematic_validation"
    print(f"  Risk mitigation: {cautious_plan.risk_mitigation}")
    print(f"  Preparation time: {cautious_plan.preparation_time}")
    
    # Adventurous agent approach  
    adventurous_plan = adventurous_agent.plan_approach(risky_task)
    print("\n🚀 Adventurous Agent Plan:")
    print(f"  Strategy: {adventurous_plan.strategy}")  # Likely: "rapid_exploration"
    print(f"  Risk tolerance: {adventurous_plan.risk_tolerance}")
    print(f"  Innovation focus: {adventurous_plan.innovation_focus}")

demonstrate_personality_effects()
```

## Cognitive Configuration

### Memory Settings

Configure memory systems for optimal performance:

```python
from cognito_sim_engine import MemoryConfig

# Configure memory systems
memory_config = MemoryConfig(
    # Working memory settings
    working_memory_capacity=7,
    working_memory_decay=0.1,
    
    # Long-term memory settings
    episodic_capacity=10000,
    semantic_capacity=50000,
    procedural_capacity=1000,
    
    # Consolidation settings
    consolidation_threshold=0.7,
    sleep_consolidation=True,
    
    # Forgetting curves
    episodic_forgetting="power_law",
    semantic_forgetting="exponential",
    
    # Retrieval settings
    retrieval_noise=0.1,
    spreading_activation=True
)

# Apply to agent
configured_agent = CognitiveAgent(
    "memory_optimized",
    memory_config=memory_config
)
```

### Reasoning Configuration

Tune reasoning capabilities:

```python
from cognito_sim_engine import ReasoningConfig

# Configure reasoning engine
reasoning_config = ReasoningConfig(
    # Inference settings
    max_inference_depth=10,
    confidence_threshold=0.6,
    uncertainty_propagation=True,
    
    # Strategy settings
    default_strategy="mixed",
    strategy_selection="adaptive",
    
    # Performance settings
    timeout_seconds=5.0,
    parallel_processing=True,
    caching_enabled=True,
    
    # Bias settings
    confirmation_bias=0.1,
    availability_bias=0.05,
    anchoring_bias=0.08
)

reasoning_agent = ReasoningAgent(
    "tuned_reasoner",
    reasoning_config=reasoning_config
)
```

### Learning Configuration

Optimize learning parameters:

```python
from cognito_sim_engine import LearningConfig

# Configure learning system
learning_config = LearningConfig(
    # Algorithm settings
    learning_algorithm="q_learning",
    learning_rate=0.1,
    discount_factor=0.95,
    
    # Exploration settings
    exploration_strategy="epsilon_greedy",
    initial_epsilon=0.3,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    
    # Experience settings
    experience_replay=True,
    replay_buffer_size=10000,
    batch_size=32,
    
    # Transfer settings
    transfer_learning=True,
    similarity_threshold=0.7,
    
    # Meta-learning settings
    meta_learning=True,
    adaptation_rate=0.05
)

learning_agent = LearningAgent(
    "optimized_learner",
    learning_config=learning_config
)
```

## Goal Management

### Setting Agent Goals

```python
from cognito_sim_engine import Goal, GoalType

# Create different types of goals
achievement_goal = Goal(
    description="Master machine learning fundamentals",
    goal_type=GoalType.ACHIEVEMENT,
    priority=0.9,
    deadline="2024-12-31",
    target_facts=[
        "understand_supervised_learning",
        "understand_unsupervised_learning", 
        "apply_ml_algorithms",
        "evaluate_model_performance"
    ]
)

maintenance_goal = Goal(
    description="Stay updated with latest research",
    goal_type=GoalType.MAINTENANCE,
    priority=0.6,
    recurring=True,
    interval="weekly",
    target_facts=["read_recent_papers", "track_conferences"]
)

avoidance_goal = Goal(
    description="Avoid overfitting in models",
    goal_type=GoalType.AVOIDANCE,
    priority=0.8,
    conditions=["building_ml_models"],
    target_facts=["use_regularization", "validate_properly"]
)

# Add goals to agent
agent.add_goal(achievement_goal)
agent.add_goal(maintenance_goal) 
agent.add_goal(avoidance_goal)

# Goal prioritization and management
agent.prioritize_goals()  # Automatic prioritization
agent.schedule_goal_pursuit()  # Plan goal achievement
```

### Dynamic Goal Adaptation

```python
class AdaptiveGoalSystem:
    def __init__(self, agent):
        self.agent = agent
        self.goal_history = []
        self.adaptation_rules = []
    
    def monitor_goal_progress(self):
        """Monitor and adapt goals based on progress"""
        
        for goal in self.agent.goals:
            progress = self.agent.evaluate_goal_progress(goal)
            
            # Adapt based on progress
            if progress.completion_rate < 0.3 and progress.time_elapsed > 0.7:
                # Poor progress - simplify or extend deadline
                self.adapt_struggling_goal(goal, progress)
                
            elif progress.completion_rate > 0.8 and progress.time_remaining > 0.5:
                # Ahead of schedule - increase ambition
                self.enhance_successful_goal(goal, progress)
    
    def adapt_struggling_goal(self, goal, progress):
        """Adapt goals that are struggling"""
        
        # Option 1: Break into smaller subgoals
        if goal.complexity > 0.7:
            subgoals = self.decompose_goal(goal)
            for subgoal in subgoals:
                self.agent.add_goal(subgoal)
            self.agent.remove_goal(goal)
        
        # Option 2: Extend deadline
        elif goal.deadline:
            extended_deadline = self.calculate_extended_deadline(goal, progress)
            goal.deadline = extended_deadline
        
        # Option 3: Reduce scope
        else:
            simplified_goal = self.simplify_goal(goal)
            self.agent.replace_goal(goal, simplified_goal)
    
    def enhance_successful_goal(self, goal, progress):
        """Enhance goals that are succeeding"""
        
        # Add stretch objectives
        stretch_targets = self.generate_stretch_targets(goal)
        goal.target_facts.extend(stretch_targets)
        
        # Increase priority for high-performing goals
        goal.priority = min(1.0, goal.priority * 1.2)

# Apply adaptive goal management
adaptive_goals = AdaptiveGoalSystem(agent)
adaptive_goals.monitor_goal_progress()
```

## Agent Capabilities and Skills

### Enabling Specific Capabilities

```python
from cognito_sim_engine import Capability

# Define custom capabilities
research_capability = Capability(
    name="research_methodology",
    required_skills=["literature_review", "experiment_design", "data_analysis"],
    knowledge_domains=["scientific_method", "statistics", "domain_expertise"],
    cognitive_requirements={"reasoning_depth": 8, "memory_capacity": 2000}
)

collaboration_capability = Capability(
    name="team_collaboration",
    required_skills=["communication", "coordination", "conflict_resolution"],
    knowledge_domains=["social_dynamics", "project_management"],
    cognitive_requirements={"social_awareness": 0.8, "empathy": 0.7}
)

# Enable capabilities for agent
agent.enable_capability(research_capability)
agent.enable_capability(collaboration_capability)

# Check agent capabilities
print("🎯 Agent Capabilities:")
for capability in agent.get_capabilities():
    print(f"  • {capability.name}: {capability.proficiency_level:.2f}")
```

### Skill Development

```python
from cognito_sim_engine import Skill, SkillLevel

def develop_agent_skills(agent, target_skills):
    """Develop specific skills through practice"""
    
    for skill_name in target_skills:
        # Start with basic skill level
        skill = Skill(
            name=skill_name,
            level=SkillLevel.NOVICE,
            experience_points=0,
            practice_history=[]
        )
        
        agent.add_skill(skill)
        
        # Practice skill through exercises
        exercises = generate_skill_exercises(skill_name)
        
        for exercise in exercises:
            # Practice exercise
            result = agent.practice_skill(skill_name, exercise)
            
            # Update skill based on performance
            if result.success:
                skill.experience_points += result.points_earned
                skill.update_level()
            
            # Track practice history
            skill.practice_history.append({
                "exercise": exercise,
                "result": result,
                "timestamp": time.time()
            })

# Develop skills for research agent
research_skills = [
    "literature_review",
    "hypothesis_formation", 
    "experimental_design",
    "statistical_analysis",
    "scientific_writing"
]

develop_agent_skills(agent, research_skills)
```

## Multi-Agent Coordination

### Creating Agent Teams

```python
from cognito_sim_engine import AgentTeam, TeamRole

def create_research_team():
    """Create a coordinated research team"""
    
    # Define team roles
    roles = {
        "leader": TeamRole(
            name="research_leader",
            responsibilities=["coordination", "decision_making", "oversight"],
            authority_level=0.9
        ),
        "theorist": TeamRole(
            name="theorist", 
            responsibilities=["theory_development", "conceptual_analysis"],
            authority_level=0.7
        ),
        "experimentalist": TeamRole(
            name="experimentalist",
            responsibilities=["experiment_design", "data_collection"],
            authority_level=0.7
        ),
        "analyst": TeamRole(
            name="data_analyst",
            responsibilities=["data_analysis", "statistical_modeling"],
            authority_level=0.6
        )
    }
    
    # Create team members
    team_members = {}
    
    # Research leader (MetaCognitive for strategic planning)
    team_members["leader"] = MetaCognitiveAgent(
        "research_leader",
        personality_traits={
            "conscientiousness": 0.9,
            "extraversion": 0.8,
            "openness": 0.7
        },
        role=roles["leader"]
    )
    
    # Theorist (Reasoning agent for logical analysis)
    team_members["theorist"] = ReasoningAgent(
        "theorist",
        personality_traits={
            "openness": 0.9,
            "conscientiousness": 0.8,
            "neuroticism": 0.3
        },
        role=roles["theorist"]
    )
    
    # Experimentalist (Cognitive agent for balanced skills)
    team_members["experimentalist"] = CognitiveAgent(
        "experimentalist",
        personality_traits={
            "conscientiousness": 0.9,
            "openness": 0.7,
            "agreeableness": 0.8
        },
        role=roles["experimentalist"]
    )
    
    # Data analyst (Learning agent for pattern recognition)
    team_members["analyst"] = LearningAgent(
        "data_analyst",
        personality_traits={
            "conscientiousness": 0.8,
            "openness": 0.6,
            "neuroticism": 0.4
        },
        role=roles["analyst"]
    )
    
    # Create team
    research_team = AgentTeam(
        team_id="cognitive_research_team",
        members=list(team_members.values()),
        coordination_strategy="hierarchical",
        communication_protocol="formal"
    )
    
    return research_team

# Create and configure team
team = create_research_team()

# Set team goals
team_goal = Goal(
    description="Develop new cognitive architecture",
    goal_type=GoalType.ACHIEVEMENT,
    priority=1.0,
    collaborative=True,
    required_roles=["leader", "theorist", "experimentalist", "analyst"]
)

team.set_shared_goal(team_goal)
```

## Agent Monitoring and Debugging

### Performance Monitoring

```python
from cognito_sim_engine import AgentMonitor, PerformanceMetrics

# Create agent monitor
monitor = AgentMonitor(
    metrics_to_track=[
        "goal_achievement_rate",
        "memory_utilization",
        "reasoning_accuracy",
        "learning_progress",
        "social_effectiveness"
    ],
    monitoring_interval=10,  # Every 10 actions
    alert_thresholds={
        "goal_achievement_rate": 0.3,  # Alert if below 30%
        "memory_utilization": 0.9,     # Alert if above 90%
        "reasoning_accuracy": 0.5      # Alert if below 50%
    }
)

# Attach monitor to agent
monitor.attach_to_agent(agent)

# View real-time metrics
def display_agent_metrics(agent):
    """Display current agent performance metrics"""
    
    metrics = monitor.get_current_metrics(agent)
    
    print(f"📊 Agent Performance Metrics for {agent.agent_id}:")
    print(f"  🎯 Goal Achievement: {metrics['goal_achievement_rate']:.2f}")
    print(f"  🧠 Memory Usage: {metrics['memory_utilization']:.2f}")
    print(f"  🤔 Reasoning Accuracy: {metrics['reasoning_accuracy']:.2f}")
    print(f"  📈 Learning Progress: {metrics['learning_progress']:.2f}")
    print(f"  👥 Social Effectiveness: {metrics['social_effectiveness']:.2f}")
    
    # Show alerts if any
    alerts = monitor.get_alerts(agent)
    if alerts:
        print("⚠️ Performance Alerts:")
        for alert in alerts:
            print(f"    • {alert.metric}: {alert.message}")

# Monitor agent during simulation
def run_monitored_simulation(agent, tasks):
    """Run simulation with continuous monitoring"""
    
    for i, task in enumerate(tasks):
        # Execute task
        result = agent.execute_task(task)
        
        # Check metrics every 10 tasks
        if i % 10 == 0:
            display_agent_metrics(agent)
            
            # Auto-adjust if needed
            alerts = monitor.get_alerts(agent)
            for alert in alerts:
                if alert.metric == "memory_utilization":
                    agent.memory_manager.cleanup_old_memories()
                elif alert.metric == "goal_achievement_rate":
                    agent.revise_goal_strategies()
```

### Debugging Agent Behavior

```python
from cognito_sim_engine import AgentDebugger

# Create debugger with detailed logging
debugger = AgentDebugger(
    log_level="detailed",
    trace_components=[
        "reasoning_steps",
        "memory_access",
        "goal_processing",
        "decision_making"
    ]
)

# Attach debugger to agent
debugger.attach(agent)

# Debug specific agent behavior
def debug_agent_decision(agent, decision_context):
    """Debug why agent made specific decision"""
    
    # Enable detailed tracing
    debugger.start_trace("decision_analysis")
    
    # Let agent make decision
    decision = agent.make_decision(decision_context)
    
    # Stop tracing and analyze
    trace = debugger.stop_trace("decision_analysis")
    
    print("🔍 Decision Analysis:")
    print(f"  Decision: {decision.action}")
    print(f"  Confidence: {decision.confidence:.2f}")
    
    print("\n🧠 Reasoning Steps:")
    for step in trace.reasoning_steps:
        print(f"    {step.step_number}: {step.description}")
        print(f"      Confidence: {step.confidence:.2f}")
    
    print("\n💾 Memory Accesses:")
    for access in trace.memory_accesses:
        print(f"    {access.memory_type}: {access.query}")
        print(f"      Retrieved: {len(access.results)} items")
    
    print("\n🎯 Goal Considerations:")
    for goal_eval in trace.goal_evaluations:
        print(f"    {goal_eval.goal.description}")
        print(f"      Relevance: {goal_eval.relevance:.2f}")
        print(f"      Progress impact: {goal_eval.progress_impact:.2f}")

# Example debugging session
decision_context = {
    "situation": "Multiple research directions available",
    "time_pressure": 0.6,
    "resources": ["literature", "lab_access", "collaborators"],
    "constraints": ["budget_limited", "deadline_approaching"]
}

debug_agent_decision(agent, decision_context)
```

## Best Practices

### 1. Agent Selection

- **Match agent type to task**: Use ReasoningAgent for logical tasks, LearningAgent for adaptive scenarios
- **Consider personality fit**: Align personality traits with role requirements
- **Balance team composition**: Mix complementary personalities and capabilities

### 2. Configuration Optimization

- **Start with defaults**: Begin with standard configurations and tune based on performance
- **Monitor resource usage**: Track memory and processing utilization
- **Iterative refinement**: Adjust parameters based on observed behavior

### 3. Goal Management

- **Clear, specific goals**: Define measurable objectives with success criteria
- **Appropriate complexity**: Match goal complexity to agent capabilities
- **Regular review**: Monitor and adapt goals based on progress

### 4. Performance Tuning

- **Profile bottlenecks**: Identify performance limitations using monitoring tools
- **Optimize memory**: Tune memory configurations for task requirements
- **Balance accuracy vs speed**: Adjust reasoning depth and timeout values

---

Creating effective cognitive agents requires understanding both the technical capabilities and the cognitive science principles underlying the system. This guide provides the foundation for building sophisticated agents tailored to your specific research or application needs.

**Next**: Learn about [Environment Setup](environment-setup.md) to create rich contexts for your agents, or explore [Memory Management](memory-management.md) for optimizing agent knowledge systems.
