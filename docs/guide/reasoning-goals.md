# Reasoning & Goals

This guide covers how to configure and manage the reasoning engine and goal systems that drive intelligent agent behavior in Cognito Simulation Engine.

## Reasoning Engine Configuration

### Basic Setup

```python
from cognito_sim_engine import InferenceEngine, SymbolicReasoner

# Create reasoning engine with standard configuration
reasoning_config = {
    "max_inference_depth": 10,
    "confidence_threshold": 0.6,
    "timeout_seconds": 5.0,
    "strategy": "mixed",
    "uncertainty_handling": True,
    "parallel_processing": False
}

inference_engine = InferenceEngine(config=reasoning_config)

# Configure the symbolic reasoner
reasoner = SymbolicReasoner(
    depth_limit=10,
    breadth_limit=20,
    confidence_propagation=True,
    contradiction_detection=True
)

inference_engine.set_reasoner(reasoner)
```

### Advanced Reasoning Configuration

```python
class AdvancedReasoningEngine(InferenceEngine):
    def __init__(self, config):
        super().__init__(config)
        self.reasoning_strategies = {
            "forward_chaining": ForwardChainingStrategy(),
            "backward_chaining": BackwardChainingStrategy(),
            "abductive": AbductiveReasoningStrategy(),
            "analogical": AnalogicalReasoningStrategy(),
            "causal": CausalReasoningStrategy()
        }
        self.meta_reasoner = MetaReasoningController()
        self.reasoning_cache = ReasoningCache(size_limit=1000)
    
    def configure_adaptive_strategy_selection(self):
        """Configure strategy selection based on problem type"""
        
        strategy_rules = [
            {
                "condition": lambda problem: problem.type == "diagnostic",
                "strategy": "backward_chaining",
                "confidence": 0.8
            },
            {
                "condition": lambda problem: problem.type == "prediction",
                "strategy": "forward_chaining", 
                "confidence": 0.9
            },
            {
                "condition": lambda problem: problem.type == "explanation",
                "strategy": "abductive",
                "confidence": 0.7
            },
            {
                "condition": lambda problem: problem.uncertainty > 0.5,
                "strategy": "analogical",
                "confidence": 0.6
            }
        ]
        
        self.meta_reasoner.add_strategy_rules(strategy_rules)
    
    def reason_with_meta_control(self, goal, facts, context=None):
        """Reasoning with meta-level strategy control"""
        
        # Analyze problem characteristics
        problem_analysis = self.analyze_problem(goal, facts, context)
        
        # Select appropriate reasoning strategy
        strategy = self.meta_reasoner.select_strategy(problem_analysis)
        
        # Execute reasoning with selected strategy
        reasoning_result = self.execute_reasoning(
            strategy=strategy,
            goal=goal,
            facts=facts,
            context=context
        )
        
        # Monitor and adapt strategy if needed
        if reasoning_result.confidence < 0.5:
            alternative_strategy = self.meta_reasoner.select_alternative_strategy(
                problem_analysis, 
                failed_strategy=strategy
            )
            
            reasoning_result = self.execute_reasoning(
                strategy=alternative_strategy,
                goal=goal,
                facts=facts,
                context=context
            )
        
        # Cache successful reasoning patterns
        if reasoning_result.success:
            self.reasoning_cache.store_pattern(
                problem_type=problem_analysis.type,
                strategy=strategy,
                success_metrics=reasoning_result.metrics
            )
        
        return reasoning_result

# Create advanced reasoning engine
advanced_config = {
    "max_inference_depth": 15,
    "confidence_threshold": 0.5,
    "timeout_seconds": 10.0,
    "strategy": "adaptive",
    "uncertainty_handling": True,
    "parallel_processing": True,
    "meta_reasoning": True,
    "analogical_reasoning": True,
    "causal_reasoning": True
}

advanced_reasoning = AdvancedReasoningEngine(advanced_config)
advanced_reasoning.configure_adaptive_strategy_selection()
```

## Goal Management System

### Goal Types and Configuration

```python
from cognito_sim_engine import Goal, GoalType, GoalManager

# Create different types of goals
achievement_goal = Goal(
    goal_id="learn_ml_fundamentals",
    description="Master machine learning fundamentals",
    goal_type=GoalType.ACHIEVEMENT,
    priority=0.9,
    deadline="2024-12-31",
    success_criteria=[
        "understand_supervised_learning",
        "understand_unsupervised_learning",
        "implement_basic_algorithms",
        "evaluate_model_performance"
    ],
    measurable_metrics={
        "knowledge_coverage": 0.8,
        "practical_skill": 0.7,
        "confidence_level": 0.8
    }
)

maintenance_goal = Goal(
    goal_id="stay_current_research",
    description="Stay current with AI research",
    goal_type=GoalType.MAINTENANCE,
    priority=0.6,
    recurring=True,
    interval="weekly",
    success_criteria=[
        "read_recent_papers",
        "attend_conferences",
        "participate_discussions"
    ],
    measurable_metrics={
        "papers_read_per_week": 5,
        "conferences_attended_per_year": 3,
        "discussion_participation": 0.7
    }
)

avoidance_goal = Goal(
    goal_id="avoid_research_pitfalls",
    description="Avoid common research mistakes",
    goal_type=GoalType.AVOIDANCE,
    priority=0.8,
    conditions=["conducting_research"],
    success_criteria=[
        "avoid_confirmation_bias",
        "avoid_overfitting",
        "avoid_cherry_picking"
    ],
    violation_detection_rules=[
        "check_methodology_rigor",
        "validate_statistical_significance",
        "ensure_reproducibility"
    ]
)

# Create goal manager
goal_manager = GoalManager(
    max_active_goals=5,
    priority_update_frequency=10,  # Update priorities every 10 cycles
    goal_conflict_resolution="priority_based",
    achievement_tracking=True
)

goal_manager.add_goal(achievement_goal)
goal_manager.add_goal(maintenance_goal)
goal_manager.add_goal(avoidance_goal)
```

### Advanced Goal Processing

```python
class AdvancedGoalManager(GoalManager):
    def __init__(self, config):
        super().__init__(config)
        self.goal_decomposer = GoalDecomposer()
        self.goal_scheduler = GoalScheduler()
        self.conflict_resolver = GoalConflictResolver()
        self.progress_tracker = GoalProgressTracker()
        
    def process_complex_goal(self, complex_goal):
        """Process complex goals through decomposition and planning"""
        
        # Decompose complex goal into subgoals
        subgoals = self.goal_decomposer.decompose(complex_goal)
        
        # Create dependency graph
        dependency_graph = self.goal_decomposer.create_dependency_graph(subgoals)
        
        # Schedule goal pursuit
        schedule = self.goal_scheduler.create_schedule(subgoals, dependency_graph)
        
        # Add to active goals with scheduling information
        for subgoal in subgoals:
            subgoal.schedule_info = schedule.get_schedule_info(subgoal.goal_id)
            self.add_goal(subgoal)
        
        return subgoals
    
    def resolve_goal_conflicts(self):
        """Resolve conflicts between active goals"""
        
        active_goals = self.get_active_goals()
        conflicts = self.conflict_resolver.detect_conflicts(active_goals)
        
        for conflict in conflicts:
            resolution = self.conflict_resolver.resolve_conflict(conflict)
            
            if resolution.type == "priority_adjustment":
                for goal_id, new_priority in resolution.priority_adjustments.items():
                    self.update_goal_priority(goal_id, new_priority)
            
            elif resolution.type == "goal_modification":
                for goal_id, modifications in resolution.goal_modifications.items():
                    self.modify_goal(goal_id, modifications)
            
            elif resolution.type == "goal_suspension":
                for goal_id in resolution.suspended_goals:
                    self.suspend_goal(goal_id)
    
    def adaptive_goal_management(self, agent_state, environment_state):
        """Adaptively manage goals based on current context"""
        
        # Update goal priorities based on context
        context_priorities = self.calculate_contextual_priorities(
            agent_state, 
            environment_state
        )
        
        for goal_id, context_priority in context_priorities.items():
            current_goal = self.get_goal(goal_id)
            if current_goal:
                # Blend original priority with contextual factors
                blended_priority = (
                    current_goal.base_priority * 0.7 + 
                    context_priority * 0.3
                )
                self.update_goal_priority(goal_id, blended_priority)
        
        # Generate new goals based on opportunities
        opportunities = self.detect_goal_opportunities(agent_state, environment_state)
        for opportunity in opportunities:
            if opportunity.confidence > 0.7:
                new_goal = self.create_goal_from_opportunity(opportunity)
                self.add_goal(new_goal)
        
        # Retire completed or obsolete goals
        self.retire_obsolete_goals()
    
    def track_goal_progress(self):
        """Track and analyze goal achievement progress"""
        
        for goal in self.get_active_goals():
            progress = self.progress_tracker.assess_progress(goal)
            
            # Update goal with progress information
            goal.progress_info = progress
            
            # Trigger adaptations based on progress
            if progress.completion_rate > 0.9:
                self.prepare_goal_completion(goal)
            
            elif progress.stalled and progress.time_elapsed > goal.patience_threshold:
                self.handle_stalled_goal(goal)
            
            elif progress.ahead_of_schedule:
                self.consider_goal_enhancement(goal)

# Example usage
advanced_goal_config = {
    "max_active_goals": 8,
    "decomposition_enabled": True,
    "conflict_resolution": "sophisticated",
    "adaptive_management": True,
    "progress_tracking": "detailed"
}

advanced_goal_manager = AdvancedGoalManager(advanced_goal_config)

# Add complex goal that will be decomposed
complex_research_goal = Goal(
    goal_id="develop_agi_system",
    description="Develop a working artificial general intelligence system",
    goal_type=GoalType.ACHIEVEMENT,
    priority=1.0,
    complexity=0.95,
    estimated_duration=365 * 24 * 3600,  # 1 year in seconds
    success_criteria=[
        "design_cognitive_architecture",
        "implement_learning_systems",
        "validate_general_intelligence",
        "demonstrate_real_world_capabilities"
    ]
)

subgoals = advanced_goal_manager.process_complex_goal(complex_research_goal)
print(f"Complex goal decomposed into {len(subgoals)} subgoals")
```

## Reasoning-Goal Integration

### Goal-Directed Reasoning

```python
class GoalDirectedReasoning:
    def __init__(self, reasoning_engine, goal_manager):
        self.reasoning_engine = reasoning_engine
        self.goal_manager = goal_manager
        self.reasoning_goal_cache = {}
    
    def reason_toward_goal(self, goal, available_facts, context=None):
        """Perform reasoning specifically directed toward achieving a goal"""
        
        # Create reasoning objective from goal
        reasoning_objective = self.goal_to_reasoning_objective(goal)
        
        # Filter facts relevant to goal
        relevant_facts = self.filter_goal_relevant_facts(goal, available_facts)
        
        # Add goal-specific reasoning rules
        goal_specific_rules = self.generate_goal_specific_rules(goal)
        
        # Perform goal-directed inference
        reasoning_result = self.reasoning_engine.infer(
            objective=reasoning_objective,
            facts=relevant_facts,
            additional_rules=goal_specific_rules,
            context=context
        )
        
        # Evaluate reasoning contribution to goal
        goal_contribution = self.evaluate_goal_contribution(reasoning_result, goal)
        
        # Update goal progress based on reasoning results
        if goal_contribution.positive_contribution:
            self.goal_manager.update_goal_progress(
                goal.goal_id, 
                progress_delta=goal_contribution.progress_amount
            )
        
        return reasoning_result
    
    def goal_to_reasoning_objective(self, goal):
        """Convert goal into reasoning objective"""
        
        objective = ReasoningObjective(
            target_conclusions=goal.success_criteria,
            confidence_threshold=0.6,
            reasoning_type="goal_achievement",
            context={"goal_id": goal.goal_id, "goal_type": goal.goal_type}
        )
        
        return objective
    
    def generate_goal_specific_rules(self, goal):
        """Generate reasoning rules specific to goal achievement"""
        
        goal_rules = []
        
        # Rules based on goal type
        if goal.goal_type == GoalType.ACHIEVEMENT:
            goal_rules.extend(self.generate_achievement_rules(goal))
        elif goal.goal_type == GoalType.MAINTENANCE:
            goal_rules.extend(self.generate_maintenance_rules(goal))
        elif goal.goal_type == GoalType.AVOIDANCE:
            goal_rules.extend(self.generate_avoidance_rules(goal))
        
        # Rules based on goal domain
        domain_rules = self.generate_domain_specific_rules(goal.domain)
        goal_rules.extend(domain_rules)
        
        return goal_rules
    
    def evaluate_goal_contribution(self, reasoning_result, goal):
        """Evaluate how reasoning results contribute to goal achievement"""
        
        contribution = GoalContribution()
        
        # Check if reasoning conclusions match goal criteria
        matching_criteria = 0
        for criterion in goal.success_criteria:
            if any(conclusion.matches(criterion) for conclusion in reasoning_result.conclusions):
                matching_criteria += 1
        
        # Calculate contribution metrics
        contribution.criterion_satisfaction = matching_criteria / len(goal.success_criteria)
        contribution.confidence_boost = reasoning_result.overall_confidence
        contribution.progress_amount = contribution.criterion_satisfaction * 0.1
        contribution.positive_contribution = contribution.progress_amount > 0.05
        
        return contribution

# Example goal-directed reasoning
goal_directed = GoalDirectedReasoning(reasoning_engine, goal_manager)

# Perform reasoning toward specific goal
research_goal = goal_manager.get_goal("learn_ml_fundamentals")
current_facts = agent.memory_manager.get_relevant_facts(research_goal.description)

reasoning_result = goal_directed.reason_toward_goal(
    goal=research_goal,
    available_facts=current_facts,
    context={"learning_phase": "fundamentals"}
)

print(f"Reasoning toward goal completed with confidence: {reasoning_result.overall_confidence:.2f}")
```

### Reasoning About Goals

```python
class MetaGoalReasoning:
    def __init__(self, reasoning_engine):
        self.reasoning_engine = reasoning_engine
        self.goal_reasoning_rules = self.create_goal_reasoning_rules()
    
    def reason_about_goal_priorities(self, goals, context):
        """Reason about which goals should have higher priority"""
        
        # Create facts about current goals
        goal_facts = []
        for goal in goals:
            goal_facts.extend(self.goal_to_facts(goal))
        
        # Add context facts
        context_facts = self.context_to_facts(context)
        
        # Reason about priorities
        priority_reasoning = self.reasoning_engine.infer(
            objective=ReasoningObjective(
                target_conclusions=["optimal_goal_priority(?goal, ?priority)"],
                reasoning_type="goal_prioritization"
            ),
            facts=goal_facts + context_facts,
            additional_rules=self.goal_reasoning_rules
        )
        
        # Extract priority recommendations
        priority_recommendations = self.extract_priority_recommendations(
            priority_reasoning.conclusions
        )
        
        return priority_recommendations
    
    def reason_about_goal_conflicts(self, conflicting_goals):
        """Reason about how to resolve goal conflicts"""
        
        # Model conflict situation
        conflict_facts = []
        for i, goal1 in enumerate(conflicting_goals):
            for goal2 in conflicting_goals[i+1:]:
                conflict_type = self.analyze_conflict_type(goal1, goal2)
                conflict_facts.append(
                    Fact("goal_conflict", [goal1.goal_id, goal2.goal_id, conflict_type])
                )
        
        # Reason about resolution strategies
        resolution_reasoning = self.reasoning_engine.infer(
            objective=ReasoningObjective(
                target_conclusions=["resolve_conflict(?goal1, ?goal2, ?strategy)"],
                reasoning_type="conflict_resolution"
            ),
            facts=conflict_facts,
            additional_rules=self.conflict_resolution_rules
        )
        
        return self.extract_resolution_strategies(resolution_reasoning.conclusions)
    
    def reason_about_goal_achievement_strategies(self, goal, available_resources):
        """Reason about strategies for achieving a specific goal"""
        
        # Model goal achievement problem
        goal_facts = self.goal_to_facts(goal)
        resource_facts = self.resources_to_facts(available_resources)
        
        # Add strategy knowledge
        strategy_facts = self.get_strategy_knowledge(goal.domain)
        
        # Reason about strategies
        strategy_reasoning = self.reasoning_engine.infer(
            objective=ReasoningObjective(
                target_conclusions=["effective_strategy(?goal, ?strategy, ?effectiveness)"],
                reasoning_type="strategy_selection"
            ),
            facts=goal_facts + resource_facts + strategy_facts,
            additional_rules=self.strategy_reasoning_rules
        )
        
        return self.extract_strategy_recommendations(strategy_reasoning.conclusions)
    
    def create_goal_reasoning_rules(self):
        """Create rules for reasoning about goals"""
        
        rules = [
            # Priority rules
            Rule(
                conditions=[
                    Fact("goal", ["?g"]),
                    Fact("deadline_approaching", ["?g"]),
                    Fact("high_importance", ["?g"])
                ],
                conclusion=Fact("high_priority", ["?g"]),
                confidence=0.9,
                name="urgent_important_priority"
            ),
            
            # Resource allocation rules
            Rule(
                conditions=[
                    Fact("goal", ["?g"]),
                    Fact("resource_intensive", ["?g"]),
                    Fact("limited_resources", [])
                ],
                conclusion=Fact("lower_priority", ["?g"]),
                confidence=0.7,
                name="resource_constraint_priority"
            ),
            
            # Dependency rules
            Rule(
                conditions=[
                    Fact("goal", ["?g1"]),
                    Fact("goal", ["?g2"]),
                    Fact("depends_on", ["?g1", "?g2"]),
                    Fact("not_achieved", ["?g2"])
                ],
                conclusion=Fact("blocked", ["?g1"]),
                confidence=0.95,
                name="dependency_blocking"
            )
        ]
        
        return rules

# Example meta-goal reasoning
meta_goal_reasoning = MetaGoalReasoning(reasoning_engine)

# Reason about goal priorities
current_goals = goal_manager.get_active_goals()
current_context = {
    "time_pressure": 0.7,
    "available_resources": ["computational", "human"],
    "external_deadlines": ["conference_submission"],
    "recent_progress": 0.6
}

priority_recommendations = meta_goal_reasoning.reason_about_goal_priorities(
    current_goals, 
    current_context
)

print("🎯 Goal Priority Recommendations:")
for rec in priority_recommendations:
    print(f"  {rec.goal_id}: {rec.recommended_priority:.2f} (reason: {rec.rationale})")
```

## Advanced Reasoning Techniques

### Analogical Reasoning

```python
class AnalogicalReasoning:
    def __init__(self, memory_manager, reasoning_engine):
        self.memory_manager = memory_manager
        self.reasoning_engine = reasoning_engine
        self.analogy_cache = AnalogicalCache()
    
    def find_analogous_situations(self, current_situation, similarity_threshold=0.7):
        """Find analogous situations from memory"""
        
        # Search episodic memory for similar situations
        similar_episodes = self.memory_manager.episodic_memory.find_similar_episodes(
            current_situation,
            similarity_threshold=similarity_threshold
        )
        
        # Extract structural similarities
        analogies = []
        for episode in similar_episodes:
            analogy = self.extract_structural_analogy(current_situation, episode)
            if analogy.structural_similarity > similarity_threshold:
                analogies.append(analogy)
        
        return sorted(analogies, key=lambda x: x.overall_similarity, reverse=True)
    
    def reason_by_analogy(self, current_problem, analogous_cases):
        """Reason about current problem using analogous cases"""
        
        reasoning_results = []
        
        for analogy in analogous_cases:
            # Map solution from analogous case to current problem
            mapped_solution = self.map_solution(
                analogy.source_solution,
                analogy.mapping,
                current_problem
            )
            
            # Evaluate mapped solution validity
            validity_assessment = self.assess_analogy_validity(
                analogy,
                current_problem,
                mapped_solution
            )
            
            # Create reasoning result
            reasoning_result = ReasoningResult(
                conclusion=mapped_solution,
                confidence=validity_assessment.confidence,
                reasoning_type="analogical",
                source_analogy=analogy,
                validity_factors=validity_assessment.factors
            )
            
            reasoning_results.append(reasoning_result)
        
        return reasoning_results
    
    def learn_from_analogical_reasoning(self, analogy_result, actual_outcome):
        """Learn from analogical reasoning experience"""
        
        # Update analogy effectiveness
        analogy = analogy_result.source_analogy
        success = self.evaluate_reasoning_success(analogy_result, actual_outcome)
        
        # Update analogy cache with feedback
        self.analogy_cache.update_analogy_effectiveness(
            analogy.analogy_id,
            success_score=success.score,
            feedback=success.feedback
        )
        
        # Learn new analogical patterns
        if success.score > 0.8:
            pattern = self.extract_successful_pattern(analogy, analogy_result)
            self.analogy_cache.add_successful_pattern(pattern)

# Example analogical reasoning
analogical_reasoner = AnalogicalReasoning(memory_manager, reasoning_engine)

# Current research problem
current_problem = {
    "type": "optimization_challenge",
    "domain": "neural_architecture_search",
    "constraints": ["computational_budget", "accuracy_target"],
    "resources": ["dataset", "computing_cluster"],
    "goal": "find_optimal_architecture"
}

# Find analogous situations
analogies = analogical_reasoner.find_analogous_situations(current_problem)

# Reason by analogy
analogical_solutions = analogical_reasoner.reason_by_analogy(current_problem, analogies)

print(f"Found {len(analogical_solutions)} analogical solutions:")
for i, solution in enumerate(analogical_solutions[:3]):
    print(f"  {i+1}. {solution.conclusion} (confidence: {solution.confidence:.2f})")
```

### Causal Reasoning

```python
class CausalReasoning:
    def __init__(self, reasoning_engine, memory_manager):
        self.reasoning_engine = reasoning_engine
        self.memory_manager = memory_manager
        self.causal_model = CausalModel()
    
    def infer_causal_relationships(self, observations):
        """Infer causal relationships from observations"""
        
        # Build causal hypotheses
        causal_hypotheses = self.generate_causal_hypotheses(observations)
        
        # Test hypotheses using available data
        tested_hypotheses = []
        for hypothesis in causal_hypotheses:
            test_result = self.test_causal_hypothesis(hypothesis, observations)
            tested_hypotheses.append({
                "hypothesis": hypothesis,
                "evidence": test_result.evidence,
                "confidence": test_result.confidence,
                "strength": test_result.causal_strength
            })
        
        # Build causal model from validated hypotheses
        validated_hypotheses = [
            h for h in tested_hypotheses 
            if h["confidence"] > 0.6
        ]
        
        causal_network = self.build_causal_network(validated_hypotheses)
        self.causal_model.update_network(causal_network)
        
        return causal_network
    
    def reason_about_interventions(self, desired_outcome, causal_network):
        """Reason about interventions to achieve desired outcomes"""
        
        # Find causal paths to desired outcome
        causal_paths = causal_network.find_paths_to_outcome(desired_outcome)
        
        # Evaluate intervention points
        intervention_options = []
        for path in causal_paths:
            for node in path.nodes:
                if node.interventable:
                    intervention = self.evaluate_intervention(node, desired_outcome, path)
                    intervention_options.append(intervention)
        
        # Rank interventions by effectiveness and feasibility
        ranked_interventions = sorted(
            intervention_options,
            key=lambda x: x.expected_effectiveness * x.feasibility,
            reverse=True
        )
        
        return ranked_interventions
    
    def counterfactual_reasoning(self, scenario, alternative_conditions):
        """Perform counterfactual reasoning about alternative scenarios"""
        
        # Create counterfactual scenario
        counterfactual_scenario = self.create_counterfactual(scenario, alternative_conditions)
        
        # Reason about likely outcomes under alternative conditions
        counterfactual_outcomes = self.causal_model.predict_outcomes(counterfactual_scenario)
        
        # Compare with actual scenario
        comparison = self.compare_scenarios(scenario, counterfactual_scenario)
        
        return {
            "counterfactual_outcomes": counterfactual_outcomes,
            "scenario_comparison": comparison,
            "insights": self.extract_counterfactual_insights(comparison)
        }

# Example causal reasoning
causal_reasoner = CausalReasoning(reasoning_engine, memory_manager)

# Research productivity observations
productivity_observations = [
    {"factor": "sleep_hours", "value": 7, "productivity": 0.8},
    {"factor": "sleep_hours", "value": 5, "productivity": 0.4},
    {"factor": "collaboration_frequency", "value": 0.6, "productivity": 0.9},
    {"factor": "interruptions_per_hour", "value": 3, "productivity": 0.3},
    {"factor": "coffee_consumption", "value": 2, "productivity": 0.7}
]

# Infer causal relationships
causal_network = causal_reasoner.infer_causal_relationships(productivity_observations)

# Reason about interventions to improve productivity
interventions = causal_reasoner.reason_about_interventions(
    desired_outcome={"productivity": 0.9},
    causal_network=causal_network
)

print("🔬 Causal Intervention Recommendations:")
for intervention in interventions[:3]:
    print(f"  • {intervention.description}")
    print(f"    Expected effect: {intervention.expected_effectiveness:.2f}")
    print(f"    Feasibility: {intervention.feasibility:.2f}")
```

## Integration with Agent Architecture

### Complete Reasoning-Goal Integration

```python
def integrate_reasoning_goals_memory(agent):
    """Complete integration of reasoning, goals, and memory systems"""
    
    # Configure goal-directed reasoning
    goal_directed_reasoning = GoalDirectedReasoning(
        agent.reasoning_engine,
        agent.goal_manager
    )
    
    # Configure meta-goal reasoning
    meta_goal_reasoning = MetaGoalReasoning(agent.reasoning_engine)
    
    # Configure memory-guided reasoning
    agent.reasoning_engine.set_memory_interface(agent.memory_manager)
    agent.reasoning_engine.enable_memory_guided_inference()
    
    # Set up reasoning-memory feedback loops
    agent.reasoning_engine.set_memory_update_callbacks([
        lambda result: agent.memory_manager.store_reasoning_episode(result),
        lambda result: agent.memory_manager.update_concept_activations(result),
        lambda result: agent.memory_manager.learn_from_reasoning_patterns(result)
    ])
    
    # Configure goal adaptation based on reasoning
    agent.goal_manager.set_reasoning_interface(agent.reasoning_engine)
    agent.goal_manager.enable_reasoning_based_adaptation()
    
    # Create integrated cognitive cycle
    def integrated_cognitive_cycle():
        # 1. Update goal priorities based on current context
        current_context = agent.get_current_context()
        priority_updates = meta_goal_reasoning.reason_about_goal_priorities(
            agent.goal_manager.get_active_goals(),
            current_context
        )
        agent.goal_manager.apply_priority_updates(priority_updates)
        
        # 2. Select highest priority goal
        current_goal = agent.goal_manager.get_highest_priority_goal()
        
        if current_goal:
            # 3. Retrieve relevant knowledge from memory
            relevant_facts = agent.memory_manager.retrieve_goal_relevant_facts(current_goal)
            
            # 4. Perform goal-directed reasoning
            reasoning_result = goal_directed_reasoning.reason_toward_goal(
                goal=current_goal,
                available_facts=relevant_facts,
                context=current_context
            )
            
            # 5. Update memories based on reasoning
            agent.memory_manager.consolidate_reasoning_results(reasoning_result)
            
            # 6. Update goal progress
            agent.goal_manager.update_goal_progress_from_reasoning(
                current_goal.goal_id,
                reasoning_result
            )
            
            # 7. Generate actions based on reasoning conclusions
            actions = agent.action_generator.generate_actions_from_reasoning(
                reasoning_result,
                current_goal
            )
            
            return actions
        
        return []
    
    # Set the integrated cycle as the agent's main cognitive cycle
    agent.set_cognitive_cycle(integrated_cognitive_cycle)

# Apply integration to agent
integrate_reasoning_goals_memory(agent)
```

## Best Practices

### 1. Reasoning Configuration

- **Match strategy to problem type**: Use appropriate reasoning strategies for different problem types
- **Set realistic time limits**: Balance thoroughness with computational constraints
- **Enable uncertainty handling**: Real-world problems involve uncertainty
- **Use confidence thresholds**: Filter out low-confidence conclusions

### 2. Goal Management

- **Clear success criteria**: Define measurable goal achievement criteria
- **Appropriate decomposition**: Break complex goals into manageable subgoals
- **Regular priority updates**: Adapt goal priorities based on changing context
- **Conflict resolution**: Handle goal conflicts systematically

### 3. Integration

- **Memory-guided reasoning**: Use past experiences to guide current reasoning
- **Goal-directed reasoning**: Focus reasoning efforts on goal achievement
- **Feedback loops**: Create learning loops between reasoning, goals, and memory
- **Meta-reasoning**: Reason about reasoning strategies and goal priorities

### 4. Performance Optimization

- **Cache reasoning results**: Avoid re-solving similar problems
- **Limit reasoning depth**: Prevent infinite reasoning loops
- **Parallel processing**: Use parallel reasoning when appropriate
- **Strategy adaptation**: Learn which strategies work best for different problems

---

The reasoning and goal systems work together to create intelligent, purposeful behavior in cognitive agents. By properly configuring and integrating these systems, you can create agents that exhibit sophisticated problem-solving and goal-directed behavior.

**Next**: Explore [CLI Usage](cli-usage.md) for command-line tools to manage reasoning and goals, or see [API Reference](../api/reasoning.md) for detailed technical documentation.
