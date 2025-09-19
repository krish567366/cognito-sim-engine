# Environment Setup

Environments in Cognito Simulation Engine provide the context and world in which cognitive agents operate. This guide covers how to create, configure, and manage rich simulation environments.

## Quick Start: Creating Your First Environment

```python
from cognito_sim_engine import Environment, CognitiveAgent

# Create a basic research environment
env = Environment(
    environment_id="research_lab",
    environment_type="collaborative_workspace",
    physical_properties={
        "space_size": "large",
        "layout": "open_office", 
        "resources": ["computers", "whiteboards", "meeting_rooms"],
        "noise_level": 0.3
    },
    temporal_properties={
        "time_scale": "real_time",
        "work_hours": "09:00-17:00",
        "timezone": "UTC"
    }
)

# Add an agent to the environment
agent = CognitiveAgent("researcher_alice")
env.add_agent(agent)

# Start the environment
env.start()

print(f"Environment '{env.environment_id}' is running")
print(f"Agents in environment: {[a.agent_id for a in env.agents]}")
```

## Environment Types

### 1. Collaborative Workspace

Best for: Team research, group problem-solving, social interaction studies

```python
from cognito_sim_engine import CollaborativeEnvironment

# Create collaborative research environment
research_env = CollaborativeEnvironment(
    environment_id="ai_research_lab",
    workspace_config={
        "shared_resources": [
            "research_database",
            "computation_cluster", 
            "visualization_tools",
            "meeting_spaces"
        ],
        "communication_channels": [
            "direct_message",
            "group_chat",
            "video_calls", 
            "whiteboard_sessions"
        ],
        "collaboration_tools": [
            "shared_documents",
            "version_control",
            "task_boards",
            "peer_review_system"
        ]
    }
)

# Configure team dynamics
research_env.set_team_dynamics({
    "hierarchy_level": 0.3,        # Relatively flat structure
    "communication_openness": 0.8,  # Open communication
    "knowledge_sharing": 0.9,       # High knowledge sharing
    "competition_level": 0.2        # Low internal competition
})
```

### 2. Learning Environment

Best for: Educational simulations, skill development, adaptive learning

```python
from cognito_sim_engine import LearningEnvironment

# Create adaptive learning environment
learning_env = LearningEnvironment(
    environment_id="ml_bootcamp",
    curriculum_config={
        "learning_objectives": [
            "understand_ml_fundamentals",
            "implement_algorithms",
            "evaluate_models",
            "apply_to_real_problems"
        ],
        "difficulty_progression": "adaptive",
        "feedback_frequency": "immediate",
        "assessment_methods": ["quiz", "project", "peer_review"]
    }
)

# Configure learning progression
learning_env.set_progression_rules({
    "prerequisite_enforcement": True,
    "mastery_threshold": 0.8,
    "retry_allowed": True,
    "hint_system": True,
    "collaborative_learning": True
})

# Add learning materials
learning_materials = [
    {
        "id": "ml_basics",
        "type": "interactive_tutorial",
        "difficulty": 0.3,
        "estimated_time": 120,  # minutes
        "prerequisites": []
    },
    {
        "id": "supervised_learning",
        "type": "hands_on_exercise", 
        "difficulty": 0.5,
        "estimated_time": 180,
        "prerequisites": ["ml_basics"]
    },
    {
        "id": "deep_learning",
        "type": "project",
        "difficulty": 0.8,
        "estimated_time": 480,
        "prerequisites": ["supervised_learning"]
    }
]

for material in learning_materials:
    learning_env.add_learning_material(material)
```

### 3. Problem-Solving Environment

Best for: Research challenges, complex problem solving, innovation studies

```python
from cognito_sim_engine import ProblemSolvingEnvironment

# Create challenging problem environment
problem_env = ProblemSolvingEnvironment(
    environment_id="agi_challenge",
    problem_config={
        "domain": "artificial_general_intelligence",
        "complexity_level": 0.9,
        "solution_space": "open_ended",
        "evaluation_criteria": [
            "novelty",
            "feasibility", 
            "impact_potential",
            "theoretical_soundness"
        ]
    }
)

# Define the core problem
agi_problem = {
    "title": "Develop human-level reasoning system",
    "description": """
    Create a cognitive architecture that can:
    1. Learn from few examples like humans
    2. Transfer knowledge across domains
    3. Reason about novel situations
    4. Explain its decision-making process
    """,
    "constraints": [
        "Computationally feasible",
        "Interpretable outputs", 
        "Safe and controllable",
        "Builds on existing research"
    ],
    "success_metrics": [
        "Performance on cognitive benchmarks",
        "Generalization capability",
        "Learning efficiency",
        "Explanation quality"
    ]
}

problem_env.set_core_problem(agi_problem)

# Add problem-solving resources
problem_env.add_resources([
    "research_literature_database",
    "computational_resources",
    "experimental_datasets",
    "evaluation_frameworks",
    "expert_knowledge_base"
])
```

### 4. Social Simulation Environment

Best for: Social dynamics, communication studies, group behavior research

```python
from cognito_sim_engine import SocialEnvironment

# Create social simulation environment
social_env = SocialEnvironment(
    environment_id="academic_conference",
    social_config={
        "social_structure": "network",
        "interaction_patterns": [
            "formal_presentations",
            "informal_discussions",
            "networking_events",
            "collaborative_sessions"
        ],
        "social_norms": {
            "respect_speaking_time": 0.9,
            "acknowledge_contributions": 0.8,
            "share_knowledge_openly": 0.7,
            "support_junior_researchers": 0.8
        }
    }
)

# Configure social dynamics
social_env.configure_dynamics({
    "group_formation": "interest_based",
    "influence_propagation": True,
    "reputation_system": True,
    "social_learning": True,
    "conflict_resolution": "mediated"
})

# Add social events
conference_events = [
    {
        "name": "keynote_presentation",
        "duration": 60,
        "participants": "all",
        "interaction_type": "broadcast"
    },
    {
        "name": "poster_session", 
        "duration": 120,
        "participants": "voluntary",
        "interaction_type": "small_groups"
    },
    {
        "name": "panel_discussion",
        "duration": 90,
        "participants": "selected_panelists_plus_audience",
        "interaction_type": "moderated_discussion"
    }
]

for event in conference_events:
    social_env.schedule_event(event)
```

## Environment Configuration

### Physical Properties

Configure the physical aspects of the environment:

```python
def configure_physical_environment():
    """Configure detailed physical environment properties"""
    
    physical_config = {
        # Spatial properties
        "dimensions": {
            "length": 100,  # meters
            "width": 80,
            "height": 4
        },
        "layout": {
            "type": "open_office_with_private_spaces",
            "work_areas": 20,
            "meeting_rooms": 5,
            "common_areas": 3,
            "quiet_zones": 2
        },
        
        # Environmental conditions
        "lighting": {
            "natural_light": 0.7,
            "artificial_light": 0.3,
            "adjustable": True
        },
        "acoustics": {
            "base_noise_level": 0.3,
            "reverberation": 0.2,
            "sound_isolation": 0.6
        },
        "climate": {
            "temperature": 22,  # Celsius
            "humidity": 0.45,
            "air_quality": 0.9
        },
        
        # Resources and tools
        "computing_resources": {
            "workstations": 25,
            "high_performance_cluster": 1,
            "cloud_access": True,
            "software_licenses": ["research_tools", "analysis_software"]
        },
        "physical_tools": [
            "whiteboards",
            "projection_systems", 
            "3d_printers",
            "laboratory_equipment"
        ],
        "information_resources": [
            "digital_library",
            "research_databases",
            "archive_systems"
        ]
    }
    
    return physical_config

# Apply physical configuration
env = Environment("advanced_research_facility")
env.configure_physical_properties(configure_physical_environment())
```

### Temporal Properties

Configure time and scheduling:

```python
from cognito_sim_engine import TemporalConfig, TimeScale

def configure_temporal_environment():
    """Configure time-related environment properties"""
    
    temporal_config = TemporalConfig(
        # Time scale settings
        time_scale=TimeScale.ACCELERATED,  # Faster than real-time
        acceleration_factor=10,  # 10x speed
        
        # Work schedule
        work_schedule={
            "monday": {"start": "09:00", "end": "17:00"},
            "tuesday": {"start": "09:00", "end": "17:00"},
            "wednesday": {"start": "09:00", "end": "17:00"},
            "thursday": {"start": "09:00", "end": "17:00"},
            "friday": {"start": "09:00", "end": "15:00"},  # Half day Friday
            "saturday": "optional",
            "sunday": "off"
        },
        
        # Special time periods
        special_periods=[
            {
                "name": "conference_week",
                "start": "2024-03-15",
                "end": "2024-03-22", 
                "modifications": {
                    "extended_hours": True,
                    "increased_collaboration": 0.3,
                    "external_visitors": True
                }
            },
            {
                "name": "paper_deadline",
                "start": "2024-06-01",
                "end": "2024-06-15",
                "modifications": {
                    "work_intensity": 1.5,
                    "meeting_frequency": 0.5,  # Fewer meetings
                    "focus_mode": True
                }
            }
        ],
        
        # Rhythm and cycles
        daily_rhythms={
            "peak_productivity": ["10:00-12:00", "14:00-16:00"],
            "collaborative_time": ["13:00-14:00", "16:00-17:00"],
            "quiet_time": ["08:00-09:00", "12:00-13:00"]
        },
        
        # Event scheduling
        recurring_events=[
            {
                "name": "team_standup",
                "frequency": "daily",
                "time": "09:15",
                "duration": 15,
                "participants": "team_members"
            },
            {
                "name": "research_seminar",
                "frequency": "weekly", 
                "day": "friday",
                "time": "15:00",
                "duration": 60,
                "participants": "all_researchers"
            }
        ]
    )
    
    return temporal_config

# Apply temporal configuration
temporal_settings = configure_temporal_environment()
env.configure_temporal_properties(temporal_settings)
```

### Information Environment

Configure information flow and knowledge availability:

```python
from cognito_sim_engine import InformationEnvironment

def setup_information_environment():
    """Setup rich information environment"""
    
    info_env = InformationEnvironment(
        # Knowledge bases
        knowledge_bases=[
            {
                "name": "research_literature",
                "type": "academic_papers",
                "size": 1000000,  # 1M papers
                "update_frequency": "daily",
                "access_method": "search_and_browse",
                "quality_score": 0.85
            },
            {
                "name": "experimental_data", 
                "type": "datasets",
                "size": 50000,  # 50K datasets
                "update_frequency": "weekly",
                "access_method": "query_based",
                "quality_score": 0.9
            },
            {
                "name": "code_repositories",
                "type": "source_code",
                "size": 100000,  # 100K repos
                "update_frequency": "continuous",
                "access_method": "version_control",
                "quality_score": 0.7
            }
        ],
        
        # Information flow patterns
        information_flow={
            "formal_channels": [
                "research_presentations",
                "published_papers",
                "technical_reports",
                "official_announcements"
            ],
            "informal_channels": [
                "hallway_conversations",
                "coffee_break_discussions",
                "lunch_meetings", 
                "social_media_interactions"
            ],
            "collaborative_channels": [
                "shared_workspaces",
                "version_control_systems",
                "collaborative_documents",
                "peer_review_platforms"
            ]
        },
        
        # Information quality and filtering
        quality_control={
            "peer_review": True,
            "fact_checking": 0.8,
            "source_credibility": 0.9,
            "information_freshness": 0.7,
            "relevance_filtering": 0.8
        },
        
        # Access permissions and restrictions
        access_control={
            "public_information": 0.6,    # 60% publicly accessible
            "institutional_access": 0.3,  # 30% requires institutional access
            "restricted_access": 0.1      # 10% highly restricted
        }
    )
    
    return info_env

# Setup information environment
info_env = setup_information_environment()
env.integrate_information_environment(info_env)
```

## Dynamic Environment Features

### Adaptive Environmental Changes

Create environments that evolve based on agent behavior:

```python
class AdaptiveEnvironment:
    def __init__(self, base_environment):
        self.base_env = base_environment
        self.adaptation_rules = []
        self.environmental_state = {}
        self.change_history = []
    
    def add_adaptation_rule(self, trigger, change_function, name):
        """Add rule for environmental adaptation"""
        
        rule = {
            "name": name,
            "trigger": trigger,
            "change_function": change_function,
            "activation_count": 0
        }
        self.adaptation_rules.append(rule)
    
    def monitor_and_adapt(self):
        """Monitor agent behavior and adapt environment"""
        
        # Collect behavioral data
        agent_behaviors = self.collect_agent_behaviors()
        
        # Check adaptation triggers
        for rule in self.adaptation_rules:
            if rule["trigger"](agent_behaviors, self.environmental_state):
                # Apply environmental change
                changes = rule["change_function"](agent_behaviors, self.environmental_state)
                self.apply_changes(changes)
                
                # Record adaptation
                rule["activation_count"] += 1
                self.change_history.append({
                    "rule": rule["name"],
                    "timestamp": time.time(),
                    "changes": changes,
                    "trigger_data": agent_behaviors
                })
    
    def collect_agent_behaviors(self):
        """Collect aggregated agent behavior data"""
        
        behaviors = {
            "collaboration_frequency": 0,
            "information_seeking": 0,
            "problem_solving_attempts": 0,
            "knowledge_sharing": 0,
            "stress_levels": [],
            "productivity_metrics": [],
            "social_interactions": 0
        }
        
        for agent in self.base_env.agents:
            # Aggregate behavioral metrics
            behaviors["collaboration_frequency"] += agent.get_collaboration_frequency()
            behaviors["information_seeking"] += agent.get_information_seeking_rate()
            behaviors["problem_solving_attempts"] += agent.get_problem_solving_attempts()
            behaviors["knowledge_sharing"] += agent.get_knowledge_sharing_frequency()
            behaviors["stress_levels"].append(agent.get_stress_level())
            behaviors["productivity_metrics"].append(agent.get_productivity_score())
            behaviors["social_interactions"] += agent.get_social_interaction_count()
        
        # Calculate averages
        num_agents = len(self.base_env.agents)
        if num_agents > 0:
            behaviors["avg_stress"] = np.mean(behaviors["stress_levels"])
            behaviors["avg_productivity"] = np.mean(behaviors["productivity_metrics"])
            behaviors["collaboration_frequency"] /= num_agents
            behaviors["information_seeking"] /= num_agents
        
        return behaviors

# Example adaptation rules
def setup_adaptive_rules(adaptive_env):
    """Setup common environmental adaptation rules"""
    
    # Rule 1: Reduce noise when stress levels are high
    def high_stress_trigger(behaviors, env_state):
        return behaviors.get("avg_stress", 0) > 0.7
    
    def reduce_noise_change(behaviors, env_state):
        return {
            "acoustics.base_noise_level": max(0.1, env_state.get("acoustics.base_noise_level", 0.3) - 0.1),
            "lighting.natural_light": min(1.0, env_state.get("lighting.natural_light", 0.7) + 0.1)
        }
    
    adaptive_env.add_adaptation_rule(
        high_stress_trigger,
        reduce_noise_change,
        "stress_reduction"
    )
    
    # Rule 2: Increase collaboration spaces when collaboration is high
    def high_collaboration_trigger(behaviors, env_state):
        return behaviors.get("collaboration_frequency", 0) > 0.8
    
    def expand_collaboration_change(behaviors, env_state):
        return {
            "layout.meeting_rooms": env_state.get("layout.meeting_rooms", 5) + 1,
            "layout.common_areas": env_state.get("layout.common_areas", 3) + 1
        }
    
    adaptive_env.add_adaptation_rule(
        high_collaboration_trigger,
        expand_collaboration_change,
        "collaboration_expansion"
    )
    
    # Rule 3: Adjust information access based on seeking behavior
    def high_info_seeking_trigger(behaviors, env_state):
        return behaviors.get("information_seeking", 0) > 0.9
    
    def improve_info_access_change(behaviors, env_state):
        return {
            "information_resources.search_speed": 1.2,  # 20% faster
            "information_resources.access_broadness": min(1.0, 
                env_state.get("information_resources.access_broadness", 0.6) + 0.1)
        }
    
    adaptive_env.add_adaptation_rule(
        high_info_seeking_trigger,
        improve_info_access_change,
        "information_access_improvement"
    )

# Create adaptive environment
adaptive_env = AdaptiveEnvironment(env)
setup_adaptive_rules(adaptive_env)

# Run adaptive monitoring
def run_adaptive_simulation(adaptive_env, duration_steps=1000):
    """Run simulation with environmental adaptation"""
    
    for step in range(duration_steps):
        # Normal environment step
        adaptive_env.base_env.step()
        
        # Check for adaptations every 50 steps
        if step % 50 == 0:
            adaptive_env.monitor_and_adapt()
            
            # Log changes if any occurred
            if adaptive_env.change_history:
                latest_change = adaptive_env.change_history[-1]
                if latest_change["timestamp"] > time.time() - 60:  # Recent change
                    print(f"🔄 Environmental adaptation: {latest_change['rule']}")
                    for change, value in latest_change["changes"].items():
                        print(f"    {change}: {value}")

# Run adaptive simulation
run_adaptive_simulation(adaptive_env)
```

### Event-Driven Environment

Create environments with dynamic events:

```python
from cognito_sim_engine import EventDrivenEnvironment, EnvironmentalEvent

class EventDrivenEnvironment:
    def __init__(self, base_environment):
        self.base_env = base_environment
        self.event_queue = []
        self.event_handlers = {}
        self.active_events = {}
    
    def schedule_event(self, event, trigger_time):
        """Schedule an environmental event"""
        
        scheduled_event = {
            "event": event,
            "trigger_time": trigger_time,
            "scheduled_time": time.time()
        }
        self.event_queue.append(scheduled_event)
        self.event_queue.sort(key=lambda x: x["trigger_time"])
    
    def register_event_handler(self, event_type, handler_function):
        """Register handler for specific event types"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler_function)
    
    def process_events(self, current_time):
        """Process any events that should trigger now"""
        
        triggered_events = []
        
        # Check for events to trigger
        while self.event_queue and self.event_queue[0]["trigger_time"] <= current_time:
            event_data = self.event_queue.pop(0)
            event = event_data["event"]
            
            # Trigger event
            self.trigger_event(event)
            triggered_events.append(event)
        
        return triggered_events
    
    def trigger_event(self, event):
        """Trigger an environmental event"""
        
        # Call registered handlers
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                handler(event, self.base_env)
        
        # Add to active events if it has duration
        if event.duration > 0:
            end_time = time.time() + event.duration
            self.active_events[event.event_id] = {
                "event": event,
                "start_time": time.time(),
                "end_time": end_time
            }
        
        # Notify agents
        for agent in self.base_env.agents:
            agent.perceive_environmental_event(event)

# Define environmental events
conference_event = EnvironmentalEvent(
    event_id="ai_conference_2024",
    event_type="external_conference",
    description="Major AI conference brings external visitors and ideas",
    duration=7 * 24 * 3600,  # 7 days in seconds
    effects={
        "external_visitors": 50,
        "knowledge_influx": 0.8,
        "networking_opportunities": 0.9,
        "distraction_level": 0.3
    }
)

equipment_failure_event = EnvironmentalEvent(
    event_id="server_maintenance",
    event_type="resource_disruption",
    description="Server maintenance reduces computational resources",
    duration=6 * 3600,  # 6 hours
    effects={
        "computing_resources_available": 0.3,  # Only 30% available
        "work_disruption": 0.4,
        "collaboration_increase": 0.2  # People work together more
    }
)

breakthrough_event = EnvironmentalEvent(
    event_id="research_breakthrough",
    event_type="knowledge_discovery",
    description="Major breakthrough in related field affects research direction",
    duration=30 * 24 * 3600,  # 30 days
    effects={
        "research_excitement": 0.9,
        "paradigm_shift": 0.7,
        "collaboration_motivation": 0.8,
        "publication_pressure": 0.6
    }
)

# Create event-driven environment
event_env = EventDrivenEnvironment(env)

# Schedule events
event_env.schedule_event(conference_event, time.time() + 7 * 24 * 3600)  # In 1 week
event_env.schedule_event(equipment_failure_event, time.time() + 3 * 24 * 3600)  # In 3 days
event_env.schedule_event(breakthrough_event, time.time() + 14 * 24 * 3600)  # In 2 weeks

# Register event handlers
def handle_conference_event(event, environment):
    """Handle conference event effects"""
    print(f"🎯 Conference event: {event.description}")
    
    # Temporary environmental changes
    environment.modify_properties({
        "social_dynamics.networking_opportunities": event.effects["networking_opportunities"],
        "information_flow.external_knowledge": event.effects["knowledge_influx"],
        "workspace.visitor_access": True
    })
    
    # Notify all agents
    for agent in environment.agents:
        agent.receive_notification(f"Conference starting: {event.description}")

def handle_resource_disruption(event, environment):
    """Handle resource disruption events"""
    print(f"⚠️ Resource disruption: {event.description}")
    
    # Reduce available resources
    environment.modify_properties({
        "computing_resources.availability": event.effects["computing_resources_available"],
        "work_efficiency.baseline": 1.0 - event.effects["work_disruption"]
    })

event_env.register_event_handler("external_conference", handle_conference_event)
event_env.register_event_handler("resource_disruption", handle_resource_disruption)
```

## Environment Monitoring and Analysis

### Real-time Environment Metrics

```python
from cognito_sim_engine import EnvironmentMonitor

class EnvironmentMonitor:
    def __init__(self, environment):
        self.environment = environment
        self.metrics_history = []
        self.alert_thresholds = {}
        self.monitoring_active = False
    
    def start_monitoring(self, collection_interval=60):
        """Start continuous environment monitoring"""
        
        self.monitoring_active = True
        self.collection_interval = collection_interval
        
        # Start monitoring thread
        import threading
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
    
    def collect_metrics(self):
        """Collect current environment metrics"""
        
        current_time = time.time()
        
        # Environmental state metrics
        environmental_metrics = {
            "timestamp": current_time,
            "active_agents": len(self.environment.agents),
            "environmental_state": self.environment.get_current_state(),
            "resource_utilization": self.calculate_resource_utilization(),
            "information_flow_rate": self.calculate_information_flow(),
            "collaboration_index": self.calculate_collaboration_index(),
            "productivity_score": self.calculate_environment_productivity(),
            "stress_indicators": self.calculate_stress_indicators()
        }
        
        # Agent-environment interaction metrics
        interaction_metrics = {
            "agent_satisfaction": self.calculate_agent_satisfaction(),
            "environmental_adaptation_rate": self.calculate_adaptation_rate(),
            "resource_conflicts": self.detect_resource_conflicts(),
            "communication_efficiency": self.calculate_communication_efficiency()
        }
        
        # Combine all metrics
        all_metrics = {**environmental_metrics, **interaction_metrics}
        self.metrics_history.append(all_metrics)
        
        return all_metrics
    
    def calculate_collaboration_index(self):
        """Calculate overall collaboration level in environment"""
        
        if not self.environment.agents:
            return 0.0
        
        total_collaboration = 0.0
        total_possible_collaborations = 0
        
        for i, agent1 in enumerate(self.environment.agents):
            for agent2 in self.environment.agents[i+1:]:
                # Check if agents are collaborating
                collaboration_strength = agent1.get_collaboration_strength(agent2)
                total_collaboration += collaboration_strength
                total_possible_collaborations += 1
        
        if total_possible_collaborations == 0:
            return 0.0
        
        return total_collaboration / total_possible_collaborations
    
    def calculate_environment_productivity(self):
        """Calculate overall environmental productivity"""
        
        if not self.environment.agents:
            return 0.0
        
        # Aggregate agent productivity scores
        agent_productivities = [
            agent.get_productivity_score() 
            for agent in self.environment.agents
        ]
        
        individual_productivity = np.mean(agent_productivities)
        
        # Environmental factors affecting productivity
        resource_availability = self.environment.get_resource_availability()
        information_accessibility = self.environment.get_information_accessibility()
        distraction_level = self.environment.get_distraction_level()
        
        # Combined productivity score
        environmental_multiplier = (
            resource_availability * 0.4 +
            information_accessibility * 0.3 +
            (1.0 - distraction_level) * 0.3
        )
        
        return individual_productivity * environmental_multiplier
    
    def generate_environment_report(self, time_period="last_24_hours"):
        """Generate comprehensive environment analysis report"""
        
        # Filter metrics for time period
        current_time = time.time()
        if time_period == "last_24_hours":
            start_time = current_time - 24 * 3600
        elif time_period == "last_week":
            start_time = current_time - 7 * 24 * 3600
        else:
            start_time = 0  # All time
        
        relevant_metrics = [
            m for m in self.metrics_history 
            if m["timestamp"] >= start_time
        ]
        
        if not relevant_metrics:
            return "No metrics available for specified time period"
        
        # Analyze trends
        report = self._generate_detailed_report(relevant_metrics)
        
        return report
    
    def _generate_detailed_report(self, metrics):
        """Generate detailed analysis report"""
        
        report = {
            "summary": {
                "time_period": f"{len(metrics)} data points",
                "average_agents": np.mean([m["active_agents"] for m in metrics]),
                "average_productivity": np.mean([m["productivity_score"] for m in metrics]),
                "average_collaboration": np.mean([m["collaboration_index"] for m in metrics])
            },
            
            "trends": {
                "productivity_trend": self._calculate_trend([m["productivity_score"] for m in metrics]),
                "collaboration_trend": self._calculate_trend([m["collaboration_index"] for m in metrics]),
                "satisfaction_trend": self._calculate_trend([m["agent_satisfaction"] for m in metrics])
            },
            
            "alerts": self._check_alert_conditions(metrics[-1] if metrics else {}),
            
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report

# Setup environment monitoring
monitor = EnvironmentMonitor(env)
monitor.start_monitoring(collection_interval=300)  # Every 5 minutes

# Set alert thresholds
monitor.alert_thresholds = {
    "productivity_score": {"min": 0.4, "max": 1.0},
    "collaboration_index": {"min": 0.3, "max": 1.0},
    "agent_satisfaction": {"min": 0.5, "max": 1.0},
    "stress_indicators": {"min": 0.0, "max": 0.7}
}

# Generate reports
def print_environment_status():
    """Print current environment status"""
    
    current_metrics = monitor.collect_metrics()
    
    print("🌍 Environment Status Report")
    print(f"  Active Agents: {current_metrics['active_agents']}")
    print(f"  Productivity Score: {current_metrics['productivity_score']:.2f}")
    print(f"  Collaboration Index: {current_metrics['collaboration_index']:.2f}")
    print(f"  Agent Satisfaction: {current_metrics['agent_satisfaction']:.2f}")
    print(f"  Resource Utilization: {current_metrics['resource_utilization']:.2f}")
    
    # Check for alerts
    alerts = monitor._check_alert_conditions(current_metrics)
    if alerts:
        print("⚠️ Environment Alerts:")
        for alert in alerts:
            print(f"    • {alert}")

# Periodic status updates
print_environment_status()
```

## Best Practices

### 1. Environment Design

- **Match complexity to purpose**: Simple environments for basic studies, complex for realistic simulations
- **Consider scalability**: Design environments that can handle varying numbers of agents
- **Plan for adaptation**: Build in mechanisms for environmental change and evolution

### 2. Performance Optimization

- **Resource management**: Monitor and optimize computational resource usage
- **Event processing**: Efficient event handling for dynamic environments
- **State management**: Optimize environment state storage and updates

### 3. Validation and Testing

- **Environmental validity**: Ensure environments realistically represent target domains
- **Agent-environment fit**: Verify that agents can effectively operate in the environment
- **Behavioral emergence**: Test whether intended behaviors emerge from environment design

### 4. Monitoring and Maintenance

- **Continuous monitoring**: Track environment metrics and agent interactions
- **Performance analysis**: Regular analysis of environment effectiveness
- **Adaptive improvement**: Use feedback to improve environment design

---

Environment setup is crucial for creating meaningful cognitive simulations. Well-designed environments provide the context that enables sophisticated agent behaviors and meaningful research insights.

**Next**: Learn about [Memory Management](memory-management.md) to optimize agent knowledge systems, or explore [Reasoning & Goals](reasoning-goals.md) for advanced cognitive architectures.
