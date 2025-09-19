# Memory Management

Effective memory management is crucial for creating realistic and performant cognitive simulations. This guide covers how to configure, optimize, and monitor memory systems in Cognito Simulation Engine.

## Memory Architecture Overview

The memory system consists of multiple interconnected components:

```python
from cognito_sim_engine import MemoryManager, WorkingMemory, EpisodicMemory, SemanticMemory

# Create comprehensive memory manager
memory_manager = MemoryManager(
    working_memory=WorkingMemory(capacity=7),
    episodic_memory=EpisodicMemory(capacity=10000),
    semantic_memory=SemanticMemory(capacity=50000),
    procedural_memory=ProceduralMemory(capacity=1000)
)

# Configure integration between memory systems
memory_manager.configure_integration(
    consolidation_rate=0.05,
    transfer_learning=True,
    cross_memory_activation=True
)
```

## Working Memory Configuration

### Basic Setup

```python
from cognito_sim_engine import WorkingMemory, WorkingMemoryConfig

# Configure working memory with realistic constraints
wm_config = WorkingMemoryConfig(
    capacity=7,                    # Miller's 7±2 items
    decay_rate=0.1,               # Natural forgetting
    interference_factor=0.05,      # New items interfere with old
    rehearsal_boost=0.3,          # Active maintenance strength
    attention_focus_bonus=0.5     # Attention strengthens items
)

working_memory = WorkingMemory(config=wm_config)

# Add items with different importance levels
working_memory.add_item(
    content="Current research goal: Develop AGI architecture",
    importance=0.9,
    activation=0.8
)

working_memory.add_item(
    content="Meeting scheduled at 2 PM",
    importance=0.6,
    activation=0.7
)

working_memory.add_item(
    content="Coffee cup on desk",
    importance=0.1,
    activation=0.3
)
```

### Advanced Working Memory Features

```python
class AdvancedWorkingMemory(WorkingMemory):
    def __init__(self, config):
        super().__init__(config)
        self.chunking_enabled = True
        self.attention_allocation = {}
        self.cognitive_load = 0.0
    
    def add_item_with_chunking(self, content, related_items=None):
        """Add item with automatic chunking of related content"""
        
        if self.chunking_enabled and related_items:
            # Create chunk from related items
            chunk = self.create_chunk(content, related_items)
            return self.add_item(chunk, importance=0.8)
        else:
            return self.add_item(content)
    
    def create_chunk(self, main_content, related_items):
        """Create meaningful chunks to overcome capacity limits"""
        
        chunk = MemoryChunk(
            main_content=main_content,
            elements=related_items,
            chunk_type="semantic_grouping"
        )
        
        return chunk
    
    def allocate_attention(self, item_id, attention_amount):
        """Allocate attention to specific working memory items"""
        
        if item_id in self.items:
            self.attention_allocation[item_id] = attention_amount
            
            # Attention strengthens items
            self.items[item_id].activation += attention_amount * 0.3
            self.items[item_id].activation = min(1.0, self.items[item_id].activation)
    
    def calculate_cognitive_load(self):
        """Calculate current cognitive load based on working memory state"""
        
        # Base load from number of items
        item_load = len(self.items) / self.capacity
        
        # Complexity load from item complexity
        complexity_load = sum(
            item.complexity_score for item in self.items.values()
        ) / len(self.items) if self.items else 0
        
        # Interference load
        interference_load = self.calculate_interference_level()
        
        total_load = (item_load * 0.4 + 
                     complexity_load * 0.3 + 
                     interference_load * 0.3)
        
        self.cognitive_load = min(1.0, total_load)
        return self.cognitive_load

# Example usage
advanced_wm = AdvancedWorkingMemory(wm_config)

# Add related items as a chunk
ml_concepts = [
    "supervised learning",
    "unsupervised learning", 
    "reinforcement learning",
    "deep learning"
]

advanced_wm.add_item_with_chunking(
    "Machine learning fundamentals",
    related_items=ml_concepts
)

# Monitor cognitive load
load = advanced_wm.calculate_cognitive_load()
print(f"Current cognitive load: {load:.2f}")
```

## Long-Term Memory Management

### Episodic Memory Configuration

```python
from cognito_sim_engine import EpisodicMemory, Episode, MemoryContext

# Configure episodic memory with realistic parameters
episodic_config = {
    "capacity": 10000,
    "consolidation_threshold": 0.7,
    "forgetting_curve": "power_law",
    "context_binding_strength": 0.8,
    "emotional_enhancement": True
}

episodic_memory = EpisodicMemory(config=episodic_config)

# Store rich episodic memories
def store_research_session(session_data):
    """Store a research session as episodic memory"""
    
    episode = Episode(
        content=session_data["description"],
        temporal_context={
            "start_time": session_data["start_time"],
            "duration": session_data["duration"],
            "time_of_day": session_data["time_of_day"]
        },
        spatial_context={
            "location": session_data["location"],
            "environment_type": session_data["environment"],
            "participants": session_data["participants"]
        },
        emotional_context={
            "valence": session_data["emotional_valence"],
            "arousal": session_data["arousal_level"],
            "satisfaction": session_data["satisfaction"]
        },
        causal_context={
            "triggering_events": session_data["triggers"],
            "outcomes": session_data["outcomes"],
            "goal_progress": session_data["goal_progress"]
        }
    )
    
    episode_id = episodic_memory.store_episode(episode)
    return episode_id

# Example research session
session = {
    "description": "Breakthrough in neural architecture design",
    "start_time": "2024-01-15T14:30:00",
    "duration": 3600,
    "time_of_day": "afternoon",
    "location": "Research Lab A",
    "environment": "collaborative",
    "participants": ["Dr. Smith", "Alice", "Bob"],
    "emotional_valence": 0.8,
    "arousal_level": 0.7,
    "satisfaction": 0.9,
    "triggers": ["Previous approach failed", "New insight emerged"],
    "outcomes": ["Novel architecture proposed", "Experiments planned"],
    "goal_progress": 0.6
}

episode_id = store_research_session(session)
```

### Semantic Memory Optimization

```python
from cognito_sim_engine import SemanticMemory, ConceptGraph, KnowledgeExtraction

class OptimizedSemanticMemory(SemanticMemory):
    def __init__(self, config):
        super().__init__(config)
        self.concept_graph = ConceptGraph()
        self.knowledge_extractor = KnowledgeExtraction()
        self.activation_history = {}
    
    def add_knowledge_from_experience(self, episodic_memory):
        """Extract semantic knowledge from episodic experiences"""
        
        # Get recent episodes
        recent_episodes = episodic_memory.get_recent_episodes(days=7)
        
        # Extract concepts and relations
        for episode in recent_episodes:
            concepts = self.knowledge_extractor.extract_concepts(episode.content)
            relations = self.knowledge_extractor.extract_relations(episode.content)
            
            # Add to semantic network
            for concept in concepts:
                self.add_concept(concept)
            
            for relation in relations:
                self.add_relation(relation)
    
    def optimize_knowledge_structure(self):
        """Optimize semantic knowledge organization"""
        
        # Identify frequently co-accessed concepts
        co_access_patterns = self.analyze_co_access_patterns()
        
        # Strengthen connections between frequently accessed concepts
        for (concept1, concept2), frequency in co_access_patterns.items():
            if frequency > 5:  # Threshold for strengthening
                self.strengthen_connection(concept1, concept2, strength=0.1)
        
        # Prune weak connections
        self.prune_weak_connections(threshold=0.1)
        
        # Create higher-level abstractions
        self.create_abstractions()
    
    def semantic_search_with_context(self, query, context=None):
        """Context-aware semantic search"""
        
        # Basic concept matching
        base_results = self.concept_graph.search(query)
        
        # Apply context filtering if provided
        if context:
            context_filtered = self.filter_by_context(base_results, context)
            
            # Boost contextually relevant results
            for result in context_filtered:
                result.relevance_score *= 1.3
        
        # Apply spreading activation
        activated_concepts = self.spread_activation(
            source_concepts=[r.concept for r in base_results],
            max_hops=3,
            decay_factor=0.7
        )
        
        # Combine and rank results
        all_results = base_results + activated_concepts
        ranked_results = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)
        
        return ranked_results[:10]  # Top 10 results

# Setup optimized semantic memory
semantic_config = {
    "capacity": 50000,
    "organization": "hierarchical_network",
    "spreading_activation": True,
    "concept_learning": True,
    "relation_extraction": True
}

semantic_memory = OptimizedSemanticMemory(semantic_config)
```

## Memory Integration and Coordination

### Cross-Memory System Coordination

```python
class MemoryCoordinator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.transfer_rules = []
        self.consolidation_scheduler = ConsolidationScheduler()
    
    def coordinate_memory_systems(self):
        """Coordinate information flow between memory systems"""
        
        # Working memory to long-term transfer
        self.transfer_working_to_longterm()
        
        # Episodic to semantic extraction
        self.extract_semantic_from_episodic()
        
        # Cross-memory activation
        self.activate_related_memories()
        
        # Memory consolidation
        self.consolidate_memories()
    
    def transfer_working_to_longterm(self):
        """Transfer important working memory items to long-term storage"""
        
        wm = self.memory_manager.working_memory
        important_items = [
            item for item in wm.get_all_items()
            if item.importance > 0.7 and item.activation > 0.5
        ]
        
        for item in important_items:
            # Determine appropriate long-term memory system
            if self.is_episodic_content(item):
                episode = self.convert_to_episode(item)
                self.memory_manager.episodic_memory.store_episode(episode)
                
            elif self.is_semantic_content(item):
                concept = self.convert_to_concept(item)
                self.memory_manager.semantic_memory.add_concept(concept)
                
            elif self.is_procedural_content(item):
                procedure = self.convert_to_procedure(item)
                self.memory_manager.procedural_memory.add_procedure(procedure)
    
    def extract_semantic_from_episodic(self):
        """Extract general knowledge from episodic experiences"""
        
        episodic = self.memory_manager.episodic_memory
        semantic = self.memory_manager.semantic_memory
        
        # Get episodes for analysis
        recent_episodes = episodic.get_episodes_since(days_back=30)
        
        # Find patterns across episodes
        patterns = self.identify_patterns(recent_episodes)
        
        # Convert patterns to semantic knowledge
        for pattern in patterns:
            if pattern.frequency >= 3:  # Seen at least 3 times
                concept = self.pattern_to_concept(pattern)
                semantic.add_concept(concept)
    
    def schedule_consolidation(self, memory_type, trigger_conditions):
        """Schedule memory consolidation based on conditions"""
        
        consolidation_task = {
            "memory_type": memory_type,
            "trigger_conditions": trigger_conditions,
            "consolidation_function": self.get_consolidation_function(memory_type)
        }
        
        self.consolidation_scheduler.add_task(consolidation_task)

# Example memory coordination
coordinator = MemoryCoordinator(memory_manager)

# Set up automatic coordination
def periodic_coordination():
    """Run memory coordination periodically"""
    coordinator.coordinate_memory_systems()

# Schedule coordination every 100 simulation cycles
memory_manager.add_periodic_task(periodic_coordination, interval=100)
```

## Memory Performance Optimization

### Memory Cleanup and Garbage Collection

```python
class MemoryGarbageCollector:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.cleanup_strategies = {
            "working_memory": self.cleanup_working_memory,
            "episodic_memory": self.cleanup_episodic_memory,
            "semantic_memory": self.cleanup_semantic_memory
        }
    
    def run_garbage_collection(self, aggressive=False):
        """Run memory cleanup across all systems"""
        
        cleanup_stats = {}
        
        for memory_type, cleanup_func in self.cleanup_strategies.items():
            before_count = self.get_memory_count(memory_type)
            cleanup_func(aggressive=aggressive)
            after_count = self.get_memory_count(memory_type)
            
            cleanup_stats[memory_type] = {
                "before": before_count,
                "after": after_count,
                "removed": before_count - after_count
            }
        
        return cleanup_stats
    
    def cleanup_working_memory(self, aggressive=False):
        """Clean up working memory"""
        
        wm = self.memory_manager.working_memory
        
        # Remove items below activation threshold
        threshold = 0.1 if aggressive else 0.05
        wm.remove_items_below_threshold(threshold)
        
        # Remove very old items (if not rehearsed)
        max_age = 300 if aggressive else 600  # seconds
        wm.remove_old_items(max_age)
    
    def cleanup_episodic_memory(self, aggressive=False):
        """Clean up episodic memory"""
        
        em = self.memory_manager.episodic_memory
        
        if aggressive:
            # Remove low-importance episodes
            em.remove_episodes_below_importance(threshold=0.3)
            
            # Remove very old, unaccessed episodes
            em.remove_unaccessed_episodes(days_threshold=365)
        else:
            # Conservative cleanup
            em.remove_episodes_below_importance(threshold=0.1)
            em.remove_unaccessed_episodes(days_threshold=730)
    
    def cleanup_semantic_memory(self, aggressive=False):
        """Clean up semantic memory"""
        
        sm = self.memory_manager.semantic_memory
        
        # Remove concepts with very low activation
        threshold = 0.05 if aggressive else 0.02
        sm.remove_concepts_below_activation(threshold)
        
        # Prune weak connections
        connection_threshold = 0.1 if aggressive else 0.05
        sm.prune_weak_connections(connection_threshold)

# Setup automatic garbage collection
gc_manager = MemoryGarbageCollector(memory_manager)

# Run periodic cleanup
def scheduled_cleanup():
    stats = gc_manager.run_garbage_collection(aggressive=False)
    print("🧹 Memory cleanup completed:")
    for memory_type, stat in stats.items():
        print(f"  {memory_type}: {stat['removed']} items removed")

# Schedule cleanup every 1000 cycles
memory_manager.add_periodic_task(scheduled_cleanup, interval=1000)
```

### Memory Performance Monitoring

```python
class MemoryPerformanceMonitor:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.performance_history = []
        self.alert_thresholds = {
            "working_memory_utilization": 0.9,
            "episodic_memory_utilization": 0.8,
            "semantic_memory_utilization": 0.8,
            "average_retrieval_time": 1.0,  # seconds
            "memory_fragmentation": 0.7
        }
    
    def collect_performance_metrics(self):
        """Collect comprehensive memory performance metrics"""
        
        current_time = time.time()
        
        metrics = {
            "timestamp": current_time,
            
            # Utilization metrics
            "working_memory_utilization": self.calculate_wm_utilization(),
            "episodic_memory_utilization": self.calculate_em_utilization(),
            "semantic_memory_utilization": self.calculate_sm_utilization(),
            
            # Performance metrics
            "average_retrieval_time": self.calculate_avg_retrieval_time(),
            "retrieval_success_rate": self.calculate_retrieval_success_rate(),
            "memory_fragmentation": self.calculate_memory_fragmentation(),
            
            # Quality metrics
            "memory_coherence": self.calculate_memory_coherence(),
            "cross_memory_consistency": self.calculate_consistency(),
            
            # Resource metrics
            "total_memory_usage": self.calculate_total_memory_usage(),
            "memory_access_frequency": self.calculate_access_frequency()
        }
        
        self.performance_history.append(metrics)
        return metrics
    
    def calculate_wm_utilization(self):
        """Calculate working memory utilization"""
        wm = self.memory_manager.working_memory
        return len(wm.items) / wm.capacity
    
    def calculate_avg_retrieval_time(self):
        """Calculate average memory retrieval time"""
        recent_retrievals = self.memory_manager.get_recent_retrievals(count=100)
        
        if not recent_retrievals:
            return 0.0
        
        total_time = sum(r.retrieval_time for r in recent_retrievals)
        return total_time / len(recent_retrievals)
    
    def calculate_memory_coherence(self):
        """Calculate overall memory system coherence"""
        
        # Check for contradictions
        contradictions = self.detect_memory_contradictions()
        
        # Check for consistency across systems
        consistency_score = self.calculate_cross_system_consistency()
        
        # Check for temporal consistency
        temporal_consistency = self.calculate_temporal_consistency()
        
        coherence_score = (
            (1.0 - min(1.0, len(contradictions) / 10)) * 0.4 +
            consistency_score * 0.3 +
            temporal_consistency * 0.3
        )
        
        return coherence_score
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        
        if not self.performance_history:
            return "No performance data available"
        
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements
        
        report = {
            "summary": {
                "avg_wm_utilization": np.mean([m["working_memory_utilization"] for m in recent_metrics]),
                "avg_retrieval_time": np.mean([m["average_retrieval_time"] for m in recent_metrics]),
                "avg_coherence": np.mean([m["memory_coherence"] for m in recent_metrics])
            },
            
            "trends": {
                "utilization_trend": self.calculate_trend([m["working_memory_utilization"] for m in recent_metrics]),
                "performance_trend": self.calculate_trend([m["average_retrieval_time"] for m in recent_metrics])
            },
            
            "alerts": self.check_performance_alerts(recent_metrics[-1]),
            
            "recommendations": self.generate_optimization_recommendations(recent_metrics)
        }
        
        return report
    
    def check_performance_alerts(self, current_metrics):
        """Check for performance issues requiring attention"""
        
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in current_metrics:
                value = current_metrics[metric]
                
                if value > threshold:
                    alerts.append({
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "high" if value > threshold * 1.2 else "medium"
                    })
        
        return alerts

# Setup performance monitoring
perf_monitor = MemoryPerformanceMonitor(memory_manager)

# Monitor performance periodically
def monitor_memory_performance():
    metrics = perf_monitor.collect_performance_metrics()
    
    # Check for alerts
    alerts = perf_monitor.check_performance_alerts(metrics)
    
    if alerts:
        print("⚠️ Memory Performance Alerts:")
        for alert in alerts:
            print(f"  {alert['metric']}: {alert['value']:.3f} (threshold: {alert['threshold']:.3f})")

# Schedule monitoring
memory_manager.add_periodic_task(monitor_memory_performance, interval=100)
```

## Memory-Based Learning and Adaptation

### Adaptive Memory Configuration

```python
class AdaptiveMemoryManager(MemoryManager):
    def __init__(self, base_config):
        super().__init__(base_config)
        self.performance_tracker = MemoryPerformanceTracker()
        self.adaptation_history = []
        self.learning_rate = 0.1
    
    def adapt_memory_parameters(self):
        """Automatically adapt memory parameters based on performance"""
        
        current_performance = self.performance_tracker.get_current_performance()
        
        # Adapt working memory capacity
        if current_performance["cognitive_overload"] > 0.8:
            self.reduce_working_memory_load()
        elif current_performance["cognitive_underutilization"] > 0.7:
            self.increase_working_memory_efficiency()
        
        # Adapt consolidation rates
        if current_performance["forgetting_rate"] > 0.6:
            self.increase_consolidation_rate()
        elif current_performance["memory_interference"] > 0.5:
            self.adjust_interference_handling()
        
        # Adapt retrieval strategies
        if current_performance["retrieval_accuracy"] < 0.7:
            self.optimize_retrieval_strategies()
    
    def reduce_working_memory_load(self):
        """Reduce cognitive load when overwhelmed"""
        
        # Increase chunking aggressiveness
        self.working_memory.chunking_threshold *= 0.9
        
        # Increase forgetting rate for low-importance items
        self.working_memory.decay_rate *= 1.1
        
        # Prioritize high-importance items
        self.working_memory.importance_boost *= 1.2
        
        self.log_adaptation("reduced_wm_load")
    
    def optimize_retrieval_strategies(self):
        """Optimize memory retrieval based on performance"""
        
        # Analyze retrieval failures
        failed_retrievals = self.get_failed_retrievals()
        
        for failure in failed_retrievals:
            # Strengthen relevant pathways
            if failure.failure_type == "pathway_weak":
                self.strengthen_retrieval_pathway(failure.query, failure.target)
            
            # Add alternative retrieval cues
            elif failure.failure_type == "insufficient_cues":
                self.add_retrieval_cues(failure.target, failure.context)
        
        self.log_adaptation("optimized_retrieval")
    
    def learn_from_memory_usage(self):
        """Learn optimal memory configurations from usage patterns"""
        
        usage_patterns = self.analyze_memory_usage_patterns()
        
        # Learn optimal capacity allocations
        optimal_capacities = self.calculate_optimal_capacities(usage_patterns)
        
        # Gradually adjust toward optimal values
        for memory_type, optimal_capacity in optimal_capacities.items():
            current_capacity = self.get_memory_capacity(memory_type)
            adjustment = (optimal_capacity - current_capacity) * self.learning_rate
            
            new_capacity = current_capacity + adjustment
            self.set_memory_capacity(memory_type, new_capacity)
        
        # Learn optimal transfer timing
        optimal_transfer_timing = self.calculate_optimal_transfer_timing(usage_patterns)
        self.update_transfer_schedules(optimal_transfer_timing)

# Create adaptive memory system
adaptive_config = {
    "base_working_memory_capacity": 7,
    "adaptation_enabled": True,
    "learning_rate": 0.1,
    "performance_monitoring": True
}

adaptive_memory = AdaptiveMemoryManager(adaptive_config)

# Enable continuous adaptation
adaptive_memory.enable_continuous_adaptation(interval=500)  # Every 500 cycles
```

## Integration with Cognitive Architecture

### Memory-Reasoning Integration

```python
def integrate_memory_with_reasoning(memory_manager, reasoning_engine):
    """Integrate memory systems with reasoning engine"""
    
    # Configure memory-based reasoning
    reasoning_engine.set_memory_interface(memory_manager)
    
    # Enable memory-guided inference
    reasoning_engine.enable_memory_guided_inference(
        use_episodic_analogies=True,
        use_semantic_activation=True,
        use_procedural_priming=True
    )
    
    # Configure memory updates from reasoning
    reasoning_engine.set_memory_update_rules([
        "store_reasoning_chains",
        "update_concept_activations",
        "learn_from_failures"
    ])

# Example integration
integrate_memory_with_reasoning(memory_manager, agent.reasoning_engine)
```

## Best Practices

### 1. Memory Configuration

- **Start with realistic parameters**: Use cognitive science research as a guide
- **Monitor performance**: Track utilization and performance metrics
- **Adapt gradually**: Make incremental adjustments based on observed behavior

### 2. Performance Optimization

- **Regular cleanup**: Implement periodic garbage collection
- **Efficient retrieval**: Use appropriate indexing and caching strategies
- **Memory hierarchy**: Leverage different memory systems appropriately

### 3. Integration

- **Cross-system coordination**: Ensure memory systems work together effectively
- **Reasoning integration**: Connect memory with reasoning and decision-making
- **Learning integration**: Use memory to support learning and adaptation

### 4. Debugging and Analysis

- **Performance monitoring**: Track memory system performance continuously
- **Usage analysis**: Understand how memory is being used
- **Coherence checking**: Verify memory consistency and coherence

---

Effective memory management is essential for creating realistic and powerful cognitive simulations. By understanding and properly configuring the memory systems, you can create agents that exhibit human-like memory behaviors while maintaining computational efficiency.

**Next**: Explore [Reasoning & Goals](reasoning-goals.md) to learn how memory integrates with reasoning systems, or see [CLI Usage](cli-usage.md) for command-line tools to manage memory systems.
