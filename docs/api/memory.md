# Memory API Reference

The Memory API provides comprehensive memory systems for cognitive agents, including working memory, episodic memory, semantic memory, and long-term memory with advanced consolidation and retrieval capabilities.

## Memory Architecture Overview

The memory system consists of several interconnected components:

- **WorkingMemory** - Short-term active memory for immediate processing
- **EpisodicMemory** - Memory for personal experiences and events
- **SemanticMemory** - Memory for general knowledge and concepts
- **LongTermMemory** - Consolidated memory for persistent knowledge
- **MemoryManager** - Orchestrates all memory systems

## MemoryManager

The central coordinator for all memory systems in a cognitive agent.

### Class Definition

```python
class MemoryManager:
    """
    Central manager for all memory systems.
    
    Coordinates working memory, episodic memory, semantic memory, and long-term memory,
    providing unified interfaces for storage, retrieval, and consolidation.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Memory configuration object
        """
```

### Core Methods

#### store

```python
def store(
    self,
    content: Any,
    memory_type: MemoryType,
    activation: float = 1.0,
    tags: List[str] = None,
    context: Dict[str, Any] = None
) -> str:
    """
    Store content in appropriate memory system.
    
    Args:
        content: Content to store
        memory_type: Type of memory to store in
        activation: Initial activation level
        tags: Tags for categorization
        context: Additional context information
        
    Returns:
        str: Memory item ID
        
    Raises:
        MemoryError: If storage fails
    """
```

#### retrieve

```python
def retrieve(
    self,
    query: str,
    memory_types: List[MemoryType] = None,
    limit: int = 10,
    activation_threshold: float = 0.1,
    context: Dict[str, Any] = None
) -> List[MemoryItem]:
    """
    Retrieve memories matching query.
    
    Args:
        query: Search query
        memory_types: Types of memory to search
        limit: Maximum number of results
        activation_threshold: Minimum activation level
        context: Context for retrieval
        
    Returns:
        List[MemoryItem]: Retrieved memory items
    """
```

**Example Usage:**

```python
from cognito_sim_engine import MemoryManager, MemoryType, MemoryConfig

# Configure memory system
config = MemoryConfig(
    working_memory_capacity=7,
    episodic_memory_capacity=10000,
    semantic_memory_enabled=True,
    consolidation_enabled=True
)

# Create memory manager
memory_manager = MemoryManager(config)

# Store different types of memories
# Working memory - current task information
working_memory_id = memory_manager.store(
    content="researching neural networks",
    memory_type=MemoryType.WORKING,
    activation=1.0,
    tags=["current_task", "research"]
)

# Episodic memory - personal experience
episode_id = memory_manager.store(
    content={
        "event": "attended_ml_conference",
        "location": "Stanford University",
        "insights": ["learned about transformers", "met researchers"],
        "emotions": {"excitement": 0.8, "curiosity": 0.9}
    },
    memory_type=MemoryType.EPISODIC,
    tags=["conference", "learning", "networking"]
)

# Semantic memory - general knowledge
concept_id = memory_manager.store(
    content={
        "concept": "neural_network",
        "definition": "computational model inspired by biological neural networks",
        "properties": ["weights", "biases", "activation_functions"],
        "applications": ["image_recognition", "nlp", "game_playing"]
    },
    memory_type=MemoryType.SEMANTIC,
    tags=["ml_concept", "neural_networks"]
)

# Retrieve memories
research_memories = memory_manager.retrieve(
    query="neural network research",
    memory_types=[MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC],
    limit=20
)

print(f"Found {len(research_memories)} relevant memories")
```

#### consolidate

```python
def consolidate(self, force: bool = False) -> ConsolidationResult:
    """
    Consolidate memories from working to long-term memory.
    
    Args:
        force: Force consolidation even if conditions not met
        
    Returns:
        ConsolidationResult: Results of consolidation process
    """
```

#### forget

```python
def forget(
    self,
    criteria: Dict[str, Any],
    memory_types: List[MemoryType] = None,
    preserve_important: bool = True
) -> int:
    """
    Remove memories based on criteria.
    
    Args:
        criteria: Forgetting criteria
        memory_types: Types of memory to process
        preserve_important: Preserve high-importance memories
        
    Returns:
        int: Number of memories removed
    """
```

## WorkingMemory

Short-term memory for active information processing with limited capacity and decay.

### Class Definition

```python
class WorkingMemory:
    """
    Working memory with limited capacity and activation decay.
    
    Implements the cognitive constraint of limited working memory capacity
    with realistic decay and interference patterns.
    """
    
    def __init__(
        self,
        capacity: int = 7,
        decay_rate: float = 0.1,
        interference_factor: float = 0.05
    ):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum number of items (Miller's 7±2)
            decay_rate: Rate of activation decay per time step
            interference_factor: Interference between similar items
        """
```

### Core Methods

#### store

```python
def store(
    self,
    item: MemoryItem,
    activation: float = 1.0,
    replace_strategy: str = "lru"
) -> bool:
    """
    Store item in working memory.
    
    Args:
        item: Memory item to store
        activation: Initial activation level
        replace_strategy: Strategy for replacing items when full
        
    Returns:
        bool: True if stored successfully
    """
```

#### retrieve

```python
def retrieve(
    self,
    query: str,
    activation_boost: float = 0.2,
    limit: int = None
) -> List[MemoryItem]:
    """
    Retrieve items from working memory.
    
    Args:
        query: Search query
        activation_boost: Boost activation of retrieved items
        limit: Maximum number of items to retrieve
        
    Returns:
        List[MemoryItem]: Retrieved items ordered by activation
    """
```

#### update_activations

```python
def update_activations(self, time_step: float = 1.0) -> None:
    """
    Update activation levels with decay and interference.
    
    Args:
        time_step: Time elapsed since last update
    """
```

**Example Usage:**

```python
from cognito_sim_engine import WorkingMemory, MemoryItem

# Create working memory with realistic constraints
working_memory = WorkingMemory(
    capacity=7,
    decay_rate=0.1,
    interference_factor=0.05
)

# Store current task information
task_item = MemoryItem(
    content="solve optimization problem",
    activation=1.0,
    timestamp=time.time(),
    tags=["current_task"]
)

success = working_memory.store(task_item, activation=1.0)
print(f"Task stored: {success}")

# Store relevant facts
for i, fact in enumerate([
    "gradient descent finds local minima",
    "learning rate affects convergence speed",
    "regularization prevents overfitting"
]):
    fact_item = MemoryItem(
        content=fact,
        activation=0.8,
        timestamp=time.time(),
        tags=["optimization_fact"]
    )
    working_memory.store(fact_item)

# Simulate time passing with activation decay
for step in range(10):
    working_memory.update_activations(time_step=1.0)
    
    # Retrieve active information
    active_items = working_memory.retrieve("optimization")
    print(f"Step {step}: {len(active_items)} active items")

# Check capacity constraints
print(f"Working memory capacity: {working_memory.capacity}")
print(f"Current items: {len(working_memory.items)}")
```

## EpisodicMemory

Memory system for storing and retrieving personal experiences and events.

### Class Definition

```python
class EpisodicMemory:
    """
    Episodic memory for personal experiences and events.
    
    Stores contextualized episodes with temporal, spatial, and emotional information.
    Supports narrative retrieval and episodic reasoning.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        similarity_threshold: float = 0.7,
        temporal_decay: bool = True
    ):
        """
        Initialize episodic memory.
        
        Args:
            capacity: Maximum number of episodes
            similarity_threshold: Threshold for episode similarity
            temporal_decay: Enable temporal decay of episode accessibility
        """
```

### Core Methods

#### store_episode

```python
def store_episode(
    self,
    episode: Episode,
    consolidate: bool = True
) -> str:
    """
    Store an episode in memory.
    
    Args:
        episode: Episode to store
        consolidate: Whether to consolidate similar episodes
        
    Returns:
        str: Episode ID
    """
```

#### retrieve_episodes

```python
def retrieve_episodes(
    self,
    query: str,
    temporal_range: Optional[Tuple[datetime, datetime]] = None,
    emotional_filter: Optional[Dict[str, float]] = None,
    similarity_threshold: float = 0.5,
    limit: int = 10
) -> List[Episode]:
    """
    Retrieve episodes matching criteria.
    
    Args:
        query: Search query
        temporal_range: Time range for episodes
        emotional_filter: Emotional state filter
        similarity_threshold: Minimum similarity score
        limit: Maximum number of episodes
        
    Returns:
        List[Episode]: Retrieved episodes
    """
```

#### find_similar_episodes

```python
def find_similar_episodes(
    self,
    target_episode: Episode,
    similarity_threshold: float = 0.7,
    context_weight: float = 0.3
) -> List[Tuple[Episode, float]]:
    """
    Find episodes similar to target episode.
    
    Args:
        target_episode: Episode to find similarities for
        similarity_threshold: Minimum similarity score
        context_weight: Weight of contextual similarity
        
    Returns:
        List[Tuple[Episode, float]]: Episodes with similarity scores
    """
```

**Example Usage:**

```python
from cognito_sim_engine import EpisodicMemory, Episode, EmotionalState
from datetime import datetime, timedelta

# Create episodic memory
episodic_memory = EpisodicMemory(
    capacity=10000,
    similarity_threshold=0.7,
    temporal_decay=True
)

# Create and store research episode
research_episode = Episode(
    episode_id="research_session_001",
    agent_id="researcher_001",
    timestamp=datetime.now(),
    events=[
        {"action": "read_paper", "object": "attention_is_all_you_need"},
        {"action": "take_notes", "content": "transformer architecture insights"},
        {"action": "implement_experiment", "success": True}
    ],
    context={
        "location": "research_lab",
        "goal": "understand_transformers",
        "duration": 7200,  # 2 hours
        "collaborators": ["colleague_a", "colleague_b"]
    },
    emotional_state=EmotionalState(
        valence=0.8,  # positive
        arousal=0.6,  # moderately excited
        dominance=0.7  # feeling in control
    ),
    outcomes={
        "knowledge_gained": "transformer_architecture",
        "confidence_increase": 0.3,
        "questions_generated": [
            "how do transformers handle long sequences?",
            "what are the computational trade-offs?"
        ]
    }
)

# Store the episode
episode_id = episodic_memory.store_episode(research_episode)
print(f"Stored episode: {episode_id}")

# Retrieve similar research experiences
similar_episodes = episodic_memory.retrieve_episodes(
    query="research transformer",
    temporal_range=(datetime.now() - timedelta(days=30), datetime.now()),
    emotional_filter={"valence": 0.5},  # positive experiences
    limit=5
)

print(f"Found {len(similar_episodes)} similar research episodes")

# Find episodes similar to current one
current_episode = Episode(
    episode_id="current_research",
    agent_id="researcher_001",
    timestamp=datetime.now(),
    events=[{"action": "study_attention_mechanism"}],
    context={"goal": "understand_attention"}
)

similar_with_scores = episodic_memory.find_similar_episodes(
    target_episode=current_episode,
    similarity_threshold=0.6
)

print("Similar episodes:")
for episode, score in similar_with_scores:
    print(f"  {episode.episode_id}: {score:.3f} similarity")
```

## SemanticMemory

Knowledge representation system for concepts, facts, and their relationships.

### Class Definition

```python
class SemanticMemory:
    """
    Semantic memory for general knowledge and concepts.
    
    Implements a graph-based knowledge representation with concepts,
    relations, and inference capabilities.
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reasoning_enabled: bool = True
    ):
        """
        Initialize semantic memory.
        
        Args:
            knowledge_graph: External knowledge graph
            reasoning_enabled: Enable inference over knowledge
        """
```

### Core Methods

#### store_concept

```python
def store_concept(
    self,
    concept: Concept,
    relations: List[Relation] = None,
    confidence: float = 1.0
) -> str:
    """
    Store a concept with optional relations.
    
    Args:
        concept: Concept to store
        relations: Relations to other concepts
        confidence: Confidence in the concept
        
    Returns:
        str: Concept ID
    """
```

#### retrieve_concepts

```python
def retrieve_concepts(
    self,
    query: str,
    relation_filter: Optional[str] = None,
    confidence_threshold: float = 0.5,
    expand_relations: bool = True,
    limit: int = 20
) -> List[Concept]:
    """
    Retrieve concepts matching query.
    
    Args:
        query: Search query
        relation_filter: Filter by relation type
        confidence_threshold: Minimum confidence
        expand_relations: Include related concepts
        limit: Maximum concepts to return
        
    Returns:
        List[Concept]: Retrieved concepts
    """
```

#### get_related_concepts

```python
def get_related_concepts(
    self,
    concept_id: str,
    relation_types: List[str] = None,
    max_distance: int = 2,
    strength_threshold: float = 0.3
) -> Dict[str, List[Tuple[Concept, float]]]:
    """
    Get concepts related to given concept.
    
    Args:
        concept_id: ID of source concept
        relation_types: Types of relations to follow
        max_distance: Maximum relation distance
        strength_threshold: Minimum relation strength
        
    Returns:
        Dict[str, List[Tuple[Concept, float]]]: Related concepts by relation type
    """
```

**Example Usage:**

```python
from cognito_sim_engine import SemanticMemory, Concept, Relation

# Create semantic memory
semantic_memory = SemanticMemory(reasoning_enabled=True)

# Store machine learning concepts
ml_concept = Concept(
    concept_id="machine_learning",
    name="Machine Learning",
    definition="Field of study that gives computers the ability to learn",
    properties={
        "domain": "computer_science",
        "complexity": "high",
        "applications": ["classification", "regression", "clustering"]
    },
    examples=["neural_networks", "decision_trees", "svm"]
)

neural_network_concept = Concept(
    concept_id="neural_network",
    name="Neural Network",
    definition="Computing system inspired by biological neural networks",
    properties={
        "type": "ml_algorithm",
        "learning_type": "supervised",
        "components": ["neurons", "weights", "biases"]
    }
)

# Store concepts with relations
ml_id = semantic_memory.store_concept(ml_concept)
nn_id = semantic_memory.store_concept(neural_network_concept)

# Add relations between concepts
ml_nn_relation = Relation(
    relation_id="ml_includes_nn",
    source_concept=ml_id,
    target_concept=nn_id,
    relation_type="includes",
    strength=0.9,
    properties={"specificity": "high"}
)

semantic_memory.add_relation(ml_nn_relation)

# Retrieve related concepts
related_to_ml = semantic_memory.get_related_concepts(
    concept_id=ml_id,
    relation_types=["includes", "related_to"],
    max_distance=2
)

print("Concepts related to Machine Learning:")
for relation_type, concepts in related_to_ml.items():
    print(f"  {relation_type}:")
    for concept, strength in concepts:
        print(f"    {concept.name} ({strength:.2f})")

# Query semantic memory
ml_concepts = semantic_memory.retrieve_concepts(
    query="machine learning algorithms",
    expand_relations=True,
    limit=10
)

print(f"Found {len(ml_concepts)} ML-related concepts")
```

## LongTermMemory

Consolidated memory system for persistent knowledge and experiences.

### Class Definition

```python
class LongTermMemory:
    """
    Long-term memory with consolidation and retrieval.
    
    Implements memory consolidation from other memory systems,
    with organized storage and efficient retrieval mechanisms.
    """
    
    def __init__(
        self,
        consolidation_threshold: float = 0.7,
        retrieval_cue_sensitivity: float = 0.5,
        forgetting_curve_enabled: bool = True
    ):
        """
        Initialize long-term memory.
        
        Args:
            consolidation_threshold: Threshold for memory consolidation
            retrieval_cue_sensitivity: Sensitivity to retrieval cues
            forgetting_curve_enabled: Enable Ebbinghaus forgetting curve
        """
```

### Core Methods

#### consolidate_memory

```python
def consolidate_memory(
    self,
    source_memory: Union[MemoryItem, Episode, Concept],
    consolidation_strength: float,
    schema: Optional[MemorySchema] = None
) -> str:
    """
    Consolidate memory from other systems.
    
    Args:
        source_memory: Memory to consolidate
        consolidation_strength: Strength of consolidation
        schema: Memory schema for organization
        
    Returns:
        str: Consolidated memory ID
    """
```

#### retrieve_consolidated

```python
def retrieve_consolidated(
    self,
    retrieval_cues: List[str],
    context: Dict[str, Any] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    strength_threshold: float = 0.3
) -> List[ConsolidatedMemory]:
    """
    Retrieve consolidated memories.
    
    Args:
        retrieval_cues: Cues for memory retrieval
        context: Contextual information
        time_range: Time range for memories
        strength_threshold: Minimum consolidation strength
        
    Returns:
        List[ConsolidatedMemory]: Retrieved memories
    """
```

**Example Usage:**

```python
from cognito_sim_engine import LongTermMemory, MemorySchema

# Create long-term memory
ltm = LongTermMemory(
    consolidation_threshold=0.7,
    retrieval_cue_sensitivity=0.5,
    forgetting_curve_enabled=True
)

# Define memory schema for research experiences
research_schema = MemorySchema(
    schema_id="research_experience",
    categories=["methodology", "findings", "insights", "applications"],
    consolidation_rules={
        "importance_weight": 0.4,
        "frequency_weight": 0.3,
        "recency_weight": 0.3
    }
)

# Consolidate important research episode
important_episode = episodic_memory.get_episode("breakthrough_discovery")
consolidated_id = ltm.consolidate_memory(
    source_memory=important_episode,
    consolidation_strength=0.9,
    schema=research_schema
)

print(f"Consolidated important discovery: {consolidated_id}")

# Retrieve related consolidated memories
research_memories = ltm.retrieve_consolidated(
    retrieval_cues=["research", "discovery", "methodology"],
    context={"domain": "machine_learning"},
    strength_threshold=0.5
)

print(f"Retrieved {len(research_memories)} consolidated research memories")
```

## Memory Configuration

### MemoryConfig

```python
@dataclass
class MemoryConfig:
    # Working memory settings
    working_memory_capacity: int = 7
    working_memory_decay_rate: float = 0.1
    
    # Episodic memory settings
    episodic_memory_capacity: int = 10000
    episodic_temporal_decay: bool = True
    episodic_consolidation_enabled: bool = True
    
    # Semantic memory settings
    semantic_memory_enabled: bool = True
    semantic_reasoning_enabled: bool = True
    knowledge_graph_enabled: bool = True
    
    # Long-term memory settings
    long_term_memory_enabled: bool = True
    consolidation_threshold: float = 0.7
    consolidation_frequency: int = 100  # steps
    
    # General settings
    memory_cleanup_enabled: bool = True
    cleanup_interval: int = 1000
    max_total_memory_mb: int = 1000
    
    # Advanced features
    memory_interference_enabled: bool = True
    associative_retrieval: bool = True
    context_dependent_retrieval: bool = True
    emotional_memory_modulation: bool = True
```

## Memory Integration Patterns

### Cross-Memory Retrieval

```python
def integrated_memory_retrieval(memory_manager, query, context=None):
    """
    Retrieve from all memory systems with integration.
    """
    
    # Start with working memory (most active)
    working_results = memory_manager.working_memory.retrieve(query)
    
    # Get related episodic experiences
    episodic_results = memory_manager.episodic_memory.retrieve_episodes(
        query=query,
        context=context
    )
    
    # Get semantic knowledge
    semantic_results = memory_manager.semantic_memory.retrieve_concepts(
        query=query,
        expand_relations=True
    )
    
    # Get consolidated long-term memories
    ltm_results = memory_manager.long_term_memory.retrieve_consolidated(
        retrieval_cues=query.split(),
        context=context
    )
    
    # Integrate and rank results
    integrated_results = IntegratedMemoryResults(
        working_memory=working_results,
        episodic_memory=episodic_results,
        semantic_memory=semantic_results,
        long_term_memory=ltm_results
    )
    
    return integrated_results.rank_by_relevance()

# Example usage
query = "machine learning optimization techniques"
context = {"current_goal": "improve_model_performance"}

integrated_memories = integrated_memory_retrieval(
    memory_manager=agent.memory_manager,
    query=query,
    context=context
)

print("Integrated memory retrieval results:")
for memory_type, results in integrated_memories.items():
    print(f"{memory_type}: {len(results)} items")
```

### Memory-Guided Reasoning

```python
def memory_guided_inference(memory_manager, reasoning_query):
    """
    Use memory to guide reasoning process.
    """
    
    # Retrieve relevant memories
    relevant_memories = memory_manager.retrieve(
        query=reasoning_query,
        memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC]
    )
    
    # Extract facts from memories
    facts = []
    for memory in relevant_memories:
        if memory.memory_type == MemoryType.SEMANTIC:
            facts.extend(extract_facts_from_concept(memory.content))
        elif memory.memory_type == MemoryType.EPISODIC:
            facts.extend(extract_facts_from_episode(memory.content))
    
    # Use facts for reasoning
    reasoning_result = reasoning_engine.infer(
        facts=facts,
        goal=reasoning_query
    )
    
    # Store reasoning result as new episodic memory
    reasoning_episode = Episode(
        episode_id=f"reasoning_{uuid.uuid4()}",
        agent_id=agent.agent_id,
        timestamp=datetime.now(),
        events=[
            {"action": "reasoning", "query": reasoning_query},
            {"action": "conclusion", "result": reasoning_result.conclusion}
        ],
        context={"reasoning_type": "memory_guided"}
    )
    
    memory_manager.store_episode(reasoning_episode)
    
    return reasoning_result

# Example usage
reasoning_query = "What are the best practices for training neural networks?"
result = memory_guided_inference(memory_manager, reasoning_query)
print(f"Memory-guided reasoning conclusion: {result.conclusion}")
```

## Performance Optimization

### Memory Efficiency

```python
# Configure memory for efficiency
efficient_config = MemoryConfig(
    working_memory_capacity=5,  # Smaller capacity
    episodic_memory_capacity=5000,  # Reasonable size
    memory_cleanup_enabled=True,
    cleanup_interval=500,  # Frequent cleanup
    max_total_memory_mb=500  # Memory limit
)

# Monitor memory usage
def monitor_memory_usage(memory_manager):
    stats = memory_manager.get_memory_statistics()
    
    if stats.total_memory_mb > 400:
        # Trigger cleanup
        memory_manager.cleanup_old_memories()
        
    if stats.working_memory_utilization > 0.9:
        # Consolidate working memory
        memory_manager.consolidate_working_memory()
    
    return stats

# Regular monitoring
stats = monitor_memory_usage(memory_manager)
print(f"Memory usage: {stats.total_memory_mb:.2f} MB")
```

### Retrieval Optimization

```python
# Optimize retrieval with caching
class CachedMemoryManager(MemoryManager):
    def __init__(self, config=None):
        super().__init__(config)
        self.retrieval_cache = {}
        self.cache_size_limit = 1000
    
    def retrieve(self, query, **kwargs):
        # Check cache first
        cache_key = self.create_cache_key(query, kwargs)
        
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        # Perform retrieval
        results = super().retrieve(query, **kwargs)
        
        # Cache results
        if len(self.retrieval_cache) < self.cache_size_limit:
            self.retrieval_cache[cache_key] = results
        
        return results
    
    def invalidate_cache(self):
        self.retrieval_cache.clear()

# Use cached memory manager
cached_memory = CachedMemoryManager(config)
```

---

The Memory API provides sophisticated memory systems that enable realistic cognitive behavior in artificial agents. Use these components to create agents with human-like memory characteristics and capabilities.

**Related APIs:**

- [Agents API](agents.md) - Agent integration with memory systems
- [Reasoning API](reasoning.md) - Memory-guided reasoning
- [Engine API](engine.md) - Memory system orchestration
