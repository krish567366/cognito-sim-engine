# CLI Usage Guide

This guide covers how to use the Cognito Simulation Engine command-line interface (CLI) for managing cognitive simulations, agents, and environments.

## Installation and Setup

### Install Cognito Simulation Engine

```bash
# Install from PyPI
pip install cognito-sim-engine

# Verify installation
cognito-sim --version
```

### Basic CLI Structure

```bash
cognito-sim [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

## Global Options

```bash
# Show help
cognito-sim --help

# Enable verbose output
cognito-sim --verbose COMMAND

# Set configuration file
cognito-sim --config path/to/config.yaml COMMAND

# Enable debug mode
cognito-sim --debug COMMAND

# Set log level
cognito-sim --log-level DEBUG COMMAND
```

## Core Commands

### 1. Agent Management

#### Create Agents

```bash
# Create a basic cognitive agent
cognito-sim agent create \
  --name "research_assistant" \
  --type cognitive \
  --personality openness:0.8,conscientiousness:0.9 \
  --output agents/research_assistant.json

# Create a learning agent with specific capabilities
cognito-sim agent create \
  --name "ml_student" \
  --type learning \
  --learning-rate 0.01 \
  --memory-capacity 10000 \
  --reasoning-depth 5 \
  --goals "learn_ml_fundamentals,complete_projects" \
  --output agents/ml_student.json

# Create multiple agents from template
cognito-sim agent create-batch \
  --template templates/student_template.yaml \
  --count 5 \
  --name-prefix "student_" \
  --output-dir agents/classroom/

# Create specialized research agents
cognito-sim agent create \
  --name "researcher" \
  --type cognitive \
  --specialization research \
  --domain "artificial_intelligence" \
  --reasoning-strategies "analytical,creative,critical" \
  --memory-types "episodic,semantic,working" \
  --collaboration-style "cooperative"
```

#### Manage Agents

```bash
# List all agents
cognito-sim agent list

# Show agent details
cognito-sim agent show research_assistant

# Update agent configuration
cognito-sim agent update research_assistant \
  --personality conscientiousness:0.95 \
  --add-goal "publish_research_paper"

# Clone an existing agent
cognito-sim agent clone research_assistant \
  --name "research_assistant_v2" \
  --modify personality.openness:0.9

# Delete agent
cognito-sim agent delete research_assistant --confirm
```

#### Agent Capabilities

```bash
# Test agent reasoning
cognito-sim agent test-reasoning research_assistant \
  --problem "How to improve machine learning model accuracy?" \
  --facts "current_accuracy:0.85,dataset_size:10000" \
  --output reasoning_test.json

# Evaluate agent memory
cognito-sim agent test-memory research_assistant \
  --memory-type episodic \
  --query "research experiences" \
  --output memory_test.json

# Test agent goal processing
cognito-sim agent test-goals research_assistant \
  --scenario "research_deadline_approaching" \
  --output goal_test.json
```

### 2. Environment Management

#### Create Environments

```bash
# Create a research laboratory environment
cognito-sim environment create \
  --name "ai_research_lab" \
  --type collaborative \
  --size 1000 \
  --resources "computing_cluster,datasets,libraries" \
  --dynamics "knowledge_sharing,peer_review" \
  --output environments/ai_lab.json

# Create a learning environment
cognito-sim environment create \
  --name "online_classroom" \
  --type educational \
  --capacity 30 \
  --learning-resources "lectures,assignments,forums" \
  --assessment-system "automated" \
  --collaboration "study_groups"

# Create competitive environment
cognito-sim environment create \
  --name "ml_competition" \
  --type competitive \
  --competition-type "kaggle_style" \
  --evaluation-metric "accuracy" \
  --time-limit "7_days" \
  --leaderboard "public"
```

#### Manage Environments

```bash
# List environments
cognito-sim environment list

# Show environment details
cognito-sim environment show ai_research_lab

# Update environment
cognito-sim environment update ai_research_lab \
  --add-resource "new_gpu_cluster" \
  --modify dynamics.collaboration_frequency:0.8

# Add agents to environment
cognito-sim environment add-agents ai_research_lab \
  research_assistant ml_student researcher

# Remove agents from environment
cognito-sim environment remove-agents ai_research_lab \
  ml_student
```

#### Environment Monitoring

```bash
# Monitor environment state
cognito-sim environment monitor ai_research_lab \
  --duration 3600 \
  --interval 60 \
  --metrics "agent_interactions,knowledge_exchange,goal_progress" \
  --output monitoring_log.json

# Generate environment report
cognito-sim environment report ai_research_lab \
  --period "last_week" \
  --include-agents \
  --include-interactions \
  --output reports/lab_report.html
```

### 3. Simulation Management

#### Run Simulations

```bash
# Run basic simulation
cognito-sim simulation run \
  --environment ai_research_lab \
  --agents research_assistant,ml_student \
  --duration 3600 \
  --output simulation_results.json

# Run educational simulation
cognito-sim simulation run \
  --environment online_classroom \
  --scenario "machine_learning_course" \
  --duration 7200 \
  --real-time-factor 0.1 \
  --save-state simulation_state.pkl

# Run competition simulation
cognito-sim simulation run \
  --environment ml_competition \
  --scenario "computer_vision_challenge" \
  --participants 10 \
  --time-limit 604800 \
  --evaluation-frequency 3600
```

#### Advanced Simulation Options

```bash
# Run simulation with custom configuration
cognito-sim simulation run \
  --config simulations/research_study_config.yaml \
  --parameters "learning_rate:0.01,exploration_factor:0.1" \
  --checkpoint-interval 600 \
  --resume-from checkpoint_001.pkl

# Run batch simulations
cognito-sim simulation batch \
  --config-template templates/experiment_template.yaml \
  --parameter-grid parameters/grid_search.yaml \
  --parallel-jobs 4 \
  --output-dir batch_results/

# Run interactive simulation
cognito-sim simulation interactive \
  --environment ai_research_lab \
  --agents research_assistant \
  --step-mode \
  --debug-mode
```

#### Simulation Control

```bash
# Pause simulation
cognito-sim simulation pause simulation_001

# Resume simulation
cognito-sim simulation resume simulation_001

# Stop simulation
cognito-sim simulation stop simulation_001

# Get simulation status
cognito-sim simulation status simulation_001

# List running simulations
cognito-sim simulation list --status running
```

### 4. Memory and Knowledge Management

#### Memory Operations

```bash
# Import knowledge into agent memory
cognito-sim memory import research_assistant \
  --source "knowledge_base.json" \
  --memory-type semantic \
  --confidence-threshold 0.7

# Export agent memory
cognito-sim memory export research_assistant \
  --memory-types "episodic,semantic" \
  --format json \
  --output agent_memory_backup.json

# Search agent memory
cognito-sim memory search research_assistant \
  --query "machine learning algorithms" \
  --memory-types "all" \
  --max-results 20

# Clean up agent memory
cognito-sim memory cleanup research_assistant \
  --remove-duplicates \
  --confidence-threshold 0.3 \
  --age-threshold 86400
```

#### Knowledge Base Management

```bash
# Create knowledge base
cognito-sim knowledge create \
  --name "ml_knowledge_base" \
  --domain "machine_learning" \
  --sources "textbooks,papers,tutorials" \
  --structure "hierarchical"

# Add knowledge to base
cognito-sim knowledge add ml_knowledge_base \
  --source "new_research_papers.json" \
  --validate \
  --update-existing

# Query knowledge base
cognito-sim knowledge query ml_knowledge_base \
  --question "What are the best practices for neural network training?" \
  --context "beginner_level" \
  --format "summary"

# Share knowledge base with agents
cognito-sim knowledge share ml_knowledge_base \
  --agents "research_assistant,ml_student" \
  --access-level "read_write"
```

### 5. Analysis and Reporting

#### Generate Reports

```bash
# Agent performance report
cognito-sim report agent-performance research_assistant \
  --period "last_month" \
  --metrics "goal_achievement,learning_progress,interaction_quality" \
  --format html \
  --output reports/agent_performance.html

# Simulation analysis report
cognito-sim report simulation-analysis simulation_001 \
  --include-agent-behaviors \
  --include-environment-dynamics \
  --include-goal-progression \
  --output reports/simulation_analysis.pdf

# Comparative analysis
cognito-sim report compare \
  --agents "research_assistant,ml_student" \
  --period "last_week" \
  --metrics "reasoning_efficiency,memory_usage,goal_achievement" \
  --output reports/agent_comparison.html
```

#### Data Export

```bash
# Export simulation data
cognito-sim export simulation simulation_001 \
  --format csv \
  --include "agent_states,interactions,events" \
  --output data/simulation_001_export.csv

# Export agent data
cognito-sim export agent research_assistant \
  --format json \
  --include "memory,goals,personality,history" \
  --output data/research_assistant_export.json

# Export environment data
cognito-sim export environment ai_research_lab \
  --format yaml \
  --include "configuration,agents,resources,dynamics" \
  --output data/ai_lab_export.yaml
```

### 6. Configuration Management

#### Configuration Files

```bash
# Generate default configuration
cognito-sim config generate \
  --type full \
  --output cognito_config.yaml

# Validate configuration
cognito-sim config validate cognito_config.yaml

# Show current configuration
cognito-sim config show

# Set configuration values
cognito-sim config set \
  --key "simulation.default_duration" \
  --value "3600"

# Reset configuration to defaults
cognito-sim config reset --confirm
```

#### Profile Management

```bash
# Create configuration profile
cognito-sim profile create research_profile \
  --base-config research_config.yaml \
  --description "Configuration for research simulations"

# Use profile
cognito-sim --profile research_profile simulation run \
  --environment ai_research_lab

# List profiles
cognito-sim profile list

# Delete profile
cognito-sim profile delete research_profile --confirm
```

## Advanced Usage Examples

### 1. Research Study Simulation

```bash
# Set up complete research study
#!/bin/bash

# Create research environment
cognito-sim environment create \
  --name "cognitive_research_lab" \
  --type collaborative \
  --resources "compute_cluster,datasets,visualization_tools" \
  --dynamics "peer_review,knowledge_sharing,hypothesis_testing"

# Create diverse research team
for i in {1..5}; do
  cognito-sim agent create \
    --name "researcher_$i" \
    --type cognitive \
    --specialization "research" \
    --personality "openness:0.9,conscientiousness:0.8" \
    --reasoning-strategies "analytical,creative" \
    --goals "conduct_research,publish_papers,collaborate"
done

# Add agents to environment
cognito-sim environment add-agents cognitive_research_lab \
  researcher_1 researcher_2 researcher_3 researcher_4 researcher_5

# Run research simulation
cognito-sim simulation run \
  --environment cognitive_research_lab \
  --scenario "agi_research_project" \
  --duration 86400 \
  --checkpoint-interval 3600 \
  --output research_simulation.json

# Generate comprehensive report
cognito-sim report simulation-analysis research_simulation \
  --include-collaboration-patterns \
  --include-knowledge-evolution \
  --include-breakthrough-events \
  --output reports/research_study_results.html
```

### 2. Educational Assessment

```bash
# Educational simulation with assessment
#!/bin/bash

# Create classroom environment
cognito-sim environment create \
  --name "ml_classroom" \
  --type educational \
  --capacity 20 \
  --curriculum "machine_learning_fundamentals" \
  --assessment-system "continuous"

# Create diverse student agents
cognito-sim agent create-batch \
  --template templates/student_template.yaml \
  --count 20 \
  --personality-variation "high" \
  --learning-rate-range "0.001,0.1" \
  --output-dir agents/students/

# Create instructor agent
cognito-sim agent create \
  --name "ml_instructor" \
  --type teaching \
  --expertise "machine_learning" \
  --teaching-style "adaptive" \
  --assessment-capability "comprehensive"

# Run educational simulation
cognito-sim simulation run \
  --environment ml_classroom \
  --scenario "ml_course_semester" \
  --duration 2592000 \
  --real-time-factor 0.001 \
  --save-checkpoints

# Analyze learning outcomes
cognito-sim report educational-assessment ml_classroom \
  --metrics "learning_progress,engagement,collaboration" \
  --individual-reports \
  --output reports/educational_assessment/
```

### 3. Cognitive Architecture Testing

```bash
# Test different cognitive architectures
#!/bin/bash

# Create test environment
cognito-sim environment create \
  --name "cognitive_test_arena" \
  --type experimental \
  --challenges "reasoning,memory,learning,adaptation"

# Create agents with different architectures
architectures=("symbolic" "connectionist" "hybrid" "emergent")

for arch in "${architectures[@]}"; do
  cognito-sim agent create \
    --name "agent_${arch}" \
    --architecture "$arch" \
    --reasoning-depth 10 \
    --memory-capacity 50000 \
    --learning-rate 0.01 \
    --goals "solve_challenges,adapt_strategies"
done

# Run comparative tests
cognito-sim simulation batch \
  --environment cognitive_test_arena \
  --scenarios "cognitive_challenges.yaml" \
  --agents "agent_symbolic,agent_connectionist,agent_hybrid,agent_emergent" \
  --repetitions 10 \
  --output-dir architecture_comparison/

# Generate comparative analysis
cognito-sim report architecture-comparison \
  --simulation-set architecture_comparison/ \
  --metrics "reasoning_efficiency,memory_utilization,learning_speed,adaptability" \
  --statistical-analysis \
  --output reports/architecture_comparison.html
```

## CLI Best Practices

### 1. Configuration Management

```bash
# Use configuration files for complex setups
cognito-sim --config production_config.yaml simulation run

# Use profiles for different use cases
cognito-sim --profile research_profile agent create

# Validate configurations before use
cognito-sim config validate custom_config.yaml
```

### 2. Resource Management

```bash
# Monitor system resources during long simulations
cognito-sim simulation run --monitor-resources

# Use batch processing for multiple experiments
cognito-sim simulation batch --parallel-jobs 4

# Save checkpoints for long-running simulations
cognito-sim simulation run --checkpoint-interval 600
```

### 3. Data Management

```bash
# Regular backups of important agents and environments
cognito-sim export agent important_agent --output backups/

# Clean up old simulation data
cognito-sim cleanup --older-than 30d --simulation-data

# Compress large datasets
cognito-sim export simulation large_sim --compress --output compressed/
```

### 4. Debugging and Development

```bash
# Use debug mode for development
cognito-sim --debug simulation run

# Interactive mode for testing
cognito-sim simulation interactive --step-mode

# Verbose logging for troubleshooting
cognito-sim --log-level DEBUG --verbose simulation run
```

## Integration with Other Tools

### 1. Jupyter Notebooks

```bash
# Export simulation data for Jupyter analysis
cognito-sim export simulation sim_001 --format jupyter

# Generate notebook template for analysis
cognito-sim generate notebook-template \
  --simulation sim_001 \
  --analysis-type "agent_behavior" \
  --output analysis_template.ipynb
```

### 2. External Data Sources

```bash
# Import data from external sources
cognito-sim import data \
  --source "external_dataset.csv" \
  --target-agent research_assistant \
  --memory-type semantic

# Connect to databases
cognito-sim connect database \
  --type postgresql \
  --connection-string "postgresql://user:pass@host:port/db" \
  --agent research_assistant
```

### 3. Visualization Tools

```bash
# Generate visualization data
cognito-sim export visualization sim_001 \
  --type "network_analysis" \
  --output viz_data.json

# Create interactive dashboards
cognito-sim generate dashboard sim_001 \
  --metrics "agent_interactions,goal_progress,memory_usage" \
  --output dashboard.html
```

## Troubleshooting

### Common Issues

```bash
# Check system requirements
cognito-sim doctor

# Validate agent configurations
cognito-sim agent validate research_assistant

# Test environment connectivity
cognito-sim environment test ai_research_lab

# Debug simulation issues
cognito-sim simulation debug sim_001 --verbose
```

### Performance Optimization

```bash
# Profile simulation performance
cognito-sim simulation profile \
  --environment test_env \
  --duration 300 \
  --output performance_profile.json

# Optimize agent configurations
cognito-sim agent optimize research_assistant \
  --metric "reasoning_efficiency" \
  --output optimized_config.json

# Memory usage analysis
cognito-sim memory analyze research_assistant \
  --report memory_usage_report.html
```

---

The CLI provides comprehensive tools for managing all aspects of cognitive simulations. Use these commands to create sophisticated research studies, educational simulations, and cognitive architecture experiments.

**Next**: Explore [API Reference](../api/overview.md) for programmatic access, or see [Examples](../examples/overview.md) for complete simulation scenarios.
