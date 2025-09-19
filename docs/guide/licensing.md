# Licensing System

The Cognito Simulation Engine includes a comprehensive licensing system powered by QuantumMeta License Manager. This system ensures proper usage authorization while providing clear guidance for license activation and support.

## Overview

The licensing system protects different tiers of functionality:

- **Core**: Basic cognitive simulation features
- **Pro**: Advanced reasoning and memory systems  
- **Enterprise**: Large-scale simulations and distributed computing
- **Research**: Academic research and publication features

## Machine ID

Every installation generates a unique Machine ID that is used for license validation and support. You can retrieve your Machine ID using:

```python
from cognito_sim_engine import get_machine_id
print(f"Machine ID: {get_machine_id()}")
```

Or via CLI:

```bash
cogsim license-info
```

## License Integration

### Class-Level Licensing

The main classes inherit from `LicensedClass` and validate licenses during instantiation:

```python
from cognito_sim_engine import CognitiveEngine, CognitiveAgent

# Core license required
engine = CognitiveEngine(license_tier="core")
agent = CognitiveAgent("agent_001", license_tier="core")

# Pro license required  
pro_agent = CognitiveAgent("pro_agent", license_tier="pro")

# Enterprise license required
enterprise_engine = CognitiveEngine(license_tier="enterprise")
```

### Method-Level Licensing

Specific methods require higher license tiers:

```python
# Pro license required for advanced reasoning
result = agent.advanced_reasoning(
    problem="Complex AI problem",
    reasoning_depth=20
)

# Research license required for research insights
insights = agent.generate_research_insights(
    domain="artificial_intelligence"
)

# Enterprise license required for collaboration
collab_result = agent.collaborate_with_agents(
    other_agents=[other_agent],
    collaboration_goal="Joint research project"
)
```

## Error Handling

License errors provide comprehensive information:

```python
from cognito_sim_engine import CognitoLicenseError

try:
    pro_agent = CognitiveAgent("agent", license_tier="pro")
except CognitoLicenseError as e:
    print(f"License Error: {e}")
    # Error includes:
    # - Machine ID
    # - Support contact (bajpaikrishna715@gmail.com)
    # - Error code
    # - Clear resolution steps
```

## CLI Commands

### License Information

```bash
cogsim license-info
```

Displays:

- Current license status
- Machine ID
- Available features
- Support contact information

### License Activation

```bash
cogsim activate-license /path/to/license.qkey
```

## License Tiers

### Core License

- Basic cognitive agent creation
- Simple reasoning and memory operations
- Basic environment interactions
- Standard simulation capabilities

### Pro License

- Advanced reasoning with enhanced depth
- Sophisticated memory management
- Performance analytics and optimization
- Advanced metacognitive capabilities

### Enterprise License

- Large-scale distributed simulations
- Multi-agent collaboration systems
- Load balancing and cluster computing
- Enterprise-grade analytics

### Research License

- Academic research capabilities
- Research insight generation
- Hypothesis formation tools
- Publication support features

## Support and Contact

For licensing questions, activation issues, or technical support:

**Email**: bajpaikrishna715@gmail.com

**Required Information**:

- Your Machine ID (use `cogsim license-info`)
- Description of the issue
- License tier needed
- Intended use case

## Security Features

- **No Development Mode**: No bypass mechanisms in production
- **Machine Binding**: Licenses are tied to specific machine IDs
- **Secure Validation**: Uses QuantumMeta's secure validation system
- **Grace Period**: 1-day grace period for license renewal
- **Tamper Protection**: License validation cannot be bypassed

## Example Usage

```python
#!/usr/bin/env python3
from cognito_sim_engine import (
    CognitiveEngine, 
    CognitiveAgent,
    CognitoLicenseError,
    get_machine_id
)

def main():
    print(f"Machine ID: {get_machine_id()}")
    
    try:
        # Basic usage (Core license)
        agent = CognitiveAgent("demo_agent", license_tier="core")
        print("✅ Core agent created")
        
        # Advanced features (Pro license)
        result = agent.advanced_reasoning("AI problem")
        print("✅ Pro reasoning completed")
        
    except CognitoLicenseError as e:
        print(f"License required: {e}")
        print("Contact: bajpaikrishna715@gmail.com")

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **License Not Found**
   - Contact support with your Machine ID
   - Ensure license file is properly activated

2. **Feature Not Licensed**
   - Upgrade to appropriate license tier
   - Contact sales for license upgrade

3. **License Expired**
   - Renew license through support
   - Contact bajpaikrishna715@gmail.com

4. **Installation Issues**
   - Ensure `quantummeta-license` is installed
   - Check Python version compatibility (3.9+)

### Getting Help

Always include your Machine ID when contacting support:

```python
from cognito_sim_engine import get_machine_id
print(f"My Machine ID: {get_machine_id()}")
```

Contact: **bajpaikrishna715@gmail.com**
