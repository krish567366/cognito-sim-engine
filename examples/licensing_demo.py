#!/usr/bin/env python3
"""
Example script demonstrating Cognito Simulation Engine licensing integration.

This script shows how the licensing system protects different features
and provides clear error messages with machine ID and support contact.
"""

from cognito_sim_engine import (
    CognitiveEngine, 
    CognitiveAgent, 
    CognitoLicenseError,
    get_license_info,
    display_license_info,
    get_machine_id
)

def main():
    print("🧠 Cognito Simulation Engine - Licensing Demo")
    print("=" * 50)
    
    # Display current license status
    print("\n1. Current License Status:")
    try:
        display_license_info()
    except Exception as e:
        print(f"Error displaying license info: {e}")
    
    print(f"\n2. Machine ID: {get_machine_id()}")
    
    # Test basic functionality (Core license)
    print("\n3. Testing Core Features:")
    try:
        # This should work with Core license or during 1-day grace period
        agent = CognitiveAgent(
            agent_id="demo_agent_001",
            name="Demo Agent",
            license_tier="core"
        )
        print("✅ Core agent created successfully")
        
        engine = CognitiveEngine(license_tier="core")
        print("✅ Core engine created successfully")
        
    except CognitoLicenseError as e:
        print(f"❌ Core feature failed: {e}")
    
    # Test Pro features
    print("\n4. Testing Pro Features:")
    try:
        # This requires Pro license
        pro_agent = CognitiveAgent(
            agent_id="pro_agent_001",
            name="Pro Agent",
            license_tier="pro"
        )
        
        # Test advanced reasoning
        result = pro_agent.advanced_reasoning(
            problem="How to optimize neural network training?",
            reasoning_depth=20
        )
        print("✅ Pro advanced reasoning successful")
        print(f"   Result: {result.get('success', False)}")
        
    except CognitoLicenseError as e:
        print(f"❌ Pro feature failed:")
        print(f"   {str(e)}")
    
    # Test Research features
    print("\n5. Testing Research Features:")
    try:
        research_agent = CognitiveAgent(
            agent_id="research_agent_001",
            name="Research Agent",
            license_tier="research"
        )
        
        insights = research_agent.generate_research_insights(
            domain="artificial_intelligence"
        )
        print("✅ Research insights generated successfully")
        print(f"   Insights: {len(insights.get('insights', []))}")
        
    except CognitoLicenseError as e:
        print(f"❌ Research feature failed:")
        print(f"   {str(e)}")
    
    # Test Enterprise features
    print("\n6. Testing Enterprise Features:")
    try:
        enterprise_engine = CognitiveEngine(license_tier="enterprise")
        
        result = enterprise_engine.run_distributed_simulation(
            worker_count=8,
            load_balancing="dynamic"
        )
        print("✅ Enterprise distributed simulation initiated")
        
    except CognitoLicenseError as e:
        print(f"❌ Enterprise feature failed:")
        print(f"   {str(e)}")
    
    # Test collaboration (Enterprise feature)
    print("\n7. Testing Agent Collaboration:")
    try:
        agent1 = CognitiveAgent("collab_agent_1", license_tier="enterprise")
        agent2 = CognitiveAgent("collab_agent_2", license_tier="enterprise")
        
        collab_result = agent1.collaborate_with_agents(
            other_agents=[agent2],
            collaboration_goal="Develop new AI architecture"
        )
        print("✅ Agent collaboration successful")
        print(f"   Collaboration ID: {collab_result.get('collaboration_id')}")
        
    except CognitoLicenseError as e:
        print(f"❌ Collaboration feature failed:")
        print(f"   {str(e)}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTo activate a license:")
    print("1. Contact support: bajpaikrishna715@gmail.com")
    print(f"2. Provide your Machine ID: {get_machine_id()}")
    print("3. Use: cogsim activate-license <license_file>")


if __name__ == "__main__":
    main()
