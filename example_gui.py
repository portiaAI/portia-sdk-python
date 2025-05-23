#!/usr/bin/env python3
"""Example script demonstrating the Portia Terminal GUI.

This script shows how to use the Portia Terminal GUI with different
configurations and queries.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path so we can import our GUI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portia_gui import run_portia_gui
from portia import Config, LogLevel, Portia, example_tool_registry
from portia.errors import InvalidConfigError


def check_api_keys():
    """Check if necessary API keys are configured."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    
    if not (has_openai or has_anthropic or has_google):
        print("❌ Error: No LLM API keys found!")
        print("\nTo use the Portia GUI, you need to set up at least one LLM provider:")
        print("\n1. Create a .env file in this directory with one of:")
        print("   OPENAI_API_KEY=your_openai_api_key")
        print("   ANTHROPIC_API_KEY=your_anthropic_api_key")
        print("   GOOGLE_API_KEY=your_google_api_key")
        print("\n2. Or export the environment variable:")
        print("   export OPENAI_API_KEY=your_openai_api_key")
        print("\n3. Get API keys from:")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        print("   - Anthropic: https://console.anthropic.com/")
        print("   - Google AI: https://makersuite.google.com/app/apikey")
        return False
    
    print("✅ API key found, proceeding...")
    return True


def create_portia_safely():
    """Create a Portia instance with error handling."""
    try:
        return Portia(
            Config.from_default(default_log_level=LogLevel.INFO),
            tools=example_tool_registry,
        )
    except InvalidConfigError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nThis usually means you need to set up API keys for an LLM provider.")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None


def main():
    """Main function to run different examples."""
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 Portia Terminal GUI - Example Script")
    print("=" * 50)
    
    # Check API keys first
    if not check_api_keys():
        return
    
    # Try to create Portia instance
    portia = create_portia_safely()
    if not portia:
        return
    
    # Example queries
    queries = {
        "1": {
            "name": "Simple math calculation",
            "query": "Add 42 and 58, then multiply the result by 3",
            "description": "Basic arithmetic operations"
        },
        "2": {
            "name": "Weather information query",
            "query": "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
            "description": "Weather data retrieval and calculation (requires weather tools)"
        },
        "3": {
            "name": "Complex multi-step query",
            "query": "Please calculate the square root of 144, then add 10 to it, and finally tell me if the result is greater than 20",
            "description": "Multi-step mathematical reasoning"
        }
    }
    
    # Display available examples
    print("\nChoose an example to run:")
    for key, example in queries.items():
        print(f"{key}. {example['name']}")
        print(f"   Query: {example['query']}")
        print(f"   Description: {example['description']}")
        print()
    
    print("4. Custom query (enter your own)")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice in queries:
            example = queries[choice]
            print(f"\n🎯 Running: {example['name']}")
            print(f"📝 Query: {example['query']}")
            print("\n🖥️  Starting GUI... (Press Ctrl+C in the GUI to quit)")
            run_portia_gui(portia, example['query'])
            
        elif choice == "4":
            custom_query = input("\n📝 Enter your custom query: ").strip()
            if custom_query:
                print(f"\n🎯 Running custom query: {custom_query}")
                print("\n🖥️  Starting GUI... (Press Ctrl+C in the GUI to quit)")
                run_portia_gui(portia, custom_query)
            else:
                print("❌ Empty query. Exiting.")
                
        else:
            print("❌ Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 