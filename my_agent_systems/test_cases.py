# Test cases for travel_brainstormer
travel_brainstormer_tests = [
    {
        "input": "I want adventure and nature",
        "expected_keywords": ["mountains", "adventure", "trek", "hiking"]
    },
    {
        "input": "I want shopping and luxury",
        "expected_keywords": ["shopping", "luxury", "malls"]
    }
]

# Test cases for attractions_planner
attractions_planner_tests = [
    {
        "input": "I want to visit Japan",
        "expected_keywords": ["Tokyo", "Kyoto", "Mount Fuji"]
    },
    {
        "input": "Show me attractions in France",
        "expected_keywords": ["Paris", "Eiffel", "Louvre"]
    }
]

# Test cases for root agent routing
root_agent_tests = [
    {
        "input": "Help me decide where to travel",
        "expected_agent": "travel_brainstormer"
    },
    {
        "input": "I want to visit Italy",
        "expected_agent": "attractions_planner"
    }
]
