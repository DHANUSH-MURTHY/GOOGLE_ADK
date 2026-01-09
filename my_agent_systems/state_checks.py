def validate_attractions_state(agent, user_input):
    response = agent.run(user_input)

    state = response.session.state
    attractions = state.get("attractions", [])

    return {
        "saved": bool(attractions),
        "count": len(attractions),
        "attractions": attractions
    }
