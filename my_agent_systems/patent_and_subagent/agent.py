import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from callback_logging import log_query_to_model, log_model_response
except ImportError:
    import traceback
    traceback.print_exc()
    print(f"Sys path: {sys.path}")
    raise

from dotenv import load_dotenv
from google.adk import Agent
from google.genai import types
from typing import Optional, List, Dict

from google.adk.tools.tool_context import ToolContext

load_dotenv()

# Ensure MODEL environment variable is set
MODEL = os.getenv("MODEL")
if not MODEL:
    raise ValueError("MODEL environment variable is not set")


# Tools 
def save_attractions_to_state(
    tool_context: ToolContext,
    attractions: List[str]
) -> dict[str, str]:
    """Saves the list of attractions to state["attractions"].

    Args:
        attractions [str]: a list of strings to add to the list of attractions

    Returns:
        None
    """
    # Load existing attractions from state. If none exist, start an empty list
    existing_attractions = tool_context.state.get("attractions", [])

    # Update the 'attractions' key with a combo of old and new lists.
    # corresponding updates in the session's state.
    tool_context.state["attractions"] = existing_attractions + attractions

    return {"status": "success"}


# Agents

attractions_planner = Agent(
    name="attractions_planner",
    model=MODEL,
    description="Build a list of attractions to visit in a country.",
    instruction="""
        - Provide the user options for attractions to visit within their selected country.
        - When they reply, use your tool to save their selected attraction
             and then provide more possible attractions.
        - If they ask to view the list, provide a bulleted list of
            { attractions? } and then suggest some more.""",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    # tools parameter 
    tools=[save_attractions_to_state]

    )

travel_brainstormer = Agent(
    name="travel_brainstormer",
    model=MODEL,
    description="Help a user decide what country to visit.",
    instruction="""
        Provide a few suggestions of popular countries for travelers.
        
        Help a user identify their primary goals of travel:
        adventure, leisure, learning, shopping, or viewing art

        Identify countries that would make great destinations
        based on their priorities.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

root_agent = Agent(
    name="steering",
    model=MODEL,
    description="Start a user on a travel adventure.",
    instruction="""
        Ask the user if they know where they'd like to travel
        or if they need some help deciding.If they need help deciding, send them to
        'travel_brainstormer'.
        If they know what country they'd like to visit,
        send them to the 'attractions_planner'.
        """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    sub_agents=[travel_brainstormer, attractions_planner]

)
