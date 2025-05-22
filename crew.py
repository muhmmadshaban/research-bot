from crewai import Agent, Task, Process, Crew
from tools import google_search_tool
from agents import (
    introduction_agent,
    literature_review_agent,
    methodology_agent,
    implementation_agent,
    results_analytics_agent,
    reference_agent,
    proof_agent
)
from tasks import (
    introduction_task,
    literature_review_task,
    methodology_task,
    implementation_task,
    results_analytics_task,
    reference_task,
    proof_read_task
)

# Initialize the Crew with agents and tasks
crew = Crew(
    agents=[
        introduction_agent,
        literature_review_agent,
        methodology_agent,
        implementation_agent,
        results_analytics_agent,
        reference_agent,
        proof_agent
    ],
    tasks=[
        introduction_task,
        literature_review_task,
        methodology_task,
        implementation_task,
        results_analytics_task,
        reference_task,
        proof_read_task
    ],
    process=Process.sequential
)

# Define the research topic
topic = "Ai in Healthcare"

try:
    # Kick off the crew process with the specified topic
    result = crew.kickoff(inputs={"topic": topic})
    print(result)  # Output the result of the process
except Exception as e:
    print(f"An error occurred: {e}")
