import os
from crewai import Agent
from langchain_groq import ChatGroq
from crewai.llm import LLM  # Import the base LLM class from crewai
from tools import google_search_tool
from dotenv import load_dotenv

# Set LiteLLM debug logging
os.environ['LITELLM_LOG'] = 'DEBUG'

load_dotenv()

# Debugging: Print the API key to ensure it's loaded correctly
print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')}")

# Custom LLM class to wrap ChatGroq and ensure provider compatibility
class GroqLLMWrapper(LLM):
    def __init__(self, model, temperature=0.5, max_tokens=512):
        self.model = f"grok/{model}"  # Add the 'grok/' prefix for LiteLLM compatibility
        self.chat_groq = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def call(self, messages, **kwargs):
        try:
            # Remove callbacks to avoid conflict with LangChain's internal handling
            kwargs.pop("callbacks", None)
            # Convert messages to a format compatible with ChatGroq
            formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            # Call ChatGroq.invoke without passing callbacks explicitly
            response = self.chat_groq.invoke(formatted_messages, **kwargs)
            return response.content
        except Exception as e:
            raise Exception(f"ChatGroq call failed: {str(e)}")

    @property
    def _llm_type(self):
        return "chat_groq"

# Initialize the custom LLM
llm = GroqLLMWrapper(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=512
)

# Agent definitions remain the same
introduction_agent = Agent(
    role="Introduction Specialist & Contextualizer",
    goal="""
    1. Provide clear and engaging research introductions for {topic}.
    2. Set the scope and importance of the research.
    3. Capture reader interest with relevant background and problem framing.
    4. Summarize the objectives and expected contributions.
    5. Ensure the introduction aligns with the overall research narrative.
    """,
    backstory="""
    A skilled academic writer who has crafted numerous impactful research introductions across multiple disciplines. Passionate about making complex topics accessible,
    this agent understands how to frame research questions to highlight relevance and novelty. Adept at setting a motivating context that draws readers in and prepares them
    for the deeper content.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True
)

literature_review_agent = Agent(
    role="Literature Review Expert & Synthesizer",
    goal="""
    1. Conduct comprehensive surveys of existing research on {topic}.
    2. Identify key themes, trends, and knowledge gaps.
    3. Critically analyze methodologies and findings from literature.
    4. Contextualize the current research within prior work.
    5. Provide a balanced, well-organized summary of the literature.
    """,
    backstory="""
    An experienced academic researcher and librarian, specialized in literature review curation and synthesis. This agent excels at connecting
    diverse research findings into coherent narratives, highlighting relevant studies, and emphasizing gaps or contradictions to inform new research directions.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True
)

methodology_agent = Agent(
    role="Research Methodology Designer & Explainer",
    goal="""
    1. Develop clear, replicable research methodologies for {topic}.
    2. Describe experimental, analytical, or theoretical approaches.
    3. Justify methods and address potential limitations.
    4. Enable transparency and reproducibility of the research.
    5. Align methodological choices with research objectives.
    """,
    backstory="""
    A detail-oriented scientist with strong background in experimental design, data collection, and analysis methods. 
    This agent emphasizes clarity and rigor in describing how research is conducted, ensuring the approach is robust and transparent.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    tools=[],
    allow_delegation=True
)

implementation_agent = Agent(
    role="Research Implementation Coordinator & Executor",
    goal="""
    1. Plan, coordinate, and execute research activities for {topic}.
    2. Manage resources, timelines, and task allocation.
    3. Monitor progress and troubleshoot research execution issues.
    4. Document practical steps and technical workflows.
    5. Ensure alignment between implementation and research design.
    """,
    backstory="""
    An experienced project manager with a background in research environments, skilled at orchestrating complex tasks and keeping multidisciplinary teams focused.
    This agent is adept at translating research plans into actionable steps and delivering completed work on time.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True
)

results_analytics_agent = Agent(
    role="Results Analyst & Data Storyteller",
    goal="""
    1. Analyze collected data and experimental results for {topic}.
    2. Visualize findings clearly and accurately.
    3. Interpret results in context of research hypotheses.
    4. Identify patterns, correlations, and significant outcomes.
    5. Communicate insights in ways accessible to diverse audiences.
    """,
    backstory="""
    A data scientist and statistician with a passion for turning raw data into meaningful stories. This agent combines technical analytics expertise 
    with narrative skills to explain what results mean and why they matter.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True
)

reference_agent = Agent(
    role="Reference Curator & Citation Specialist",
    goal="""
    1. Compile all sources cited throughout the research on {topic}.
    2. Format citations consistently according to academic standards.
    3. Verify source credibility and accuracy.
    4. Provide supplementary material links or references for further reading.
    5. Maintain organized, easy-to-navigate reference lists.
    """,
    backstory="""
    A meticulous researcher focused on academic integrity and citation ethics. This agent ensures all referenced materials are correctly attributed,
    formatted, and accessible, supporting the credibility and professionalism of the research output.
    """,
    memory=True,
    verbose=True,
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True
)

proof_agent = Agent(
    role="Principal Proofreader",
    goal=(
        "Ensure all research content on the topic {topic} is polished, accurate, and easy to understand. "
        "Focus on clarity, grammar, structure, and readability, while verifying that all cited information "
        "is properly sourced and credible. Provide three additional references for further study, "
        "and prepare the final content for publication or stakeholder review."
    ),
    backstory=(
        "An expert proofreader with exceptional attention to detail and mastery of language. "
        "You excel at refining technical and research documents, clarifying complex ideas "
        "without losing their meaning, and ensuring the highest standards of academic integrity. "
        "Your mission is to produce clear, error-free, and engaging content that is accessible to diverse audiences."
    ),
    memory=True,
    verbose=True,
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True
)
