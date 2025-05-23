
# 🧠 Multi-Agent Research Assistant using CrewAI, LangChain & Groq

This project is a sophisticated AI-powered multi-agent research assistant built with [CrewAI](https://docs.crewai.com/), [LangChain](https://www.langchain.com/), and [Groq](https://groq.com/) models via `ChatGroq`. It automates research tasks such as writing introductions, literature reviews, methodology design, implementation planning, result analysis, citation management, and proofreading.

## 🚀 Features

- 🔍 **Introduction Agent**: Crafts engaging, contextualized research introductions.
- 📚 **Literature Review Agent**: Synthesizes existing research and identifies gaps.
- 🧪 **Methodology Agent**: Designs replicable and justified research methods.
- 🛠️ **Implementation Agent**: Coordinates execution of the research plan.
- 📊 **Results Analytics Agent**: Analyzes data and highlights key findings.
- 📖 **Reference Agent**: Manages citations and ensures academic integrity.
- 📝 **Proof Agent**: Finalizes content with proofreading and additional references.

## 🧩 Tech Stack

- 🧠 [CrewAI](https://docs.crewai.com/): Multi-agent orchestration framework.
- 🔗 [LangChain](https://www.langchain.com/): LLM framework for structured prompts.
- ⚡ [ChatGroq](https://groq.com/): Ultra-fast inference using Groq hardware and LLMs (like LLaMA 3).
- 🌐 Google Search Tool: Integrated for live web research.

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/muhmmadshaban/research-bot
cd research-bot
```

### 2. Create a `.env` File

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> Make sure to include these packages in `requirements.txt`:  
> `crewai`, `langchain`, `langchain-groq`, `python-dotenv`, and any others used.

### 4. Run the Agents

You can instantiate and use agents individually, or orchestrate them using a task manager such as `Crew` (not shown in code but available in CrewAI):

```python
from agents import introduction_agent, literature_review_agent, ...

result = introduction_agent.run(input="Your research topic here")
print(result)
```

## 🔧 Custom LLM Wrapper

To ensure compatibility with CrewAI and LangChain, a custom `GroqLLMWrapper` class wraps the `ChatGroq` API and manages message formatting and invocation.

## 🧠 Agent Responsibilities

Each agent has a specific research goal and a role, along with contextual backstories to improve alignment and task fidelity.

## 📎 Tools

Most agents use a `google_search_tool` to fetch relevant information dynamically.

## 📌 Notes

- You must have access to Groq API to use `ChatGroq`.
- Internet access is simulated via tools (like `google_search_tool`) – actual results depend on your implementation.
- You can extend this setup using `Crew` to coordinate the agents as a team.

## 📜 License

This project is open-source under the [MIT License](LICENSE).
