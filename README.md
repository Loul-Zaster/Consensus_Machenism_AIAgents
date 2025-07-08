# Consensus Mechanism AI Agents for Medical Diagnosis

A multi-agent system for medical diagnosis using LangGraph framework and IO.net Intelligence API.

## Overview

This project implements a consensus mechanism for medical diagnosis using multiple AI agents. Each agent specializes in a different aspect of the medical diagnosis process, and they work together to provide a comprehensive diagnosis and treatment plan.

The system leverages the LangGraph framework for agent orchestration and the IO.net Intelligence API for high-quality language model capabilities.

## Features

- Multiple specialized agents (researcher, diagnostician, treatment advisor, consensus builder)
- Consensus mechanism to synthesize opinions from multiple agents
- Integration with IO.net Intelligence API or OpenAI API
- Support for medical research, diagnosis, and treatment recommendations
- **Real-time web search** for up-to-date medical information
- **Source verification** to evaluate the credibility of medical information sources
- **Interactive Streamlit interface** for easy use and visualization
- **Markdown formatting** for clear, structured diagnosis and treatment plans
- **Automatic reasoning process** that shows medical thinking before conclusions
- **Trusted domains filtering** to ensure reliable medical information

## Architecture

The system consists of the following components:

1. **Agents**:
   - **Researcher Agent**: Researches medical topics and provides comprehensive information using real-time web search
   - **Source Verification Agent**: Evaluates the credibility of information sources
   - **Diagnostician Agents**: Analyze symptoms, medical history, and research findings to provide diagnoses
   - **Treatment Advisor Agents**: Recommend treatment options based on diagnoses and research findings
   - **Consensus Agent**: Synthesizes opinions from multiple agents to provide a unified diagnosis and treatment plan

2. **LangGraph Workflow**:
   - Orchestrates the agents in a directed graph
   - Manages state transitions and data flow between agents
   - Implements the consensus mechanism

3. **Web Search Integration**:
   - Google Custom Search API for finding recent medical information
   - Web scraping capabilities for extracting content from medical websites
   - SerpAPI integration for alternative search results

4. **Source Verification**:
   - Evaluates the credibility of medical information sources
   - Assigns credibility scores to individual sources and overall research
   - Influences the consensus process based on source reliability

5. **IO.net Intelligence API Integration**:
   - Provides high-quality language model capabilities
   - Supports medical reasoning and knowledge

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/consensus-mechanism-ai-agents.git
   cd consensus-mechanism-ai-agents
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -e .
   ```

4. Set up environment variables:
   - Create a `.env` file based on the `env.example` file
   - Add your IO.net Intelligence API key and other required credentials
   - (Optional) Add Google API key, CSE ID, and SerpAPI key for web search functionality

## Usage

### Running with Streamlit

The easiest way to use the system is through the Streamlit interface:

```
streamlit run app/streamlit_app.py
```

This will open a web interface where you can:
1. Enter medical topics for research
2. Provide symptoms, medical history, and test results
3. Enable real-time web search
4. View detailed diagnosis, treatment plans, and research findings

### Running from Command Line

To run the system from command line:

```
python -m app.langraph.main
```

### Environment Variables

Create a `.env` file based on the `env.example` file with the following variables:

```
# LLM API Configuration (use either OPENAI or IOINTELLIGENCE)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# OR 

# IO.net Intelligence API Configuration
IOINTELLIGENCE_API_KEY=your_io_intelligence_api_key_here
IOINTELLIGENCE_BASE_URL=https://api.intelligence.io.solutions/api/v1/
IOINTELLIGENCE_DEFAULT_MODEL=deepseek-ai/DeepSeek-R1-0528

# Web Search API Key
SERPER_API_KEY=your_serper_api_key_here

# Timeout & Retry Configuration
DEFAULT_REQUEST_TIMEOUT=30
MAX_RETRIES=3
RESEARCH_TIMEOUT=120
DIAGNOSIS_TIMEOUT=90
CONSENSUS_TIMEOUT=90
```

### Output

The system generates a comprehensive medical diagnosis report that includes:

1. Research findings on the medical topic
2. Source verification and credibility assessment
3. Multiple diagnoses from different diagnostician agents
4. Multiple treatment recommendations from different treatment advisor agents
5. A consensus diagnosis and treatment plan that synthesizes the opinions of all agents

The report is saved as a Markdown file in the current directory.

## Technical Details

### LangGraph Implementation

This project uses LangGraph which implements a state machine for agent orchestration:

- Each agent updates part of the global state
- Conditional edges determine the next agent to run based on the state
- The workflow manages timeouts and prevents infinite loops

### Web Search Reliability

The system includes several reliability features:
- List of trusted medical domains for filtering search results
- Session-based requests with retry mechanisms
- Automatic fallback to simulated results if web search fails
- Cache mechanism to avoid redundant API calls

### Requirements

- Python 3.9+
- LangGraph 0.5.1+
- LangChain 0.3.0+
- IO.net Intelligence API access
- (Optional) Google Custom Search API key and CSE ID
- (Optional) SerpAPI key

## Development

### Project Structure

```
Consensus_Machenism_AIAgents/
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   └── __init__.py
│   ├── langraph/
│   │   ├── __init__.py
│   │   ├── agents.py    # Agent implementations
│   │   ├── graph.py     # Workflow graph definition
│   │   └── main.py      # Entry point for CLI
│   ├── models/
│   │   ├── __init__.py
│   │   └── llm_client.py # LLM client implementations
│   ├── streamlit_app.py  # Streamlit interface
│   └── tools/
│       ├── __init__.py
│       └── web_search.py # Web search implementations
├── env.example           # Environment variables template
├── README.md
├── requirements.txt
└── setup.py
```

### Adding New Agents

To add a new agent to the system:

1. Create a new agent class in `app/langraph/agents.py`
2. Update the graph in `app/langraph/graph.py` to include the new agent
3. Update the state handling in `app/langraph/graph.py` to include the new agent's outputs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the agent orchestration framework
- [IO.net](https://io.net) for the Intelligence API
- [LangChain](https://github.com/langchain-ai/langchain) for the language model integration