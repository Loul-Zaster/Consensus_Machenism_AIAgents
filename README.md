# Consensus Mechanism AI Agents for Medical Diagnosis

A multi-agent system for medical diagnosis using LangGraph framework and IO.net Intelligence API.

## Overview

This project implements a consensus mechanism for medical diagnosis using multiple AI agents. Each agent specializes in a different aspect of the medical diagnosis process, and they work together to provide a comprehensive diagnosis and treatment plan.

The system leverages the LangGraph framework for agent orchestration and the IO.net Intelligence API for high-quality language model capabilities.

## Features

- Multiple specialized agents (researcher, diagnosticians, treatment advisors)
- Consensus mechanism to synthesize opinions from multiple agents
- Integration with IO.net Intelligence API
- Support for medical research, diagnosis, and treatment recommendations
- **Real-time web search** for up-to-date medical information
- **Source verification** to evaluate the credibility of medical information sources
- Integration with Google Search API and SerpAPI for comprehensive research

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

### Running the System

To run the system, use the `main.py` script:

```
python main.py "Type 2 Diabetes" --symptoms "Frequent urination, excessive thirst" --medical-history "Family history of diabetes"
```

### Command Line Arguments

- `topic`: The medical topic to research (required)
- `--symptoms`, `-s`: Patient symptoms (optional)
- `--medical-history`, `-m`: Patient medical history (optional)
- `--test-results`, `-t`: Patient test results (optional)
- `--output`, `-o`: Output file name (default: medical_diagnosis_results.md) (optional)
- `--realtime`, `-r`: Enable real-time web search for the most current information (optional)

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

This project uses LangGraph 0.5.1+ which introduces several changes to the API:

- Nodes return partial state updates instead of modifying the entire state
- State management is handled differently from previous versions
- Streaming API has been updated

### Web Search Integration

The system integrates with multiple web search APIs:

- **Google Custom Search API**: Provides access to Google search results
- **SerpAPI**: Alternative search engine API that provides comprehensive results
- **Web Scraper**: Extracts content from medical websites and journals

### Source Verification

The system includes a source verification component that:

- Identifies sources mentioned in research findings
- Evaluates the credibility of each source based on type, reputation, and content
- Calculates an overall credibility score for the research
- Influences the consensus process based on source reliability

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
consensus-mechanism-ai-agents/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── langraph/
│   │   ├── __init__.py
│   │   ├── agents.py
│   │   ├── graph.py
│   │   └── main.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   └── tools/
│       ├── __init__.py
│       └── web_search.py
├── env.example
├── main.py
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