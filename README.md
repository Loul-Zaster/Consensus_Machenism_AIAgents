# Cancer Consensus AI Agents

A multi-agent system for cancer diagnosis using LangGraph framework and IO.net Intelligence API.

## Overview

This project implements a consensus mechanism for cancer diagnosis using multiple AI agents. Each agent specializes in a different aspect of the oncology diagnosis process, and they work together to provide a comprehensive cancer diagnosis and treatment plan.

The system leverages the LangGraph framework for agent orchestration and the IO.net Intelligence API for high-quality language model capabilities.

## Features

- Multiple specialized oncology agents (researcher, diagnostician, treatment advisor, consensus builder)
- Consensus mechanism to synthesize opinions from multiple cancer specialists
- Integration with IO.net Intelligence API or OpenAI API
- Support for cancer research, diagnosis, and treatment recommendations
- **Real-time web search** for up-to-date cancer information
- **Source verification** to evaluate the credibility of oncology information sources
- **Interactive Streamlit interface** for easy use and visualization
- **Markdown formatting** for clear, structured cancer diagnosis and treatment plans
- **Automatic reasoning process** that shows oncological thinking before conclusions
- **Trusted domains filtering** to ensure reliable cancer information from sources like cancer.gov, asco.org, etc.
- **🌐 Multi-language Translation** using IO Intelligence framework for global accessibility (35+ languages supported)

## Architecture

The system consists of the following components:

1. **Cancer-Specialized Agents**:
   - **Researcher Agent**: Researches cancer topics and provides comprehensive information using real-time web search
   - **Source Verification Agent**: Evaluates the credibility of oncology information sources
   - **Diagnostician Agents**: Analyze symptoms, medical history, and research findings to provide cancer diagnoses
   - **Treatment Advisor Agents**: Recommend cancer treatment options based on diagnoses and research findings
   - **Consensus Agent**: Synthesizes opinions from multiple agents to provide a unified cancer diagnosis and treatment plan
   - **Translation Agent**: Translates medical consensus reports to multiple languages using IO Intelligence framework

2. **LangGraph Workflow**:
   - Orchestrates the agents in a directed graph
   - Manages state transitions and data flow between agents
   - Implements the consensus mechanism

3. **Cancer-Focused Web Search Integration**:
   - Google Custom Search API for finding recent cancer information
   - Web scraping capabilities for extracting content from oncology websites
   - SerpAPI integration for alternative search results
   - Prioritizes trusted cancer domains like cancer.gov, nci.nih.gov, asco.org, etc.

4. **Oncology Source Verification**:
   - Evaluates the credibility of cancer information sources
   - Assigns credibility scores to individual sources and overall research
   - Influences the consensus process based on source reliability
   - Prioritizes peer-reviewed oncology journals and authoritative cancer organizations

5. **IO.net Intelligence API Integration**:
   - Provides high-quality language model capabilities
   - Supports oncological reasoning and knowledge
   - Enables multi-language translation of medical reports

6. **Translation System**:
   - Supports 35+ languages including medical terminology
   - Preserves medical accuracy and terminology during translation
   - Maintains document structure and formatting
   - Provides translation metadata and status tracking
   - Handles various API response formats for reliable translation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cancer-consensus-ai-agents.git
   cd cancer-consensus-ai-agents
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
1. Enter cancer topics for research
2. Provide cancer-related symptoms, medical history, and test results
3. Enable real-time web search
4. **Select target language for translation** (35+ languages supported)
5. View detailed cancer diagnosis, treatment plans, and research findings in your chosen language

### Translation Features

The system includes advanced translation capabilities:

- **Multi-language Support**: Translate results to 35+ languages including Spanish, French, German, Chinese, Japanese, Arabic, and more
- **Medical Accuracy**: Specialized medical translation that preserves terminology and accuracy
- **Document Structure**: Maintains formatting and structure of medical reports
- **Translation Status**: Clear indication of translation status and metadata
- **Robust Error Handling**: Handles various API response formats for reliable translation

To use translation:
1. Select your desired language from the sidebar in the Streamlit interface
2. Run the analysis as usual
3. Results will be automatically translated to your selected language
4. Translation information is displayed at the top of the results

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

The system generates a comprehensive cancer diagnosis report that includes:

1. Cancer research findings on the oncology topic
2. Source verification and credibility assessment of oncology sources
3. Multiple cancer diagnoses from different diagnostician agents
4. Multiple cancer treatment recommendations from different treatment advisor agents
5. A consensus cancer diagnosis and treatment plan that synthesizes the opinions of all agents
6. Translation of all results to the selected language (if translation is enabled)

The report is displayed in the Streamlit interface and can be saved as a Markdown file.

## Technical Details

### LangGraph Implementation

This project uses LangGraph which implements a state machine for agent orchestration:

- Each agent updates part of the global state
- Conditional edges determine the next agent to run based on the state
- The workflow manages timeouts and prevents infinite loops

### Cancer Web Search Reliability

The system includes several reliability features:
- List of trusted oncology domains for filtering search results (cancer.gov, asco.org, etc.)
- Session-based requests with retry mechanisms
- Automatic fallback to simulated results if web search fails
- Cache mechanism to avoid redundant API calls

### Requirements

- Python 3.9+
- LangGraph 0.5.1+
- LangChain 0.3.0+
- IO Intelligence framework (`iointel>=0.1.0`)
- IO.net Intelligence API access or OpenAI API key
- (Optional) Google Custom Search API key and CSE ID
- (Optional) SerpAPI key

## Development

### Project Structure

```
Consensus_Machenism_AIAgents/
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── translation_agent.py  # IO Intelligence translation agent
│   ├── langraph/
│   │   ├── __init__.py
│   │   ├── agents.py    # Cancer agent implementations
│   │   ├── graph.py     # Workflow graph definition
│   │   └── main.py      # Entry point for CLI
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clinical_trial_finder.py
│   │   ├── llm_client.py
│   │   ├── lung_cancer_classifier.py
│   │   ├── lung_cancer_prognosis.py
│   │   ├── lung_cancer_stager.py
│   │   └── lung_cancer_treatment_advisor.py
│   ├── streamlit_app.py  # Streamlit interface
│   └── tools/
│       ├── __init__.py
│       └── web_search.py # Cancer-focused web search implementations
├── env.example           # Environment variables template
├── README.md
├── requirements.txt
└── setup.py
```

### Adding New Cancer Specialists

To add a new cancer specialist agent to the system:

1. Create a new agent class in `app/langraph/agents.py`
2. Update the graph in `app/langraph/graph.py` to include the new agent
3. Update the state handling in `app/langraph/graph.py` to include the new agent's outputs

## License

This project is licensed under the MIT License - see the LICENSE file for details.