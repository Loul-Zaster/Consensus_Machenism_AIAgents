# Cancer Consensus AI Agents

![Cancer Consensus AI Agents](/LOGO.png)

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Agent Workflow](#-agent-workflow)
- [Translation Features](#-translation-features)
- [Cancer Research Specialization](#-cancer-research-specialization)
- [Evaluation Dashboard](#-evaluation-dashboard)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)
- [Requirements](#-requirements)
- [License](#-license)

## 🔬 Introduction

Cancer Consensus AI Agents is a sophisticated multi-agent system designed specifically for cancer diagnosis, analysis, and treatment recommendations. The system implements a consensus mechanism using specialized AI agents that work collaboratively to analyze cancer cases and provide comprehensive diagnostic reports.

The project leverages the LangGraph framework for agent orchestration and supports both the IO.net Intelligence API and OpenAI API for high-quality language model capabilities. It includes real-time web search functionality, source verification, and multi-language translation features.

## 🌟 Features

### Core Capabilities
- **Multi-Agent Consensus System**: Utilizes multiple specialized oncology agents that collaborate to build consensus
- **Expert Cancer Agents**: Each agent specializes in different aspects of oncology (research, diagnosis, treatment, etc.)
- **Real-time Web Search**: Retrieves up-to-date cancer information from trusted medical sources
- **Source Verification**: Evaluates the credibility of oncology information sources with a scoring system
- **Comprehensive Analysis**: Provides detailed cancer diagnosis, staging, treatment options, and prognosis
- **Interactive Dashboard**: User-friendly Streamlit interface for easy interaction and visualization
- **Evaluation Tools**: Built-in RAGAS evaluation framework to assess RAG system performance

### Advanced Features
- **Multi-language Support**: Translates cancer analysis reports to 35+ languages
- **Specialized Lung Cancer Analysis**: Detailed classification, staging, and treatment recommendations
- **Clinical Trial Matching**: Identifies relevant clinical trials for specific cancer cases
- **Genetic Marker Analysis**: Incorporates genetic testing results for precision medicine
- **Trusted Domain Filtering**: Ensures reliable cancer information from authoritative sources
- **Comprehensive Documentation**: Detailed markdown-formatted reports with structured sections

## 🏗 Architecture

The system architecture consists of several key components:

### 1. Core Agents

- **ResearcherAgent**: Researches cancer topics using web search or simulation mode
  - Performs targeted searches on cancer types, symptoms, and treatments
  - Filters results for high-quality oncology information
  - Gathers information from multiple sources to ensure comprehensive coverage

- **SourceVerifier**: Assesses the credibility of cancer information sources
  - Assigns credibility scores to each source (0-10)
  - Prioritizes authoritative cancer organizations and peer-reviewed journals
  - Provides an overall research credibility score

- **Diagnostician**: Analyzes symptoms, history, and research to suggest cancer diagnoses
  - Evaluates likelihood of different cancer diagnoses
  - Considers staging, histopathology, and molecular profiles
  - Recommends confirmatory tests for each potential diagnosis

- **TreatmentAdvisor**: Recommends evidence-based cancer treatments
  - Creates comprehensive treatment plans based on diagnoses
  - Covers primary interventions, supportive care, and follow-up protocols
  - Suggests clinical trials and addresses contraindications

- **ConsensusBuilder**: Synthesizes inputs from all agents to build a unified assessment
  - Weighs different diagnoses and treatment options
  - Resolves conflicts between different agent opinions
  - Produces a comprehensive consensus report with all key sections

- **LungCancerSpecialistAgent**: Provides specialized analysis for lung cancer cases
  - Classifies lung cancer types and subtypes
  - Determines cancer stages using TNM or other appropriate systems
  - Recommends targeted treatments based on genetic markers
  - Predicts prognosis based on multiple factors
  - Finds matching clinical trials

- **TranslationAgent**: Translates the consensus report to other languages
  - Preserves medical terminology and document structure
  - Supports 35+ languages with high accuracy
  - Uses specialized medical translation capabilities

### 2. System Architecture
- **LangGraph Workflow**: Orchestrates agents in a directed graph with state management
- **IO.net Intelligence API**: Provides high-quality language model capabilities
- **Web Search Integration**: Retrieves recent cancer information from trusted sources
- **Evaluation Framework**: Assesses RAG system performance using RAGAS metrics
- **Streamlit Interface**: Provides user-friendly interaction and visualization

## 📁 Project Structure

```
Consensus_Machenism_AIAgents/
├── app/                            # Main application code
│   ├── __init__.py
│   ├── agents/                     # Agent implementation
│   │   ├── __init__.py
│   │   └── translation_agent.py    # Multi-language translation agent
│   ├── langraph/                   # LangGraph implementation
│   │   ├── __init__.py
│   │   ├── agents.py               # Agent definitions
│   │   ├── graph.py                # LangGraph workflow
│   │   └── main.py                 # CLI entry point
│   ├── models/                     # Specialized cancer models
│   │   ├── __init__.py
│   │   ├── clinical_trial_finder.py
│   │   ├── llm_client.py           # LLM API client
│   │   ├── lung_cancer_classifier.py
│   │   ├── lung_cancer_prognosis.py
│   │   ├── lung_cancer_stager.py
│   │   └── lung_cancer_treatment_advisor.py
│   ├── streamlit_app.py            # Individual Streamlit app
│   └── tools/                      # Utility tools
│       ├── __init__.py
│       └── web_search.py           # Web search implementation
├── chroma_db/                      # Vector database for RAG
├── data/                           # Sample cancer data and documents
├── evaluation_dataset.json         # Dataset for evaluating the system
├── evaluation_ragas.py             # RAGAS evaluation implementation
├── run_dashboard.py                # Wrapper script with path fixing
├── agent.py                        # RAG agent implementation
├── dashboard.py                    # Main Streamlit dashboard 
├── prompts.py                      # System prompts
├── requirements.txt                # Project dependencies
├── run.py                          # CLI runner
├── run.sh                          # Shell script runner
├── run.bat                         # Windows batch runner
├── setup.py                        # Package configuration
├── streamlit_evaluation.py         # Evaluation dashboard
└── README.md                       # This documentation
```

## 🔧 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- git (optional, for cloning the repository)

### Step 1: Clone or Download the Repository
```bash
git clone https://github.com/intelligence-io/Consensus_Mechanism_AIAgents.git
cd Consensus_Mechanism_AIAgents
```

### Step 2: Set Up a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install as Development Package (Optional)
```bash
pip install -e .
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root directory based on the provided `env.example`:

```
# IO.net Intelligence API Configuration
IOINTELLIGENCE_API_KEY=your_io_intelligence_api_key_here
IOINTELLIGENCE_BASE_URL=https://api.intelligence.io.solutions/api/v1/
IOINTELLIGENCE_DEFAULT_MODEL=meta-llama/Llama-3.3-70B-Instruct

# Web Search API Keys (optional)
SERPER_API_KEY=your_serpapi_api_key_here

# OpenAI API Key (alternative to IO.net Intelligence)
OPENAI_API_KEY=your_openai_api_key_here

# Additional settings (optional)
DEFAULT_REQUEST_TIMEOUT=30
MAX_RETRIES=3
RESEARCH_TIMEOUT=120
DIAGNOSIS_TIMEOUT=90
CONSENSUS_TIMEOUT=90
```

### API Keys

1. **IO.net Intelligence API Key**:
   - Required for full functionality including translation
   - Sign up at [IO.net Intelligence](https://intelligence.io.solutions)

2. **OpenAI API Key** (alternative):
   - Can be used as fallback if IO.net Intelligence API is not available
   - Sign up at [OpenAI](https://platform.openai.com)

3. **SerpAPI Key** (optional for real-time search):
   - Enables real-time web search for cancer information
   - Sign up at [SerpAPI](https://serpapi.com)

## 🚀 Usage

### Running the Main Application

The main command to run the application is:

```bash
streamlit run dashboard.py
```

This will launch the integrated dashboard that combines both the cancer analysis system and evaluation tools in a single interface.

If you encounter module import errors, use the wrapper script that ensures correct Python paths:

```bash
python run_dashboard.py
```

### Using the Cancer Analysis System

1. When the dashboard loads, you'll see two mode options:
   - **Cancer Analysis System**: For analyzing cancer cases
   - **RAG Evaluation Dashboard**: For evaluating system performance

2. Click on the **Cancer Analysis System** card

3. Fill in the patient information:
   - **Cancer Concern**: Type of cancer (e.g., "Lung cancer")
   - **Symptoms**: Patient's symptoms (e.g., "Persistent cough, shortness of breath")
   - **Medical History**: Patient's medical history (e.g., "60-year-old male, 40 pack-year smoking history")
   - **Test Results**: Results from medical tests (e.g., "CT scan shows 3.5 cm mass in right upper lobe")

4. Optional settings:
   - Check **Use real-time search** to enable web search for the latest information (requires SerpAPI key)
   - In the sidebar, select a target language for translation (requires IO.net Intelligence API)

5. Click **Start Analysis** to begin processing

6. The system will display progress through several stages:
   - Cancer Research
   - Source Verification
   - Cancer Analysis
   - Treatment Analysis
   - Consensus Building

7. View the results in the structured report with sections for:
   - Consensus diagnosis
   - Oncological reasoning
   - Comprehensive cancer care plan
   - Patient guidance
   - Source credibility assessment

### Using the Evaluation Dashboard

1. Click on the **RAG Evaluation Dashboard** card

2. Navigate using the tabs:
   - **Overview**: Shows summary statistics and main metrics
   - **Details**: Provides detailed evaluation results
   - **Explanations**: Explains each metric and how to improve

3. Click **Start Evaluation** to begin the evaluation process

4. After evaluation completes, you'll see the results displayed with:
   - Metric scores for faithfulness, answer relevancy, context precision, and context recall
   - Visualizations of the performance metrics
   - Detailed breakdown of each evaluation sample

## 🔄 Agent Workflow

The consensus mechanism follows this workflow:

1. **Research Phase**:
   - The ResearcherAgent gathers information about the specified cancer topic
   - It searches for relevant, recent, and reliable cancer information
   - If real-time search is enabled, it uses web search APIs to find information

2. **Source Verification Phase**:
   - The SourceVerifier evaluates the credibility of each source
   - It assigns credibility scores based on source authority and reliability
   - Sources from trusted cancer domains receive higher scores

3. **Specialized Analysis** (for lung cancer):
   - If the topic is lung cancer, the LungCancerSpecialistAgent is activated
   - It provides detailed classification, staging, and treatment recommendations
   - It analyzes genetic markers and finds matching clinical trials

4. **Diagnosis Phase**:
   - The Diagnostician analyzes the patient information and research findings
   - It suggests potential cancer diagnoses with likelihood assessment
   - It recommends confirmatory tests for each diagnosis

5. **Treatment Phase**:
   - The TreatmentAdvisor recommends appropriate cancer treatments
   - It considers the diagnoses, patient factors, and latest research
   - It provides comprehensive treatment plans and supportive care options

6. **Consensus Building Phase**:
   - The ConsensusBuilder synthesizes inputs from all agents
   - It resolves conflicts and weighs evidence
   - It produces a unified consensus report with all key sections

7. **Translation Phase** (if target language selected):
   - The TranslationAgent translates the consensus report
   - It preserves medical terminology and document structure
   - It provides translation metadata and status tracking

## 🌐 Translation Features

The system supports translation of cancer analysis reports to 35+ languages:

- **Supported Languages**: Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Vietnamese, and more
- **Medical Accuracy**: Specialized medical translation preserving terminology
- **Formatting Preservation**: Maintains document structure and formatting
- **Requirements**: Translation requires the IO.net Intelligence API key

To use translation:
1. Select your desired language from the sidebar dropdown menu
2. Run the analysis as normal
3. Results will be automatically translated to your selected language

## 🧬 Cancer Research Specialization

The system is specialized in cancer research with these features:

- **Trusted Domain Filtering**: Prioritizes authoritative cancer sources like cancer.gov, nci.nih.gov, asco.org
- **Source Credibility Assessment**: Evaluates sources on a 0-10 scale based on authority and reliability
- **Specialized Lung Cancer Analysis**: Provides detailed lung cancer classification, staging, and treatment recommendations
- **Genetic Marker Analysis**: Incorporates genetic testing results for precision medicine approaches
- **Clinical Trial Matching**: Identifies relevant clinical trials based on cancer type, stage, and genetic markers

## 📊 Evaluation Dashboard

The system includes a built-in evaluation dashboard using the RAGAS framework:

- **Faithfulness**: Measures factual accuracy of generated answers compared to the provided context
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Context Precision**: Measures the precision of retrieved context
- **Context Recall**: Measures how well the retrieved context covers necessary information

To run the evaluation:
1. Select the **RAG Evaluation Dashboard** mode
2. Click **Start Evaluation** to begin the process
3. View the results in the metrics display and charts

## ❓ Troubleshooting

### Import Errors
If you encounter "Could not import evaluation_ragas module" or similar import errors:
```bash
# Use the wrapper script that fixes Python paths
python run_dashboard.py
```

### API Key Issues
If you see "Missing LLM API key" errors:
1. Ensure you have created a `.env` file with your API keys
2. Verify that the API keys are correctly formatted
3. Check that either IOINTELLIGENCE_API_KEY or OPENAI_API_KEY is set

### Web Search Not Working
If real-time web search isn't working:
1. Ensure you have set SERPER_API_KEY in your `.env` file
2. Check your internet connection
3. Verify that your API key is valid and has sufficient credits

### Translation Not Working
If translation isn't working:
1. Ensure you have set IOINTELLIGENCE_API_KEY in your `.env` file
2. Note that translation is only available with the IO.net Intelligence API

## 🛠 Development

### Adding New Cancer Specialists

To add a new cancer specialist agent:

1. Create a new agent class in `app/langraph/agents.py`:
```python
class NewCancerSpecialist:
    """Agent for specialized cancer analysis."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Implement specialist logic
        return updated_state
```

2. Update the graph in `app/langraph/graph.py` to include the new agent:
```python
cancer_workflow.add_node("new_specialist", NewCancerSpecialist())
cancer_workflow.add_edge("diagnosis", "new_specialist")
cancer_workflow.add_edge("new_specialist", "treatment")
```

3. Update state handling to include the new agent's outputs

### Customizing the Dashboard

To customize the Streamlit dashboard:

1. Modify `dashboard.py` to add new UI elements or features
2. Update CSS styles in the st.markdown sections with unsafe_allow_html=True
3. Add new tabs, metrics, or visualization components as needed

## 📋 Requirements

The project requires the following main dependencies:
- Python 3.9+
- langchain>=0.3.0
- langgraph>=0.5.1
- streamlit>=1.32.0
- openai>=1.13.3 (alternative to iointel)
- iointel>=0.1.0 (for IO.net Intelligence API)
- ragas>=0.1.0 (for evaluation)
- chromadb
- pandas>=2.0.0
- matplotlib>=3.10.0

See `requirements.txt` for the complete list of dependencies.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Developed with 🧠 by Enzo & Zaster