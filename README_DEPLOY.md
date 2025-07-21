# Deployment and Running Guide for Consensus Mechanism AI Agents

## Introduction

Cancer Consensus AI Agents is a multi-agent artificial intelligence system for cancer analysis and diagnosis based on LangGraph and IO.net Intelligence API. The system includes two main applications:

1. **Cancer Analysis System**: Analyzes cancer cases based on symptoms, medical history, and test results.
2. **RAG Evaluation Dashboard**: Evaluates the performance of the RAG system using the RAGAS framework.

## System Requirements

- Python 3.9 or higher
- Pip package manager
- LangChain, Streamlit and related libraries (see `requirements.txt`)
- API key from IO.net Intelligence or OpenAI (optional: SerpAPI for web search)

## Installation

1. Clone or download the repository:
   ```
   git clone <repository-url>
   cd Consensus_Machenism_AIAgents
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

3. Create a `.env` file based on `env.example` and fill in your API keys:
   ```
   # Primary API: IO.net Intelligence API Configuration
   IOINTELLIGENCE_API_KEY=your_io_intelligence_api_key_here
   IOINTELLIGENCE_BASE_URL=https://api.intelligence.io.solutions/api/v1/
   IOINTELLIGENCE_DEFAULT_MODEL=meta-llama/Llama-3.3-70B-Instruct
   
   # Fallback API: OpenAI API Key
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: SerpAPI for web search
   SERPER_API_KEY=your_serper_api_key_here
   ```

## Running the Application

### Method 1: Using the automatic launch script

We've created an automated script to easily launch the application.

#### On Windows:

Run the `run.bat` file:
```
run.bat        # Run both applications
run.bat app    # Run only Cancer Analysis System
run.bat eval   # Run only RAG Evaluation Dashboard
```

#### On Linux/macOS:

Run the `run.sh` file (ensure it has execution permissions):
```
chmod +x run.sh    # Grant execution permission (only needed once)

./run.sh           # Run both applications
./run.sh app       # Run only Cancer Analysis System
./run.sh eval      # Run only RAG Evaluation Dashboard
```

### Method 2: Using Python directly

```
python run.py           # Run both applications
python run.py app       # Run only Cancer Analysis System
python run.py eval      # Run only RAG Evaluation Dashboard
```

### Method 3: Running individual Streamlit applications

```
streamlit run app/streamlit_app.py      # Run Cancer Analysis System
streamlit run streamlit_evaluation.py   # Run RAG Evaluation Dashboard
```

## Accessing the Applications

After running the script, the applications will automatically open in your web browser:

- **Cancer Analysis System**: http://localhost:8501
- **RAG Evaluation Dashboard**: http://localhost:8502 (or 8501 if running individually)

## API Key Configuration

The system will prioritize API keys in the following order:

1. **IO.net Intelligence API Key**: First priority, if configured
2. **OpenAI API Key**: Fallback if IO.net Intelligence API Key is not available
3. **SerpAPI Key**: Optional, used for real-time web search

## Evaluation Features (RAG Evaluation Dashboard)

The evaluation feature uses RAGAS to assess RAG system performance, including the following metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

## Notes

- To stop the application, press `Ctrl+C` in the terminal or close the browser tabs.
- If you encounter errors, check the API key configuration and installation of dependencies.

## Support

If you encounter issues, please check:
1. API keys are properly configured in the `.env` file
2. All dependencies have been installed
3. Python and Streamlit are correctly installed

For more details, please refer to the full documentation in the original `README.md` file. 