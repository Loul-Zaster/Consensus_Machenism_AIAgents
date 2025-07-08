from setuptools import setup, find_packages

setup(
    name="consensus_mechanism_ai_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.0",
        "langchain_openai>=0.2.0",
        "langchain-core>=0.3.0",
        "langsmith>=0.3.0",
        "langchain-experimental>=0.3.0",
        "langserve>=0.1.0",
        "langchainhub>=0.1.0",
        "langchain-community>=0.3.0",
        "langchain-text-splitters>=0.3.0",
        "langgraph>=0.0.20",
        "openai>=1.13.3",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
    ],
) 