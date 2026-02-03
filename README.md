# 🦜🔗 LangChain Learning

A hands-on learning repository exploring LangChain components and modern LLM application development patterns.

## 📌 Overview

This repository contains practical examples and experiments with various LangChain components, serving as a personal reference for building LLM-powered applications.

## 🧩 Components Covered

- **Prompt Templates** - PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
- **Chains** - LLMChain, SequentialChain, TransformChain
- **Output Parsers** - StructuredOutputParser, PydanticOutputParser, JSONOutputParser
- **Retrievals** - Vector stores, document loaders, text splitters
- **Agents & Tools** - ReAct agents, tool calling, custom tools
- **Memory** - Conversation history, buffer memory, summary memory
- **Runnables** - LCEL (LangChain Expression Language), custom runnables

## 🛠️ Tech Stack

- Python 3.11
- LangChain
- LangChain Community
- Google Gemini (`langchain-google-genai`)
- Pydantic
- python-dotenv

## ⚙️ Setup

1. **Clone the repository**
```bash
   git clone https://github.com/Sohanuu66/langchain_learning.git
   cd langchain_learning
```

2. **Create virtual environment**
```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Configure environment**
   
   Create a `.env` file in the root directory:
```env
   GEMINI_API_KEY=your_api_key_here
```