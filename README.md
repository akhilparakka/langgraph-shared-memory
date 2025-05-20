# Task Master

An intelligent todo list manager powered by LangChain and GPT-4, featuring persistent memory and natural language task management.

## Features

- Natural language task management
- Persistent memory across conversations for:
  - User profiles
  - Todo lists
  - Task management preferences
- Built with modern AI tools (GPT-4, LangChain, LangGraph)
- FastAPI backend with CORS support
- Intelligent task understanding and organization

## Tech Stack

- Python 3.12+
- FastAPI
- LangChain & LangGraph
- OpenAI GPT-4
- TrustCall
- uv (Python package installer and virtualenv manager)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd task-master
```

2. Set up the environment using uv:

```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
uv sync
```

4. Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Interaction

Send a POST request to `/chat`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"config": {"user_id": "user123"}, "message": "Add a task to buy groceries tomorrow"}'
```

## Project Structure

```
task-master/
├── app/
│   └── graph/
│       ├── builder.py      # Core graph logic and memory management
│       ├── graph_state.py  # State definitions and data models
│       ├── spy.py         # Tool call tracking
│       └── utils.py       # Helper functions
├── main.py                # FastAPI application
└── README.md
```

## Development

The project uses a state graph architecture with three types of memory:

1. Profile Memory - Stores user information
2. Todo Memory - Manages tasks and their metadata
3. Instruction Memory - Stores user preferences for todo management
