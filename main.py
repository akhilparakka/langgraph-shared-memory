from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from app.graph.builder import graph

app = FastAPI(
    title="Task Master", description="The Destroyer of worlds", version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    config: Dict[str, Any]  # Changed from 'any' to 'Any'
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        input_message = HumanMessage(content=request.message)

        response_chunks = []
        for chunk in graph.stream(
            {"messages": [input_message]}, request.config, stream_mode="values"
        ):
            if "messages" in chunk and not isinstance(
                chunk["messages"][-1], HumanMessage
            ):
                response_chunks.append(chunk["messages"][-1].content)

        full_response = " ".join(response_chunks)
        return ChatResponse(response=full_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
