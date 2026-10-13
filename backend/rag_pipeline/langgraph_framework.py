import os
import sqlite3
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import from your dedicated utilities module
from rag_pipeline.utils.page_renderer import render_pages
from rag_pipeline.retrieval import TwoStageRetriever
from rag_pipeline import config  # Centralized source of truth

# -------------------------------------------------------------
# 1. ENTERPRISE STATE MANAGEMENT SCHEMA
# -------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_query: str
    doc_stem_name: str  
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_pages: List[int]
    retrieved_images: List[str]
    system_errors: List[str]

# -------------------------------------------------------------
# 2. MODEL-AGNOSTIC INFRASTRUCTURE INITIALIZATION
# -------------------------------------------------------------
vision_llm = ChatOllama(
    base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434"),
    model=config.VISION_LLM_MODEL,
    temperature=config.VISION_LLM_TEMPERATURE,
    streaming=True
)

query_normalizer_llm = ChatOllama(
    base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434"),
    model=getattr(config, "NORMALIZER_LLM_MODEL", "qwen2.5:3b"),
    temperature=getattr(config, "NORMALIZER_LLM_TEMPERATURE", 0.0)
)

REAL_RETRIEVER = None

def get_retriever():
    global REAL_RETRIEVER
    if REAL_RETRIEVER is None:
        REAL_RETRIEVER = TwoStageRetriever()
    return REAL_RETRIEVER

# -------------------------------------------------------------
# 3. DATA ENCODING UTILITIES
# -------------------------------------------------------------
def encode_local_image_to_base64(image_path_str: str) -> str:
    path = Path(image_path_str)
    if not path.exists():
        raise FileNotFoundError(f"Target file missing for base64 encoding at: {image_path_str}")
    
    mime_type, _ = mimetypes.guess_type(image_path_str)
    if not mime_type:
        mime_type = "image/png"
        
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    return f"data:{mime_type};base64,{encoded_string}"

# -------------------------------------------------------------
# 4. ROBUST GRAPH NODES
# -------------------------------------------------------------
def pre_process_node(state: AgentState) -> Dict[str, Any]:
    print("\n[NODE: PRE-PROCESS] Normalizing user query payload...")
    messages = state.get("messages", [])
    
    if not messages:
        return {
            "current_query": state.get("current_query", "").strip(),
            "system_errors": []
        }
        
    last_message = str(messages[-1].content).strip()
    
    if len(messages) <= 1:
        print(" -> Initial thread query. Skipping normalization check.")
        return {
            "current_query": last_message,
            "system_errors": []
        }
        
    conversational_triggers = {
        "its", "it", "they", "them", "their", "this", "that", "these", "those", 
        "he", "she", "him", "her", "why", "how", "advantages", "disadvantages", 
        "pros", "cons", "applications", "methods", "compare", "difference"
    }
    
    query_words = set(last_message.lower().split())
    
    if not query_words.intersection(conversational_triggers):
        print(" -> Query evaluated as standalone. Skipping rewrite pass.")
        return {
            "current_query": last_message,
            "system_errors": []
        }

    print(" -> Ambiguous tracking detected. Reformulating with Query Normalizer...")
    
    system_prompt = (
        "You are an elite query normalization component. Analyze the recent chat history and the latest user prompt. "
        "If the prompt is already a complete, standalone question that can be searched in a vector database without context, "
        "return it exactly unchanged. If it relies on pronouns ('its', 'this') or context from previous turns, "
        "rewrite it into a single, comprehensive search query. Do NOT answer the question. Output ONLY the raw string."
    )
    
    chat_context = []
    for msg in messages[-4:-1]:
        role = "USER" if isinstance(msg, HumanMessage) else "ASSISTANT"
        chat_context.append(f"{role}: {msg.content}")
        
    user_payload = (
        f"Conversation Memory:\n"
        f"{'\n'.join(chat_context)}\n\n"
        f"Latest User Prompt: {last_message}"
    )
    
    try:
        response = query_normalizer_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ])
        resolved_query = response.content.strip().strip("'\"")
        print(f" -> Normalized Target: '{resolved_query}'")
        
        return {
            "current_query": resolved_query,
            "system_errors": []
        }
    except Exception as e:
        print(f" -> Normalizer parsing anomaly: {e}. Falling back to raw text.")
        return {
            "current_query": last_message,
            "system_errors": [f"Normalizer fallback: {str(e)}"]
        }

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    print("[NODE: RETRIEVAL] Launching visual vector search across document spaces...")
    query = state["current_query"]
    doc_stem = state.get("doc_stem_name", "")
    
    if not doc_stem:
        return {
            "retrieved_chunks": [], 
            "retrieved_pages": [],
            "system_errors": ["No active document context provided (doc_stem_name)."]
        }

    top_k_value = getattr(config, "RETRIEVAL_TOP_K", 10)

    try:
        retriever_instance = get_retriever()
        result = retriever_instance.search(
            query_text=query, 
            doc_stem_name=doc_stem, 
            top_k=top_k_value
        )
        return {
            "retrieved_chunks": result.get("chunks", []),
            "retrieved_pages": result.get("page_numbers", [])
        }
    except Exception as e:
        error_msg = f"ColPali active runtime execution call failed: {str(e)}"
        print(f" -> Execution Warning: {error_msg}")
        return {
            "retrieved_chunks": [], 
            "retrieved_pages": [],
            "system_errors": [error_msg]
        }

def visual_context_node(state: AgentState) -> Dict[str, Any]:
    print("[NODE: VISUAL CONTEXT] Generating dynamic page canvases...")
    target_pages = state.get("retrieved_pages", [])[:1]
    images = render_pages(
        state["doc_stem_name"],
        target_pages
    )
    return {
        "retrieved_images": images
    }

# def generate_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
#     print("[NODE: GENERATOR] Executing Config-Driven Ranked Multimodal Generation Layer...")
#     original_query = state["messages"][-1].content if state["messages"] else state["current_query"]
#     chunks = state.get("retrieved_chunks", [])
#     local_image_paths = state.get("retrieved_images", [])

#     streaming_callback = None
#     if config and "configurable" in config:
#         streaming_callback = config["configurable"].get("streaming_callback", None)
        
#     if not chunks:
#         msg = "I could not find relevant details in the document matching that query."
#         if streaming_callback:
#             streaming_callback(msg)
#         return {"messages": [AIMessage(content=msg)]}

#     sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
#     max_text_chars = getattr(config, "MAX_TEXT_CONTEXT_CHARS", 5500)
    
#     processed_chunks = []
#     total_chars = 0

#     for idx, c in enumerate(sorted_chunks):
#         chunk_text = str(c.get('text', c.get('chunk_text', ''))).strip()
#         display_text = f"[Rank {idx+1} | Page {c.get('page_num')}]: {chunk_text}"
        
#         if total_chars + len(display_text) > max_text_chars:
#             remaining_budget = max_text_chars - total_chars
#             if remaining_budget > 150:
#                 processed_chunks.append(f"{display_text[:remaining_budget]}... [Truncated for Token Size Safety]")
#             break
#         processed_chunks.append(display_text)
#         total_chars += len(display_text)

#     context_payload = "\n\n".join(processed_chunks)

#     system_prompt = """
# You are an intelligent offline document assistant. Your task is to answer the user's question by reasoning over BOTH the retrieved document text context and the attached page images.

# The retrieved text and page images are complementary sources of information. Analyze them together before generating an answer.

# Strict Operational Directives:

# 1. Treat both retrieved text and attached page images as sources of truth.
#    - Use text context to understand definitions, explanations, and detailed descriptions.
#    - Use page images to understand figures, diagrams, tables, charts, layouts, labels, and visual relationships.
#    - Do not ignore visual information when it is relevant to the user's question.

# 2. When the user asks about a figure, table, diagram, image, architecture, flowchart, or any visual element:
#    - Carefully inspect the attached page images first.
#    - Extract information from visible labels, captions, arrows, structures, legends, and relationships.
#    - Use retrieved text only as supporting context.
#    - Do not answer only from surrounding paragraphs if the requested information exists in the image.

# 3. When answering normal text-based questions:
#    - Use the retrieved text context as the main reference.
#    - Use page images to verify or supplement information when useful.

# 4. Combine information from multiple sources into a single coherent answer.
#    Do not answer chunk-by-chunk or describe separate pieces of evidence independently.

# 5. Preserve analytical and technical precision.
#    If the available text or images only partially answer the question:
#    - Clearly explain what information is available.
#    - State what specific information is not available.
#    - Do not fill missing details using assumptions.

# 6. If the information is missing from BOTH the retrieved text context and the attached page images, reply exactly with:
# "I couldn't find sufficient information in the document to answer that question."

# 7. Structural formatting expectations:
#    - Synthesize duplicated or overlapping information into one clear explanation.
#    - Reference page locations naturally within sentences (for example: "As shown on page 3...").
#    - Use bold bullet points when explaining features, components, methodologies, comparisons, or structured information.
#    - For figures and diagrams, describe the visual structure first, then explain its meaning.

# 8. Strictly avoid meta-commentary:
#    Never mention internal systems such as embeddings, vector search indexes, retrieval pipelines, chunks, or document processing systems.

# 9. Before producing the final answer, internally verify:
#    - Did I inspect the attached image when the question refers to visual content?
#    - Does my answer match the evidence present in the document?
#    - Am I answering the user's actual question rather than only summarizing nearby text?
# """
#     vision_content = [
#         {
#             "type": "text",
#             "text": f"{system_prompt}\n\nUser Question: {original_query}\n\nRetrieved Document Context (Ranked):\n{context_payload}"
#         }
#     ]

# # 1. Strip the data URI prefix from your base64 strings
#     cleaned_images = []
#     if total_chars < (max_text_chars - 500):
#         for path in local_image_paths:
#             try:
#                 base64_uri = encode_local_image_to_base64(path)
#                 # Ollama requires raw base64 without the "data:image/png;base64," header
#                 if "," in base64_uri:
#                     raw_base64 = base64_uri.split(",")[1]
#                 else:
#                     raw_base64 = base64_uri
                    
#                 cleaned_images.append(raw_base64)
#                 print(f" -> Image frame {path} prepared for Ollama successfully.")
#             except Exception as e:
#                 print(f" -> Non-fatal image embedding skip for {path}: {str(e)}")
#     else:
#         print(" -> Alert: Large context density detected. Dropping image payload dynamically.")

#     # 2. Build the strict Ollama-compliant HumanMessage structure
#     full_text_prompt = f"{system_prompt}\n\nUser Question: {original_query}\n\nRetrieved Document Context (Ranked):\n{context_payload}"
    
#     # LangChain Ollama relies on the explicit 'images' keyword argument containing raw base64 strings
#     formatted_message = HumanMessage(
#         content=full_text_prompt,
#         images=cleaned_images if cleaned_images else None
#     )

#     try:
#         content = ""
#         # Pass the cleanly structured message object directly to your vision LLM stream
#         for chunk in vision_llm.stream([formatted_message]):
#             token = chunk.content
#             content += token
#             if streaming_callback:
#                 streaming_callback(token)
#             print(token, end="", flush=True)

#         return {"messages": [AIMessage(content=content)]}

#     except Exception as e:
#         if "exceed" in str(e).lower() or "400" in str(e).lower():
#             print(" -> Context Fault Caught. Triggering final fallback mitigation layer...")

#             fallback_text = [
#                 {
#                     "type": "text",
#                     "text": f"{system_prompt}\n\nUser Question: {original_query}\n\nRetrieved Context:\n{context_payload[:3500]}"
#                 }
#             ]

#             content = ""
#             for chunk in vision_llm.stream([HumanMessage(content=fallback_text)]):
#                 token = chunk.content
#                 content += token
#                 if streaming_callback:
#                     streaming_callback(token)
#                 print(token, end="", flush=True)

#             return {"messages": [AIMessage(content=content)]}
            
#         print(f" -> Runtime generation fault context: {e}")
#         err_msg = "An internal text synthesis overflow exception occurred during local rendering."
#         if streaming_callback:
#             streaming_callback(err_msg)
#         return {"messages": [AIMessage(content=err_msg)]}

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage

def generate_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
    print("[NODE: GENERATOR] Executing Config-Driven Ranked Multimodal Generation Layer...")
    original_query = state["messages"][-1].content if state["messages"] else state["current_query"]
    chunks = state.get("retrieved_chunks", [])
    local_image_paths = state.get("retrieved_images", [])

    streaming_callback = None
    if config and "configurable" in config:
        streaming_callback = config["configurable"].get("streaming_callback", None)
        
    if not chunks:
        msg = "I could not find relevant details in the document matching that query."
        if streaming_callback:
            streaming_callback(msg)
        return {"messages": [AIMessage(content=msg)]}

    sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
    max_text_chars = getattr(config, "MAX_TEXT_CONTEXT_CHARS", 5500)
    
    processed_chunks = []
    total_chars = 0

    for idx, c in enumerate(sorted_chunks):
        chunk_text = str(c.get('text', c.get('chunk_text', ''))).strip()
        display_text = f"[Rank {idx+1} | Page {c.get('page_num')}]: {chunk_text}"
        
        if total_chars + len(display_text) > max_text_chars:
            remaining_budget = max_text_chars - total_chars
            if remaining_budget > 150:
                processed_chunks.append(f"{display_text[:remaining_budget]}... [Truncated for Token Size Safety]")
            break
        processed_chunks.append(display_text)
        total_chars += len(display_text)

    context_payload = "\n\n".join(processed_chunks)

    system_prompt = """
You are an intelligent offline document assistant. Your task is to answer the user's question by reasoning over BOTH the retrieved document text context and the attached page images.

The retrieved text and page images are complementary sources of information. Analyze them together before generating an answer.

Strict Operational Directives:

1. Treat both retrieved text and attached page images as sources of truth.
   - Use text context to understand definitions, explanations, and detailed descriptions.
   - Use page images to understand figures, diagrams, tables, charts, layouts, labels, and visual relationships.
   - Do not ignore visual information when it is relevant to the user's question.

2. When the user asks about a figure, table, diagram, image, architecture, flowchart, or any visual element:
   - The figure caption alone is not enough evidence. Inspect the actual visual content inside the page image, including tables, labels, columns, arrows, and text embedded inside figures.
   - Carefully inspect the attached page images first.
   - Extract information from visible labels, captions, arrows, structures, legends, and relationships.
   - Use retrieved text only as supporting context.
   - Do not answer only from surrounding paragraphs if the requested information exists in the image.

3. When answering normal text-based questions:
   - Use the retrieved text context as the main reference.
   - Use page images to verify or supplement information when useful.

4. Combine information from multiple sources into a single coherent answer.
   Do not answer chunk-by-chunk or describe separate pieces of evidence independently.

5. Preserve analytical and technical precision.
   If the available text or images only partially answer the question:
   - Clearly explain what information is available.
   - State what specific information is not available.
   - Do not fill missing details using assumptions.

6. If the information is missing from BOTH the retrieved text context and the attached page images, reply exactly with:
"I couldn't find sufficient information in the document to answer that question."

7. Structural formatting expectations:
   - Synthesize duplicated or overlapping information into one clear explanation.
   - Reference page locations naturally within sentences (for example: "As shown on page 3...").
   - Use bold bullet points when explaining features, components, methodologies, comparisons, or structured information.
   - For figures and diagrams, describe the visual structure first, then explain its meaning.

8. Strictly avoid meta-commentary:
   Never mention internal systems such as embeddings, vector search indexes, retrieval pipelines, chunks, or document processing systems.

9. Before producing the final answer, internally verify:
   - Did I inspect the attached image when the question refers to visual content?
   - Does my answer match the evidence present in the document?
   - Am I answering the user's actual question rather than only summarizing nearby text?
"""

    # Always prepare images when available to maintain robust visual context
    cleaned_images = []
    for path in local_image_paths:
        try:
            base64_uri = encode_local_image_to_base64(path)
            if "," in base64_uri:
                raw_base64 = base64_uri.split(",")[1]
            else:
                raw_base64 = base64_uri
                
            cleaned_images.append(raw_base64)
            print(f" -> Image frame {path} prepared for Ollama successfully.")
        except Exception as e:
            print(f" -> Non-fatal image embedding skip for {path}: {str(e)}")

    # Assemble primary structured prompts
    full_text_prompt = f"{system_prompt}\n\nUser Question: {original_query}\n\nRetrieved Document Context (Ranked):\n{context_payload}"
    
    message_content = [{"type": "text", "text": full_text_prompt}]
    #message_content = [{ "type": "text", "text": f"User Question: {original_query}" }]
# 2. Append each image block dynamically if they exist
    if cleaned_images:
        if "," in base64_uri:
            raw_base64 = base64_uri.split(",")[1]
        for raw_base64 in cleaned_images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{raw_base64}"}
            })

    # 3. Instantiate the HumanMessage passing the structured list directly to content
    formatted_message = HumanMessage(content=message_content)

    try:
        content = ""
        print(" -> Attempting primary execution path stream...")
        for chunk in vision_llm.stream([formatted_message]):
            token = chunk.content
            content += token
            if streaming_callback:
                streaming_callback(token)
            print(token, end="", flush=True)

        return {"messages": [AIMessage(content=content)]}

    except Exception as e:
        if "exceed" in str(e).lower() or "400" in str(e).lower():
            print(" -> Context Fault Caught. Triggering robust fallback mitigation layer...")

            # Cleaned string fallback mapping to ensure consistent Langchain HumanMessage structural typing
            fallback_prompt_text = f"{system_prompt}\n\nUser Question: {original_query}\n\nRetrieved Context (Truncated):\n{context_payload[:3500]}"
            
            fallback_message = HumanMessage(
                content=fallback_prompt_text,
                images=cleaned_images if cleaned_images else None
            )

            content = ""
            for chunk in vision_llm.stream([fallback_message]):
                token = chunk.content
                content += token
                if streaming_callback:
                    streaming_callback(token)
                print(token, end="", flush=True)

            return {"messages": [AIMessage(content=content)]}
            
        print(f" -> Runtime generation fault context: {e}")
        err_msg = "An internal text synthesis overflow exception occurred during local rendering."
        if streaming_callback:
            streaming_callback(err_msg)
        return {"messages": [AIMessage(content=err_msg)]}
# -------------------------------------------------------------
# 5. COMPILING THE WORKFLOW WITH PERSISTENT AGENT MEMORY SAVER
# -------------------------------------------------------------
builder = StateGraph(AgentState)

# Add remaining essential nodes
builder.add_node("pre_process", pre_process_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("visual_context", visual_context_node)
builder.add_node("generate", generate_node)

# Linear flow connection: pre_process -> retrieve -> visual_context -> generate
builder.add_edge(START, "pre_process")
builder.add_edge("pre_process", "retrieve")
builder.add_edge("retrieve", "visual_context")
builder.add_edge("visual_context", "generate")
builder.add_edge("generate", END)

checkpointer = MemorySaver()
orchestration_engine = builder.compile(checkpointer=checkpointer)