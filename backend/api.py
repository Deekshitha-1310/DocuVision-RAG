# api.py
import json
import os
import queue
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

from engines import pipeline_factory

app = FastAPI(title="Offline Multi-Mode RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionManager:
    def __init__(self, filepath=None):
        if filepath is None:
            self.filepath = str(Path(__file__).resolve().parent / "sessions.json")
        else:
            self.filepath = filepath
        self.sessions = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except Exception:
                self.sessions = {}
        if not self.sessions:
            self._auto_discover_existing_docs()

    def _auto_discover_existing_docs(self):
        base_dir = Path(__file__).resolve().parent
        processed_dir = base_dir.parent / "processed_data"
        discovered = {}
        
        # Discover from processed_data outputs
        if processed_dir.exists():
            for out_dir in processed_dir.glob("*_output"):
                stem = out_dir.stem.replace("_output", "")
                meta_file = out_dir / "document_metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        assets = data.get("extracted_assets", [])
                        total_pages = data.get("total_pages", 1)
                        table_count = sum(1 for a in assets if a.get("type") == "table")
                        image_count = sum(1 for a in assets if a.get("type") == "image")
                        
                        pdf_name = data.get("pdf_source", f"{stem}.pdf")
                        discovered[stem] = {
                            "doc_stem": stem,
                            "doc_fullname": pdf_name,
                            "mode": "text",
                            "metadata": {
                                "pages": total_pages,
                                "tables": table_count,
                                "images": image_count,
                                "chunks": 26
                            }
                        }
                    except Exception:
                        pass
                        
        if discovered:
            self.sessions = discovered
            self._save_to_disk()

    def _save_to_disk(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=4)
        except Exception:
            pass

    def set_session(self, session_id, doc_stem, doc_fullname, mode, metadata=None):
        self.sessions[session_id] = {
            "doc_stem": doc_stem,
            "doc_fullname": doc_fullname,
            "mode": mode,
            "metadata": metadata or {}
        }
        self._save_to_disk()

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def get_all_sessions(self):
        return self.sessions

manager = SessionManager()

# Mount processed_data for serving extracted images & figures
from fastapi.staticfiles import StaticFiles
from rag_pipeline import config

assets_path = Path(config.OUTPUT_DIR)
assets_path.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

class ChatRequest(BaseModel):
    query: str
    sessionId: Optional[str] = None
    mode: Optional[str] = "text"

class SwitchSessionRequest(BaseModel):
    sessionId: str

@app.get("/health")
async def health_check():
    return {"status": "online"}

@app.get("/documents")
async def list_documents():
    """Returns deduplicated library of all previously ingested documents."""
    sessions = manager.get_all_sessions()
    
    # Deduplicate by document stem/filename so each PDF appears cleanly once
    unique_docs = {}
    for sid, sdata in sessions.items():
        doc_stem = sdata.get("doc_stem", "")
        if not doc_stem:
            continue
        unique_docs[doc_stem] = {
            "sessionId": sid,
            "filename": sdata.get("doc_fullname", f"{doc_stem}.pdf"),
            "doc_stem": doc_stem,
            "mode": sdata.get("mode", "vision"),
            "metadata": sdata.get("metadata", {})
        }
        
    return {"documents": list(unique_docs.values())}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Query(...), mode: str = Query("vision")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    staging_dir = Path("./staging_docs")
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_path = staging_dir / file.filename

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    doc_stem = target_path.stem

    def upload_stream_generator():
        try:
            def make_frame(progress: int, status: str, payload: dict = None):
                out = {"progress": progress, "status": status}
                if payload: out["payload"] = payload
                return f"data: {json.dumps(out)}\n\n"

            yield make_frame(2, "Staging completed. Initiating pipeline worker...")
            
            progress_queue = queue.Queue()
            
            if mode == "vision":
                from rag_pipeline.ingestion import ingest_document
                def vision_worker():
                    try:
                        res = ingest_document(str(target_path), progress_callback=lambda p, m: progress_queue.put(("progress", p, m)))
                        progress_queue.put(("done", res))
                    except Exception as ex:
                        progress_queue.put(("error", str(ex)))

                worker_thread = threading.Thread(target=vision_worker, daemon=True)
            else:
                worker_thread = threading.Thread(
                    target=pipeline_factory.run_text_ingestion_sync,
                    args=(str(target_path), progress_queue),
                    daemon=True
                )
                
            worker_thread.start()

            metadata = {}
            while True:
                try:
                    event_data = progress_queue.get(timeout=0.2)
                except queue.Empty:
                    if not worker_thread.is_alive() and progress_queue.empty():
                        break
                    continue

                if event_data[0] == "progress":
                    yield make_frame(event_data[1], event_data[2])
                elif event_data[0] == "done":
                    metadata = event_data[1] or {}
                    break
                elif event_data[0] == "error":
                    raise RuntimeError(event_data[1])

            manager.set_session(session_id=session_id, doc_stem=doc_stem, doc_fullname=file.filename, mode=mode, metadata=metadata)

            final_payload = {
                "status": "success",
                "filename": file.filename,
                "pages": metadata.get("pages", 0),
                "tables": metadata.get("tables", 0),
                "images": metadata.get("images", 0),
                "chunks": metadata.get("chunks", 0),
                "sessionId": session_id,
            }
            # Final upload frame includes done flag
            out = {
                "progress": 100,
                "status": "Ingestion completed successfully.",
                "payload": final_payload,
                "done": True
            }
            yield f"data: {json.dumps(out)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'progress': 100, 'status': f'Fault: {str(e)}', 'done': True})}\n\n"

    return StreamingResponse(upload_stream_generator(), media_type="text/event-stream")

@app.post("/chat")
async def chat_with_document(payload: ChatRequest):
    session = manager.get_session(payload.sessionId) if payload.sessionId else None
    if session is None:
        # Fallback to the most recent document session in manager if available
        all_sessions = manager.get_all_sessions()
        if all_sessions:
            last_sid = list(all_sessions.keys())[-1]
            session = all_sessions[last_sid]
            payload.sessionId = last_sid

    if session is None:
        raise HTTPException(
            status_code=400,
            detail="No active document session found. Please upload a PDF or select an indexed document first."
        )

    doc_stem = session["doc_stem"]
    active_mode = session.get("mode", "vision")

    def chat_stream_generator():
        stream_queue = queue.Queue()

        def make_frame(progress: int, status: str, answer: str = None, sources: list = None):
            out = {"progress": progress, "status": status}
            if answer is not None: out["answer"] = answer
            if sources is not None: out["sources"] = sources
            return f"data: {json.dumps(out)}\n\n"

        def run_orchestrator():
            try:
                if active_mode == "vision":
                    initial_input = {"messages": [("user", payload.query)], "doc_stem_name": doc_stem}
                    config_run = {"configurable": {"thread_id": payload.sessionId, "streaming_callback": lambda t: stream_queue.put(("token", t))}}
                    for event in pipeline_factory.vision_engine.stream(initial_input, config=config_run):
                        for node_name, _ in event.items():
                            if node_name == "pre_process": stream_queue.put(("status", 20, "Analyzing query syntax..."))
                            elif node_name == "retrieve": stream_queue.put(("status", 45, "Searching visual indices..."))
                            elif node_name == "grade": stream_queue.put(("status", 65, "Validating context metrics..."))
                            elif node_name == "visual_context": stream_queue.put(("status", 80, "Extracting page layout matrices..."))
                            elif node_name == "generate": stream_queue.put(("status", 95, "Streaming generated text tokens..."))
                else:
                    pipeline_factory.answer_text_query_sync(payload.query, stream_queue, doc_stem=doc_stem)
                
            except Exception as exc:
                stream_queue.put(("error", str(exc)))

        worker_thread = threading.Thread(target=run_orchestrator, daemon=True)
        worker_thread.start()

        full_compiled_answer = ""
        generation_finished = False

        while True:
            try:
                event_data = stream_queue.get(timeout=0.1)
            except queue.Empty:
                # Exit only after done is received, the worker has finished,
                # and there are no more queued events.
                if (
                    generation_finished
                    and not worker_thread.is_alive()
                    and stream_queue.empty()
                ):
                    break
                continue

            if event_data[0] == "status":
                yield make_frame(event_data[1], event_data[2])

            elif event_data[0] == "sources":
                yield make_frame(
                    50,
                    "Formatting reference sources...",
                    sources=event_data[1]
                )

            elif event_data[0] == "token":
                full_compiled_answer += event_data[1]
                yield make_frame(
                    98,
                    "Streaming text tokens...",
                    full_compiled_answer
                )

            elif event_data[0] == "done":
                print("DONE RECEIVED")
                generation_finished = True

            elif event_data[0] == "error":
                yield f"data: {json.dumps({'progress':100,'status':event_data[1],'error':True,'done':True})}\n\n"
                return

        print("SENDING FINAL FRAME")

        out = {
            "progress": 100,
            "status": "Processing complete.",
            "answer": full_compiled_answer,
            "done": True
        }

        yield f"data: {json.dumps(out)}\n\n"
        print("FINAL FRAME SENT")

    return StreamingResponse(chat_stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)