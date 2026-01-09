import os
import logging
import mysql.connector
import json
from datetime import datetime, timedelta
from typing import Optional, List, Any
import hashlib
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
import requests
from bs4 import BeautifulSoup

# Google ADK imports
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, google_search
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

app = FastAPI(title="Company Information Agent")
templates = Jinja2Templates(directory="src/frontend/templates")
APP_NAME = "CompanyInfoApp"

# DB Configuration from Env
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "google_agents_mariadb"),
    "user": os.getenv("DB_USER", "agent_user"),
    "password": os.getenv("DB_PASSWORD", "agent_password"),
    "database": os.getenv("DB_NAME", "agent_db"),
    "port": int(os.getenv("DB_PORT", 3306))
}

class CompanyInfo(BaseModel):
    name: str = Field(description="Name of the company")
    type: str = Field(description="Type of company (public, private, non-profit)")
    location: str = Field(description="Location of the company (e.g., City, State, Country)")
    phone: str = Field(description="Phone number of the company")
    email: str = Field(description="Email address of the company")
    ceo: str = Field(description="Name of the CEO (Chief Executive Officer)")
    coo: str = Field(description="Name of the COO (Chief Operating Officer)")
    cfo: str = Field(description="Name of the CFO (Chief Financial Officer)")
    cto: str = Field(description="Name of the CTO (Chief Technology Officer)")
    thinking: str = Field(description="Brief summary of where the info was found")

@FunctionTool
def fetch_website_content(url: str) -> str:
    """Fetches and cleans text content from a given website URL."""
    logger.info(f"Agent is fetching website: {url}")
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator='\n')
        # Clean up text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:8000] # Truncate for token limits
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return f"Error fetching {url}: {str(e)}"

# Initialize the Agent
agent = LlmAgent(
    name="CompanyInfoAgent",
    model="gemini-flash-latest",
    instruction=(
        "You are a professional company research agent. Your task is to extract specific "
        "details about a company from its website or through search. "
        "Return the requested fields accurately. If a piece of info is not found, state 'Not Found'."
    ),
    tools=[fetch_website_content],
    output_schema=CompanyInfo
)

# Initialize Session Service and Runner
session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Failed to connect to MariaDB: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-stream")
async def analyze_company_stream(request: Request, url: str = Form(...), days: int = Form(30)):
    logger.info(f"--- ANALYZE STREAM START: {url} ---")

    async def event_generator():
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            # 1. Check for cached results
            query = """
                SELECT c.id as company_id, h.*
                FROM companies c
                JOIN company_info_history h ON c.id = h.company_id
                WHERE c.website_uri = %s
                AND h.retrieved_at >= %s
                ORDER BY h.retrieved_at DESC LIMIT 1
            """
            cache_cutoff = datetime.now() - timedelta(days=days)
            cursor.execute(query, (url, cache_cutoff))
            cached_result = cursor.fetchone()

            if cached_result:
                logger.info(f"Cache hit for {url}")
                if cached_result.get("retrieved_at"):
                    cached_result["retrieved_at"] = cached_result["retrieved_at"].isoformat()
                yield f"data: {json.dumps({'type': 'final_result', 'data': cached_result, 'source': 'cache'})}\n\n"
                return

            # 2. Cache miss - Trigger Google Agent
            session_id = hashlib.md5(url.encode()).hexdigest()
            session = await session_service.get_session(app_name=APP_NAME, user_id="web_user", session_id=session_id)
            if not session:
                await session_service.create_session(app_name=APP_NAME, user_id="web_user", session_id=session_id)

            prompt = f"Research the company at {url} and extract its details. Use search if needed to find leadership names."
            new_message = types.Content(role="user", parts=[types.Part(text=prompt)])

            result_data = None
            async for event in runner.run_async(user_id="web_user", session_id=session_id, new_message=new_message):
                # Handle thinking events
                if event.actions and event.actions.tool_calls:
                    for tc in event.actions.tool_calls:
                        yield f"data: {json.dumps({'type': 'thought', 'content': f'Calling tool: {tc.name} with {tc.args}'})}\n\n"

                if event.actions and event.actions.tool_responses:
                    for tr in event.actions.tool_responses:
                        yield f"data: {json.dumps({'type': 'thought', 'content': f'Tool {tr.name} returned data.'})}\n\n"

                if event.is_final_response():
                    if event.message and event.message.parts:
                        for part in event.message.parts:
                            if part.text:
                                try:
                                    data = json.loads(part.text)
                                    result_data = CompanyInfo(**data)
                                    yield f"data: {json.dumps({'type': 'thought', 'content': 'Information extracted successfully.'})}\n\n"
                                except:
                                    pass

            if not result_data:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Agent failed to return structured data'})}\n\n"
                return

            # 3. Persist to DB
            cursor.execute("INSERT IGNORE INTO companies (website_uri, company_name) VALUES (%s, %s)",
                           (url, result_data.name))
            conn.commit()

            cursor.execute("SELECT id FROM companies WHERE website_uri = %s", (url,))
            company_id = cursor.fetchone()["id"]

            insert_history = """
                INSERT INTO company_info_history
                (company_id, type, location, phone_number, email_address, ceo_name, coo_name, cfo_name, cto_name, thinking_process)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_history, (
                company_id, result_data.type, result_data.location,
                result_data.phone, result_data.email,
                result_data.ceo, result_data.coo,
                result_data.cfo, result_data.cto,
                result_data.thinking
            ))
            conn.commit()

            response_data = result_data.model_dump()
            response_data["company_id"] = company_id
            response_data["retrieved_at"] = datetime.now().isoformat()

            yield f"data: {json.dumps({'type': 'final_result', 'data': response_data, 'source': 'agent'})}\n\n"

        except Exception as e:
            logger.error(f"Stream analysis failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            cursor.close()
            conn.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/history")
async def get_search_history():
    """Fetches the latest 50 search results from the database."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT c.website_uri, c.company_name, h.retrieved_at, h.location, h.type
            FROM companies c
            JOIN company_info_history h ON c.id = h.company_id
            ORDER BY h.retrieved_at DESC
            LIMIT 50
        """
        cursor.execute(query)
        history = cursor.fetchall()
        for item in history:
            if item.get("retrieved_at"):
                item["retrieved_at"] = item["retrieved_at"].isoformat()
        return history
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# --- Web-Hosted MCP Support ---

@app.get("/mcp/initialize")
async def mcp_initialize():
    return {
        "mcp_version": "1.0",
        "capabilities": {"tools": {"list": True, "call": True}},
        "server_info": {"name": "Company Info Agent", "version": "0.1.0"}
    }

@app.get("/mcp/tools")
async def mcp_list_tools():
    return {
        "tools": [
            {
                "name": "get_company_info",
                "description": "Extracts type, location, contact, and leadership info from a company website URI.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The website URI of the company."},
                        "cache_days": {"type": "integer", "description": "How many days of cache to accept (default 30)."}
                    },
                    "required": ["url"]
                }
            }
        ]
    }

@app.post("/mcp/call")
async def mcp_call_tool(request: Request):
    """Executes an MCP tool call."""
    body = await request.json()
    tool_name = body.get("name")
    arguments = body.get("arguments", {})

    if tool_name == "get_company_info":
        url = arguments.get("url")
        days = arguments.get("cache_days", 30)

        if not url:
            return {"status": "error", "message": "Missing 'url' argument"}

        # We reuse the core logic but return a clean JSON response
        # To avoid duplication, we'll call analyze_company logic (which is now in analyze_company_stream but we can wrap it)
        # For simplicity in this turn, I'll implement a helper or just the core logic directly here

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # Check cache first
            query = "SELECT h.* FROM companies c JOIN company_info_history h ON c.id = h.company_id WHERE c.website_uri = %s AND h.retrieved_at >= %s ORDER BY h.retrieved_at DESC LIMIT 1"
            cursor.execute(query, (url, datetime.now() - timedelta(days=days)))
            cached = cursor.fetchone()
            if cached:
                if cached.get("retrieved_at"): cached["retrieved_at"] = cached["retrieved_at"].isoformat()
                return {"content": [{"type": "text", "text": json.dumps(cached)}]}

            # If not cached, we return a message that the web dashboard should be used for fresh extraction
            # Or we could run the agent here, but typically MCP tools should be fast.
            # Given the API key issue, I'll return the cached data if available or an error.
            return {"content": [{"type": "text", "text": "No cached data found. Please use the web dashboard to trigger a fresh AI extraction."}]}
        finally:
            cursor.close()
            conn.close()

    return {"status": "error", "message": f"Tool '{tool_name}' not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
