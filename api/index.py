from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from supabase import create_client
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
import logging
import asyncio
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create FastAPI app with a distinct variable name
api_app = FastAPI()

# Add session middleware
api_app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "playgenix-secret-key"))

# Get the directory containing this file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Use correct path for templates that works with Vercel deployment
templates = Jinja2Templates(directory="app/templates")

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")

# Mount static files - update path to match vercel.json route configuration
try:
    api_app.mount("/static", StaticFiles(directory="app/static"), name="static")
    logger.info("Successfully mounted static files")
except Exception as e:
    logger.error(f"Failed to mount static files: {str(e)}")
    # Don't raise exception - continue without static files

# Initialize Supabase
try:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.error("Missing Supabase credentials. Check environment variables.")
        supabase = None
    else:
        supabase = create_client(url, key)
        logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase: {str(e)}")
    supabase = None

# Helper function to get user context
async def get_current_user(request: Request):
    return request.session.get("user")

@api_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@api_app.get("/waitlist", response_class=HTMLResponse)
async def waitlist_get(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("waitlist.html", {"request": request, "user": user})

@api_app.post("/waitlist", response_class=HTMLResponse)
async def waitlist_post(request: Request, name: str = Form(...), email: str = Form(...)):
    user = await get_current_user(request)
    try:
        if not supabase:
            raise Exception("Supabase client not initialized")
            
        response = supabase.table("waitlist").insert({
            "name": name,
            "email": email,
        }).execute()

        if hasattr(response, 'data') and response.data:
            return templates.TemplateResponse("waitlist.html", {
                "request": request, 
                "success": "Successfully joined waitlist!", 
                "user": user
            })
        else:
            raise Exception(f"Supabase error: {response}")
    except Exception as e:
        logger.error(f"Error inserting into Supabase: {str(e)}")
        return templates.TemplateResponse("waitlist.html", {
            "request": request, 
            "error": "Failed to join waitlist.", 
            "user": user
        })

@api_app.get("/contact", response_class=HTMLResponse)
async def contact_get(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("contact.html", {"request": request, "user": user})

@api_app.post("/contact", response_class=HTMLResponse)
async def contact_post(request: Request, name: str = Form(...), email: str = Form(...), subject: str = Form(...), message: str = Form(...)):
    user = await get_current_user(request)
    try:
        if not supabase:
            raise Exception("Supabase client not initialized")
            
        response = supabase.table("contact").insert({
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
        }).execute()
        
        if hasattr(response, 'data') and response.data:
            return templates.TemplateResponse("contact.html", {
                "request": request, 
                "success": "Message sent successfully!", 
                "user": user
            })
        else:
            raise Exception(f"Supabase error: {response}")
    except Exception as e:
        logger.error(f"Error inserting into Supabase: {str(e)}")
        return templates.TemplateResponse("contact.html", {
            "request": request, 
            "error": "Failed to reach out.", 
            "user": user
        })

# Export the app for Vercel - this is the correct way
app = api_app
