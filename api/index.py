from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from supabase import create_client
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "playgenix-secret-key"))

# Get the directory containing this file, then go up one level to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Use absolute paths for templates and static directories
templates_dir = os.path.join(project_root, "templates")
static_dir = os.path.join(project_root, "app/static")

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"Templates directory exists: {os.path.exists(templates_dir)}")
logger.info(f"Static directory exists: {os.path.exists(static_dir)}")

# Mount static files
try:
    app.mount("/app/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Successfully mounted static files from {static_dir}")
except Exception as e:
    logger.error(f"Failed to mount static files: {str(e)}")
    # Don't raise exception - continue without static files

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)

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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/waitlist", response_class=HTMLResponse)
async def waitlist_get(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("waitlist.html", {"request": request, "user": user})

@app.post("/waitlist", response_class=HTMLResponse)
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

@app.get("/contact", response_class=HTMLResponse)
async def contact_get(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("contact.html", {"request": request, "user": user})

@app.post("/contact", response_class=HTMLResponse)
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

# Export the app for Vercel
handler = app
