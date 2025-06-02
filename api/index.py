from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
from supabase import create_client
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "playgenix-secret-key"))

# Handle templates for Vercel deployment
try:
    # Try multiple possible template locations
    if os.path.exists("templates"):
        templates = Jinja2Templates(directory="templates")
    elif os.path.exists("../templates"):
        templates = Jinja2Templates(directory="../templates")
    else:
        # Create a fallback directory structure
        templates = Jinja2Templates(directory=".")
except Exception as e:
    print(f"Template loading error: {e}")
    templates = Jinja2Templates(directory=".")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

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
        print(f"Error inserting into Supabase: {e}")
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
        print(f"Error inserting into Supabase: {e}")
        return templates.TemplateResponse("contact.html", {
            "request": request, 
            "error": "Failed to reach out.", 
            "user": user
        })

# Export the app for Vercel
handler = app
