from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from supabase import create_client
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from ability_cv import main as process_video

app = FastAPI()
load_dotenv()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "playgenix-secret-key"))

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)
if not supabase:
    raise Exception("Failed to initialize Supabase client.")

UPLOAD_DIR = Path() / 'app' / 'static' / 'uploads'

# Helper function to get user context
async def get_current_user(request: Request):
    return request.session.get("user")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/waitlist", response_class=HTMLResponse)
async def waitlist(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("waitlist.html", {"request": request, "user": user})

@app.post("/waitlist", response_class=HTMLResponse)
async def waitlist(request: Request, name: str = Form(...), email: str = Form(...)):
    user = await get_current_user(request)
    try:
        response = supabase.table("waitlist").insert({
            "name": name,
            "email": email,
        }).execute()

        if response.get("status_code") != 200:
            raise Exception(f"Supabase error: {response}")
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        return templates.TemplateResponse("waitlist.html", {"request": request, "error": "Failed to join waitlist.", "user": user})

@app.post("/waitlist", response_class=HTMLResponse)
def waitlist(request: Request, name: str = Form(...), email: str = Form(...)):
    try:
        response = supabase.table("waitlist").insert({
            "name": name,
            "email": email,
        }).execute()

        if response.get("status_code") != 200:
            raise Exception(f"Supabase error: {response}")
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        return templates.TemplateResponse("waitlist.html", {"request": request, "error": "Failed to join waitlist."})

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("contact.html", {"request": request, "user": user})

@app.post("/contact", response_class=HTMLResponse)
async def contact(request: Request, name: str = Form(...), email: str = Form(...), subject: str = Form(...), message: str = Form(...)):
    user = await get_current_user(request)
    try:
        response = supabase.table("contact").insert({
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
        }).execute()
        if response.get("status_code") != 200:
            raise Exception(f"Supabase error: {response}")
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        return templates.TemplateResponse("contact.html", {"request": request, "error": "Failed to reach out.", "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "user": user})

@app.post("/login")
async def login(request: Request):
    response = supabase.auth.sign_in_with_oauth(
                {
                    "provider": "google",
                    "options": {
                        "redirect_to": "http://127.0.0.1:8000/auth/callback",
                    }
                }
            )
    redirect_url = response.url
    if redirect_url:
        return RedirectResponse(url=redirect_url, status_code=303)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    code = request.query_params.get("code")
    if code:
        session = supabase.auth.exchange_code_for_session({"auth_code": code})
        user_data = session.user
        if user_data:
            # Get user profile from database
            user_profile = supabase.table("user_profiles").select("*").eq("user_id", user_data.id).execute()
            
            # If profile exists, use it. Otherwise use default values.
            profile_data = user_profile.data[0] if user_profile.data else {"videos_remaining": 5}
            
            # Store user info in session
            request.session["user"] = {
                "name": user_data.user_metadata.get("full_name", "User"),
                "email": user_data.email,
                "videos_remaining": profile_data.get("videos_remaining", 5)
            }
    return RedirectResponse(url="/", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/", status_code=303)

@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})

@app.post("/upload")
async def upload(request: Request, video_file: UploadFile = File(...)):
    # user = await get_current_user(request)
    data = await video_file.read()
    video_path = UPLOAD_DIR / video_file.filename
    with open(video_path, "wb") as f:
        f.write(data)
    # saved to app/static/uploads/video_file.filename
    process_video(video_file.filename, visualize=False)
    return {"filename:": video_file.filename, "content_type": video_file.content_type}

@app.post("/contact", response_class=HTMLResponse)
def contact(request: Request, name: str = Form(...), email: str = Form(...), subject: str = Form(...), message: str = Form(...)):
    try:
        response = supabase.table("contact").insert({
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
        }).execute()
        if response.get("status_code") != 200:
            raise Exception(f"Supabase error: {response}")
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        return templates.TemplateResponse("contact.html", {"request": request, "error": "Failed to reach out."})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)