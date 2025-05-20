from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from supabase import create_client
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)
if not supabase:
    raise Exception("Failed to initialize Supabase client.")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/waitlist", response_class=HTMLResponse)
def waitlist(request: Request):
    return templates.TemplateResponse("waitlist.html", {"request": request})

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
def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

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

@app.get("/login", response_class=HTMLResponse)
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request):
    response = supabase.auth.sign_in_with_oauth(
                {
                    "provider": "google",
                    "options": {
                        "redirect_to": "http://127.0.0.1:8000/contact",
                    }
                }
            )
    redirect_url = response.url
    if redirect_url:
        return RedirectResponse(url=redirect_url, status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)