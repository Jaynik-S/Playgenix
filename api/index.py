from app.main import app

# This is the entry point for Vercel
def handler(request, response):
    return app(request, response)
