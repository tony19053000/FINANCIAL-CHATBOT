from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from logging.handlers import RotatingFileHandler

# ============================
# ✅ Internal API Routers
# ============================
from api import (
    stocks,
    crypto,
    predictions,
    users,
    investments,
    visualization,  # ✅ Includes /visualize routes
    chat
)

# ============================
# ✅ External API Routers
# ============================
from api.yahoo_finance import router as yahoo_finance_router
from api.coin_gecko import router as coin_gecko_router
from api.financial_news import router as financial_news_router

# ============================
# ✅ Create FastAPI App
# ============================
app = FastAPI(
    title="Financial AI Chatbot API",
    description="Backend for insights, visualizations, predictions, and chatbot",
    version="1.0.0"
)

# ============================
# ✅ Logging Setup
# ============================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=20)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# ============================
# ✅ CORS Middleware
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# ✅ Include Routers
# ============================
app.include_router(stocks.router, prefix="/stocks", tags=["Stocks"])
app.include_router(crypto.router, prefix="/crypto", tags=["Cryptocurrencies"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(investments.router, prefix="/investments", tags=["Investments"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization & Predictions"])
app.include_router(chat.router, prefix="/chatbot", tags=["Gemini AI Chatbot"])

# ✅ External APIs
app.include_router(yahoo_finance_router, prefix="/finance", tags=["Finance"])
app.include_router(coin_gecko_router, prefix="/crypto-prices", tags=["Live Crypto Prices"])
app.include_router(financial_news_router, prefix="/financial-news", tags=["Financial News"])

# ============================
# ✅ Root & Health Check
# ============================
@app.get("/")
def read_root():
    logger.info("Root endpoint hit")
    return {"message": "Welcome to the Financial AI Chatbot API"}

@app.get("/health")
def health_check():
    logger.info("Health check passed")
    return {"status": "API is running smoothly"}

# ============================
# ✅ Global Error Handler
# ============================
@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    logger.error(f"An error occurred: {exc}")
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})