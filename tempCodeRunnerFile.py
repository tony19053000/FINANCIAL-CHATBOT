from fastapi import FastAPI

# ============================
# ✅ Internal API Routers
# ============================
from api import (
    stocks,
    crypto,
    predictions,
    users,
    investments,
    visualization,  # Now includes visualizations & prediction
    chat             # ✅ Gemini AI Chatbot
)

# ============================
# ✅ External (Third-Party) API Routers
# ============================
from api.yahoo_finance import router as yahoo_finance_router
from api.coin_gecko import router as coin_gecko_router
from api.financial_news import router as financial_news_router  # ✅ RSS-based news

# ============================
# ✅ Create FastAPI app instance
# ============================
app = FastAPI(
    title="Financial AI Chatbot API",
    description="Backend for investment insights, visualizations, predictions, chat, and more.",
    version="1.0.0"
)

# ============================
# ✅ Include Internal API Routes
# ============================
app.include_router(stocks.router, prefix="/stocks", tags=["Stocks"])
app.include_router(crypto.router, prefix="/crypto", tags=["Tracked Cryptos"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(investments.router, prefix="/investments", tags=["Investments"])
app.include_router(visualization.router, prefix="/visualize", tags=["Visualization & Predictions"])
app.include_router(chat.router, prefix="/chatbot", tags=["Gemini AI Chatbot"])  # ✅ AI Chatbot endpoint

# ============================
# ✅ Include External Data Routes
# ============================
app.include_router(yahoo_finance_router, prefix="/finance", tags=["Finance"])
app.include_router(coin_gecko_router, prefix="/crypto-prices", tags=["Live Crypto Prices"])
app.include_router(financial_news_router, prefix="/financial-news", tags=["Financial News"])

# ============================
# ✅ Root and Health Endpoints
# ============================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Financial AI Chatbot API"}

@app.get("/health")
def health_check():
    return {"status": "API is running smoothly"}
