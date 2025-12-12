"""
Research Paper Recommender System - Production API
Uses actual recommender.py functions with smart unified endpoint
"""

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyodbc
import warnings
import uvicorn
import logging
from functools import lru_cache
from fastapi.concurrency import run_in_threadpool

# Import functions from your recommender.py
from recommender import (
    load_data_from_db,
    build_content_vectors,
    get_user_preferences,
    hybrid_recommend,
    display_user_profile,
    add_review_to_db,
    add_download_to_db,
    add_search_to_db,
    calculate_popularity_scores,
    get_user_interest_papers
)

warnings.filterwarnings('ignore')


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============================================================================
# SETTINGS & CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application settings"""

    # Database Configuration
    DB_NAME: str = "Insight"
    DB_USER: str = "sa"
    DB_PASSWORD: str = "12345678"
    DB_HOST: str = "MOSTAFA_RADY\\SQLEXPRESS"
    DB_PORT: int ="1433"
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"

    # API Configuration
    API_TITLE: str = "Research Paper Recommender API"
    API_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: str = "*"

    # Performance
    CONNECTION_POOL_SIZE: int = 10
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================

class DatabasePool:
    """Database connection pool manager"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection_string = self._build_connection_string()
        self._pool = []
        self._pool_size = settings.CONNECTION_POOL_SIZE
        logger.info("Database pool initialized")

    def _build_connection_string(self) -> str:
        server_part = self.settings.DB_HOST
        # Only append port if not a named instance (no backslash) and no port specified (no comma)
        if '\\' not in server_part and ',' not in server_part:
            server_part += f',{self.settings.DB_PORT}'

        return (
            f'DRIVER={{{self.settings.DB_DRIVER}}};'
            f'SERVER={server_part};'
            f'DATABASE={self.settings.DB_NAME};'
            f'UID={self.settings.DB_USER};'
            f'PWD={self.settings.DB_PASSWORD};'
            'TrustServerCertificate=yes;'
        )

    def get_connection(self):
        try:
            if self._pool:
                return self._pool.pop()
            conn = pyodbc.connect(self.connection_string, timeout=30)
            logger.debug("Created new database connection")
            return conn
        except pyodbc.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database connection failed")

    def return_connection(self, conn):
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            self._pool.append(conn)
        except:
            try:
                conn.close()
            except:
                pass

    def close_all(self):
        for conn in self._pool:
            try:
                conn.close()
            except:
                pass
        self._pool.clear()


# Global database pool
db_pool: Optional[DatabasePool] = None


# ============================================================================
# CACHE
# ============================================================================

class InMemoryCache:
    """Simple in-memory cache with TTL"""

    def __init__(self):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._ttl = get_settings().CACHE_TTL

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._ttl):
                logger.debug(f"Cache HIT: {key}")
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = (value, datetime.now())
        logger.debug(f"Cache SET: {key}")

    def clear(self):
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        valid = sum(1 for _, (_, ts) in self._cache.items() 
                   if datetime.now() - ts < timedelta(seconds=self._ttl))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "ttl_seconds": self._ttl
        }


cache = InMemoryCache()


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_pool
    settings = get_settings()
    db_pool = DatabasePool(settings)
    logger.info(f"🚀 API started - Environment: {settings.ENVIRONMENT}")
    logger.info(f"📊 Database: {settings.DB_NAME}@{settings.DB_HOST}")

    yield

    # Shutdown
    if db_pool:
        db_pool.close_all()
    logger.info("👋 API shutting down")


# ============================================================================
# FASTAPI APP
# ============================================================================

settings = get_settings()
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Smart API using advanced hybrid recommendation algorithms",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
)

# CORS
allowed_origins = settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserInterests(BaseModel):
    """User interests input"""
    interests: List[str] = Field(..., min_items=1, max_items=10)


class PaperRecommendation(BaseModel):
    """Paper recommendation response"""
    paper_id: int
    title: str
    authors: str
    abstract: str
    keywords: str
    field_name: str
    publication_date: Optional[str]
    hybrid_score: Optional[float] = None
    content_score: Optional[float] = None
    behavior_score: Optional[float] = None
    popularity_score: Optional[float] = None


class SmartRecommendationResponse(BaseModel):
    """Smart recommendation response"""
    user_id: Optional[int]
    user_type: str
    total_recommendations: int
    accuracy_score: Optional[float]
    interest_fields: List[str] = []
    recommendations: List[PaperRecommendation]


class UserProfileResponse(BaseModel):
    """User profile response"""
    user_id: int
    name: str
    email: str
    affiliation: str
    specialization: str
    is_new_user: bool
    rated_papers_count: int
    downloaded_papers_count: int
    avg_rating: float
    interest_fields: List[Dict[str, Any]]
    favorite_authors: List[str]


class ReviewInput(BaseModel):
    """Review input"""
    user_id: int
    paper_id: int
    rating: int = Field(..., ge=1, le=5)


class DownloadInput(BaseModel):
    """Download input"""
    user_id: int
    paper_id: int





# ============================================================================
# DATABASE DEPENDENCIES
# ============================================================================

async def get_db():
    """Get database connection"""
    conn = await run_in_threadpool(db_pool.get_connection)
    try:
        yield conn
    finally:
        await run_in_threadpool(db_pool.return_connection, conn)


async def get_cached_data(conn):
    """Get cached data and features"""
    cache_key = "recommender_data"
    cached = cache.get(cache_key)

    if cached:
        return cached

    # Load data using recommender.py function
    def _load():
        data = load_data_from_db(conn)
        vectorizer, tfidf_matrix, paper_ids = build_content_vectors(data)
        return data, vectorizer, tfidf_matrix, paper_ids

    data, vectorizer, tfidf_matrix, paper_ids = await run_in_threadpool(_load)
    cache.set(cache_key, (data, vectorizer, tfidf_matrix, paper_ids))

    logger.info("Data loaded and cached")
    return data, vectorizer, tfidf_matrix, paper_ids


# ============================================================================
# MAIN UNIFIED RECOMMENDATION ENDPOINT
# ============================================================================

@app.get("/api/recommend", response_model=SmartRecommendationResponse)
async def smart_recommend(
    user_id: Optional[int] = Query(None, description="User ID (optional for new users)"),
    top_n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    conn = Depends(get_db)
):
    """
    🎯 SMART UNIFIED RECOMMENDATION ENDPOINT

    Automatically detects user type and serves appropriate recommendations:

    **Existing User (has interaction history):**
    - Uses hybrid recommendation (content + behavior + popularity)
    - Filters by user's interest fields
    - High accuracy score (8-10/10)

    **New User or User without history:**
    - Returns popular papers from all fields
    - Medium accuracy score (6/10)

    **Examples:**
    ```
    # Existing user with history
    GET /api/recommend?user_id=1&top_n=10

    # New user or no history (popular papers)
    GET /api/recommend?user_id=999&top_n=10
    GET /api/recommend?top_n=10
    ```
    """
    try:
        # Get cached data
        data, vectorizer, tfidf_matrix, paper_ids = await get_cached_data(conn)

        # SCENARIO DETECTION
        user_type = ""
        accuracy_score = 0.0
        recommendations = pd.DataFrame()
        interest_fields = []

        # Check if user exists and has interactions
        if user_id is not None:
            # Check if user exists in researchers table
            user_exists = not data['researchers'][data['researchers']['User_ID'] == user_id].empty

            if user_exists:
                # Get user preferences
                def _get_prefs():
                    return get_user_preferences(user_id, data)

                preferences = await run_in_threadpool(_get_prefs)

                # Check if user has interactions
                if not preferences['is_new_user'] and len(preferences['all_interest_fields']) > 0:
                    # SCENARIO 1: Existing user with history - Use hybrid recommendation
                    logger.info(f"Using hybrid recommendation for user {user_id}")

                    def _recommend():
                        return hybrid_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n)

                    recommendations, accuracy_score = await run_in_threadpool(_recommend)
                    user_type = "existing_user_personalized"

                    # Get interest field names
                    if preferences['all_interest_fields']:
                        for field_id in preferences['all_interest_fields']:
                            field = data['fields'][data['fields']['Field_ID'] == field_id]
                            if not field.empty:
                                interest_fields.append(field.iloc[0]['FieldName'])

                else:
                    # User exists but no history - treat as new user
                    logger.info(f"User {user_id} has no interaction history")
                    user_id = None
            else:
                logger.info(f"User {user_id} not found")
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        # SCENARIO 2: New user or no history - Popular papers
        if user_id is None or recommendations.empty:
            logger.info("Returning popular papers")

            def _popular():
                return calculate_popularity_scores(data, field_filter=None, days=30)

            popularity_df = await run_in_threadpool(_popular)

            # Merge with paper details
            recommendations = popularity_df.merge(
                data['papers'][['Paper_ID', 'Title', 'Abstract', 'Keywords', 'PublicationDate', 'Field_ID']],
                on='Paper_ID',
                how='left'
            )

            # Add field names
            recommendations = recommendations.merge(
                data['fields'][['Field_ID', 'FieldName']],
                on='Field_ID',
                how='left'
            )

            # Add authors
            author_names = []
            for paper_id in recommendations['Paper_ID']:
                paper_authors = data['write'][data['write']['Paper_ID'] == paper_id]['Author_ID'].tolist()
                authors_list = []
                for author_id in paper_authors:
                    author = data['authors'][data['authors']['Author_ID'] == author_id]
                    if not author.empty:
                        authors_list.append(f"{author.iloc[0]['FName']} {author.iloc[0]['LName']}")
                author_names.append(', '.join(authors_list) if authors_list else 'Unknown')

            recommendations['Authors'] = author_names
            recommendations = recommendations.head(top_n)

            user_type = "new_user_popular"
            accuracy_score = 6.0

        # Format response
        if recommendations.empty:
            raise HTTPException(
                status_code=404,
                detail="No recommendations available"
            )

        result = []
        for _, paper in recommendations.iterrows():
            result.append(PaperRecommendation(
                paper_id=int(paper['Paper_ID']),
                title=paper['Title'],
                authors=paper.get('Authors', 'Unknown'),
                abstract=paper['Abstract'][:300] + "..." if len(str(paper['Abstract'])) > 300 else str(paper['Abstract']),
                keywords=paper.get('Keywords', ''),
                field_name=paper.get('FieldName', 'General'),
                publication_date=paper['PublicationDate'].strftime('%Y-%m-%d') if pd.notna(paper.get('PublicationDate')) else None,
                hybrid_score=float(round(paper['hybrid_score'] * 10, 2)) if 'hybrid_score' in paper.index and pd.notna(paper['hybrid_score']) else None,
                content_score=float(round(paper['content_score'] * 10, 2)) if 'content_score' in paper.index and pd.notna(paper['content_score']) else None,
                behavior_score=float(round(paper['behavior_score'] * 10, 2)) if 'behavior_score' in paper.index and pd.notna(paper['behavior_score']) else None,
                popularity_score=float(round(paper['popularity_score'], 2)) if 'popularity_score' in paper.index and pd.notna(paper['popularity_score']) else None
            ))

        return SmartRecommendationResponse(
            user_id=user_id,
            user_type=user_type,
            total_recommendations=len(result),
            accuracy_score=accuracy_score,
            interest_fields=interest_fields,
            recommendations=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart_recommend: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


# ============================================================================
# USER PROFILE ENDPOINT
# ============================================================================

@app.get("/api/user/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: int,
    conn = Depends(get_db)
):
    """
    Get detailed user profile and preferences

    Returns:
    - User information
    - Activity statistics
    - Interest fields (selected + derived)
    - Favorite authors
    """
    try:
        data, vectorizer, tfidf_matrix, paper_ids = await get_cached_data(conn)

        # Check if user exists
        user = data['users'][data['users']['User_ID'] == user_id]
        if user.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        user = user.iloc[0]

        researcher = data['researchers'][data['researchers']['User_ID'] == user_id]
        if researcher.empty:
            raise HTTPException(status_code=404, detail=f"Researcher {user_id} not found")

        researcher = researcher.iloc[0]

        # Get preferences using recommender.py function
        def _get_prefs():
            return get_user_preferences(user_id, data)

        preferences = await run_in_threadpool(_get_prefs)

        # Build interest fields list
        interest_fields = []
        for field_id in preferences['all_interest_fields']:
            field = data['fields'][data['fields']['Field_ID'] == field_id]
            if not field.empty:
                field_info = {
                    "field_id": int(field_id),
                    "field_name": field.iloc[0]['FieldName'],
                    "source": "selected" if field_id in preferences['selected_fields'] 
                             else "history" if field_id in preferences['favorite_fields']
                             else "specialization"
                }
                interest_fields.append(field_info)

        # Get favorite author names
        favorite_authors = []
        for author_id in preferences['favorite_authors']:
            author = data['authors'][data['authors']['Author_ID'] == author_id]
            if not author.empty:
                favorite_authors.append(f"{author.iloc[0]['FName']} {author.iloc[0]['LName']}")

        return UserProfileResponse(
            user_id=user_id,
            name=user['Name'],
            email=user['Email'],
            affiliation=researcher['Affiliation'],
            specialization=researcher['Specialization'],
            is_new_user=preferences['is_new_user'],
            rated_papers_count=len(preferences['rated_papers']),
            downloaded_papers_count=len(preferences['downloaded_papers']),
            avg_rating=float(preferences['avg_rating']),
            interest_fields=interest_fields,
            favorite_authors=favorite_authors
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class SearchInput(BaseModel):
    """Search input"""
    user_id: Optional[int]
    query: str


# ============================================================================
# INTERACTION LOGGING ENDPOINTS
# ============================================================================

@app.post("/api/interaction/review")
async def add_review(
    review: ReviewInput,
    conn = Depends(get_db)
):
    """
    Add or update a paper review

    This will improve future recommendations for the user
    """
    try:
        def _add():
            return add_review_to_db(conn, review.user_id, review.paper_id, review.rating)

        success = await run_in_threadpool(_add)

        if success:
            # Clear cache to refresh recommendations
            cache.clear()
            return {
                "success": True,
                "message": "Review added successfully",
                "user_id": review.user_id,
                "paper_id": review.paper_id,
                "rating": review.rating
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add review")

    except Exception as e:
        logger.error(f"Error adding review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interaction/download")
async def log_download(
    download: DownloadInput,
    conn = Depends(get_db)
):
    """
    Log a paper download

    This will improve future recommendations for the user
    """
    try:
        def _add():
            return add_download_to_db(conn, download.user_id, download.paper_id)

        success = await run_in_threadpool(_add)

        if success:
            cache.clear()
            return {
                "success": True,
                "message": "Download logged successfully",
                "user_id": download.user_id,
                "paper_id": download.paper_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to log download")

    except Exception as e:
        logger.error(f"Error logging download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interaction/search")
async def log_search(
    search: SearchInput,
    conn = Depends(get_db)
):
    """
    Log a search query

    This will improve future recommendations for the user
    """
    try:
        if not search.user_id:
            # Anonymous search - just return success but don't log to DB
            return {
                "success": True,
                "message": "Anonymous search logged (skipped DB)",
                "user_id": None,
                "query": search.query
            }

        def _add():
            return add_search_to_db(conn, search.user_id, search.query)

        success = await run_in_threadpool(_add)

        if success:
            cache.clear()
            return {
                "success": True,
                "message": "Search logged successfully",
                "user_id": search.user_id,
                "query": search.query
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to log search")

    except Exception as e:
        logger.error(f"Error logging search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Research Paper Recommender API",
        "version": settings.API_VERSION,
        "description": "Production API using advanced hybrid recommendation algorithms",
        "features": [
            "Hybrid recommendation (content + behavior + popularity)",
            "Field-based filtering",
            "User preference learning",
            "Automatic user type detection",
            "Interaction tracking"
        ],
        "endpoints": {
            "smart_recommend": "/api/recommend?user_id={id}&top_n={n}",
            "user_profile": "/api/user/{user_id}/profile",
            "add_review": "POST /api/interaction/review",
            "log_download": "POST /api/interaction/download",
            "log_search": "POST /api/interaction/search",
            "health": "/api/health",
            "cache_stats": "/api/cache/stats",
            "cache_clear": "POST /api/cache/clear"
        }
    }


@app.get("/api/health")
async def health_check(conn = Depends(get_db)):
    """Health check"""
    try:
        def _test():
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

        await run_in_threadpool(_test)

        return {
            "status": "healthy",
            "database": "connected",
            "cache_stats": cache.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "cache_stats": cache.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/cache/clear")
async def clear_cache_endpoint():
    """Clear application cache"""
    cache.clear()
    logger.info("Cache cleared via API")
    return {
        "message": "Cache cleared successfully",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 Research Paper Recommender API - Production Ready")
    print("="*80)
    print(f"\n🎯 Environment: {settings.ENVIRONMENT}")
    print(f"🗄️  Database: {settings.DB_NAME}@{settings.DB_HOST}")
    print(f"\n📖 Documentation: http://localhost:8000/docs")
    print(f"💚 Health Check: http://localhost:8000/api/health")
    print(f"\n🔥 Main Features:")
    print("   ✓ Hybrid recommendation algorithm")
    print("   ✓ Field-based filtering")
    print("   ✓ User preference learning")
    print("   ✓ Interaction tracking")
    print("\n⚠️  Press CTRL+C to stop\n")
    print("="*80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
