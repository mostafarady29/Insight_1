"""
Research Paper Recommender System - Enhanced Version
Non-OOP Implementation using Functional Programming
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import warnings
import sys
import io

# Set encoding
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# ============================================================================
# DATABASE CONNECTION AND DATA LOADING
# ============================================================================

def load_data_from_db(conn):
    """
    Load all necessary data from database into pandas DataFrames
    """
    print("\n" + "="*80)
    print("Loading data from the database...")
    print("="*80)

    queries = {
        'users': "SELECT * FROM [User]",
        'researchers': "SELECT * FROM [Researcher]",
        'papers': "SELECT * FROM [Paper]",
        'authors': "SELECT * FROM [Author]",
        'write': "SELECT * FROM [Author_Paper]",
        'fields': "SELECT * FROM [Field]",
        'reviews': "SELECT * FROM [Review]",
        'downloads': "SELECT * FROM [Download]",
        'searches': "SELECT * FROM [Search]",
        'paper_keywords': "SELECT * FROM [Paper_Keywords]",
        'researcher_fields': "SELECT * FROM [Researcher_Field]"
    }

    data = {}
    
    for table_name, query in queries.items():
        try:
            df = pd.read_sql_query(query, conn)

            # Column name standardization
            rename_map = {}

            for col in df.columns:
                if table_name == 'researchers' and col == 'Researcher_ID':
                    rename_map[col] = 'User_ID'
                elif col == 'First_Name':
                    rename_map[col] = 'FName'
                elif col == 'Last_Name':
                    rename_map[col] = 'LName'
                elif col == 'Publication_Date':
                    rename_map[col] = 'PublicationDate'
                elif col == 'Review_Date':
                    rename_map[col] = 'ReviewDate'
                elif col == 'Download_Date':
                    rename_map[col] = 'DownloadDate'
                elif col == 'Search_Date':
                    rename_map[col] = 'SearchDate'
                elif col == 'Join_Date':
                    rename_map[col] = 'JoinDate'
                elif table_name in ['reviews', 'downloads', 'searches'] and col == 'Researcher_ID':
                    rename_map[col] = 'User_ID'
                elif table_name == 'researcher_fields' and col == 'Researcher_ID':
                    rename_map[col] = 'User_ID'
                elif col == 'Field_Name':
                    rename_map[col] = 'FieldName'
                elif col == 'numeber_of_papers':
                    rename_map[col] = 'No_Papers'

            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            data[table_name] = df
            print(f"✓ Loaded {table_name}: {len(df)} records")

        except Exception as e:
            print(f"X Error loading {table_name}: {e}")
            data[table_name] = pd.DataFrame()

    # Build Keywords column from Paper_Keywords table
    if not data['paper_keywords'].empty and not data['papers'].empty:
        keyword_col = None
        possible_names = ['Keyword', 'Keywords', 'Keyword_Text', 'KeywordName', 'Keyword_Name']
        
        for col_name in possible_names:
            if col_name in data['paper_keywords'].columns:
                keyword_col = col_name
                break
        
        if keyword_col is None:
            cols = data['paper_keywords'].columns.tolist()
            if len(cols) >= 3:
                keyword_col = cols[2]
            elif len(cols) >= 2:
                keyword_col = cols[1]
        
        if keyword_col:
            try:
                keywords_grouped = data['paper_keywords'].groupby('Paper_ID')[keyword_col].apply(
                    lambda x: ', '.join(x.astype(str))
                ).reset_index()
                keywords_grouped.columns = ['Paper_ID', 'Keywords']
                
                data['papers'] = data['papers'].merge(keywords_grouped, on='Paper_ID', how='left')
                data['papers']['Keywords'] = data['papers']['Keywords'].fillna('')
                print(f"✓ Built Keywords column from Paper_Keywords table using '{keyword_col}' column")
            except Exception as e:
                print(f"⚠ Warning: Error building Keywords column: {e}")
                data['papers']['Keywords'] = ''
        else:
            data['papers']['Keywords'] = ''
            print(f"⚠ Warning: Cannot detect keyword column")
    else:
        data['papers']['Keywords'] = ''
        print(f"⚠ Warning: Paper_Keywords table is empty")

    print("="*80 + "\n")
    return data


# ============================================================================
# DATABASE UPDATE FUNCTIONS
# ============================================================================

def add_new_researcher_to_db(conn, name, email, password, affiliation, specialization, field_ids):
    """
    Add a new researcher to the database
    """
    cursor = conn.cursor()
    
    try:
        # Get next User_ID
        cursor.execute("SELECT ISNULL(MAX(User_ID), 0) + 1 FROM [User]")
        new_user_id = cursor.fetchone()[0]
        
        # Insert into User table
        cursor.execute("""
            INSERT INTO [User] (User_ID, Name, Email, Password, Role)
            VALUES (?, ?, ?, ?, 'Researcher')
        """, (new_user_id, name, email, password))
        
        # Insert into Researcher table
        join_date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            INSERT INTO Researcher (Researcher_ID, Affiliation, Specialization, Join_Date)
            VALUES (?, ?, ?, ?)
        """, (new_user_id, affiliation, specialization, join_date))
        
        # Insert field interests into Researcher_Field table
        if field_ids:
            for field_id in field_ids:
                try:
                    cursor.execute("""
                        INSERT INTO Researcher_Field (Researcher_ID, Field_ID)
                        VALUES (?, ?)
                    """, (new_user_id, field_id))
                except:
                    pass
        
        conn.commit()
        print(f"✅ New researcher added successfully! User ID: {new_user_id}")
        return new_user_id
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error adding new researcher: {e}")
        return None


def add_review_to_db(conn, user_id, paper_id, rating):
    """
    Add or update a paper review in database
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT Review_ID FROM Review 
            WHERE Researcher_ID = ? AND Paper_ID = ?
        """, (user_id, paper_id))
        
        existing = cursor.fetchone()
        review_date = datetime.now().strftime('%Y-%m-%d')
        
        if existing:
            cursor.execute("""
                UPDATE Review 
                SET Rating = ?, Review_Date = ?
                WHERE Researcher_ID = ? AND Paper_ID = ?
            """, (rating, review_date, user_id, paper_id))
            print("✅ Review updated successfully!")
        else:
            cursor.execute("""
                INSERT INTO Review (Rating, Review_Date, Paper_ID, Researcher_ID)
                VALUES (?, ?, ?, ?)
            """, (rating, review_date, paper_id, user_id))
            print("✅ Review added successfully!")
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error adding review: {e}")
        return False


def add_download_to_db(conn, user_id, paper_id):
    """
    Log a paper download in database
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT Download_ID FROM Download 
            WHERE Researcher_ID = ? AND Paper_ID = ?
        """, (user_id, paper_id))
        
        if cursor.fetchone():
            print("ℹ Paper already downloaded by this user")
            return True
        
        download_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO Download (Download_Date, Paper_ID, Researcher_ID)
            VALUES (?, ?, ?)
        """, (download_date, paper_id, user_id))
        
        conn.commit()
        print("✅ Download logged successfully!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error logging download: {e}")
        return False


def add_search_to_db(conn, user_id, query):
    """
    Log a search query in database
    """
    cursor = conn.cursor()
    
    try:
        search_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO Search (Query, Search_Date, Researcher_ID)
            VALUES (?, ?, ?)
        """, (query, search_date, user_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error logging search: {e}")
        return False


# ============================================================================
# USER PROFILE FUNCTIONS
# ============================================================================

def get_user_preferences(user_id, data):
    """
    Extract user preferences from historical behavior and profile
    محسّن: يركز على المجالات المختارة والاهتمامات الفعلية
    """
    preferences = {
        'user_id': user_id,
        'favorite_authors': [],
        'favorite_fields': [],
        'rated_papers': [],
        'downloaded_papers': [],
        'avg_rating': 0.0,
        'specialization': None,
        'is_new_user': False,
        'search_keywords': [],
        'selected_fields': [],
        'all_interest_fields': []  # NEW: Combined list of all fields of interest
    }

    # Get user specialization
    user_spec = data['researchers'][data['researchers']['User_ID'] == user_id]
    if not user_spec.empty:
        preferences['specialization'] = user_spec.iloc[0]['Specialization']

    # PRIORITY 1: Get explicitly selected fields from Researcher_Field table
    if 'researcher_fields' in data and not data['researcher_fields'].empty:
        user_selected_fields = data['researcher_fields'][data['researcher_fields']['User_ID'] == user_id]
        if not user_selected_fields.empty:
            preferences['selected_fields'] = user_selected_fields['Field_ID'].tolist()

    # Get rated papers and calculate average rating
    user_reviews = data['reviews'][data['reviews']['User_ID'] == user_id]
    if not user_reviews.empty:
        preferences['rated_papers'] = user_reviews['Paper_ID'].tolist()
        preferences['avg_rating'] = user_reviews['Rating'].mean()

    # Get downloaded papers
    user_downloads = data['downloads'][data['downloads']['User_ID'] == user_id]
    if not user_downloads.empty:
        preferences['downloaded_papers'] = user_downloads['Paper_ID'].tolist()

    # Get search keywords
    user_searches = data['searches'][data['searches']['User_ID'] == user_id]
    if not user_searches.empty:
        preferences['search_keywords'] = user_searches['Query'].tolist()

    # Identify favorite authors (from downloaded/rated papers)
    interacted_papers = list(set(preferences['rated_papers'] + preferences['downloaded_papers']))
    if interacted_papers:
        author_papers = data['write'][data['write']['Paper_ID'].isin(interacted_papers)]
        author_counts = author_papers['Author_ID'].value_counts()
        preferences['favorite_authors'] = author_counts.head(5).index.tolist()

    # PRIORITY 2: Identify favorite fields from interaction history
    if interacted_papers:
        field_papers = data['papers'][data['papers']['Paper_ID'].isin(interacted_papers)]
        field_counts = field_papers['Field_ID'].value_counts()
        preferences['favorite_fields'] = field_counts.head(3).index.tolist()

    # NEW: Create combined list of ALL fields of interest (selected + favorite + specialization-based)
    all_fields = set()
    
    # Add selected fields (highest priority)
    if preferences['selected_fields']:
        all_fields.update(preferences['selected_fields'])
    
    # Add favorite fields from history
    if preferences['favorite_fields']:
        all_fields.update(preferences['favorite_fields'])
    
    # Add specialization-based fields (lowest priority)
    if preferences['specialization']:
        spec_fields = get_fields_from_specialization(preferences['specialization'], data)
        all_fields.update(spec_fields)
        
    # NEW: Add fields inferred from search history
    if preferences['search_keywords']:
        # Combine all search queries into one text for matching
        search_text = ' '.join(preferences['search_keywords'])
        search_fields = get_fields_from_specialization(search_text, data) # Reuse the matching logic
        all_fields.update(search_fields)
    
    preferences['all_interest_fields'] = list(all_fields)

    # Check if new user (no interactions or searches)
    # User is NOT new if they have rated/downloaded papers OR have search history
    if len(interacted_papers) == 0 and not preferences['search_keywords']:
        preferences['is_new_user'] = True

    return preferences


def get_fields_from_specialization(specialization, data):
    """
    NEW: Map specialization to field IDs
    يطابق التخصص مع المجالات المتاحة
    """
    field_ids = []
    
    if not specialization or data['fields'].empty:
        return field_ids
    
    spec_words = specialization.lower().split()
    
    for idx, row in data['fields'].iterrows():
        field_name = str(row['FieldName']).lower()
        field_desc = str(row.get('Description', '')).lower() if 'Description' in row else ''
        
        # Check if any word from specialization matches field name or description
        if any(word in field_name or word in field_desc for word in spec_words):
            field_ids.append(row['Field_ID'])
    
    return field_ids


def get_user_interest_papers(user_id, data, exclude_interacted=True):
    """
    NEW: Get all papers in user's fields of interest
    جلب جميع الأوراق في المجالات التي يهتم بها المستخدم
    
    Args:
        user_id: User identifier
        data: Dictionary of dataframes
        exclude_interacted: Whether to exclude already rated/downloaded papers
    
    Returns:
        list: Paper IDs in user's interest fields
    """
    preferences = get_user_preferences(user_id, data)
    interest_fields = preferences['all_interest_fields']
    
    if not interest_fields:
        return []
    
    # Get all papers in interest fields
    interest_papers = data['papers'][data['papers']['Field_ID'].isin(interest_fields)]['Paper_ID'].tolist()
    
    if exclude_interacted:
        # Remove already interacted papers
        interacted = set(preferences['rated_papers'] + preferences['downloaded_papers'])
        interest_papers = [pid for pid in interest_papers if pid not in interacted]
    
    return interest_papers


# ============================================================================
# CONTENT-BASED FILTERING - محسّن
# ============================================================================

def preprocess_text(text):
    """
    Preprocess text for vectorization
    """
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def build_content_vectors(data):
    """
    Build TF-IDF vectors for all papers
    """
    papers = data['papers'].copy()

    # Combine abstract and keywords for richer content representation
    papers['content'] = (papers['Abstract'].apply(preprocess_text) + ' ' + 
                         papers['Keywords'].apply(preprocess_text))

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(papers['content'])

    return vectorizer, tfidf_matrix, papers['Paper_ID'].tolist()


def content_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n=10):
    """
    محسّن: يركز فقط على الأوراق في مجالات اهتمام المستخدم
    Generate content-based recommendations ONLY from user's interest fields
    """
    preferences = get_user_preferences(user_id, data)
    
    # STEP 1: Get candidate papers (only from interest fields)
    candidate_paper_ids = get_user_interest_papers(user_id, data, exclude_interacted=True)
    
    if not candidate_paper_ids:
        print(f"⚠ No more papers available in your interest fields.")
        return pd.DataFrame()
    
    print(f"📊 Found {len(candidate_paper_ids)} papers in your interest fields")

    # STEP 2: Build user profile vector
    interacted_papers = list(set(preferences['rated_papers'] + preferences['downloaded_papers']))

    if interacted_papers:
        # Existing user: Use interaction history
        paper_indices = [paper_ids.index(pid) for pid in interacted_papers if pid in paper_ids]
        if paper_indices:
            user_profile_vector = np.asarray(tfidf_matrix[paper_indices].mean(axis=0))
        else:
            # Fallback to specialization
            user_profile_vector = build_profile_from_interests(user_id, data, vectorizer)
    else:
        # New user: Use specialization and selected fields
        user_profile_vector = build_profile_from_interests(user_id, data, vectorizer)

    # STEP 3: Calculate similarity only for candidate papers
    candidate_indices = [paper_ids.index(pid) for pid in candidate_paper_ids if pid in paper_ids]
    
    if not candidate_indices:
        return pd.DataFrame()
    
    candidate_matrix = tfidf_matrix[candidate_indices]
    similarities = cosine_similarity(user_profile_vector, candidate_matrix).flatten()

    # STEP 4: Create recommendations dataframe
    recommendations = pd.DataFrame({
        'Paper_ID': [paper_ids[idx] for idx in candidate_indices],
        'content_score': similarities
    })

    # Sort by similarity score
    recommendations = recommendations.sort_values('content_score', ascending=False).head(top_n)

    return recommendations


def build_profile_from_interests(user_id, data, vectorizer):
    """
    NEW: Build user profile vector from specialization and selected fields
    بناء profile المستخدم من التخصص والمجالات المختارة
    """
    preferences = get_user_preferences(user_id, data)
    profile_text_parts = []
    
    # Add specialization
    if preferences['specialization']:
        profile_text_parts.append(preferences['specialization'].lower())
    
    # Add field names and descriptions
    if preferences['all_interest_fields']:
        for field_id in preferences['all_interest_fields']:
            field_row = data['fields'][data['fields']['Field_ID'] == field_id]
            if not field_row.empty:
                field_name = field_row.iloc[0]['FieldName']
                profile_text_parts.append(field_name.lower())
                
                if 'Description' in field_row.columns and pd.notna(field_row.iloc[0]['Description']):
                    profile_text_parts.append(field_row.iloc[0]['Description'].lower())
    
    # Add search keywords
    if preferences['search_keywords']:
        profile_text_parts.extend([kw.lower() for kw in preferences['search_keywords'][:5]])
    
    # Combine all text
    profile_text = ' '.join(profile_text_parts) if profile_text_parts else 'research papers'
    
    # Create profile vector
    try:
        spec_vector = vectorizer.transform([profile_text])
        return np.asarray(spec_vector.toarray())
    except:
        # Fallback to mean vector
        return np.asarray([[0.0] * len(vectorizer.get_feature_names_out())])


# ============================================================================
# BEHAVIOR-BASED FILTERING - محسّن
# ============================================================================

def calculate_author_preference_score(user_id, data, paper_ids):
    """
    Calculate author preference scores for papers
    يحسب درجة تفضيل المؤلفين
    """
    preferences = get_user_preferences(user_id, data)
    favorite_authors = preferences['favorite_authors']

    if not favorite_authors:
        return {pid: 0.0 for pid in paper_ids}

    scores = {}
    for pid in paper_ids:
        paper_authors = data['write'][data['write']['Paper_ID'] == pid]['Author_ID'].tolist()
        overlap = len(set(paper_authors) & set(favorite_authors))
        scores[pid] = overlap / len(favorite_authors) if favorite_authors else 0.0

    return scores


def calculate_field_preference_score(user_id, data, paper_ids):
    """
    محسّن: يعطي درجة 1.0 فقط للأوراق في المجالات المفضلة
    Calculate field preference scores - STRICT matching
    """
    preferences = get_user_preferences(user_id, data)
    interest_fields = preferences['all_interest_fields']

    if not interest_fields:
        return {pid: 0.0 for pid in paper_ids}

    # STRICT: Papers in interest fields get 1.0, others get 0.0
    scores = {}
    for pid in paper_ids:
        paper_field_df = data['papers'][data['papers']['Paper_ID'] == pid]
        if not paper_field_df.empty:
            paper_field_id = paper_field_df.iloc[0]['Field_ID']
            scores[pid] = 1.0 if paper_field_id in interest_fields else 0.0
        else:
            scores[pid] = 0.0

    return scores


def calculate_rating_boost(user_id, data, paper_ids):
    """
    Calculate rating-based boost for papers
    """
    preferences = get_user_preferences(user_id, data)

    if not preferences['rated_papers']:
        return {pid: 0.0 for pid in paper_ids}

    user_reviews = data['reviews'][data['reviews']['User_ID'] == user_id]
    highly_rated = user_reviews[user_reviews['Rating'] >= 4]['Paper_ID'].tolist()

    if not highly_rated:
        return {pid: 0.0 for pid in paper_ids}

    scores = {}
    for pid in paper_ids:
        paper_authors = set(data['write'][data['write']['Paper_ID'] == pid]['Author_ID'])
        paper_field_df = data['papers'][data['papers']['Paper_ID'] == pid]
        paper_fields = set(paper_field_df['Field_ID'].tolist()) if not paper_field_df.empty else set()

        boost = 0.0
        for rated_pid in highly_rated:
            rated_authors = set(data['write'][data['write']['Paper_ID'] == rated_pid]['Author_ID'])
            rated_field_df = data['papers'][data['papers']['Paper_ID'] == rated_pid]
            rated_fields = set(rated_field_df['Field_ID'].tolist()) if not rated_field_df.empty else set()

            author_overlap = len(paper_authors & rated_authors)
            field_overlap = len(paper_fields & rated_fields)

            boost += (author_overlap * 0.6 + field_overlap * 0.4) / len(highly_rated)

        scores[pid] = boost

    return scores


def behavior_recommend(user_id, data, top_n=10):
    """
    محسّن: يوصي فقط بالأوراق في مجالات الاهتمام
    Generate behavior-based recommendations ONLY from interest fields
    """
    # Get candidate papers (only from interest fields)
    candidate_papers = get_user_interest_papers(user_id, data, exclude_interacted=True)
    
    if not candidate_papers:
        return pd.DataFrame()

    # Calculate different behavior scores
    author_scores = calculate_author_preference_score(user_id, data, candidate_papers)
    field_scores = calculate_field_preference_score(user_id, data, candidate_papers)
    rating_boosts = calculate_rating_boost(user_id, data, candidate_papers)

    # Combine scores (weighted average)
    behavior_scores = {}
    for pid in candidate_papers:
        behavior_scores[pid] = (
            0.3 * author_scores[pid] +
            0.5 * field_scores[pid] +  # Higher weight for field matching
            0.2 * rating_boosts[pid]
        )

    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'Paper_ID': list(behavior_scores.keys()),
        'behavior_score': list(behavior_scores.values())
    })

    # Sort by behavior score
    recommendations = recommendations.sort_values('behavior_score', ascending=False).head(top_n)

    return recommendations


# ============================================================================
# POPULARITY-BASED (WITHIN INTEREST FIELDS)
# ============================================================================

def calculate_popularity_scores(data, field_filter=None, days=30):
    """
    محسّن: يحسب الشعبية فقط ضمن المجالات المحددة
    Calculate popularity scores (optionally filtered by fields)
    
    Args:
        data: Dictionary of dataframes
        field_filter: List of field IDs to filter by (None = all fields)
        days: Number of days to consider
    
    Returns:
        pd.DataFrame: Papers with popularity scores
    """
    # Filter downloads from last N days
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_downloads = data['downloads'].copy()
    recent_downloads['DownloadDate'] = pd.to_datetime(recent_downloads['DownloadDate'])
    recent_downloads = recent_downloads[recent_downloads['DownloadDate'] >= cutoff_date]

    # Count downloads per paper
    download_counts = recent_downloads['Paper_ID'].value_counts().reset_index()
    download_counts.columns = ['Paper_ID', 'download_count']

    # Get average ratings
    avg_ratings = data['reviews'].groupby('Paper_ID')['Rating'].mean().reset_index()
    avg_ratings.columns = ['Paper_ID', 'avg_rating']

    # Start with papers (optionally filtered by field)
    if field_filter:
        papers_base = data['papers'][data['papers']['Field_ID'].isin(field_filter)][['Paper_ID']].copy()
    else:
        papers_base = data['papers'][['Paper_ID']].copy()
    
    # Combine download count and ratings
    popularity = papers_base.merge(download_counts, on='Paper_ID', how='left')
    popularity = popularity.merge(avg_ratings, on='Paper_ID', how='left')

    popularity['download_count'] = popularity['download_count'].fillna(0)
    popularity['avg_rating'] = popularity['avg_rating'].fillna(3.0)

    # Normalize scores
    max_downloads = popularity['download_count'].max()
    if max_downloads > 0:
        popularity['norm_downloads'] = popularity['download_count'] / max_downloads
    else:
        popularity['norm_downloads'] = 0.0

    popularity['norm_rating'] = popularity['avg_rating'] / 5.0

    # Calculate popularity score (weighted combination)
    popularity['popularity_score'] = (
        0.6 * popularity['norm_downloads'] +
        0.4 * popularity['norm_rating']
    )

    return popularity[['Paper_ID', 'popularity_score', 'download_count', 'avg_rating']]


# ============================================================================
# HYBRID RECOMMENDATION - محسّن للغاية
# ============================================================================

def calculate_recommendation_accuracy(user_id, recommendations, preferences, data):
    """
    محسّن: حساب دقة التوصيات
    Calculate recommendation accuracy score out of 10
    """
    if recommendations.empty:
        return 0.0
    
    score = 0.0
    max_score = 10.0
    
    # Factor 1: Field match (4 points - INCREASED)
    interest_fields = preferences['all_interest_fields']
    
    if interest_fields:
        field_matches = recommendations[recommendations['Field_ID'].isin(interest_fields)]
        field_score = (len(field_matches) / len(recommendations)) * 4.0
        score += field_score
    else:
        score += 2.0
    
    # Factor 2: Author preference match (2 points)
    if preferences['favorite_authors']:
        author_match_count = 0
        for paper_id in recommendations['Paper_ID']:
            paper_authors = set(data['write'][data['write']['Paper_ID'] == paper_id]['Author_ID'])
            if paper_authors & set(preferences['favorite_authors']):
                author_match_count += 1
        author_score = (author_match_count / len(recommendations)) * 2.0
        score += author_score
    else:
        score += 1.0
    
    # Factor 3: Keyword relevance (2 points)
    relevant_keywords = preferences['search_keywords'] if preferences['search_keywords'] else []
    if not relevant_keywords and preferences['specialization']:
        relevant_keywords = preferences['specialization'].lower().split()
    
    if relevant_keywords:
        keyword_match_count = 0
        for idx, row in recommendations.iterrows():
            keywords = str(row['Keywords']).lower()
            abstract = str(row.get('Abstract', '')).lower()
            combined_text = keywords + ' ' + abstract
            if any(keyword.lower() in combined_text for keyword in relevant_keywords):
                keyword_match_count += 1
        keyword_score = (keyword_match_count / len(recommendations)) * 2.0
        score += keyword_score
    else:
        score += 1.0
    
    # Factor 4: Rating history alignment (1.5 points)
    if preferences['rated_papers'] and preferences['avg_rating'] > 0:
        highly_rated = data['reviews'][
            (data['reviews']['User_ID'] == user_id) & 
            (data['reviews']['Rating'] >= 4)
        ]['Paper_ID'].tolist()
        
        if highly_rated:
            similarity_count = 0
            for paper_id in recommendations['Paper_ID']:
                paper_field_df = data['papers'][data['papers']['Paper_ID'] == paper_id]
                if not paper_field_df.empty:
                    paper_field = paper_field_df.iloc[0]['Field_ID']
                    
                    for rated_pid in highly_rated:
                        rated_field_df = data['papers'][data['papers']['Paper_ID'] == rated_pid]
                        if not rated_field_df.empty:
                            if rated_field_df.iloc[0]['Field_ID'] == paper_field:
                                similarity_count += 1
                                break
            
            rating_score = min((similarity_count / len(recommendations)) * 1.5, 1.5)
            score += rating_score
        else:
            score += 0.75
    else:
        score += 0.75
    
    # Factor 5: Content quality (0.5 points)
    if 'hybrid_score' in recommendations.columns:
        avg_hybrid_score = recommendations['hybrid_score'].mean()
        content_quality_score = min(avg_hybrid_score * 1.0, 0.5)
        score += content_quality_score
    else:
        score += 0.25
    
    # Ensure score is between 0 and 10
    score = max(0.0, min(score, max_score))
    
    return round(score, 2)


def hybrid_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n=10):
    """
    محسّن بشكل كامل: توصيات هجينة تركز فقط على مجالات اهتمام المستخدم
    Generate hybrid recommendations ONLY from user's interest fields
    NO RANDOM RECOMMENDATIONS - Only papers in user's interest fields
    
    Returns:
        tuple: (recommendations DataFrame, accuracy_score out of 10)
    """
    preferences = get_user_preferences(user_id, data)
    interest_fields = preferences['all_interest_fields']
    
    # Check if user has any interest fields
    if not interest_fields:
        print(f"\n⚠ No interest fields found. Please update your profile with research interests.")
        return pd.DataFrame(), 0.0
    
    print(f"\n{'─'*100}")
    print(f"🎯 Your Interest Fields: {len(interest_fields)} field(s)")
    for field_id in interest_fields:
        field_row = data['fields'][data['fields']['Field_ID'] == field_id]
        if not field_row.empty:
            print(f"   • {field_row.iloc[0]['FieldName']}")
    print(f"{'─'*100}")

    # Check for available papers in interest fields
    available_papers = get_user_interest_papers(user_id, data, exclude_interacted=True)
    
    if not available_papers:
        print(f"\n⚠ No more unread papers available in your interest fields.")
        print(f"💡 Suggestion: Add more research fields to your profile or rate more papers.")
        return pd.DataFrame(), 0.0
    
    print(f"📊 Found {len(available_papers)} unread papers in your interest fields")
    
    interacted_papers = list(set(preferences['rated_papers'] + preferences['downloaded_papers']))
    
    if len(interacted_papers) == 0:
        print(f"[New User Mode] Generating recommendations based on your selected interests...")
    else:
        print(f"[Active User Mode] Using your {len(interacted_papers)} interactions for better recommendations...")
    
    # Get content-based recommendations (only from interest fields)
    content_recs = content_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n=50)
    
    if content_recs.empty:
        print(f"\n⚠ Could not generate content-based recommendations.")
        return pd.DataFrame(), 0.0
    
    # Get behavior-based recommendations (only from interest fields)
    behavior_recs = behavior_recommend(user_id, data, top_n=50)
    
    # Get popularity scores (filtered by interest fields)
    popularity = calculate_popularity_scores(data, field_filter=interest_fields)
    
    # Merge all scores
    hybrid = content_recs.merge(behavior_recs, on='Paper_ID', how='outer')
    hybrid = hybrid.merge(popularity[['Paper_ID', 'popularity_score']], on='Paper_ID', how='left')
    
    # Fill missing scores
    hybrid['content_score'] = hybrid['content_score'].fillna(0)
    hybrid['behavior_score'] = hybrid['behavior_score'].fillna(0)
    hybrid['popularity_score'] = hybrid['popularity_score'].fillna(0)
    
    # Calculate hybrid score with adjusted weights
    if len(interacted_papers) == 0:
        # New users: Prioritize content and popularity
        hybrid['hybrid_score'] = (
            0.5 * hybrid['content_score'] +
            0.2 * hybrid['behavior_score'] +
            0.3 * hybrid['popularity_score']
        )
    else:
        # Existing users: Balance all factors
        hybrid['hybrid_score'] = (
            0.4 * hybrid['content_score'] +
            0.4 * hybrid['behavior_score'] +
            0.2 * hybrid['popularity_score']
        )
    
    # Sort by hybrid score and take top N
    hybrid = hybrid.sort_values('hybrid_score', ascending=False).head(top_n)
    
    # Add paper details
    hybrid = add_paper_details(hybrid, data)
    
    # Filter one more time to ensure only interest field papers (safety check)
    hybrid = hybrid[hybrid['Field_ID'].isin(interest_fields)]
    
    if hybrid.empty:
        print(f"\n⚠ No recommendations could be generated from your interest fields.")
        return pd.DataFrame(), 0.0
    
    # Calculate accuracy score
    accuracy_score = calculate_recommendation_accuracy(user_id, hybrid, preferences, data)
    
    print(f"\n✅ Generated {len(hybrid)} recommendations from your interest fields")
    
    return hybrid, accuracy_score


def add_paper_details(recommendations, data):
    """
    Add paper details (title, abstract, authors, field) to recommendations
    """
    if recommendations.empty:
        return recommendations
    
    existing_cols = recommendations.columns.tolist()
    
    if 'Title' not in existing_cols or 'Abstract' not in existing_cols:
        recommendations = recommendations.merge(
            data['papers'][['Paper_ID', 'Title', 'Abstract', 'Keywords', 'PublicationDate', 'Field_ID']], 
            on='Paper_ID', 
            how='left',
            suffixes=('', '_paper')
        )
        
        for col in ['Title', 'Abstract', 'Keywords', 'PublicationDate', 'Field_ID']:
            if f'{col}_paper' in recommendations.columns:
                recommendations[col] = recommendations[col].fillna(recommendations[f'{col}_paper'])
                recommendations.drop(columns=[f'{col}_paper'], inplace=True)

    if 'FieldName' not in existing_cols:
        recommendations = recommendations.merge(
            data['fields'][['Field_ID', 'FieldName']], 
            on='Field_ID', 
            how='left'
        )

    # Add author names
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

    return recommendations


# ============================================================================
# DISPLAY AND UTILITY FUNCTIONS
# ============================================================================

def display_recommendations(recommendations, accuracy_score, user_id):
    """
    Display recommendations in a readable format
    """
    print(f"\n{'='*100}")
    print(f"📋 Personalized Recommendations for Researcher (User ID: {user_id})")
    print(f"{'='*100}\n")

    print(f"📊 Recommendation Accuracy: {accuracy_score:.2f}/10")
    
    if recommendations.empty:
        print(f"\n❌ No recommendations available in your interest fields.")
        print(f"\n💡 Suggestions:")
        print(f"   • Add more research fields to your profile")
        print(f"   • Rate some papers to improve recommendations")
        print(f"   • Broaden your research interests")
        return
    
    print(f"📄 Showing {len(recommendations)} papers from your interest fields")
    print(f"{'─'*100}\n")

    for idx, row in recommendations.iterrows():
        print(f"📄 #{idx + 1}. {row.get('Title', 'Not available')}")
        print(f"    🆔 Paper ID: {row.get('Paper_ID', 'Not available')}")

        if 'Authors' in row and pd.notna(row['Authors']):
            print(f"    ✍  Authors: {row['Authors']}")

        if 'FieldName' in row and pd.notna(row['FieldName']):
            print(f"    📚 Field: {row['FieldName']}")

        if 'Keywords' in row and pd.notna(row['Keywords']) and row['Keywords']:
            print(f"    🔑 Keywords: {row['Keywords']}")

        # Display scores - ALL OUT OF 10
        if 'hybrid_score' in row and pd.notna(row['hybrid_score']):
            display_score = round(row['hybrid_score'] * 10, 2)
            print(f"    ⭐ Hybrid Score: {display_score:.2f}/10")
        if 'content_score' in row and pd.notna(row['content_score']):
            display_score = round(row['content_score'] * 10, 2)
            print(f"    📊 Content Score: {display_score:.2f}/10")
        if 'behavior_score' in row and pd.notna(row['behavior_score']):
            display_score = round(row['behavior_score'] * 10, 2)
            print(f"    🎯 Behavior Score: {display_score:.2f}/10")
        if 'popularity_score' in row and pd.notna(row['popularity_score']):
            display_score = round(row['popularity_score'] * 10, 2)
            print(f"    🔥 Popularity Score: {display_score:.2f}/10")
        
        if 'avg_rating' in row and pd.notna(row['avg_rating']):
            print(f"    ⭐ Average Rating: {row['avg_rating']:.2f}/5")

        if 'Abstract' in row and pd.notna(row['Abstract']):
            abstract_preview = str(row['Abstract'])[:200]
            print(f"    📝 Abstract: {abstract_preview}...")

        print()
    
    print(f"{'─'*100}")
    print(f"💡 All recommendations are from your selected research fields")
    print(f"{'─'*100}")


def display_user_profile(user_id, data):
    """
    Display user profile and preferences
    """
    preferences = get_user_preferences(user_id, data)

    user = data['users'][data['users']['User_ID'] == user_id]
    if user.empty:
        print(f"\n❌ User {user_id} not found.")
        return
    user = user.iloc[0]

    researcher = data['researchers'][data['researchers']['User_ID'] == user_id]
    if researcher.empty:
        print(f"\n❌ Researcher {user_id} not found in researchers table.")
        return
    researcher = researcher.iloc[0]

    print(f"\n{'='*100}")
    print(f"👤 Researcher Profile: {user['Name']} (ID: {user_id})")
    print(f"{'='*100}")
    print(f"📧 Email: {user['Email']}")
    print(f"🏛 Affiliation: {researcher['Affiliation']}")
    print(f"🎓 Specialization: {researcher['Specialization']}")

    if 'JoinDate' in researcher.index:
        print(f"📅 Join Date: {researcher['JoinDate']}")
    
    print(f"\n{'─'*100}")
    print(f"📊 Activity Statistics:")
    print(f"{'─'*100}")
    print(f"User Status: {'👶 New User' if preferences['is_new_user'] else '✅ Active User'}")
    print(f"Rated Papers: {len(preferences['rated_papers'])}")
    print(f"Downloaded Papers: {len(preferences['downloaded_papers'])}")
    if preferences['avg_rating'] > 0:
        print(f"Average Rating Given: {preferences['avg_rating']:.2f}/5")

    # Show ALL interest fields
    if preferences['all_interest_fields'] and not data['fields'].empty:
        print(f"\n🎯 Your Research Interest Fields ({len(preferences['all_interest_fields'])} total):")
        for field_id in preferences['all_interest_fields']:
            field = data['fields'][data['fields']['Field_ID'] == field_id]
            if not field.empty:
                field = field.iloc[0]
                # Mark selected vs derived fields
                if field_id in preferences['selected_fields']:
                    print(f"  ✓ {field['FieldName']} (ID: {field_id}) [Selected]")
                elif field_id in preferences['favorite_fields']:
                    print(f"  • {field['FieldName']} (ID: {field_id}) [From History]")
                else:
                    print(f"  ○ {field['FieldName']} (ID: {field_id}) [From Specialization]")

    if preferences['search_keywords']:
        print(f"\n🔍 Recent Search Keywords:")
        for keyword in list(set(preferences['search_keywords']))[:5]:
            print(f"  • {keyword}")

    if preferences['favorite_authors'] and not data['authors'].empty:
        print(f"\n✍  Favorite Authors (Top 5):")
        for author_id in preferences['favorite_authors']:
            author = data['authors'][data['authors']['Author_ID'] == author_id]
            if not author.empty:
                author = author.iloc[0]
                print(f"  • {author['FName']} {author['LName']} (ID: {author_id})")

    print()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_recommender_system(use_db=True, db_conn=None):
    """
    Main function to run the recommender system
    """
    if use_db and db_conn:
        print("\n🔄 Loading data from database...")
        data = load_data_from_db(db_conn)
    else:
        print("\n❌ Database connection required")
        return None, None, None, None

    print("✅ Data loaded successfully!")
    print("🔨 Building content vectors...")
    vectorizer, tfidf_matrix, paper_ids = build_content_vectors(data)
    print("✅ Content vectors built!")

    return data, vectorizer, tfidf_matrix, paper_ids
