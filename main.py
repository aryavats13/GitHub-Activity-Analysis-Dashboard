import streamlit as st
import os
import os.path
from pathlib import Path
from auth import init_authentication, show_login_page, login_user, register_user, logout_user

st.set_page_config(
    page_title="GitHub Activity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_authentication()

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re
from collections import Counter
import warnings
from statsmodels.tsa.arima.model import ARIMA

@st.cache_resource
def initialize_nltk():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    try:
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        nltk.download('wordnet', download_dir=nltk_data_dir)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
        return True
    except Exception as e:
        st.error(f"nltk download failed: {str(e)}")
        return False

nltk_initialized = initialize_nltk()

def safe_preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    try:
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        
        if not nltk_initialized:
            return ' '.join(text.split())
        
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            
            filtered_tokens = [
                lemmatizer.lemmatize(word) for word in tokens
                if word not in stop_words and len(word) > 2
            ]
            return ' '.join(filtered_tokens)
        except Exception:
            # fallback to basic split
            return ' '.join(text.split())
    except Exception as e:
        st.warning(f"text processing error: {str(e)}")
        return text

# warnings
warnings.filterwarnings('ignore')


load_dotenv()

class GitHubAnalyzer:
    def __init__(self, username, token=None):
        """setup github connection"""
        self.username = username
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.base_url = "https://api.github.com"
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if self.token:

            self.headers['Authorization'] = f'Bearer {self.token}'  # Fixed syntax error
            
        # Checking rate limit before starting
        self.check_rate_limit()
        
        self.repos_data = None
        self.commits_data = None
        self.issues_data = None
        self.pull_requests_data = None
        
    def check_rate_limit(self):
        """Check GitHub API rate limit status."""
        try:
            response = requests.get(f"{self.base_url}/rate_limit", headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                core_rate = data['resources']['core']
                remaining = core_rate['remaining']
                reset_time = datetime.fromtimestamp(core_rate['reset'])
                
                if remaining < 10:  #  threshold
                    reset_in = reset_time - datetime.now()
                    minutes = reset_in.total_seconds() / 60
                    if remaining == 0:
                        st.error(f"⚠️ API rate limit exceeded. Reset in {minutes:.0f} minutes. Please provide a GitHub token.")
                    else:
                        st.warning(f"⚠️ Only {remaining} API calls remaining. Reset in {minutes:.0f} minutes.")
                return remaining > 0
            return False
        except Exception as e:
            st.error(f"Error checking rate limit: {str(e)}")
            return False

    def get_user_repos(self, per_page=100, max_pages=10):
        """Fetch all repositories for the user."""
        if not self.check_rate_limit():
            return pd.DataFrame()
            
        all_repos = []
        page = 1
        
        while page <= max_pages:
            url = f"{self.base_url}/users/{self.username}/repos"
            params = {'per_page': per_page, 'page': page}
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                st.error(f"Error fetching repositories: {response.json().get('message', 'Unknown error')}")
                break
                
            repos = response.json()
            all_repos.extend(repos)
            
            if len(repos) < per_page:
                break
                
            page += 1
        
        repos_df = pd.DataFrame([{
            'repo_id': repo['id'],
            'name': repo['name'],
            'description': repo['description'] or '',
            'created_at': repo['created_at'],
            'updated_at': repo['updated_at'],
            'pushed_at': repo['pushed_at'],
            'language': repo['language'],
            'stars': repo['stargazers_count'],
            'forks': repo['forks_count'],
            'open_issues': repo['open_issues_count'],
            'size': repo['size'],
            'is_fork': repo['fork'],
            'url': repo['html_url'],
            'topics': ','.join(repo.get('topics', [])),
            'default_branch': repo['default_branch']
        } for repo in all_repos])
        
        if not repos_df.empty:
            for col in ['created_at', 'updated_at', 'pushed_at']:
                repos_df[col] = pd.to_datetime(repos_df[col])
        
        self.repos_data = repos_df
        return repos_df
    
    def get_commits_for_repo(self, repo_name, per_page=100, max_pages=5):
        """Fetch commits for a specific repository."""
        all_commits = []
        page = 1
        
        while page <= max_pages:
            url = f"{self.base_url}/repos/{self.username}/{repo_name}/commits"
            params = {'per_page': per_page, 'page': page}
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                break
                
            commits = response.json()
            all_commits.extend(commits)
            
            if len(commits) < per_page:
                break
                
            page += 1
        
        processed_commits = []
        for commit in all_commits:
            try:
                commit_data = {
                    'repo_name': repo_name,
                    'sha': commit['sha'],
                    'author': commit['commit']['author']['name'],
                    'author_email': commit['commit']['author']['email'],
                    'date': commit['commit']['author']['date'],
                    'message': commit['commit']['message'],
                    'url': commit.get('html_url', ''),
                }
                
                if commit.get('author') and commit['author'].get('login'):
                    commit_data['github_username'] = commit['author']['login']
                else:
                    commit_data['github_username'] = 'Unknown'
                    
                processed_commits.append(commit_data)
            except (KeyError, TypeError):
                continue
                
        return processed_commits
    
    def get_all_commits(self, max_repos=None):
        """Fetch commits from all repositories or a subset."""
        if self.repos_data is None:
            self.get_user_repos()
            
        if self.repos_data.empty:
            return pd.DataFrame()
        
        all_commits = []
        repos_to_process = self.repos_data['name'].head(max_repos).tolist() if max_repos else self.repos_data['name'].tolist()
        
        with st.spinner(f"Fetching commits from {len(repos_to_process)} repositories..."):
            progress_bar = st.progress(0)
            
            for i, repo_name in enumerate(repos_to_process):
                commits = self.get_commits_for_repo(repo_name)
                all_commits.extend(commits)
                progress_bar.progress((i + 1) / len(repos_to_process))
        
        commits_df = pd.DataFrame(all_commits)
        
        if not commits_df.empty:
            commits_df['date'] = pd.to_datetime(commits_df['date'])
            
            commits_df['message_length'] = commits_df['message'].str.len()
            commits_df['hour'] = commits_df['date'].dt.hour
            commits_df['day'] = commits_df['date'].dt.day_name()
            commits_df['month'] = commits_df['date'].dt.month_name()
            commits_df['year'] = commits_df['date'].dt.year
            commits_df['weekday'] = commits_df['date'].dt.dayofweek
            
            commits_df['sentiment'] = commits_df['message'].apply(
                lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0
            )
            
            commits_df['is_short_message'] = commits_df['message_length'] < 10
            commits_df['has_fix_keyword'] = commits_df['message'].str.contains(
                r'\b(fix|fixes|fixed|bug|issue)\b', case=False, regex=True
            ).fillna(False)
        
        self.commits_data = commits_df
        return commits_df
    
    def get_issues_for_repo(self, repo_name, state='all', per_page=100, max_pages=5):
        """Fetch issues for a specific repository."""
        all_issues = []
        page = 1
        
        while page <= max_pages:
            url = f"{self.base_url}/repos/{self.username}/{repo_name}/issues"
            params = {'state': state, 'per_page': per_page, 'page': page}
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                break
                
            issues = response.json()
            issues = [issue for issue in issues if 'pull_request' not in issue]
            all_issues.extend(issues)
            
            if len(issues) < per_page:
                break
                
            page += 1
            
        processed_issues = []
        for issue in all_issues:
            try:
                issue_data = {
                    'repo_name': repo_name,
                    'issue_number': issue['number'],
                    'title': issue['title'],
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at'],
                    'closed_at': issue['closed_at'],
                    'user': issue['user']['login'],
                    'labels': ','.join([label['name'] for label in issue['labels']]),
                    'comments': issue['comments'],
                    'body': issue['body'] or '',
                    'url': issue['html_url']
                }
                processed_issues.append(issue_data)
            except (KeyError, TypeError):
                continue
                
        return processed_issues
    
    def analyze_commit_patterns(self):
        """Analyze commit patterns and return insights."""
        if self.commits_data is None or self.commits_data.empty:
            return {}
            
        hourly_commits = self.commits_data['hour'].value_counts().sort_index()
        daily_commits = self.commits_data['day'].value_counts()
        monthly_commits = self.commits_data['month'].value_counts()
        
        peak_hour = hourly_commits.idxmax()
        peak_day = daily_commits.idxmax()
        
        avg_message_length = self.commits_data['message_length'].mean()
        short_commits_pct = self.commits_data['is_short_message'].mean() * 100
        fix_commits_pct = self.commits_data['has_fix_keyword'].mean() * 100
        
        self.commits_data = self.commits_data.sort_values('date')
        commits_by_date = self.commits_data.groupby(self.commits_data['date'].dt.date).size()
        
        date_range = pd.date_range(
            start=commits_by_date.index.min(),
            end=commits_by_date.index.max()
        )
        commits_by_date = commits_by_date.reindex(date_range, fill_value=0)
        
        streaks = []
        current_streak = 0
        
        for count in commits_by_date:
            if count > 0:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 0
                
        streaks.append(current_streak)
        longest_streak = max(streaks) if streaks else 0
        
        gaps = []
        current_gap = 0
        
        for count in commits_by_date:
            if count == 0:
                current_gap += 1
            else:
                gaps.append(current_gap)
                current_gap = 0
                
        gaps.append(current_gap)
        longest_gap = max(gaps) if gaps else 0
        
        return {
            'total_commits': len(self.commits_data),
            'repos_with_commits': self.commits_data['repo_name'].nunique(),
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'avg_message_length': avg_message_length,
            'short_commits_pct': short_commits_pct,
            'fix_commits_pct': fix_commits_pct,
            'longest_streak': longest_streak,
            'longest_gap': longest_gap,
            'hourly_commits': hourly_commits,
            'daily_commits': daily_commits,
            'monthly_commits': monthly_commits
        }
    
    def analyze_commit_messages(self):
        """Use NLP to analyze commit message content."""
        if self.commits_data is None or self.commits_data.empty:
            return {}
        
        # Process messages with error handling
        self.commits_data['processed_message'] = self.commits_data['message'].apply(safe_preprocess_text)
        
        all_words = " ".join(self.commits_data['processed_message']).split()
        word_freq = Counter(all_words).most_common(20)
        action_words = ['add', 'update', 'fix', 'remove', 'implement', 'refactor', 'change', 'merge']
        action_counts = {
            word: sum(1 for msg in self.commits_data['message'] if re.search(fr'\b{word}\b', msg, re.IGNORECASE))
            for word in action_words
        }
        
        sentiment_dist = self.commits_data['sentiment'].describe()
        
        if len(self.commits_data['repo_name'].unique()) > 1:
            tfidf = TfidfVectorizer(max_features=100)
            repo_messages = self.commits_data.groupby('repo_name')['processed_message'].apply(' '.join)
            
            if len(repo_messages) > 1:  
                try:
                    tfidf_matrix = tfidf.fit_transform(repo_messages)
                    feature_names = tfidf.get_feature_names_out()
                    
                    repo_top_terms = {}
                    for i, repo in enumerate(repo_messages.index):
                        tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
                        repo_top_terms[repo] = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:5]
                except:
                    repo_top_terms = {}
            else:
                repo_top_terms = {}
        else:
            repo_top_terms = {}
            
        return {
            'word_freq': word_freq,
            'action_counts': action_counts,
            'sentiment_dist': sentiment_dist,
            'repo_top_terms': repo_top_terms
        }
    
    def predict_future_activity(self, days_to_predict=30):
        """Predict future commit activity using time series forecasting."""
        if self.commits_data is None or self.commits_data.empty:
            return {}
            
        commits_by_date = self.commits_data.groupby(self.commits_data['date'].dt.date).size()
        
        if len(commits_by_date) < 14:
            return {
                'enough_data': False,
                'message': 'Need at least 14 days of commit data for forecasting'
            }
            
        date_range = pd.date_range(
            start=commits_by_date.index.min(),
            end=commits_by_date.index.max()
        )
        daily_commits = commits_by_date.reindex(date_range, fill_value=0)
        
        try:
            model = ARIMA(daily_commits.values, order=(5, 1, 0))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=days_to_predict)
            forecast_dates = pd.date_range(
                start=daily_commits.index[-1] + timedelta(days=1),
                periods=days_to_predict
            )
            
            forecast = np.maximum(forecast, 0)
            
            return {
                'enough_data': True,
                'forecast': forecast,
                'forecast_dates': forecast_dates,
                'historical': daily_commits
            }
        except:
            return {
                'enough_data': False,
                'message': 'Unable to create forecast with available data'
            }
    
    def generate_recommendations(self, commit_patterns, message_analysis):
        """Generate personalized Git workflow recommendations."""
        if not commit_patterns or not message_analysis:
            return []
            
        recommendations = []
        
        peak_hour = commit_patterns.get('peak_hour')
        if peak_hour is not None:
            if 22 <= peak_hour or peak_hour <= 5:
                recommendations.append({
                    'category': 'Work Schedule',
                    'title': 'Consider adjusting your coding hours',
                    'description': f"You commit most frequently at {peak_hour}:00, which may affect your sleep schedule. Consider shifting your coding sessions to daytime hours for better work-life balance."
                })
                
        short_pct = commit_patterns.get('short_commits_pct', 0)
        if short_pct > 30:
            recommendations.append({
                'category': 'Commit Quality',
                'title': 'Improve commit message clarity',
                'description': f"{short_pct:.1f}% of your commit messages are very short (<10 chars). More descriptive commit messages make your repository history more valuable and easier to navigate."
            })
            
        fix_pct = commit_patterns.get('fix_commits_pct', 0)
        if fix_pct > 25:
            recommendations.append({
                'category': 'Testing',
                'title': 'Consider implementing more tests',
                'description': f"{fix_pct:.1f}% of your commits contain fix-related keywords. More thorough testing before commits could reduce the need for fixes and improve code quality."
            })
            
        longest_gap = commit_patterns.get('longest_gap', 0)
        if longest_gap > 14:
            recommendations.append({
                'category': 'Consistency',
                'title': 'Maintain a more consistent coding schedule',
                'description': f"Your longest gap between commits was {longest_gap} days. More consistent contributions, even if smaller, can help maintain momentum in your projects."
            })
            
        sentiment_dist = message_analysis.get('sentiment_dist', {})
        if not sentiment_dist.empty and sentiment_dist.get('mean', 0) < -0.1:
            recommendations.append({
                'category': 'Communication',
                'title': 'Consider more positive framing in commit messages',
                'description': "Your commit messages tend to have a negative sentiment. While technical accuracy is most important, positive framing can improve team morale when working with others."
            })
            
        recommendations.append({
            'category': 'Tooling',
            'title': 'Consider using commit message templates',
            'description': "Setting up commit templates can help standardize your commit messages and ensure they contain all needed information. Add them with: git config --global commit.template ~/.gitmessage"
        })
        
        return recommendations
    
    def cluster_repositories(self):
        """Cluster repositories based on their characteristics."""
        if self.repos_data is None or self.repos_data.empty or len(self.repos_data) < 3:
            return {}
            
        features = ['stars', 'forks', 'open_issues', 'size']
        
        for feature in features:
            if feature not in self.repos_data.columns:
                return {}
                
        X = self.repos_data[features].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        max_clusters = min(5, len(self.repos_data) - 1)
        if max_clusters < 2:
            return {}
            
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
        optimal_clusters = 2
        for i in range(1, len(wcss) - 1):
            if (wcss[i-1] - wcss[i]) / (wcss[i] - wcss[i+1]) > 2:
                optimal_clusters = i + 1
                break
                
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        self.repos_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_profiles = pd.DataFrame(centers, columns=features)
        
        cluster_names = []
        for i, profile in cluster_profiles.iterrows():
            if profile['stars'] > profile['forks'] * 2:
                name = "Popular/Starred"
            elif profile['open_issues'] > profile['stars']:
                name = "Active Development"
            elif profile['size'] > cluster_profiles['size'].median() * 2:
                name = "Large Projects"
            else:
                name = f"Cluster {i+1}"
            cluster_names.append(name)
            
        cluster_profiles['name'] = cluster_names
        
        return {
            'repos_with_clusters': self.repos_data[['name', 'stars', 'forks', 'open_issues', 'size', 'cluster']],
            'cluster_profiles': cluster_profiles
        }

# Streamlit app
def main():
    # Check if user is authenticated
    if not st.session_state.authenticated:
        show_login_page()
        return
        
    st.title("GitHub Activity Analysis Dashboard")
    
    # Add logout button in sidebar
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            logout_user()
            st.experimental_rerun()
    
    st.markdown("""
    Welcome to the comprehensive GitHub activity analyzer. This tool provides deep insights into your 
    GitHub contributions, coding patterns, and repository statistics.
    """)
    
    with st.sidebar:
        st.header("Configuration")
        st.markdown("""
        Configure your analysis settings below. For the best experience, please provide your GitHub credentials.
        """)
        
        username = st.text_input("GitHub Username", placeholder="Enter your GitHub username")
        
        st.markdown("---")
        st.subheader("Authentication")
        st.markdown("""
        A personal access token is strongly recommended for:
        - Higher API rate limits
        - Access to private repositories
        - More accurate analytics
        """)
        
        token = st.text_input(
            "Personal Access Token",
            type="password",
            placeholder="Enter your GitHub token",
            help="Create a token in GitHub Settings → Developer Settings → Personal Access Tokens"
        )
        
        if not token:
            st.warning(
                "Without authentication, you may experience API rate limits. "
                "Create a token with 'repo' and 'read:user' scopes for full access."
            )
        
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        **Features:**
        - Commit pattern analysis
        - Repository insights
        - Code quality metrics
        - Activity predictions
        - Personalized recommendations
        
        **Data Privacy:** All analysis is performed locally and no data is stored.
        """)
    
    if username:
        analyzer = GitHubAnalyzer(username, token)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Dashboard", "Commit Analysis", "Repository Insights", 
            "Predictions", "Recommendations"
        ])
        
        with tab1:
            st.header("Activity Dashboard")
            st.markdown("""
            Get a comprehensive overview of your GitHub activity. Click the button below to start the analysis.
            This process may take a few moments depending on your repository count.
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                analyze_button = st.button(
                    "Begin Analysis",
                    help="Fetches and analyzes your GitHub data",
                    use_container_width=True
                )
            with col2:
                st.markdown("")  # Spacing
            
            if analyze_button:
                # get repos
                with st.spinner("Fetching repositories..."):
                    repos_df = analyzer.get_user_repos()
                    
                if repos_df is not None and not repos_df.empty:
                    st.success(f"Found {len(repos_df)} repositories!")
                    
                    # get commits
                    with st.spinner("Analyzing commits..."):
                        max_repos = min(10, len(repos_df))
                        commits_df = analyzer.get_all_commits(max_repos=max_repos)
                    
                    if commits_df is not None and not commits_df.empty:
                        # Enhanced metrics display
                        st.markdown("---")
                        st.subheader("Key Metrics")
                        st.markdown("Overview of your GitHub presence and activity levels")
                        
                        # quick stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Repos", len(repos_df))
                        with col2:
                            st.metric("Total Commits", len(commits_df))
                        with col3:
                            st.metric("Active Repos", commits_df['repo_name'].nunique())
                        with col4:
                            days_active = (commits_df['date'].max() - commits_df['date'].min()).days
                            st.metric("Days Active", days_active if days_active > 0 else 0)
                            
                        # recent activity
                        st.markdown("---")
                        st.subheader("Recent Activity")
                        st.markdown("""
                        Your most recent contributions across repositories. 
                        This shows your latest work and active projects.
                        """)
                        recent_commits = commits_df.sort_values('date', ascending=False).head(5)
                        for _, commit in recent_commits.iterrows():
                            st.markdown(f"**{commit['repo_name']}** - _{commit['date'].strftime('%Y-%m-%d %H:%M')}_")
                            st.markdown(f"> {commit['message'].split('\n')[0]}")
                    else:
                        st.warning("No commits found in recent repositories")
                else:
                    st.error(f"No repositories found for {username}")

        # Enhance other tabs similarly with better descriptions and section organization
        with tab2:
            st.header("Commit Pattern Analysis")
            st.markdown("""
            Understand your coding patterns and habits through detailed commit analysis.
            This section provides insights into when and how you contribute to repositories.
            """)
            
            if analyzer.commits_data is not None and not analyzer.commits_data.empty:
                commit_patterns = analyzer.analyze_commit_patterns()
                message_analysis = analyzer.analyze_commit_messages()
                
                # Display commit timing patterns
                st.subheader("When Do You Commit?")
                col1, col2 = st.columns(2)
                with col1:
                    hourly_commits = commit_patterns['hourly_commits']
                    fig = px.bar(
                        x=hourly_commits.index,
                        y=hourly_commits.values,
                        labels={'x': 'Hour of Day', 'y': 'Number of Commits'},
                        title='Commits by Hour of Day'
                    )
                    fig.update_layout(xaxis_tickmode='linear')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_counts = commit_patterns['daily_commits'].reindex(day_order, fill_value=0)
                    fig = px.bar(
                        x=daily_counts.index,
                        y=daily_counts.values,
                        labels={'x': 'Day of Week', 'y': 'Number of Commits'},
                        title='Commits by Day of Week'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Commit message analysis
                st.subheader("Commit Message Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    word_freq = message_analysis['word_freq']
                    if word_freq:
                        words, counts = zip(*word_freq)
                        fig = px.bar(
                            x=words, 
                            y=counts,
                            labels={'x': 'Word', 'y': 'Frequency'},
                            title='Most Common Words in Commit Messages'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough commit message data for word frequency analysis.")
                with col2:
                    action_counts = message_analysis['action_counts']
                    if action_counts:
                        fig = px.bar(
                            x=list(action_counts.keys()),
                            y=list(action_counts.values()),
                            labels={'x': 'Action', 'y': 'Count'},
                            title='Common Actions in Commits'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough commit message data for action analysis.")
                
                # Commit message quality
                st.subheader("Commit Message Quality")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_length = commit_patterns.get('avg_message_length', 0)
                    st.metric("Average Message Length", f"{avg_length:.1f} chars")
                with col2:
                    short_pct = commit_patterns.get('short_commits_pct', 0)
                    st.metric("Short Messages", f"{short_pct:.1f}%")
                with col3:
                    fix_pct = commit_patterns.get('fix_commits_pct', 0)
                    st.metric("Fix-related Commits", f"{fix_pct:.1f}%")
                
                # Commit streaks
                st.subheader("Commit Consistency")
                col1, col2 = st.columns(2)
                with col1:
                    longest_streak = commit_patterns.get('longest_streak', 0)
                    st.metric("Longest Commit Streak", f"{longest_streak} days")
                with col2:
                    longest_gap = commit_patterns.get('longest_gap', 0)
                    st.metric("Longest Gap Between Commits", f"{longest_gap} days")
                
                # Commit heatmap
                st.subheader("Commit Activity Heatmap")
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=365)
                commit_counts = analyzer.commits_data.groupby(analyzer.commits_data['date'].dt.date).size()
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                heatmap_data = pd.DataFrame(index=date_range)
                heatmap_data['count'] = 0
                for date, count in commit_counts.items():
                    if date in heatmap_data.index:
                        heatmap_data.loc[date, 'count'] = count
                heatmap_data['weekday'] = heatmap_data.index.weekday
                heatmap_data['weeknum'] = heatmap_data.index.isocalendar().week
                pivot_data = heatmap_data.pivot_table(
                    index='weekday',
                    columns='weeknum',
                    values='count',
                    aggfunc='sum',
                    fill_value=0
                )
                pivot_data = pivot_data.iloc[::-1]
                
                # Create a Plotly heatmap instead of matplotlib/seaborn
                fig = px.imshow(
                    pivot_data,
                    color_continuous_scale='Blues',
                    labels=dict(x="Week", y="Day", color="Commits"),
                    height=300
                )
                fig.update_layout(
                    xaxis_title="Weeks",
                    yaxis_title="Day of Week",
                    title="Commit Activity Heatmap (Last 12 Months)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No commit data loaded. Please fetch GitHub data first.")
        
        with tab3:
            st.header("Repository Insights")
            st.markdown("""
            Gain insights into your repositories, including language distribution, statistics, and clustering analysis.
            """)
            if analyzer.repos_data is not None and not analyzer.repos_data.empty:
                st.subheader("Programming Language Distribution")
                language_counts = analyzer.repos_data['language'].value_counts()
                language_counts = language_counts[language_counts > 0]
                if not language_counts.empty:
                    fig = px.pie(
                        values=language_counts.values,
                        names=language_counts.index,
                        title='Repository Languages'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No language information available for your repositories.")
                
                st.subheader("Repository Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_stars = analyzer.repos_data['stars'].mean()
                    st.metric("Average Stars", f"{avg_stars:.1f}")
                with col2:
                    avg_forks = analyzer.repos_data['forks'].mean()
                    st.metric("Average Forks", f"{avg_forks:.1f}")
                with col3:
                    avg_issues = analyzer.repos_data['open_issues'].mean()
                    st.metric("Average Open Issues", f"{avg_issues:.1f}")
                
                st.subheader("Repository Age Analysis")
                # Fix timezone handling
                current_time = datetime.now(timezone.utc)
                repo_dates = pd.to_datetime(analyzer.repos_data['created_at'])
                # Check if dates are already tz-aware
                if repo_dates.dt.tz is None:
                    repo_dates = repo_dates.dt.tz_localize('UTC')
                analyzer.repos_data['age_days'] = (
                    (current_time - repo_dates)
                    .dt.total_seconds() / (24 * 60 * 60)
                )
                
                age_sorted = analyzer.repos_data.sort_values('age_days', ascending=False)
                fig = px.bar(
                    age_sorted.head(10),
                    x='name',
                    y='age_days',
                    labels={'name': 'Repository', 'age_days': 'Age (days)'},
                    title='Oldest Repositories'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Repository Clustering")
                clustering_results = analyzer.cluster_repositories()
                if clustering_results and 'repos_with_clusters' in clustering_results:
                    repos_with_clusters = clustering_results['repos_with_clusters']
                    cluster_profiles = clustering_results['cluster_profiles']
                    st.markdown("Repositories have been grouped into clusters based on their characteristics:")
                    for i, (idx, profile) in enumerate(cluster_profiles.iterrows()):
                        st.markdown(f"**Cluster {i+1}: {profile['name']}**")
                        metrics = {
                            'Stars': f"{profile['stars']:.1f}",
                            'Forks': f"{profile['forks']:.1f}",
                            'Issues': f"{profile['open_issues']:.1f}",
                            'Size (KB)': f"{profile['size']:.0f}"
                        }
                        cols = st.columns(len(metrics))
                        for j, (metric, value) in enumerate(metrics.items()):
                            with cols[j]:
                                st.metric(metric, value)
                        cluster_repos = repos_with_clusters[repos_with_clusters['cluster'] == i]
                        st.markdown(f"Repositories in this cluster: {', '.join(cluster_repos['name'])}")
                        st.markdown("---")
                else:
                    st.info("Not enough repository data for clustering analysis.")
            else:
                st.info("No repository data loaded. Please fetch GitHub data first.")
        
        with tab4:
            st.header("Activity Predictions")
            st.markdown("""
            Predict your future GitHub activity based on historical data. This section provides forecasts and momentum analysis.
            """)
            if analyzer.commits_data is not None and not analyzer.commits_data.empty:
                activity_prediction = analyzer.predict_future_activity(days_to_predict=30)
                if activity_prediction.get('enough_data', False):
                    st.subheader("Commit Activity Forecast (Next 30 Days)")
                    historical = activity_prediction['historical']
                    forecast = activity_prediction['forecast']
                    forecast_dates = activity_prediction['forecast_dates']
                    plot_data = pd.DataFrame({
                        'date': list(historical.index) + list(forecast_dates),
                        'commits': list(historical.values) + list(forecast),
                        'type': ['Historical'] * len(historical) + ['Forecast'] * len(forecast)
                    })
                    fig = px.line(
                        plot_data,
                        x='date',
                        y='commits',
                        color='type',
                        labels={'commits': 'Commit Count', 'date': 'Date'},
                        title='Commit Activity Forecast'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    total_forecasted = sum(forecast)
                    avg_daily = total_forecasted / len(forecast)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Forecasted Commits (30 days)", f"{total_forecasted:.0f}")
                    with col2:
                        st.metric("Avg. Daily Commits (Forecast)", f"{avg_daily:.1f}")
                    st.info("Note: This forecast is based on your historical commit patterns. Actual activity may vary based on project demands and other factors.")
                else:
                    st.info(activity_prediction.get('message', 'Not enough commit data for forecasting. Try analyzing more repositories.'))
                if analyzer.repos_data is not None and not analyzer.repos_data.empty:
                    st.subheader("Project Momentum Analysis")
                    # Fix timezone handling for pushed_at dates
                    current_time = datetime.now(timezone.utc)
                    pushed_dates = pd.to_datetime(analyzer.repos_data['pushed_at'])
                    # Make pushed_dates timezone aware
                    if pushed_dates.dt.tz is None:
                        pushed_dates = pushed_dates.dt.tz_localize('UTC')
                    
                    analyzer.repos_data['days_since_update'] = (
                        current_time - pushed_dates
                    ).dt.total_seconds() / (24 * 60 * 60)  # Convert to days
                    
                    recently_active = analyzer.repos_data.sort_values('days_since_update').head(5)
                    recently_active['momentum_score'] = (
                        recently_active['days_since_update'] / 
                        (1 + recently_active['stars'] + recently_active['forks'] * 2)
                    )
                    recently_active = recently_active.sort_values('momentum_score')
                    st.markdown("These repositories have the most momentum based on recent activity and engagement:")
                    display_data = recently_active[['name', 'days_since_update', 'stars', 'forks']]
                    display_data.columns = ['Repository', 'Days Since Update', 'Stars', 'Forks']
                    st.table(display_data)
            else:
                st.info("No commit data loaded. Please fetch GitHub data first.")
        
        with tab5:
            st.header("Git Workflow Recommendations")
            st.markdown("""
            Get personalized recommendations to improve your Git workflow and productivity.
            """)
            if analyzer.commits_data is not None and not analyzer.commits_data.empty:
                commit_patterns = analyzer.analyze_commit_patterns()
                message_analysis = analyzer.analyze_commit_messages()
                recommendations = analyzer.generate_recommendations(commit_patterns, message_analysis)
                if recommendations:
                    for rec in recommendations:
                        with st.expander(f"{rec['category']}: {rec['title']}"):
                            st.markdown(rec['description'])
                    st.subheader("Suggested Git Configuration")
                    git_aliases = [
                        "# Add these to your ~/.gitconfig file",
                        "[alias]",
                        "    # Quick status check",
                        "    st = status -s",
                        "    # Better log visualization",
                        "    lg = log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit",
                    ]
                    if commit_patterns.get('fix_commits_pct', 0) > 15:
                        git_aliases.extend([
                            "    # Quick fix commit",
                            "    fix = commit -m 'fix: '"
                        ])
                    if message_analysis.get('action_counts', {}).get('add', 0) > 5:
                        git_aliases.extend([
                            "    # Quick add commit",
                            "    feature = commit -m 'feat: '"
                        ])
                    commit_template = [
                        "# Suggested ~/.gitmessage template",
                        "# <type>: <subject>",
                        "#",
                        "# feat: Add new feature",
                        "# fix: Fix bug",
                        "# docs: Update documentation",
                        "# refactor: Refactor code",
                        "#",
                        "# Why was this change made?",
                        "#",
                        "# References/Related issues:"
                    ]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.code("\n".join(git_aliases), language="ini")
                    with col2:
                        st.code("\n".join(commit_template), language="ini")
                    st.subheader("Suggested Pre-commit Hook")
                    pre_commit = [
                        "#!/bin/sh",
                        "",
                        "# Get the commit message",
                        "commit_msg_file=$1",
                        "commit_msg=$(cat $commit_msg_file)",
                        "",
                        "# Check if commit message is too short",
                        "if [ ${#commit_msg} -lt 10 ]; then",
                        "  echo \"Error: Commit message is too short (${#commit_msg} chars). Please be more descriptive.\"",
                        "  exit 1",
                        "fi",
                        "",
                        "# Check if commit follows conventional format",
                        r"if ! echo \"$commit_msg\" | grep -qE '^(feat|fix|docs|style|refactor|test|chore)(\([a-z]+\))?: .+'; then",  # Fixed escape sequence with raw string
                        "  echo \"Warning: Commit message does not follow conventional format.\"",
                        "  echo \"Example: feat: add new feature\"",
                        "  # Not exiting with error to allow commit anyway",
                        "fi",
                        "",
                        "exit 0"
                    ]
                    st.code("\n".join(pre_commit), language="sh")
            else:
                st.info("No commit data loaded. Please fetch GitHub data first.")
    
    else:
        st.markdown("""
        ## Getting Started
        
        Welcome to the GitHub Activity Analyzer! This tool helps you:
        
        1. **Understand Your Workflow**
           - Discover your most productive coding hours
           - Track contribution patterns
           - Analyze commit quality
        
        2. **Improve Code Quality**
           - Get personalized recommendations
           - Track code maintenance metrics
           - Monitor repository health
        
        3. **Plan Better**
           - View activity forecasts
           - Track project momentum
           - Identify areas for improvement
        
        To begin, enter your GitHub username in the sidebar.
        """)

if __name__ == "__main__":
    main()
