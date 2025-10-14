import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="PM Internship Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #FF9933; /* Saffron color from Indian flag */
        font-family: 'Roboto', sans-serif;
        padding: 1.5rem 0;
        border-bottom: 2px solid #138808; /* Green color from Indian flag */
    }
    .sub-header {
        text-align: center;
        color: #333333;
        font-size: 1.1rem;
        font-family: 'Roboto', sans-serif;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: #FFFFFF; /* White background for clean look */
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #000080; /* Navy blue border */
        color: #333333;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Roboto', sans-serif;
    }
    .metric-card {
        background: #F5F5F5; /* Light grey for metrics */
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #FF9933; /* Saffron border */
        margin: 0.5rem 0;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF9933 0%, #138808 100%); /* Saffron to green gradient */
        color: #FFFFFF;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        font-family: 'Roboto', sans-serif;
        width: 100%;
        transition: background 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #138808 0%, #FF9933 100%); /* Reverse gradient on hover */
        opacity: 0.9;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #000080; /* Navy blue border */
        border-radius: 8px;
        margin-bottom: 1rem;
        background: #FFFFFF;
        font-family: 'Roboto', sans-serif;
    }
    body {
        background-color: #F5F5F5; /* Light grey background for the entire app */
        font-family: 'Roboto', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load data and models
@st.cache_data
def load_data():
    """Load the internship dataset"""
    try:
        df = pd.read_csv("company.csv")
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv("../company.csv")
            return df
        except FileNotFoundError:
            st.error("❌ Error: company.csv not found. Please ensure the file is accessible.")
            return None

@st.cache_resource
def load_models():
    """Load trained models and artifacts"""
    try:
        with open('internship_recommender_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return model, scaler, tfidf, label_encoders, feature_cols
    except FileNotFoundError:
        return None, None, None, None, None

def recommend_internship(user_profile, df_original, top_n=5):
    """
    Recommend internships based on user profile using rule-based matching
    """
    # Filter internships based on degree eligibility
    eligible_internships = df_original[
        df_original['degrees_eligible'].str.contains(user_profile['degree'], na=False, case=False)
    ].copy()
    
    if len(eligible_internships) == 0:
        return None, "No internships found for your degree. Try selecting a different degree."
    
    # Calculate match scores
    scores = []
    
    for idx, row in eligible_internships.iterrows():
        score = 0
        details = {}
        
        # Degree match (mandatory - already filtered)
        score += 25
        details['degree_match'] = True
        
        # Skills match
        required_skills = set([s.strip().lower() for s in str(row['skills_required']).split(',')])
        user_skills_set = set([s.strip().lower() for s in user_profile['skills']])
        matched_skills = required_skills.intersection(user_skills_set)
        skill_match_rate = len(matched_skills) / len(required_skills) if len(required_skills) > 0 else 0
        score += skill_match_rate * 40
        details['skill_match_rate'] = skill_match_rate
        details['matched_skills'] = list(matched_skills)
        
        # Location preference
        if user_profile['location_preference'].lower() == 'any' or \
           row['location_mode'].lower() == user_profile['location_preference'].lower():
            score += 15
            details['location_match'] = True
        elif row['location_mode'].lower() == 'remote':
            score += 10
            details['location_match'] = 'Remote'
        else:
            details['location_match'] = False
        
        # Stipend
        if row['stipend'] >= user_profile['min_stipend']:
            score += 20
            details['stipend_match'] = True
        else:
            details['stipend_match'] = False
        
        scores.append({
            'index': idx,
            'score': score,
            'details': details,
            'company': row['company_name'],
            'role': row['intern_role'],
            'stipend': row['stipend'],
            'location': row['location_mode'],
            'skills_required': row['skills_required']
        })
    
    # Sort by score
    scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    
    return scores[:top_n], None

def create_chart(fig):
    """Helper function to display matplotlib charts in Streamlit"""
    st.pyplot(fig)
    plt.close()

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">💼 PM Internship Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find Your Perfect Internship Match with AI-Powered Recommendations</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar - User Input
    with st.sidebar:
        st.markdown("### 📋 Your Profile")
        st.markdown("---")
        
        # Get available options from data
        all_skills = []
        for skills in df['skills_required'].dropna():
            all_skills.extend([s.strip() for s in str(skills).split(',')])
        top_skills = [skill for skill, _ in Counter(all_skills).most_common(30)]
        
        available_degrees = sorted(set([d.strip() for deg in df['degrees_eligible'].dropna() 
                                       for d in str(deg).split(',')]))
        
        available_locations = ['Any'] + sorted(df['location_mode'].unique().tolist())
        
        # User inputs
        user_degree = st.selectbox(
            "🎓 Your Degree",
            options=available_degrees,
            help="Select your educational qualification"
        )
        
        user_skills = st.multiselect(
            "💻 Your Skills",
            options=top_skills,
            default=[],
            help="Select all skills you possess"
        )
        
        # Allow custom skills
        custom_skills = st.text_input(
            "➕ Add Custom Skills (comma-separated)",
            placeholder="e.g., Docker, Kubernetes"
        )
        
        if custom_skills:
            user_skills.extend([s.strip() for s in custom_skills.split(',') if s.strip()])
        
        user_location = st.selectbox(
            "📍 Preferred Location",
            options=available_locations,
            help="Choose your preferred work location"
        )
        
        user_min_stipend = st.number_input(
            "💰 Minimum Expected Stipend (₹)",
            min_value=0,
            max_value=100000,
            value=15000,
            step=1000,
            help="Enter your minimum stipend expectation"
        )
        
        st.markdown("---")
        search_button = st.button("🔍 Find Internships", use_container_width=True)
    
    # Main content area
    if not search_button:
        # Dashboard view
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Internships", len(df))
        with col2:
            st.metric("Companies", df['company_name'].nunique())
        with col3:
            st.metric("Avg Stipend", f"₹{df['stipend'].mean():,.0f}")
        with col4:
            st.metric("Locations", df['location_mode'].nunique())
        
        st.markdown("---")
        
        # Display charts
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 Roles", "🏢 Companies", "📍 Locations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stipend Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df['stipend'], bins=20, color='#FF9933', edgecolor='black', alpha=0.7)
                ax.set_xlabel('Stipend (₹)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('Stipend Distribution', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                create_chart(fig)
            
            with col2:
                st.subheader("Top Skills in Demand")
                skill_counts = Counter(all_skills).most_common(10)
                skills = [s[0] for s in skill_counts]
                counts = [s[1] for s in skill_counts]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(skills, counts, color=sns.color_palette('viridis', len(skills)))
                ax.set_xlabel('Count', fontsize=12)
                ax.set_title('Top 10 Skills', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3, axis='x')
                create_chart(fig)
        
        with tab2:
            st.subheader("Internship Roles Distribution")
            role_counts = df['intern_role'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette('coolwarm', len(role_counts))
            ax.barh(role_counts.index, role_counts.values, color=colors)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_title('Top 10 Internship Roles', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            create_chart(fig)
        
        with tab3:
            st.subheader("Top Companies by Average Stipend")
            company_stipend = df.groupby('company_name')['stipend'].mean().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette('plasma', len(company_stipend))
            ax.barh(company_stipend.index, company_stipend.values, color=colors)
            ax.set_xlabel('Average Stipend (₹)', fontsize=12)
            ax.set_title('Top 10 Companies by Stipend', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            create_chart(fig)
        
        with tab4:
            st.subheader("Internships by Location Mode")
            location_counts = df['location_mode'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette('Set3', len(location_counts))
            ax.bar(location_counts.index, location_counts.values, color=colors, edgecolor='black')
            ax.set_ylabel('Count', fontsize=12)
            ax.set_xlabel('Location Mode', fontsize=12)
            ax.set_title('Distribution by Location', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            create_chart(fig)
    
    else:
        # Search results
        if not user_skills:
            st.warning("⚠️ Please select at least one skill to get better recommendations!")
            return
        
        user_profile = {
            'degree': user_degree,
            'skills': user_skills,
            'location_preference': user_location,
            'min_stipend': user_min_stipend
        }
        
        with st.spinner("🔍 Searching for your perfect internship matches..."):
            recommendations, error = recommend_internship(user_profile, df, top_n=10)
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try selecting a different degree or broadening your criteria.")
            return
        
        # Display results
        st.success(f"✅ Found {len(recommendations)} matching internships!")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_match = np.mean([r['score'] for r in recommendations])
            st.metric("Average Match Score", f"{avg_match:.1f}%")
        with col2:
            avg_stipend = np.mean([r['stipend'] for r in recommendations])
            st.metric("Average Stipend", f"₹{avg_stipend:,.0f}")
        with col3:
            remote_count = sum(1 for r in recommendations if r['location'].lower() == 'remote')
            st.metric("Remote Opportunities", remote_count)
        
        st.markdown("---")
        st.subheader("🎯 Your Top Matches")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} - {rec['company']} | {rec['role']} | Match: {rec['score']:.0f}%", expanded=(i<=3)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {rec['company']}")
                    st.markdown(f"**Role:** {rec['role']}")
                    st.markdown(f"**Stipend:** ₹{rec['stipend']:,}")
                    st.markdown(f"**Location:** {rec['location']}")
                    st.markdown(f"**Skills Required:** {rec['skills_required']}")
                    
                    if rec['details']['matched_skills']:
                        st.success(f"✅ Your Matching Skills: {', '.join(rec['details']['matched_skills'])}")
                
                with col2:
                    # Match score breakdown
                    st.markdown("**Match Breakdown:**")
                    
                    # Progress bars for different criteria
                    if rec['details']['degree_match']:
                        st.progress(1.0, text="✅ Degree Match")
                    
                    skill_rate = rec['details']['skill_match_rate']
                    st.progress(skill_rate, text=f"Skills: {skill_rate*100:.0f}%")
                    
                    if rec['details']['location_match']:
                        st.progress(1.0, text="✅ Location Match")
                    else:
                        st.progress(0.0, text="❌ Location")
                    
                    if rec['details']['stipend_match']:
                        st.progress(1.0, text="✅ Stipend Match")
                    else:
                        st.progress(0.0, text="❌ Stipend")
                
                st.markdown("---")
                st.markdown(f"**Overall Match Score: {rec['score']:.1f}/100**")
        
        # Visualization of match scores
        st.markdown("---")
        st.subheader("📊 Match Score Distribution")
        
        companies = [r['company'] for r in recommendations]
        scores = [r['score'] for r in recommendations]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn(np.array(scores) / 100)
        bars = ax.barh(companies, scores, color=colors, edgecolor='black')
        ax.set_xlabel('Match Score', fontsize=12)
        ax.set_title('Match Scores for Recommended Internships', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        create_chart(fig)
        
        # Download recommendations
        scores_df = pd.DataFrame([
            {
                'Rank': i+1,
                'Company': r['company'], 
                'Role': r['role'], 
                'Score': f"{r['score']:.1f}%",
                'Stipend': f"₹{r['stipend']:,}",
                'Location': r['location']
            } 
            for i, r in enumerate(recommendations)
        ])
        
        csv = scores_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv,
            file_name="internship_recommendations.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()