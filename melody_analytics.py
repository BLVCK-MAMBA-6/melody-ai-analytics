"""
Melody AI - Advanced Business Analytics Tool (Fixed Version)
Built for SMBs to harness the power of their data
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import os

# API Key Setup
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Melody AI - Business Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

# Custom CSS (same as before)
st.markdown("""
    <style>
    :root {
        --melody-pink: #FF1B6D;
        --melody-purple: #8B1BA8;
        --melody-dark: #1a1a2e;
        --melody-light: #f8f9fa;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .melody-header {
        background: linear-gradient(135deg, #FF1B6D 0%, #8B1BA8 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 27, 109, 0.3);
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #fff5f9 0%, #ffe8f3 100%);
        border-left: 4px solid var(--melody-pink);
    }
    .agent-message {
        background: linear-gradient(135deg, #f5f0ff 0%, #efe8ff 100%);
        border-left: 4px solid var(--melody-purple);
    }
    .chat-message strong {
        color: #1a1a2e !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="melody-header">
    <h1 class="melody-title">🎵 Melody AI</h1>
    <p class="melody-subtitle">Advanced Business Analytics for SMBs</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your business data (CSV)", 
        type=['csv'],
        help="Upload your CSV file to start analyzing"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                df_raw = pd.read_csv(uploaded_file, low_memory=False)
            
            # Optimize memory
            for col in df_raw.columns:
                if df_raw[col].dtype == 'object':
                    num_unique = df_raw[col].nunique()
                    num_total = len(df_raw)
                    if num_unique / num_total < 0.5:
                        df_raw[col] = df_raw[col].astype('category')
            
            st.session_state.df = df_raw
            memory_mb = df_raw.memory_usage(deep=True).sum() / 1024**2
            st.success(f"✅ Loaded {len(df_raw):,} rows ({memory_mb:.1f} MB)")
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df_raw.head(100), height=300)
            
            with st.expander("📈 Data Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", f"{df_raw.shape[0]:,}")
                    st.metric("Total Columns", df_raw.shape[1])
                with col2:
                    numeric_cols = len(df_raw.select_dtypes(include=[np.number]).columns)
                    st.metric("Numeric Fields", numeric_cols)
                    missing = df_raw.isnull().sum().sum()
                    st.metric("Missing Values", missing)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 💡 About Melody AI")
    st.markdown("""
    - 🔍 Explore business data
    - 📊 Generate insights
    - 📈 Create visualizations
    - 🎯 Make data-driven decisions
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Tool definitions with better error handling
def get_data_info(query: str) -> str:
    """Get information about the loaded dataset"""
    if st.session_state.df is None:
        return "No data loaded. Please upload a CSV file first."
    
    df = st.session_state.df
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist()[:20],
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist()[:10],
        "sample_data": df.head(3).to_dict('records')
    }
    return json.dumps(info, indent=2, default=str)

def get_statistics(query: str) -> str:
    """Get statistical summary of numeric columns"""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return "No numeric columns found."
    
    stats = numeric_df.describe().to_dict()
    return json.dumps(stats, default=str, indent=2)

def execute_calculation(query: str) -> str:
    """Execute calculations on the dataset"""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    query_lower = query.lower()
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return "No numeric columns available for calculations."
        
        # Find mentioned column
        target_col = None
        for col in numeric_cols:
            if col.lower() in query_lower:
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[0]  # Default to first numeric column
        
        # Perform calculation
        if "mean" in query_lower or "average" in query_lower:
            result = {f"average_{target_col}": float(df[target_col].mean())}
        elif "sum" in query_lower or "total" in query_lower:
            result = {f"total_{target_col}": float(df[target_col].sum())}
        elif "max" in query_lower:
            result = {f"max_{target_col}": float(df[target_col].max())}
        elif "min" in query_lower:
            result = {f"min_{target_col}": float(df[target_col].min())}
        else:
            result = {
                "mean": float(df[target_col].mean()),
                "sum": float(df[target_col].sum()),
                "column": target_col
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Calculation error: {str(e)}"

def create_visualization(query: str) -> str:
    """Create visualizations based on query"""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    query_lower = query.lower()
    
    try:
        melody_colors = ['#FF1B6D', '#8B1BA8', '#FF6B9D', '#B24BDB', '#FF9EC7']
        sns.set_palette(melody_colors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if "bar" in query_lower and cat_cols:
            top_n = 10
            value_counts = df[cat_cols[0]].value_counts().head(top_n)
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette=melody_colors)
            ax.set_title(f"Top {top_n} {cat_cols[0]}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Count")
            
        elif "pie" in query_lower and cat_cols:
            top_n = 5
            data = df[cat_cols[0]].value_counts().head(top_n)
            ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=melody_colors, startangle=90)
            ax.set_title(f"Distribution of {cat_cols[0]}", fontsize=14, fontweight='bold')
            
        elif "scatter" in query_lower and len(numeric_cols) >= 2:
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size)
            sns.scatterplot(data=sample_df, x=numeric_cols[0], y=numeric_cols[1], ax=ax, color='#FF1B6D', alpha=0.6)
            ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}", fontsize=14, fontweight='bold')
            
        elif "hist" in query_lower and numeric_cols:
            sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=ax, color='#FF1B6D', bins=30)
            ax.set_title(f"Distribution of {numeric_cols[0]}", fontsize=14, fontweight='bold')
            
        else:
            return "Please specify: bar chart, pie chart, scatter plot, or histogram"
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        return "Visualization created successfully!"
        
    except Exception as e:
        return f"Visualization error: {str(e)}"

def generate_insights(query: str) -> str:
    """Generate automatic business insights"""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    insights = []
    
    try:
        # Numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:3]:
            mean_val = df[col].mean()
            insights.append(f"📊 **{col}**: Average is {mean_val:,.2f}")
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols[:2]:
            if df[col].nunique() < 20:
                top_val = df[col].mode()[0]
                count = (df[col] == top_val).sum()
                insights.append(f"🏆 **{col}**: Most common is '{top_val}' ({count} occurrences)")
        
        # Data quality
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        insights.append(f"📋 **Data Quality**: {missing_pct:.1f}% missing values")
        
        return "\n\n".join(insights) if insights else "No significant patterns found."
        
    except Exception as e:
        return f"Insights error: {str(e)}"

# Create tools
tools = [
    Tool(
        name="GetDataInfo",
        func=get_data_info,
        description="Get dataset information: shape, columns, data types"
    ),
    Tool(
        name="GetStatistics",
        func=get_statistics,
        description="Get statistical summaries of numeric columns"
    ),
    Tool(
        name="ExecuteCalculation",
        func=execute_calculation,
        description="Calculate sum, mean, max, min for numeric columns"
    ),
    Tool(
        name="CreateVisualization",
        func=create_visualization,
        description="Create bar, pie, scatter, or histogram charts"
    ),
    Tool(
        name="GenerateInsights",
        func=generate_insights,
        description="Generate automatic insights and patterns"
    )
]

# Improved prompt template
template = """You are Melody AI, a business analyst assistant.

Available dataset columns: {columns_info}

Answer the user's question using the available tools. Be concise and direct.

IMPORTANT RULES:
1. Use GetDataInfo to see available columns first
2. For calculations, use ExecuteCalculation
3. For charts/graphs/plots, ALWAYS use CreateVisualization
4. For insights/overview, use GenerateInsights
5. Keep responses brief and clear

{tools}

Tool Names: {tool_names}

Question: {input}

Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "columns_info"]
)

@st.cache_resource
def initialize_agent(df=None):
    """Initialize the Melody AI agent"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Prepare column info
        if df is not None:
            cols_info = ", ".join(df.columns.tolist()[:15])
        else:
            cols_info = "No data loaded yet"
        
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt.partial(columns_info=cols_info)
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60,
            early_stopping_method="generate"
        )
        
        return agent_executor
    except Exception as e:
        st.error(f"Failed to initialize AI: {str(e)}")
        return None

# Main chat interface
st.markdown("### 💬 Chat with Melody AI")

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(
            f'<div class="chat-message user-message">'
            f'<strong>👤 You:</strong> {message["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message agent-message">'
            f'<strong>🎵 Melody AI:</strong> {message["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )

# Chat input
if st.session_state.df is not None:
    user_input = st.chat_input("Ask Melody AI about your data...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.agent_executor is None:
            st.session_state.agent_executor = initialize_agent(st.session_state.df)
        
        if st.session_state.agent_executor:
            with st.spinner("🎵 Analyzing..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": user_input})
                    agent_response = response.get('output', 'Unable to process request.')
                except Exception as e:
                    agent_response = f"I encountered an error: {str(e)}\n\nTry asking a simpler question."
            
            st.session_state.chat_history.append({"role": "agent", "content": agent_response})
            st.rerun()
else:
    st.info("👈 Upload your CSV file to start!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Explore\nDiscover patterns in your data")
    with col2:
        st.markdown("### 📊 Analyze\nGet statistical insights")
    with col3:
        st.markdown("### 🎯 Optimize\nMake data-driven decisions")

# Example queries
with st.expander("💡 Try These Questions"):
    st.markdown("""
    **Quick Stats:**
    - What insights can you find?
    - Give me an overview of the dataset
    - What's the average of [column name]?
    
    **Visualizations:**
    - Create a bar chart
    - Show me a pie chart
    - Plot a histogram
    
    **Analysis:**
    - What are the top categories?
    - Show me the distribution
    - Find correlations
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🎵 <strong>Melody AI</strong> - Empowering SMBs with Data Insights"
    "</div>",
    unsafe_allow_html=True
)
