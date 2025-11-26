"""
Melody AI - Advanced Business Analytics Tool (FIXED v3)
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

# Try to get key from Streamlit secrets, otherwise environment variable
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

# Custom CSS
st.markdown("""
    <style>
    :root {
        --melody-pink: #FF1B6D;
        --melody-purple: #8B1BA8;
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
    
    .melody-title {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .melody-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.95);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
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
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF1B6D 0%, #8B1BA8 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 27, 109, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="melody-header">'
    '<h1 class="melody-title">🎵 Melody AI</h1>'
    '<p class="melody-subtitle">Advanced Business Analytics for SMBs</p>'
    '</div>',
    unsafe_allow_html=True
)

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
                    if num_unique / len(df_raw) < 0.5:
                        df_raw[col] = df_raw[col].astype('category')
            
            st.session_state.df = df_raw
            st.session_state.agent_executor = None
            
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
                
                st.markdown("**Columns:**")
                st.text("\n".join([f"• {col}" for col in df_raw.columns[:15]]))
                if len(df_raw.columns) > 15:
                    st.text(f"... +{len(df_raw.columns)-15} more")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 💡 About Melody AI")
    st.markdown("""
    - 🔍 Explore business data
    - 📊 Generate insights
    - 📈 Create visualizations
    - 🎯 Data-driven decisions
    """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Tool definitions
def get_data_info(query: str) -> str:
    """Get information about the loaded dataset"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    return json.dumps({
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }, indent=2)

def get_statistics(query: str) -> str:
    """Get statistical summary"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return "No numeric columns found"
    
    return json.dumps(numeric_df.describe().to_dict(), default=str, indent=2)

def execute_calculation(query: str) -> str:
    """Execute calculations"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    query_lower = query.lower()
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return "No numeric columns available"
        
        # Find mentioned column
        target_col = None
        for col in numeric_cols:
            if col.lower() in query_lower:
                target_col = col
                break
        
        if target_col:
            if "mean" in query_lower or "average" in query_lower:
                return json.dumps({"average": float(df[target_col].mean()), "column": target_col})
            elif "sum" in query_lower or "total" in query_lower:
                return json.dumps({"total": float(df[target_col].sum()), "column": target_col})
            elif "max" in query_lower:
                return json.dumps({"max": float(df[target_col].max()), "column": target_col})
            elif "min" in query_lower:
                return json.dumps({"min": float(df[target_col].min()), "column": target_col})
        
        # Default: return means
        result = {col: float(df[col].mean()) for col in numeric_cols[:5]}
        return json.dumps({"averages": result})
        
    except Exception as e:
        return json.dumps({"error": str(e)})

def create_visualization(query: str) -> str:
    """Create visualizations"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    query_lower = query.lower()
    
    try:
        colors = ['#FF1B6D', '#8B1BA8', '#FF6B9D', '#B24BDB']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if "pie" in query_lower and cat_cols:
            data = df[cat_cols[0]].value_counts().head(5)
            ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=colors)
            ax.set_title(f"{cat_cols[0]} Distribution")
            
        elif "bar" in query_lower and cat_cols:
            data = df[cat_cols[0]].value_counts().head(10)
            ax.barh(data.index, data.values, color=colors[0])
            ax.set_title(f"{cat_cols[0]} Counts")
            
        elif "scatter" in query_lower and len(numeric_cols) >= 2:
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], color=colors[0], alpha=0.6)
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
            
        elif "hist" in query_lower and numeric_cols:
            ax.hist(df[numeric_cols[0]].dropna(), bins=30, color=colors[0], edgecolor='white')
            ax.set_title(f"{numeric_cols[0]} Distribution")
            ax.set_xlabel(numeric_cols[0])
            
        else:
            return "Please specify: pie, bar, scatter, or histogram"
        
        plt.tight_layout()
        st.pyplot(fig)
        return "Visualization created"
        
    except Exception as e:
        return f"Error: {str(e)}"

def generate_insights(query: str) -> str:
    """Generate insights"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    insights = []
    
    try:
        # Numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:3]:
            insights.append(f"📊 {col}: avg={df[col].mean():.2f}, max={df[col].max():.2f}")
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols[:2]:
            top = df[col].mode()[0]
            count = df[col].value_counts().iloc[0]
            insights.append(f"🏆 Top {col}: '{top}' ({count} entries)")
        
        return "\n".join(insights) if insights else "No patterns found"
    except Exception as e:
        return f"Error: {str(e)}"

# Create tools
tools = [
    Tool(name="GetDataInfo", func=get_data_info, 
         description="Get dataset information: rows, columns, types"),
    Tool(name="GetStatistics", func=get_statistics,
         description="Get statistical summaries of numeric columns"),
    Tool(name="ExecuteCalculation", func=execute_calculation,
         description="Calculate mean, sum, max, min of columns"),
    Tool(name="CreateVisualization", func=create_visualization,
         description="Create charts: pie, bar, scatter, histogram"),
    Tool(name="GenerateInsights", func=generate_insights,
         description="Generate insights and patterns from data")
]

# Initialize agent
def initialize_agent():
    if st.session_state.df is None:
        return None
    
    try:
        df = st.session_state.df
        cols_info = ", ".join(df.columns.tolist()[:15])
        if len(df.columns) > 15:
            cols_info += f"... (+{len(df.columns)-15} more)"
        
        # Create proper ReAct prompt
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Dataset columns: {columns_info}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names", "columns_info"],
            template=template
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt.partial(columns_info=cols_info)
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60
        )
    except Exception as e:
        st.error(f"Init error: {str(e)}")
        return None

# Chat interface
st.markdown("### 💬 Chat with Melody AI")

for msg in st.session_state.chat_history:
    css_class = "user-message" if msg['role'] == 'user' else "agent-message"
    icon = "👤" if msg['role'] == 'user' else "🎵"
    label = "You" if msg['role'] == 'user' else "Melody AI"
    st.markdown(
        f'<div class="chat-message {css_class}">'
        f'<strong>{icon} {label}:</strong> {msg["content"]}'
        f'</div>',
        unsafe_allow_html=True
    )

if st.session_state.df is not None:
    user_input = st.chat_input("Ask Melody AI...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.agent_executor is None:
            with st.spinner("Initializing..."):
                st.session_state.agent_executor = initialize_agent()
        
        if st.session_state.agent_executor:
            with st.spinner("🎵 Analyzing..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": user_input})
                    answer = response.get('output', 'Error processing request')
                except Exception as e:
                    answer = f"Error: {str(e)}\n\nTry: 'Give me an overview' or 'Show statistics'"
            
            st.session_state.chat_history.append({"role": "agent", "content": answer})
            st.rerun()
        else:
            st.error("Failed to initialize agent")
else:
    st.info("👈 Upload a CSV file to start!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Explore\nDiscover patterns")
    with col2:
        st.markdown("### 📊 Analyze\nGet insights")
    with col3:
        st.markdown("### 🎯 Optimize\nMake decisions")

with st.expander("💡 Example Questions"):
    st.markdown("""
    - Give me an overview of the dataset
    - What are the column names?
    - Show me statistics
    - What's the average price?
    - Create a bar chart
    - Show me a pie chart
    - Generate insights
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🎵 <strong>Melody AI</strong> - Data-Driven Insights for SMBs"
    "</div>",
    unsafe_allow_html=True
)
