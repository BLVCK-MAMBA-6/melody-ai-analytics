"""
Melody AI - Advanced Business Analytics Tool (FINAL FIX)
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

# API Key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Melody AI - Business Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'last_viz' not in st.session_state:
    st.session_state.last_viz = None

# CSS
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
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                df_raw = pd.read_csv(uploaded_file, low_memory=False)
            
            for col in df_raw.columns:
                if df_raw[col].dtype == 'object':
                    if df_raw[col].nunique() / len(df_raw) < 0.5:
                        df_raw[col] = df_raw[col].astype('category')
            
            st.session_state.df = df_raw
            st.session_state.agent_executor = None
            
            memory_mb = df_raw.memory_usage(deep=True).sum() / 1024**2
            st.success(f"✅ {len(df_raw):,} rows ({memory_mb:.1f} MB)")
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df_raw.head(100), height=300)
            
            with st.expander("📈 Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{df_raw.shape[0]:,}")
                    st.metric("Columns", df_raw.shape[1])
                with col2:
                    st.metric("Numeric", len(df_raw.select_dtypes(include=[np.number]).columns))
                    st.metric("Missing", df_raw.isnull().sum().sum())
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_viz = None
        st.rerun()

# Tool definitions with ACTUAL visualization rendering
def get_data_info(query: str) -> str:
    """Get dataset information"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    return json.dumps({
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }, indent=2)

def query_data(query: str) -> str:
    """Query specific data - find top values, filter, search"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    query_lower = query.lower()
    
    try:
        # Find highest/lowest values
        if "highest" in query_lower or "maximum" in query_lower or "top" in query_lower:
            # Find numeric column mentioned
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_col = None
            for col in numeric_cols:
                if col.lower() in query_lower:
                    target_col = col
                    break
            
            if target_col:
                # Get top N rows
                n = 3  # default
                if "top 5" in query_lower or "5 highest" in query_lower:
                    n = 5
                elif "top 10" in query_lower:
                    n = 10
                
                top_rows = df.nlargest(n, target_col)
                result = []
                for idx, row in top_rows.iterrows():
                    result.append({col: str(row[col]) for col in df.columns[:10]})
                return json.dumps({"top_results": result, "sorted_by": target_col}, indent=2)
        
        # Category analysis
        elif "category" in query_lower or "group" in query_lower:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                cat_col = cat_cols[0]
                num_col = numeric_cols[0]
                
                # Check for specific columns in query
                for col in cat_cols:
                    if col.lower() in query_lower:
                        cat_col = col
                        break
                for col in numeric_cols:
                    if col.lower() in query_lower:
                        num_col = col
                        break
                
                grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
                return json.dumps({
                    "analysis": "grouped_sum",
                    "category_column": cat_col,
                    "value_column": num_col,
                    "results": {str(k): float(v) for k, v in grouped.items()}
                }, indent=2)
        
        return json.dumps({"message": "Please specify what data you want to query"})
        
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_statistics(query: str) -> str:
    """Get statistical summaries"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return "No numeric columns"
    
    stats = {}
    for col in numeric_df.columns[:10]:
        stats[col] = {
            "mean": float(numeric_df[col].mean()),
            "median": float(numeric_df[col].median()),
            "min": float(numeric_df[col].min()),
            "max": float(numeric_df[col].max()),
            "std": float(numeric_df[col].std())
        }
    
    return json.dumps(stats, indent=2)

def create_chart(query: str) -> str:
    """Create and DISPLAY visualizations"""
    if st.session_state.df is None:
        return "No data loaded"
    
    df = st.session_state.df
    query_lower = query.lower()
    
    try:
        colors = ['#FF1B6D', '#8B1BA8', '#FF6B9D', '#B24BDB', '#FF9EC7']
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        chart_created = False
        
        # PIE CHART
        if "pie" in query_lower:
            if cat_cols and numeric_cols:
                # Find if specific columns mentioned
                cat_col = cat_cols[0]
                num_col = numeric_cols[0]
                
                for col in cat_cols:
                    if col.lower() in query_lower:
                        cat_col = col
                        break
                for col in numeric_cols:
                    if col.lower() in query_lower:
                        num_col = col
                        break
                
                # Group and get top N
                n = 5
                if "top 3" in query_lower:
                    n = 3
                elif "top 10" in query_lower:
                    n = 10
                
                data = df.groupby(cat_col)[num_col].sum().nlargest(n)
                
                wedges, texts, autotexts = ax.pie(
                    data.values, 
                    labels=data.index, 
                    autopct='%1.1f%%', 
                    colors=colors,
                    startangle=90
                )
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                ax.set_title(f"Top {n} {cat_col} by {num_col}", fontsize=14, fontweight='bold', pad=20)
                chart_created = True
        
        # BAR CHART
        elif "bar" in query_lower:
            if cat_cols and numeric_cols:
                cat_col = cat_cols[0]
                num_col = numeric_cols[0]
                
                for col in cat_cols:
                    if col.lower() in query_lower:
                        cat_col = col
                        break
                for col in numeric_cols:
                    if col.lower() in query_lower:
                        num_col = col
                        break
                
                n = 10
                if "top 3" in query_lower:
                    n = 3
                elif "top 5" in query_lower:
                    n = 5
                
                data = df.groupby(cat_col)[num_col].sum().nlargest(n).sort_values()
                
                bars = ax.barh(range(len(data)), data.values, color=colors[0])
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels(data.index)
                ax.set_xlabel(num_col, fontsize=11)
                ax.set_title(f"Top {n} {cat_col} by {num_col}", fontsize=14, fontweight='bold', pad=20)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:,.0f}', ha='left', va='center', fontsize=9)
                
                chart_created = True
        
        # HISTOGRAM
        elif "hist" in query_lower or "distribution" in query_lower:
            if numeric_cols:
                col = numeric_cols[0]
                for c in numeric_cols:
                    if c.lower() in query_lower:
                        col = c
                        break
                
                ax.hist(df[col].dropna(), bins=30, color=colors[0], edgecolor='white', alpha=0.7)
                ax.set_xlabel(col, fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3)
                chart_created = True
        
        # SCATTER
        elif "scatter" in query_lower and len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            ax.scatter(df[x_col], df[y_col], color=colors[0], alpha=0.5, s=50)
            ax.set_xlabel(x_col, fontsize=11)
            ax.set_ylabel(y_col, fontsize=11)
            ax.set_title(f"{x_col} vs {y_col}", fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            chart_created = True
        
        if chart_created:
            plt.tight_layout()
            st.session_state.last_viz = fig
            return "CHART_CREATED"
        else:
            return "Please specify: pie, bar, histogram, or scatter chart"
            
    except Exception as e:
        return f"Error creating chart: {str(e)}"

# Tools
tools = [
    Tool(
        name="GetDataInfo",
        func=get_data_info,
        description="Get dataset info: rows, columns, column names and types. Use this FIRST to understand the data structure."
    ),
    Tool(
        name="QueryData",
        func=query_data,
        description="Query and filter data. Find highest/lowest values, top performers, search for specific records. Use this to find specific data points like 'customer with highest income' or 'top 3 most expensive items'."
    ),
    Tool(
        name="GetStatistics",
        func=get_statistics,
        description="Get statistical summaries: mean, median, min, max, std for numeric columns."
    ),
    Tool(
        name="CreateChart",
        func=create_chart,
        description="Create visualizations: pie chart, bar chart, histogram, scatter plot. MUST specify chart type and optionally column names. Returns 'CHART_CREATED' when successful."
    )
]

# Agent initialization
def initialize_agent():
    if st.session_state.df is None:
        return None
    
    try:
        df = st.session_state.df
        cols_preview = ", ".join(df.columns.tolist()[:20])
        if len(df.columns) > 20:
            cols_preview += f" (and {len(df.columns)-20} more)"
        
        template = """You are Melody AI, a business analyst. Answer questions using the provided tools.

Dataset columns: {columns_info}

IMPORTANT INSTRUCTIONS:
1. ALWAYS use GetDataInfo FIRST to see available columns
2. For questions about "highest", "top", "best performing", use QueryData tool
3. For visualizations, use CreateChart and specify the chart type (pie/bar/histogram/scatter)
4. When CreateChart returns "CHART_CREATED", tell the user the chart is displayed above
5. Keep responses concise and friendly

Available tools:

{tools}

Use this format:

Question: the input question
Thought: what should I do
Action: tool name from [{tool_names}]
Action Input: input for the tool
Observation: tool output
... (repeat Thought/Action/Input/Observation as needed)
Thought: I now know the answer
Final Answer: clear, friendly response to the user

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
            prompt=prompt.partial(columns_info=cols_preview)
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=90
        )
    except Exception as e:
        st.error(f"Init error: {str(e)}")
        return None

# Chat interface
st.markdown("### 💬 Chat with Melody AI")

# Display last visualization if exists
if st.session_state.last_viz is not None:
    st.pyplot(st.session_state.last_viz)
    st.markdown("---")

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
                    answer = f"I encountered an error: {str(e)}\n\nTry: 'Show me an overview', 'What are the columns?', or 'Create a bar chart'"
            
            st.session_state.chat_history.append({"role": "agent", "content": answer})
            st.rerun()
        else:
            st.error("Failed to initialize agent")
else:
    st.info("👈 Upload a CSV file to start!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Explore\nDiscover patterns in your data")
    with col2:
        st.markdown("### 📊 Analyze\nGet statistical insights")
    with col3:
        st.markdown("### 🎯 Optimize\nMake data-driven decisions")

with st.expander("💡 Example Questions"):
    st.markdown("""
    **Data Overview:**
    - What columns does this dataset have?
    - Give me an overview
    - Show me statistics
    
    **Specific Queries:**
    - Which customer has the highest income?
    - What are the top 3 most expensive cars?
    - Which product category performs best?
    
    **Visualizations:**
    - Show me a pie chart of top 3 most expensive cars
    - Create a bar chart of top 5 companies by price
    - Make a histogram of prices
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🎵 <strong>Melody AI</strong> - Data-Driven Insights for SMBs"
    "</div>",
    unsafe_allow_html=True
)
