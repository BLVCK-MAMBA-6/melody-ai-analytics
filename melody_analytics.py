

"""
Melody AI - Advanced Business Analytics Tool
Built for SMBs to harness the power of their data
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# NEW (Secure)
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

# Custom CSS with Melody AI branding
st.markdown("""
    <style>
    /* Melody AI Brand Colors */
    :root {
        --melody-pink: #FF1B6D;
        --melody-purple: #8B1BA8;
        --melody-dark: #1a1a2e;
        --melody-light: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main header styling */
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
    
    /* Chat message styling */
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
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    /* Button styling */
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
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 27, 109, 0.05);
        border: 2px dashed var(--melody-pink);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #FF1B6D 0%, #8B1BA8 100%);
        color: white !important;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--melody-pink);
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Chat input */
    .stChatInput {
        border: 2px solid var(--melody-pink);
        border-radius: 10px;
    }
    
    /* Success message */
    .element-container:has(> .stSuccess) {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
# Header - Displaying the uploaded banner image
# Make sure the filename matches exactly what you uploaded to GitHub
st.image("header_banner.png", use_container_width=True)

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
            # Read CSV in chunks for large files
            chunk_size = 10000
            chunks = []
            
            with st.spinner("Loading data..."):
                for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False):
                    chunks.append(chunk)
                
                df_raw = pd.concat(chunks, ignore_index=True)
            
            # Optimize memory usage
            for col in df_raw.columns:
                if df_raw[col].dtype == 'object':
                    # Convert to category if few unique values
                    num_unique = df_raw[col].nunique()
                    num_total = len(df_raw)
                    
                    if num_unique / num_total < 0.5:  # If less than 50% unique
                        df_raw[col] = df_raw[col].astype('category')
                    else:
                        try:
                            numeric_col = pd.to_numeric(df_raw[col], errors='coerce')
                            if numeric_col.notna().sum() > len(df_raw) * 0.5:
                                df_raw[col] = numeric_col
                            else:
                                df_raw[col] = df_raw[col].astype(str)
                        except:
                            df_raw[col] = df_raw[col].astype(str)
            
            st.session_state.df = df_raw
            
            # Show memory usage
            memory_mb = df_raw.memory_usage(deep=True).sum() / 1024**2
            st.success(f"✅ Loaded {len(st.session_state.df):,} rows ({memory_mb:.1f} MB)")
            
            with st.expander("📋 Data Preview"):
                # Display only first 100 rows to save memory
                preview_df = st.session_state.df.head(100)
                st.dataframe(preview_df, height=300)
            
            with st.expander("📈 Data Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", f"{st.session_state.df.shape[0]:,}")
                    st.metric("Total Columns", st.session_state.df.shape[1])
                with col2:
                    numeric_cols = len(st.session_state.df.select_dtypes(include=[np.number]).columns)
                    st.metric("Numeric Fields", numeric_cols)
                    missing = st.session_state.df.isnull().sum().sum()
                    st.metric("Missing Values", missing)
                
                st.markdown("**Column Names:**")
                cols_text = ""
                for i, col in enumerate(st.session_state.df.columns[:15]):
                    cols_text += f"• {col}\n"
                if len(st.session_state.df.columns) > 15:
                    cols_text += f"... +{len(st.session_state.df.columns)-15} more"
                st.text(cols_text)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Tip: If your file is very large, try reducing its size or filtering the data first.")
    
    st.markdown("---")
    st.markdown("### 💡 About Melody AI")
    st.markdown("""
    Melody AI empowers SMBs to:
    - 🔍 Explore business data
    - 📊 Generate insights
    - 📈 Create visualizations
    - 🎯 Make data-driven decisions
    - 💼 Optimize performance
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("🤖 Google Gemini AI")
    st.markdown("🔗 LangChain")

# Tool definitions
def get_data_info(query: str) -> str:
    """Get information about the loaded dataset"""
    if st.session_state.df is None:
        return "No data loaded. Please upload a CSV file first."
    
    df = st.session_state.df
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()[:20],
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'string']).columns.tolist()[:10],
        "missing_values": {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    }
    return json.dumps(info, indent=2)

def get_statistics(query: str) -> str:
    """Get statistical summary of the dataset or specific columns"""
    if st.session_state.df is None:
        return "No data loaded. Please upload a CSV file first."
    
    df = st.session_state.df
    columns = [col for col in df.columns if col.lower() in query.lower()]
    
    if columns:
        stats = df[columns].describe().to_dict()
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats = numeric_df.describe().to_dict()
        else:
            return "No numeric columns found in the dataset."
    
    return json.dumps(stats, default=str, indent=2)

def execute_calculation(query: str) -> str:
    """Execute calculations on the dataset"""
    if st.session_state.df is None:
        return "No data loaded. Please upload a CSV file first."
    
    df = st.session_state.df
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return "No numeric columns available for calculations."
        
        # Check for max/highest queries with grouping
        if ("max" in query.lower() or "highest" in query.lower() or "expensive" in query.lower()) and "price" in query.lower():
            if "model" in query.lower():
                # Group by Model and find max price
                result = df.groupby('Model')['Price ($)'].max().sort_values(ascending=False).head(10).to_dict()
                return json.dumps({"top_10_most_expensive_models": result}, default=str, indent=2)
            else:
                max_price_row = df.loc[df['Price ($)'].idxmax()]
                result = {
                    "most_expensive_car": {
                        "model": str(max_price_row.get('Model', 'N/A')),
                        "company": str(max_price_row.get('Company', 'N/A')),
                        "price": float(max_price_row['Price ($)']),
                        "body_style": str(max_price_row.get('Body Style', 'N/A'))
                    }
                }
                return json.dumps(result, default=str, indent=2)
        
        elif "sum" in query.lower():
            result = {col: float(df[col].sum()) for col in numeric_cols[:5]}
        elif "mean" in query.lower() or "average" in query.lower():
            result = {col: float(df[col].mean()) for col in numeric_cols[:5]}
        elif "median" in query.lower():
            result = {col: float(df[col].median()) for col in numeric_cols[:5]}
        elif "correlation" in query.lower():
            corr_matrix = df[numeric_cols[:5]].corr()
            result = corr_matrix.to_dict()
        elif "count" in query.lower():
            result = {"total_rows": len(df), "total_columns": len(df.columns)}
        else:
            result = {
                "available_operations": ["sum", "mean", "median", "correlation", "count", "max", "min"],
                "numeric_columns": numeric_cols[:10]
            }
        
        return json.dumps(result, default=str, indent=2)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def filter_data(query: str) -> str:
    """Filter data based on conditions"""
    if st.session_state.df is None:
        return "No data loaded. Please upload a CSV file first."
    
    df = st.session_state.df
    
    try:
        result = {
            "total_rows": len(df),
            "columns": df.columns.tolist()[:15],
            "sample_values": {}
        }
        
        for col in df.columns[:5]:
            if df[col].dtype in ['object', 'string']:
                result["sample_values"][col] = df[col].value_counts().head(3).to_dict()
            else:
                result["sample_values"][col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
        
        return json.dumps(result, default=str, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

def create_visualization(query: str) -> str:
    """Create visualizations based on the query"""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    query = query.lower()
    
    try:
        # Set styling
        melody_colors = ['#FF1B6D', '#8B1BA8', '#FF6B9D', '#B24BDB', '#FF9EC7']
        sns.set_palette(melody_colors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        
        # Logic for PIE CHART
        if "pie" in query:
            if cat_cols and numeric_cols:
                # Usually we want a categorical column vs a numeric one (e.g., Price by Car Model)
                # Check if we need to sort (e.g., "top 3")
                if "top" in query:
                    try:
                        limit = int(''.join(filter(str.isdigit, query)))
                    except:
                        limit = 5
                    
                    # Group and sum/mean
                    data = df.groupby(cat_cols[0])[numeric_cols[0]].sum().sort_values(ascending=False).head(limit)
                    
                    ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=melody_colors, startangle=90)
                    ax.set_title(f"Top {limit} {cat_cols[0]} by {numeric_cols[0]}", fontsize=14)
                else:
                     # Simple count of a category
                    data = df[cat_cols[0]].value_counts().head(5)
                    ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=melody_colors)
                    ax.set_title(f"Distribution of {cat_cols[0]}", fontsize=14)
            else:
                return "I need both categorical and numeric data for a meaningful pie chart."

        # Logic for BAR CHART
        elif "bar" in query:
            if cat_cols and numeric_cols:
                if "top" in query or "most" in query:
                    # Sort for "top" items
                    data = df.groupby(cat_cols[0])[numeric_cols[0]].mean().sort_values(ascending=False).head(10)
                    sns.barplot(x=data.values, y=data.index, ax=ax, palette=melody_colors)
                else:
                    # Standard count plot
                    sns.countplot(y=cat_cols[0], data=df, order=df[cat_cols[0]].value_counts().iloc[:10].index, ax=ax, palette=melody_colors)
                
                ax.set_title("Bar Chart Analysis", fontsize=14)
            else:
                return "Not enough data for a bar chart."

        # Logic for SCATTER
        elif "scatter" in query and len(numeric_cols) >= 2:
            sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], ax=ax, color='#FF1B6D')
            ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}", fontsize=14)

        # Logic for HISTOGRAM
        elif "hist" in query and numeric_cols:
            sns.histplot(df[numeric_cols[0]], kde=True, ax=ax, color='#FF1B6D')
            ax.set_title(f"Distribution of {numeric_cols[0]}", fontsize=14)

        # Default to HEATMAP only if specifically asked or if correlation is mentioned
        elif "correlation" in query or "heatmap" in query:
            if len(numeric_cols) > 1:
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='RdPu', ax=ax)
                ax.set_title("Correlation Heatmap", fontsize=14)
            else:
                return "Need at least 2 numeric columns for a heatmap."
        else:
            return "Please specify if you want a Bar, Pie, Scatter, or Histogram chart."

        plt.tight_layout()
        st.pyplot(fig) # Render immediately
        return "I have created the visualization as requested."

    except Exception as e:
        return f"Error generating chart: {str(e)}"


def generate_insights(query: str) -> str:
    """Generate automatic business insights: summary stats, correlations, and top categories."""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    insights = []
    
    try:
        # 1. Numeric Insights (Trends)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Correlation check
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()
                # Select upper triangle of correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find index of feature columns with correlation greater than 0.7
                high_corr = [column for column in upper.columns if any(upper[column] > 0.7)]
                if high_corr:
                    insights.append(f"📈 **Key Correlation**: Strong relationship found involving {', '.join(high_corr[:3])}.")
            
            # Basic stats for first 3 key metrics
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                insights.append(f"📊 **{col}**: Average value is {mean_val:,.2f}")

        # 2. Categorical Insights (Top Performers)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            for col in cat_cols[:2]: # Check first 2 categorical columns
                top_val = df[col].mode()[0]
                count = df[col].value_counts().iloc[0]
                insights.append(f"🏆 **Top {col}**: '{top_val}' is the leader with {count} entries.")

        return "\n".join(insights) if insights else "No significant patterns found."
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Create tools
tools = [
    Tool(
        name="GetDataInfo",
        func=get_data_info,
        description="Get basic information about the dataset: shape, columns, data types, missing values."
    ),
    Tool(
        name="GetStatistics",
        func=get_statistics,
        description="Get statistical summaries: mean, median, std, min, max, quartiles for numeric columns."
    ),
    Tool(
        name="ExecuteCalculation",
        func=execute_calculation,
        description="Perform calculations: sum, mean, median, correlation, count, max, min. Can find most expensive items or group by categories."
    ),
    Tool(
        name="FilterData",
        func=filter_data,
        description="Get filtering information and sample values from the dataset."
    ),
    Tool(
        name="CreateVisualization",
        func=create_visualization,
        description="Create visualizations: histogram, scatter plot, bar chart, or correlation heatmap."
    ),
    Tool(
        name="GenerateInsights",
        func=generate_insights,
        description="Use this tool when the user asks for 'insights', 'overview', 'patterns', or 'what can you tell me'. It automatically finds trends and correlations."
    )
]

# 1. Create the prompt template WITH column info injection
template = """You are Melody AI, a smart business analyst. 
You are working with a dataset that has the following columns:
{columns_info}

RULES:
1. ALWAYS look at the column names above before answering.
2. If the user asks for "transaction value" or "sales", map it to the most relevant numeric column in the list above (e.g., Price, Salary, Cost).
3. If the user asks for a visualization, you MUST use the CreateVisualization tool. Do not just describe it.
4. If the user asks to "plot", "graph", or "chart", use CreateVisualization.

Tools: {tools}
Tool names: {tool_names}

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "columns_info"]
)

# 2. Update Initialize Agent to Pass the Columns
@st.cache_resource
def initialize_agent(df=None): # Add df as argument
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Using 1.5-flash as it's more stable for tools
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            max_output_tokens=1024
        )
        
        # Prepare column info string
        if df is not None:
            cols_info = ", ".join(df.columns.tolist())
            # Add type info for better context
            for col in df.columns:
                cols_info += f"\n- {col} ({df[col].dtype})"
        else:
            cols_info = "No data loaded yet."

        # Create the agent with the column info partially filled in
        agent = create_react_agent(
            llm=llm, 
            tools=tools, 
            prompt=prompt.partial(columns_info=cols_info) # Inject columns here!
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=120
        )
        
        return agent_executor
    except Exception as e:
        st.error(f"Failed to initialize Melody AI: {str(e)}")
        return None
# Main chat interface
st.markdown("### Chat with Melody AI")

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
    user_input = st.chat_input("Ask Melody AI about your business data...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.agent_executor is None:
            st.session_state.agent_executor = initialize_agent(st.session_state.df)
        
        if st.session_state.agent_executor is None:
            st.error("Failed to initialize Melody AI. Please check your API key.")
            st.stop()
        
        with st.spinner("🎵 Analyzing..."):
            try:
                response = st.session_state.agent_executor.invoke(
                    {"input": user_input},
                    config={"max_execution_time": 120}
                )
                agent_response = response.get('output', 'I encountered an error processing your request.')
            except Exception as e:
                agent_response = f"Error: {str(e)}\n\nPlease try a simpler question."
        
        st.session_state.chat_history.append({"role": "agent", "content": agent_response})
        st.rerun()

else:
    st.info("👈 Upload your business data CSV file to start your analytics journey with Melody AI!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Explore")
        st.markdown("Discover patterns and trends in your business data")
    
    with col2:
        st.markdown("### 📊 Analyze")
        st.markdown("Get statistical insights and correlations")
    
    with col3:
        st.markdown("### 🎯 Optimize")
        st.markdown("Make data-driven decisions for growth")

# Example queries
with st.expander("💡 Example Business Questions"):
    st.markdown("""
    **Sales & Revenue:**
    - What's the average transaction value?
    - Show me revenue trends over time
    - Which product category performs best?
    
    **Customer Insights:**
    - What's the customer demographic breakdown?
    - Show me customer purchase patterns
    - Analyze customer lifetime value
    
    **Performance Metrics:**
    - Create a performance dashboard visualization
    - What are the key business metrics?
    - Show correlation between sales factors
    
    **Data Exploration:**
    - Give me an overview of the dataset
    - Are there any data quality issues?
    - What insights can you find?
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🎵 <strong>Melody AI</strong> - Empowering SMBs with Data-Driven Insights<br>"
    "<small>Built with ❤️ for business growth</small>"
    "</div>", 
    unsafe_allow_html=True

)






