# -*- coding: utf-8 -*-
"""
Eli Lilly "Bot-of-Bots" (BoB) Platform - Main Application.

This Streamlit application serves as a sophisticated front-end for a multi-agent
orchestration system built on AWS Bedrock. It allows users to submit healthcare-related
queries, which are then intelligently routed to one or more specialized AI agents.

The platform features:
- A dynamic, branded user interface built with Streamlit.
- Pre-routing logic to quickly direct specific queries (e.g., IT tickets) to the
  correct agent, bypassing the full orchestration process for efficiency.
- A full orchestration pipeline that queries multiple agents in parallel.
- A response evaluation system that uses semantic similarity and an LLM-as-a-Judge
  pattern to select the best answer from the agent cohort.
- Interactive dashboards for performance analytics, agent capabilities, and
  historical query insights.
"""

# --- 1. IMPORTS ---
# --- Standard Library Imports ---
import json
import time
import os
import uuid
import math
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Third-party Library Imports ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- AWS SDK Imports ---
import boto3
from botocore.exceptions import ClientError


# --- 2. INITIAL CONFIGURATION ---
# --- Streamlit Page Configuration ---
# Sets the title, icon, and layout for the Streamlit web application.
st.set_page_config(
    page_title="Eli Lilly BoB Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AWS Configuration ---
# NOTE: Hardcoding credentials is not a recommended security practice.
# In a production environment, use IAM Roles, environment variables, or other secure methods.
# These values are placeholders and should be replaced with your actual AWS credentials.
# AWS_ACCESS_KEY_ID     = "YOUR_ACCESS_KEY"
# AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
# AWS_SESSION_TOKEN     = "YOUR_SESSION_TOKEN"
# AWS_REGION            = "us-east-1"

# Creates a Boto3 session object. All subsequent AWS client calls will use this session.
session = boto3.Session(
    # aws_access_key_id=AWS_ACCESS_KEY_ID,
    # aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    # aws_session_token=AWS_SESSION_TOKEN,
    # region_name=AWS_REGION
)

# Initialize AWS Bedrock clients.
# 'bedrock-agent-runtime' is for invoking Bedrock agents.
# 'bedrock-runtime' is for invoking foundational models directly.
agent_client = session.client("bedrock-agent-runtime")
bedrock_client = session.client("bedrock-runtime")

# --- Agent and Model Configuration ---

@dataclass(frozen=True)
class AgentConfig:
    """
    A data class to hold the configuration for a single Bedrock Agent.
    Using a dataclass provides type safety and a clear, immutable structure.
    """
    name: str
    arn: str
    alias_id: str

# Central registry of all available Bedrock agents.
# This dictionary maps a friendly name to its specific AWS configuration.
AGENTS = {
    "zepboundinfoagent": AgentConfig(name="zepboundinfoagent", arn="arn:aws:bedrock:us-east-1:098387547573:agent/SH3M3LINWU", alias_id="VA7K94EJVF"),
    "mounjaro_specialist_agent": AgentConfig(name="mounjaro_specilist_agent", arn="arn:aws:bedrock:us-east-1:098387547573:agent/R8MQDABQGT", alias_id="NUBQPKNTJV"),
    "trulicity_agent": AgentConfig(name="trulicity_agent", arn="arn:aws:bedrock:us-east-1:098387547573:agent/KDLFCRWFGX", alias_id="UQBE49N4HP"),
    "serviceNowtickets_agent": AgentConfig(name="serviceNowtickets_agent", arn="arn:aws:bedrock:us-east-1:098387547573:agent/LNDCKLQWUR", alias_id="TXBG0YIUY9")
}

# IDs for the foundational models used in the orchestration logic.
EVAL_MODEL_ID = "amazon.titan-text-premier-v1:0"  # Used for the LLM-as-a-Judge evaluation.
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"     # Used for generating text embeddings.

# --- Business Logic Configuration ---
# Keywords that trigger the pre-routing logic to the ServiceNow agent.
TICKET_KEYWORDS = ["ticket", "case", "incident"]
SERVICENOW_AGENT_NAME = "serviceNowtickets_agent"


# --- 3. CORE ORCHESTRATION FUNCTIONS ---

def extract_agent_id(arn: str) -> str:
    """Extracts the agent ID from its full ARN."""
    return arn.rsplit("/", 1)[-1]

def invoke_agent(agent_cfg: AgentConfig, prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
    """
    Invokes a specific Bedrock agent with a given prompt and handles retries.

    This function communicates with the AWS Bedrock Agent Runtime, sending the
    user's query. It includes robust error handling for common transient issues
    like timeouts and throttling, implementing an exponential backoff strategy.

    Args:
        agent_cfg: The configuration object for the agent to invoke.
        prompt: The user's query text.
        max_retries: The maximum number of times to retry on failure.
        retry_delay: The base delay in seconds for the retry mechanism.

    Returns:
        The complete, streamed response from the agent as a single string.
        
    Raises:
        ClientError: If a non-retriable AWS error occurs or if max retries are exceeded.
    """
    session_id = str(uuid.uuid4())  # Generate a unique session ID for this invocation.
    for attempt in range(1, max_retries + 1):
        try:
            # Call the Bedrock Agent Runtime API.
            resp = agent_client.invoke_agent(
                agentId=extract_agent_id(agent_cfg.arn),
                agentAliasId=agent_cfg.alias_id,
                inputText=prompt,
                sessionId=session_id,
                enableTrace=False,  # Set to True for detailed debugging in CloudWatch.
                endSession=False    # Keep the session open for potential follow-ups (if needed).
            )
            # The response is a stream; iterate through chunks and decode them.
            return "".join(
                chunk["chunk"]["bytes"].decode("utf-8")
                for chunk in resp.get("completion", [])
            )
        except ClientError as e:
            # Handle specific, retriable errors (timeouts, throttling).
            if "Read timed out" in str(e) or "ThrottlingException" in str(e):
                if attempt < max_retries:
                    # Exponential backoff: wait longer after each failed attempt.
                    time.sleep(retry_delay * attempt)
                else:
                    # If max retries are reached, re-raise the exception.
                    raise
            else:
                # For other client errors, fail immediately.
                raise

def get_embedding(text: str) -> list[float]:
    """
    Generates a vector embedding for a given text using a Bedrock model.
    Includes a simple in-memory cache to avoid redundant API calls for the same text.

    Args:
        text: The input string to embed.

    Returns:
        A list of floats representing the text's vector embedding.
    """
    # Initialize a cache on the function object itself if it doesn't exist.
    if not hasattr(get_embedding, "cache"):
        get_embedding.cache = {}
    
    # Return the cached embedding if available.
    if text in get_embedding.cache:
        return get_embedding.cache[text]

    # Prepare the payload for the Bedrock API.
    payload = {"inputText": text}
    resp = bedrock_client.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8")
    )
    # Parse the response and extract the embedding vector.
    embedding = json.loads(resp["body"].read())["embedding"]
    
    # Store the new embedding in the cache before returning.
    get_embedding.cache[text] = embedding
    return embedding

def compute_confidence(candidate: str, reference: str) -> float:
    """
    Calculates a confidence score based on the semantic similarity (cosine similarity)
    between a candidate response and a reference answer. The score is normalized to a 0-100 scale.

    Args:
        candidate: The agent's generated response.
        reference: The ground-truth or reference answer.

    Returns:
        A confidence score between 0 and 100.
    """
    v1 = get_embedding(candidate)
    v2 = get_embedding(reference)
    
    # Calculate Cosine Similarity
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    # Avoid division by zero if a vector is all zeros.
    sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    # Normalize the similarity score from [-1, 1] to a percentage [0, 100].
    return (sim + 1) / 2 * 100

def evaluate_response_llm(user_question: str, agent_answer: str, reference_answer: str) -> str:
    """
    Uses a foundational model (LLM-as-a-Judge) to evaluate an agent's response.

    This function constructs a detailed prompt asking an LLM to act as an expert
    evaluator and assess the agent's answer based on correctness, completeness,
    and clarity, comparing it against a reference answer.

    Args:
        user_question: The original question from the user.
        agent_answer: The answer provided by the agent.
        reference_answer: The ground-truth answer.

    Returns:
        A concise, paragraph-long summary of the evaluation.
    """
    # Construct the prompt for the evaluation model.
    eval_prompt = (
        f"You are an expert evaluator.\n"
        f"User asked: \"{user_question}\"\n"
        f"Agent replied: \"{agent_answer}\"\n"
        f"Reference answer: \"{reference_answer}\"\n\n"
        "Rate the agent reply on correctness, completeness, and clarity, "
        "then summarize your feedback in one concise paragraph."
    )
    
    # Prepare the payload for the Bedrock API.
    payload = {
        "inputText": eval_prompt,
        "textGenerationConfig": {
            "temperature": 0.0,  # Low temperature for deterministic, factual evaluation.
            "topP": 1.0,
            "maxTokenCount": 512,
            "stopSequences": []
        }
    }
    # Invoke the evaluation model.
    resp = bedrock_client.invoke_model(
        modelId=EVAL_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8")
    )
    # Parse and return the model's generated text.
    return json.loads(resp["body"].read())["results"][0]["outputText"]

# A predefined set of questions and their "golden" or reference answers.
# This is used for evaluating agent responses in the full orchestration pipeline.
reference_answers = {
    "hello": "Hello! Welcome to our healthcare information service. How may I assist you today?...",
    "is mounjaro yellow in colour?": "Mounjaro is described as a clear, colorless to slightly yellow solution.",
    "what are the side effects of trulicity?": "Common side effects of Trulicity include nausea, diarrhea, vomiting, abdominal pain, decreased appetite, indigestion, and fatigue.",
    "are there any open tickets related to CVS?": "There is one open ticket related to CVS in the search results.",
    "Can i take mounjaro if i have fever ?": "It is not recommended to take Mounjaro if you have a fever. You should consult your healthcare provider for personalized advice.",
}

# Pre-compute and cache embeddings for all reference questions for faster lookup.
ref_embeddings = {q: get_embedding(q) for q in reference_answers.keys()}

def find_best_reference(user_question: str, threshold: float = 0.60):
    """
    Finds the most semantically similar reference question from the predefined set.

    Args:
        user_question: The user's input query.
        threshold: The minimum similarity score required for a match.

    Returns:
        A tuple containing the best matching question, its answer, and the similarity score.
        Returns (None, None, score) if no match exceeds the threshold.
    """
    emb = get_embedding(user_question)
    best_q, best_sim = None, -1.0
    
    # Iterate through pre-computed reference embeddings to find the best match.
    for q, q_emb in ref_embeddings.items():
        # Calculate cosine similarity.
        dot = sum(a*b for a,b in zip(emb, q_emb))
        norm1 = math.sqrt(sum(a*a for a in emb))
        norm2 = math.sqrt(sum(b*b for b in q_emb))
        sim = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        if sim > best_sim:
            best_q, best_sim = q, sim
    
    # Return the best match only if it meets the similarity threshold.
    if best_sim >= threshold:
        return best_q, reference_answers[best_q], best_sim
    return None, None, best_sim

def compute_relevance(candidate: str, user_question: str) -> float:
    """
    Calculates a relevance score by measuring the semantic similarity between
    the agent's response and the original user question.

    Args:
        candidate: The agent's generated response.
        user_question: The original user question.

    Returns:
        A relevance score between 0 and 100.
    """
    # This reuses the compute_confidence function, as the underlying math is the same.
    return compute_confidence(candidate, user_question)

def handle_user_query(user_question: str):
    """
    The main orchestration logic for processing a user's query.

    This function first checks for keywords to perform pre-routing. If no keywords
    are found, it proceeds with the full orchestration pipeline:
    1. Invokes all agents in parallel.
    2. Finds a matching reference answer.
    3. If a reference is found, it scores all responses on confidence and relevance.
    4. Selects the best agent and uses an LLM to perform a final evaluation.
    5. Returns a structured dictionary with all results.

    Args:
        user_question: The query submitted by the user.

    Returns:
        A dictionary containing the results of the orchestration. The structure
        varies depending on whether pre-routing or full orchestration was used.
    """
    # --- Pre-routing Logic ---
    # If the query contains any ticket-related keywords, route directly to the ServiceNow agent.
    if any(keyword in user_question.lower() for keyword in TICKET_KEYWORDS):
        st.info(f"🎯 Ticket-related query detected. Routing directly to {SERVICENOW_AGENT_NAME}...")
        servicenow_agent = next((agent for agent in AGENTS.values() if agent.name == SERVICENOW_AGENT_NAME), None)
        
        if servicenow_agent:
            try:
                response = invoke_agent(servicenow_agent, user_question)
                return {
                    'pre_routed': True,
                    'agent': SERVICENOW_AGENT_NAME,
                    'response': response,
                    'query': user_question
                }
            except Exception as e:
                st.error(f"An error occurred while invoking {SERVICENOW_AGENT_NAME}: {e}")
        else:
            st.error(f"Error: The agent named '{SERVICENOW_AGENT_NAME}' was not found in the configuration.")
        return None
    
    # --- Full Orchestration Logic ---
    st.info("🚀 Querying all agents...")
    responses = {}
    # Use a ThreadPoolExecutor to invoke all agents concurrently for maximum speed.
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as executor:
        futures = {executor.submit(invoke_agent, ag, user_question): ag for ag in AGENTS.values()}
        for future in as_completed(futures):
            agent = futures[future]
            try:
                result = future.result()
                if result:
                    responses[agent.name] = result
            except Exception as e:
                st.warning(f"[{agent.name}] failed to generate a response: {e}")

    if not responses:
        st.error("All agents failed to provide a response.")
        return None

    # Find the best reference answer for the user's question.
    ref_q, ref_ans, sim = find_best_reference(user_question)
    
    # If a suitable reference is found, perform a full evaluation.
    if ref_ans:
        st.success(f"📚 Reference found: '{ref_q}' (Similarity: {sim:.2f}). Performing full evaluation.")
        
        # Calculate confidence and relevance scores for each agent's response.
        confidence_scores = {ag: compute_confidence(txt, ref_ans) for ag, txt in responses.items()}
        relevance_scores = {ag: compute_relevance(txt, user_question) for ag, txt in responses.items()}
        
        # Calculate a weighted combined score. Relevance is weighted more heavily.
        combined_scores = {
            ag: (0.4 * confidence_scores.get(ag, 0)) + (0.6 * relevance_scores.get(ag, 0))
            for ag in responses
        }
        
        # Identify the agent with the highest combined score.
        best_agent = max(combined_scores, key=combined_scores.get)
        best_text = responses[best_agent]
        
        # Perform the final LLM-as-a-Judge evaluation on the best response.
        evaluation = evaluate_response_llm(user_question, best_text, ref_ans)
        
        return {
            'pre_routed': False,
            'responses': responses,
            'reference': {'question': ref_q, 'answer': ref_ans, 'similarity': sim},
            'confidence_scores': confidence_scores,
            'relevance_scores': relevance_scores,
            'combined_scores': combined_scores,
            'best_agent': best_agent,
            'best_response': best_text,
            'evaluation': evaluation,
            'query': user_question
        }
    else:
        # If no reference is found, skip scoring and evaluation.
        st.warning(f"⚠️ No close reference match found (best similarity: {sim:.2f}).")
        st.info("Skipping confidence scoring and LLM evaluation. Displaying all agent responses:")
        
        return {
            'pre_routed': False,
            'responses': responses,
            'reference': None,
            'query': user_question
        }


# --- 4. UI AND STYLING ---

# Eli Lilly corporate color scheme for consistent branding.
LILLY_COLORS = {
    'primary_red': '#D2232A',
    'dark_red': '#B91C1C',
    'light_red': '#FEE2E2',
    'gradient_start': '#D2232A',
    'gradient_end': '#991B1B',
    'accent_gray': '#6B7280',
    'success_green': '#059669',
    'warning_amber': '#D97706'
}

# Custom CSS for styling the Streamlit components with the Eli Lilly brand theme.
st.markdown(f"""
<style>
    /* Main header style */
    .main-header {{
        background: linear-gradient(135deg, {LILLY_COLORS['gradient_start']} 0%, {LILLY_COLORS['gradient_end']} 100%);
        padding: 2rem; border-radius: 20px; color: white; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(210, 35, 42, 0.3);
    }}
    /* Card style for displaying information */
    .lilly-card {{
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%);
        padding: 1.5rem; border-radius: 15px; border-left: 5px solid {LILLY_COLORS['primary_red']};
        margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.2s ease;
    }}
    .lilly-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(210, 35, 42, 0.15); }}
    /* Special card for highlighting the best agent response */
    .best-agent-card {{
        background: linear-gradient(135deg, {LILLY_COLORS['primary_red']} 0%, {LILLY_COLORS['dark_red']} 100%);
        color: white; padding: 2rem; border-radius: 20px; margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(210, 35, 42, 0.4); animation: pulse-red 2s infinite alternate;
    }}
    @keyframes pulse-red {{
        from {{ box-shadow: 0 8px 30px rgba(210, 35, 42, 0.4); }}
        to {{ box-shadow: 0 12px 40px rgba(210, 35, 42, 0.6); }}
    }}
    /* Style for orchestration pipeline steps */
    .orchestration-step {{
        background: linear-gradient(90deg, {LILLY_COLORS['light_red']} 0%, #ffffff 100%);
        border-left: 4px solid {LILLY_COLORS['primary_red']}; padding: 1.2rem;
        margin: 0.8rem 0; border-radius: 10px; animation: slideInFromLeft 0.6s ease-out;
    }}
    @keyframes slideInFromLeft {{
        from {{ opacity: 0; transform: translateX(-30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    /* Alert style for pre-routing notifications */
    .pre-routing-alert {{
        background: linear-gradient(45deg, #FEF3C7, #FDE68A);
        border: 2px solid {LILLY_COLORS['warning_amber']}; border-radius: 15px;
        padding: 1.5rem; margin: 1rem 0; animation: highlightPulse 1.5s ease-in-out;
    }}
    @keyframes highlightPulse {{
        0%, 100% {{ background: linear-gradient(45deg, #FEF3C7, #FDE68A); }}
        50% {{ background: linear-gradient(45deg, #FCD34D, #F59E0B); }}
    }}
    /* Custom button style */
    .stButton > button {{
        background: linear-gradient(45deg, {LILLY_COLORS['primary_red']}, {LILLY_COLORS['dark_red']});
        color: white; border-radius: 25px; border: none; padding: 0.8rem 2.5rem;
        font-weight: 600; font-size: 1rem; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(210, 35, 42, 0.3);
    }}
    .stButton > button:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px rgba(210, 35, 42, 0.4); }}
    /* Floating brand logo */
    .lilly-brand {{
        position: fixed; top: 15px; right: 20px; background: {LILLY_COLORS['primary_red']};
        color: white; padding: 0.6rem 1.2rem; border-radius: 25px;
        font-weight: bold; font-size: 0.9rem; z-index: 1000;
        box-shadow: 0 4px 15px rgba(210, 35, 42, 0.3);
    }}
</style>
""", unsafe_allow_html=True)

# --- Streamlit Session State Initialization ---
# Session state is used to persist data across user interactions and reruns.
# query_history: Stores the results of all processed queries in the current session.
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
# processing: A flag to prevent multiple queries from running simultaneously.
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Display the floating brand logo in the top-right corner.
st.markdown("""
<div class="lilly-brand">
    🧬 Eli Lilly PoC
</div>
""", unsafe_allow_html=True)


# --- 5. STREAMLIT UI LAYOUT AND LOGIC ---

def main():
    """
    The main function that defines the overall structure of the Streamlit app.
    It sets up the header and the main tabbed interface.
    """
    # Main application header.
    st.markdown(f"""
    <div class="main-header">
        <h1>🧬 Eli Lilly BoB Platform</h1>
        <h3>Bot-of-Bots Intelligent Orchestration System</h3>
        <p>Advanced Multi-Agent Healthcare Knowledge Routing & Response Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create the tabbed navigation for the application.
    tabs = st.tabs([
        "🚀 Query Orchestration",
        "📊 Performance Analytics",
        "🧠 Agent Intelligence",
        "📈 Historical Insights"
    ])
    
    # Populate each tab with its respective content function.
    with tabs[0]:
        orchestration_interface()
    
    with tabs[1]:
        # Only show analytics if there is data to display.
        if st.session_state.query_history:
            performance_analytics()
        else:
            st.info("💡 Process some queries first to see performance analytics")
    
    with tabs[2]:
        agent_intelligence_dashboard()
    
    with tabs[3]:
        # Only show history if there is data to display.
        if st.session_state.query_history:
            historical_insights()
        else:
            st.info("📊 Query history will appear here after processing requests")

def orchestration_interface():
    """Renders the main query input interface on the first tab."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 style="color: #D2232A;">💬 Healthcare Query Interface</h3>', unsafe_allow_html=True)
        
        # Pre-defined sample queries to guide the user.
        sample_queries = [
            "What are the side effects of Mounjaro?",
            "How does Trulicity compare to other medications?",
            "I have a ticket about patient enrollment issues",
            "What is the recommended dosing for Zepbound?",
            "Are there any open tickets related to clinical trials?"
        ]
        
        selected_query = st.selectbox(
            "🔮 Try these sample queries:",
            [""] + sample_queries,
            index=0
        )
        
        user_query = st.text_area(
            "Enter your healthcare query:",
            value=selected_query,
            placeholder="Ask about Eli Lilly medications, clinical data, or support tickets...",
            height=120
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            # The main button to trigger the query processing.
            if st.button("🚀 Process Query", disabled=st.session_state.processing):
                if user_query.strip():
                    process_query_with_ui(user_query.strip())
                else:
                    st.warning("⚠️ Please enter a query first")
        
        with col_btn2:
            # Button to select a random sample query.
            if st.button("🎲 Random Query"):
                random_query = np.random.choice(sample_queries)
                # Using st.rerun() is a clean way to update the state and UI.
                st.rerun() 
        
        with col_btn3:
            # Button to clear all session data.
            if st.button("🧹 Clear Session"):
                st.session_state.query_history = []
                st.session_state.processing = False
                st.success("✅ Session cleared")
    
    with col2:
        # A sidebar-like column for live system status metrics.
        st.markdown('<h3 style="color: #D2232A;">📈 Live System Status</h3>', unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("🤖 Active Agents", len(AGENTS))
        with col_m2:
            st.metric("📋 Processed", len(st.session_state.query_history))
        
        # Display the winning agent from the last query.
        if st.session_state.query_history:
            latest = st.session_state.query_history[-1]
            st.markdown(f"""
            <div class="lilly-card">
                <h4>🏆 Last Winner</h4>
                <p><strong>{latest.get('best_agent', 'N/A')}</strong></p>
                <p>Type: {'Pre-routed' if latest.get('pre_routed') else 'Full Orchestration'}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🎯 Ready to process your first query!")
    
    # After the main interface, display the results of the latest query.
    if st.session_state.query_history:
        display_latest_results()

def process_query_with_ui(query: str):
    """
    Manages the UI during query processing to provide a better user experience.
    It shows a progress bar and status updates while the backend handle_user_query runs.
    
    Args:
        query: The user query to process.
    """
    st.session_state.processing = True # Lock the UI to prevent another submission.
    
    # Create containers for the progress bar and step-by-step updates.
    progress_container = st.container()
    progress_bar = st.progress(0)
    
    with progress_container:
        st.markdown("### 🔄 BoB Orchestration Pipeline")
        step1, step2, step3, step4, step5 = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()
    
    try:
        start_time = time.time()
        
        # --- UI Updates for each pipeline step ---
        step1.markdown('<div class="orchestration-step">...</div>', unsafe_allow_html=True)
        progress_bar.progress(20)
        
        is_ticket_query = any(keyword in query.lower() for keyword in TICKET_KEYWORDS)
        
        # Dynamically show different messages based on routing decision.
        if is_ticket_query:
            step2.markdown('<div class="pre-routing-alert">...</div>', unsafe_allow_html=True)
        else:
            step2.markdown('<div class="orchestration-step">...</div>', unsafe_allow_html=True)
        progress_bar.progress(40)
        
        # ... (further UI updates for steps 3, 4, 5) ...

        # --- Call the actual backend processing function ---
        result = handle_user_query(query)
        
        processing_time = time.time() - start_time
        progress_bar.progress(100)
        
        # Store results in session state if the query was successful.
        if result:
            result['processing_time'] = processing_time
            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.query_history.append(result)
        
        # Clean up the progress display.
        time.sleep(1) # Brief pause to let the user see the "100%".
        progress_container.empty()
        progress_bar.empty()
        
        st.success(f"✅ Query orchestrated successfully in {processing_time:.2f} seconds!")
        
    except Exception as e:
        st.error(f"❌ Orchestration error: {str(e)}")
        progress_container.empty()
        progress_bar.empty()
    
    finally:
        # IMPORTANT: Always unlock the UI, even if an error occurred.
        st.session_state.processing = False

def display_latest_results():
    """Renders the detailed results of the most recent query."""
    if not st.session_state.query_history:
        return
    
    result = st.session_state.query_history[-1]
    
    st.markdown("---")
    st.markdown('<h3 style="color: #D2232A;">📋 Latest Orchestration Results</h3>', unsafe_allow_html=True)
    
    # Display summary of the query.
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: st.markdown(f"**🔍 Query:** {result['query']}")
    with col2: st.markdown(f"**⏰ Processed:** {result.get('timestamp', 'N/A')}")
    with col3: st.markdown(f"**⚡ Time:** {result.get('processing_time', 0):.2f}s")
    
    # --- Display for Pre-routed Query ---
    if result.get('pre_routed'):
        st.markdown(f"""
        <div class="pre-routing-alert">
            ... (HTML for pre-routing result) ...
        </div>
        """, unsafe_allow_html=True)
        return # Stop here for pre-routed queries.
    
    # --- Display for Full Orchestration ---
    responses = result.get('responses', {})
    if not responses:
        st.warning("No agent responses available")
        return
    
    # Display reference match information.
    ref_info = result.get('reference')
    if ref_info:
        st.markdown(f"""<div class="lilly-card">...</div>""", unsafe_allow_html=True)
    
    # Use tabs to show each agent's response individually.
    agent_names = list(responses.keys())
    if agent_names:
        tabs = st.tabs([f"🤖 {name}" for name in agent_names])
        
        # Get all scoring data from the result dictionary.
        confidence_scores = result.get('confidence_scores', {})
        relevance_scores = result.get('relevance_scores', {})
        combined_scores = result.get('combined_scores', {})
        best_agent = result.get('best_agent')
        
        for i, agent_name in enumerate(agent_names):
            with tabs[i]:
                # Display scores for this agent.
                if confidence_scores and relevance_scores:
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("🎯 Confidence", f"{confidence_scores.get(agent_name, 0):.1f}")
                    with col2: st.metric("🔗 Relevance", f"{relevance_scores.get(agent_name, 0):.1f}")
                    with col3: st.metric("⭐ Combined", f"{combined_scores.get(agent_name, 0):.1f}")
                
                # Display the agent's text response.
                st.markdown(f"""<div class="lilly-card">{responses[agent_name]}</div>""", unsafe_allow_html=True)
                
                # Highlight the winning agent with a special card.
                if agent_name == best_agent:
                    st.markdown("""<div class="best-agent-card">...</div>""", unsafe_allow_html=True)
    
    # Display the final LLM-as-a-Judge evaluation.
    evaluation = result.get('evaluation')
    if evaluation:
        st.markdown("### 🧠 LLM Evaluation")
        st.markdown(f"""<div class="lilly-card">{evaluation}</div>""", unsafe_allow_html=True)

def performance_analytics():
    """Renders the Performance Analytics tab with charts and metrics."""
    st.markdown('<h3 style="color: #D2232A;">📊 Advanced Performance Analytics</h3>', unsafe_allow_html=True)
    if not st.session_state.query_history: return
    
    # Calculate summary metrics from the query history.
    total_queries = len(st.session_state.query_history)
    avg_time = np.mean([q.get('processing_time', 0) for q in st.session_state.query_history])
    pre_routed_count = sum(1 for q in st.session_state.query_history if q.get('pre_routed'))
    pre_route_rate = (pre_routed_count / total_queries) * 100 if total_queries > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📈 Total Queries", total_queries)
    with col2: st.metric("⚡ Avg Response", f"{avg_time:.2f}s")
    with col3: st.metric("🎯 Pre-route Rate", f"{pre_route_rate:.1f}%")
    with col4: st.metric("🚀 Efficiency", "95.5%") # Mock score for demonstration.
    
    # Plot a line chart of processing time over the session.
    if len(st.session_state.query_history) > 1:
        times = [q.get('processing_time', 2.5) for q in st.session_state.query_history]
        queries = list(range(1, len(times) + 1))
        fig = px.line(x=queries, y=times, title='⚡ Processing Time Evolution', markers=True, line_shape='spline')
        fig.update_traces(line_color=LILLY_COLORS['primary_red'])
        fig.update_layout(xaxis_title='Query Number', yaxis_title='Processing Time (s)')
        st.plotly_chart(fig, use_container_width=True)

def agent_intelligence_dashboard():
    """Renders the Agent Intelligence tab with agent-specific information."""
    st.markdown('<h3 style="color: #D2232A;">🧠 Agent Intelligence Overview</h3>', unsafe_allow_html=True)
    
    # Mock data for agent capabilities (could be fetched dynamically).
    agent_info = {
        "zepboundinfoagent": {"specialty": "Zepbound Clinical Data", "expertise": 94},
        "mounjaro_specialist_agent": {"specialty": "Mounjaro Therapeutics", "expertise": 96},
        "trulicity_agent": {"specialty": "Trulicity Treatment", "expertise": 92},
        "serviceNowtickets_agent": {"specialty": "IT Service Management", "expertise": 98}
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Agent Specializations")
        for agent_name, info in agent_info.items():
            st.markdown(f"""<div class="lilly-card">...</div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚡ Pre-routing Intelligence")
        st.markdown(f"""<div class="lilly-card">...</div>""", unsafe_allow_html=True)
        
        # Create a radar chart to visualize agent expertise levels.
        agents = list(agent_info.keys())
        scores = [agent_info[agent]["expertise"] for agent in agents]
        fig = go.Figure(go.Scatterpolar(r=scores, theta=[agent_info[agent]["specialty"] for agent in agents], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="🧠 Agent Expertise Analysis")
        st.plotly_chart(fig, use_container_width=True)

def historical_insights():
    """Renders the Historical Insights tab with aggregate data visualizations."""
    st.markdown('<h3 style="color: #D2232A;">📈 Historical Intelligence Insights</h3>', unsafe_allow_html=True)
    if len(st.session_state.query_history) < 2:
        st.info("💡 Process more queries to unlock advanced historical analytics")
        return
    
    # Calculate routing distribution.
    pre_routed = sum(1 for q in st.session_state.query_history if q.get('pre_routed'))
    full_orchestration = len(st.session_state.query_history) - pre_routed
    
    col1, col2 = st.columns(2)
    with col1:
        # Pie chart for routing distribution.
        fig = px.pie(values=[pre_routed, full_orchestration], names=['Pre-routed', 'Full Orchestration'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot with a trendline for performance.
        times = [q.get('processing_time', 2.5) for q in st.session_state.query_history]
        queries = list(range(1, len(times) + 1))
        fig = px.scatter(x=queries, y=times, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    # Display auto-generated insights.
    st.markdown("### 🧠 AI-Powered Insights")
    insights = [f"🎯 **Routing Efficiency**: ...", f"⚡ **Performance**: ..."]
    for insight in insights:
        st.markdown(f"""<div class="lilly-card">{insight}</div>""", unsafe_allow_html=True)


# --- 6. APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    # This block ensures that the main() function is called only when the script
    # is executed directly (not when imported as a module).
    main()