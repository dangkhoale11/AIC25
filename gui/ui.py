import streamlit as st
import requests
import json
from typing import List, Optional
import pandas as pd
import os
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Keyframe Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .mode-selector {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .image-placeholder {
        background: #f0f0f0; 
        height: 150px; 
        border-radius: 10px; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        border: 2px dashed #ccc;
        text-align: center; 
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def safe_image_display(image_path: str, width: int = 200, caption: str = ""):
    """
    Safely display an image with fallback options
    """
    try:
        # Method 1: Try direct path if it's a valid file
        if os.path.exists(image_path):
            st.image(image_path, width=width, caption=caption)
            return True
        
        # Method 2: Try to load from URL if it's a URL
        elif image_path.startswith(('http://', 'https://')):
            st.image(image_path, width=width, caption=caption)
            return True
        
        # Method 3: Try to get image from API
        elif hasattr(st.session_state, 'api_base_url'):
            try:
                # Assume there's an endpoint to get the image
                image_url = f"{st.session_state.api_base_url}/api/v1/keyframe/image?path={image_path}"
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, width=width, caption=caption)
                    return True
            except:
                pass
        
        return False
    except Exception as e:
        return False

def display_image_placeholder(caption: str = ""):
    """
    Display a placeholder when image can't be loaded
    """
    st.markdown(f"""
    <div class="image-placeholder">
        <div>
            🖼️<br>Image Preview<br>Not Available<br>
            <small style="font-size: 0.8em;">{caption}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'raw_search_results' not in st.session_state:
    st.session_state.raw_search_results = []
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://127.0.0.1:8000"
if 'temporal_results' not in st.session_state:
    st.session_state.temporal_results = None

# Header
st.markdown("""
<div class="search-container">
    <h1 style="margin: 0; font-size: 2.5rem;">🔍 Advanced Keyframe Search</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        A comprehensive UI for semantic, temporal, and reranked search
    </p>
</div>
""", unsafe_allow_html=True)

# API Configuration
with st.expander("⚙️ API Configuration", expanded=False):
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL for the keyframe search API"
    )
    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url
    
    image_display_method = st.selectbox(
        "Image Display Method",
        options=["Try to load from file path", "Show placeholder only", "Try to load from API endpoint", "Show path as text"],
        index=0
    )

# Main search interface
query = st.text_input(
    "🔍 Search Query",
    placeholder="Enter your search query (e.g., 'person walking in the park')",
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("📊 Max Results", 1, 500, 10) # Increased max for wider temporal search range
with col2:
    score_threshold = st.slider("🎯 Min Score", 0.0, 1.0, 0.0, 0.1)
with col3:
    batch = st.selectbox("Batch", [1, 2, 3], index=0)

# --- Search Modes ---
st.markdown("---")
st.markdown("### 🎛️ Search Modes")

search_mode = st.radio(
    "Select Search Mode",
    ["Normal Search", "Search with Exclude Group", "Search with Group and Video"],
    horizontal=True
)

if search_mode == "Search with Exclude Group":
    exclude_groups_input = st.text_input("Group IDs to exclude", placeholder="e.g., 1, 3, 7")
    exclude_groups = [int(x.strip()) for x in exclude_groups_input.split(',') if x.strip()] if exclude_groups_input else []

elif search_mode == "Search with Group and Video":
    col_grp, col_vid = st.columns(2)
    with col_grp:
        include_groups_input = st.text_input("Group IDs to include", placeholder="e.g., 2, 4")
        include_groups = [int(x.strip()) for x in include_groups_input.split(',') if x.strip()] if include_groups_input else []
        print(include_groups)
    with col_vid:
        include_videos_input = st.text_input("Video IDs to include", placeholder="e.g., 101, 203")
        include_videos = [int(x.strip()) for x in include_videos_input.split(',') if x.strip()] if include_videos_input else []


st.markdown("---")
st.markdown('### Clear Cache')
if st.button("🧹 Clear Cache", use_container_width=True):
    try:
        clear_endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/cache/clear"
        response = requests.post(clear_endpoint, timeout=10)
        if response.status_code == 200:
            st.success("✅ Cache cleared successfully!")
            st.session_state.search_results = []
            st.session_state.raw_search_results = []
            st.session_state.temporal_results = None
        else:
            st.error(f"❌ Failed to clear cache: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error clearing cache: {str(e)}")
# --- Reranking Options ---
st.markdown("---")
use_rerank = st.toggle("✨ Enable GEM Reranking")

if use_rerank:
    st.markdown("#### 💎 GEM Reranking Parameters")
    sim_metric = st.selectbox("Similarity Metric", ["cosine", "dot", "euclid"])
    p_qe = st.slider("p_qe (Query Expansion Power)", 0.0, 150.0, 3.0, 0.5)
    p_dr = st.slider("p_dr (Document Refinement Power)", 0.0, 150.0, 3.0, 0.5)
    m_neighbors = st.slider("m_neighbors (Number of Neighbors)", 0, 20, 5)

# --- Temporal Search Options ---
st.markdown("---")
use_temporal = st.toggle("🕰️ Enable Temporal Search")

if use_temporal:
    st.markdown("#### ⏳ Temporal Search Parameters")
    start_query_temporal = st.text_input("Start of Event Query", placeholder="e.g., person opens door")
    end_query_temporal = st.text_input("End of Event Query", placeholder="e.g., person closes door")
    temporal_search_range = st.slider("Results Range to Search", 0, top_k, (0, 20), help=f"Select the start and end index from the top {top_k} results to perform the temporal search on.")


# --- Search Button ---
st.markdown("---")
if st.button("🚀 Search", use_container_width=True):
    if not query.strip():
        st.error("Please enter a search query")
    elif use_temporal and (not start_query_temporal or not end_query_temporal):
        st.error("Please enter both a start and end query for temporal search.")
    else:
        with st.spinner("🔍 Searching..."):
            # Determine endpoint and base payload
            base_endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe"

            endpoint = f"{base_endpoint}/search?batch={batch}"
            payload = {"query": query, "top_k": top_k, "score_threshold": score_threshold}

            if search_mode == "Search with Exclude Group":
                endpoint = f"{base_endpoint}/search/exclude-groups?batch={batch}"
                payload["exclude_groups"] = exclude_groups
            elif search_mode == "Search with Group and Video":
                endpoint = f"{base_endpoint}/search/selected-groups-videos?batch={batch}"
                payload["include_groups"] = include_groups
                payload["include_videos"] = include_videos

            # Add rerank parameters if enabled
            if use_rerank:
                payload.update({
                    "use_rerank": True,
                    "rerank_type": "GEM",
                    "p_qe": p_qe,
                    "p_dr": p_dr,
                    "m_neighbors": m_neighbors,
                    "sim_metric": sim_metric,
                })

            # Add temporal search parameters if enabled
            if use_temporal:
                payload.update({
                    "use_temporal": True,
                    "temporal_start_query": start_query_temporal,
                    "temporal_end_query": end_query_temporal,
                    "temporal_search_range": temporal_search_range,
                })

            try:
                response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.search_results = []
                    st.session_state.raw_search_results = []
                    st.session_state.temporal_results = None

                    if 'events' in data:
                        st.session_state.temporal_results = data.get("events", [])
                        st.success(f"✅ Temporal search complete! Found {len(st.session_state.temporal_results)} events.")

                    elif 'rerank_type' in data:
                        st.session_state.search_results = data.get("results", [])
                        st.session_state.raw_search_results = data.get("raw_results", [])
                        rerank_type = data.get('rerank_type', 'N/A')
                        st.success(f"✅ Found and reranked {len(st.session_state.search_results)} results using {rerank_type} method!")

                    else:
                        st.session_state.search_results = data.get("results", [])
                        st.session_state.raw_search_results = data.get("raw_results", [])
                        st.success(f"✅ Found {len(st.session_state.search_results)} results!")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")

# --- Display Initial Results ---
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")

    # CSV export for normal search
    if st.session_state.search_results:
        sorted_results_for_csv = sorted(st.session_state.search_results, key=lambda x: x['score'], reverse=True)
        csv_data = "video_file_name,Frame Idx\n"
        for result in sorted_results_for_csv:
            try:
                path_parts = result['path'].replace('\\', '/').split('/')
                video_file_name = f"{path_parts[-3]}_{path_parts[-2]}"
                frame_idx = path_parts[-1].split('.')[0]
                csv_data += f"{video_file_name},{frame_idx}\n"
            except IndexError:
                csv_data += "unknown,unknown\n"
        st.download_button("Create File Submission (Normal Search)", csv_data, "normal_search_submission.csv", "text/csv")

    sorted_results = sorted(st.session_state.search_results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results):
        with st.container():
            col_img, col_info = st.columns([1, 3])
            with col_img:
                if not safe_image_display(result['path'], width=200, caption=f"Result {i+1}"):
                    display_image_placeholder(f"Result {i+1}")
            with col_info:
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #333;">Result #{i+1}</h4>
                        <span class="score-badge">Score: {result['score']:.3f}</span>
                    </div>
                    <p style="margin: 0.5rem 0; color: #666;"><strong>Path:</strong> {result['path']}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# --- Display Temporal Results ---
if st.session_state.temporal_results:
    st.markdown("---")
    st.markdown("### 🕰️ Temporal Search Results")

    events = st.session_state.temporal_results
    if events:
        # CSV export for temporal search
        try:
            temporal_csv_data = "video_file_name,start_Idx,end_Idx\n"
            for event in events:
                start_frame = event.get("start_frame")
                end_frame = event.get("end_frame")
                if start_frame and end_frame:
                    start_path_parts = start_frame['path'].replace('\\', '/').split('/')
                    end_path_parts = end_frame['path'].replace('\\', '/').split('/')
                    video_file_name = f"{start_path_parts[-3]}_{start_path_parts[-2]}"
                    start_idx = start_path_parts[-1].split('.')[0]
                    end_idx = end_path_parts[-1].split('.')[0]
                    temporal_csv_data += f"{video_file_name},{start_idx},{end_idx}\n"
            st.download_button("Create File Submission (Temporal Search)", temporal_csv_data, "temporal_search_submission.csv", "text/csv")
        except (IndexError, KeyError):
            st.warning("Could not generate temporal search CSV due to unexpected path format.")

        # Display frames
        for i, event in enumerate(events):
            st.markdown(f"#### Event #{i+1}")
            start_frame = event.get("start_frame")
            end_frame = event.get("end_frame")

            if start_frame and end_frame:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Start Frame")
                    if not safe_image_display(start_frame["path"]):
                        display_image_placeholder("Start Frame")
                    st.markdown(f"<p><strong>Path:</strong> {start_frame['path']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Score:</strong> {start_frame['score']:.3f}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown("##### End Frame")
                    if not safe_image_display(end_frame["path"]):
                        display_image_placeholder("End Frame")
                    st.markdown(f"<p><strong>Path:</strong> {end_frame['path']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Score:</strong> {end_frame['score']:.3f}</p>", unsafe_allow_html=True)
            else:
                st.warning(f"Event #{i+1} did not return a valid start and end frame.")
            st.markdown("---")
    else:
        st.warning("Temporal search did not return any valid events.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)