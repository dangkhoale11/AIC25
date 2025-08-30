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
if 'pivot_frame' not in st.session_state:
    st.session_state.pivot_frame = None
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

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("📊 Max Results", 1, 200, 10)
with col2:
    score_threshold = st.slider("🎯 Min Score", 0.0, 1.0, 0.0, 0.1)

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
    with col_vid:
        include_videos_input = st.text_input("Video IDs to include", placeholder="e.g., 101, 203")
        include_videos = [int(x.strip()) for x in include_videos_input.split(',') if x.strip()] if include_videos_input else []

# --- Reranking Options ---
st.markdown("---")
use_rerank = st.toggle("✨ Enable GEM Reranking")

if use_rerank:
    st.markdown("#### 💎 GEM Reranking Parameters")
    sim_metric = st.selectbox("Similarity Metric", ["cosine", "dot", "euclid"])

    st.markdown("##### Refine Query (g_qe)")
    p_qe = st.slider("p_qe (Query Expansion Power)", 0.0, 150.0, 3.0, 0.5)

    st.markdown("##### Refine Frame (g_dr)")
    p_dr = st.slider("p_dr (Document Refinement Power)", 0.0, 150.0, 3.0, 0.5)
    m_neighbors = st.slider("m_neighbors (Number of Neighbors)", 0, 20, 5)

# --- Temporal Search Options ---
st.markdown("---")
use_temporal = st.toggle("🕰️ Enable Temporal Search")

if use_temporal:
    st.markdown("#### ⏳ Temporal Search Parameters")
    start_query_temporal = st.text_input("Start of Event Query")
    end_query_temporal = st.text_input("End of Event Query")
    temporal_tolerance = st.slider("Temporal Tolerance", 0, 150, 3)

# --- Search Button ---
st.markdown("---")
if st.button("🚀 Search", use_container_width=True):
    if not query.strip():
        st.error("Please enter a search query")
    else:
        with st.spinner("🔍 Searching..."):
            try:
                # Determine endpoint and payload based on mode
                if use_rerank:
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/rerank"
                    payload = {
                        "query": query,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "rerank_type": "GEM",
                        "p_qe": p_qe,
                        "p_dr": p_dr,
                        "m_neighbors": m_neighbors,
                        "sim_metric": sim_metric,
                    }
                elif search_mode == "Normal Search":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                    payload = {"query": query, "top_k": top_k, "score_threshold": score_threshold}
                elif search_mode == "Search with Exclude Group":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                    payload = {"query": query, "top_k": top_k, "score_threshold": score_threshold, "exclude_groups": exclude_groups}
                elif search_mode == "Search with Group and Video":
                    endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                    payload = {"query": query, "top_k": top_k, "score_threshold": score_threshold, "include_groups": include_groups, "include_videos": include_videos}

                response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.search_results = data.get("results", [])
                    st.session_state.raw_search_results = data.get("raw_results", [])
                    st.success(f"✅ Found {len(st.session_state.search_results)} results!")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")

# --- Display Results ---
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")

    # CSV export for normal search
    if st.session_state.search_results:
        # Sort results by score for consistent output
        sorted_results_for_csv = sorted(st.session_state.search_results, key=lambda x: x['score'], reverse=True)

        csv_data = "video_file_name,Frame Idx\n"
        for result in sorted_results_for_csv:
            try:
                path_parts = result['path'].replace('\\', '/').split('/')
                video_file_name = f"{path_parts[-3]}/{path_parts[-2]}"
                frame_idx = path_parts[-1].split('.')[0]
                csv_data += f"{video_file_name},{frame_idx}\n"
            except IndexError:
                # Handle cases where path format is unexpected
                csv_data += "unknown,unknown\n"

        st.download_button(
           label="Create File Submission (Normal Search)",
           data=csv_data,
           file_name="normal_search_submission.csv",
           mime="text/csv",
        )

    sorted_results = sorted(st.session_state.search_results, key=lambda x: x['score'], reverse=True)

    for i, result in enumerate(sorted_results):
        with st.container():
            col_img, col_info = st.columns([1, 3])

            with col_img:
                image_displayed = safe_image_display(result['path'], width=200, caption=f"Result {i+1}")
                if not image_displayed:
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

                if use_temporal:
                    if st.button(f"Select as Pivot", key=f"pivot_{result.get('key', i)}"):
                        raw_result = next((r for r in st.session_state.raw_search_results if r.get('key') == result.get('key')), None)
                        if raw_result:
                            st.session_state.pivot_frame = raw_result
                            # Clear previous temporal results when a new pivot is selected
                            st.session_state.temporal_results = None
                            st.rerun()
                        else:
                            st.error("Could not find the raw result for the selected pivot.")
        st.markdown("<br>", unsafe_allow_html=True)

# --- Temporal Search Execution ---
if use_temporal and st.session_state.pivot_frame:
    st.markdown("---")
    st.markdown("### 🕰️ Temporal Search from Pivot")

    st.write("Pivot Frame:")
    st.json(st.session_state.pivot_frame)

    if st.button("Search Temporal Event"):
        if not start_query_temporal or not end_query_temporal:
            st.error("Please enter both a start and end query for temporal search.")
        else:
            with st.spinner("Performing temporal search..."):
                endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/temporal"
                payload = {
                    "start_query": start_query_temporal,
                    "end_query": end_query_temporal,
                    "pivot_frame": st.session_state.pivot_frame,
                    "temporal_tolerance": temporal_tolerance,
                }
                response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30)

                if response.status_code == 200:
                    st.session_state.temporal_results = response.json()
                    st.success("Temporal search complete!")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

# --- Display Temporal Results ---
if st.session_state.temporal_results and st.session_state.temporal_results.get("start_frame") and st.session_state.temporal_results.get("end_frame"):
    st.markdown("---")
    st.markdown("###  Temporal Search Results")

    start_frame = st.session_state.temporal_results["start_frame"]
    end_frame = st.session_state.temporal_results["end_frame"]

    # CSV export for temporal search
    try:
        start_path_parts = start_frame['path'].replace('\\', '/').split('/')
        end_path_parts = end_frame['path'].replace('\\', '/').split('/')

        video_file_name = f"{start_path_parts[-3]}/{start_path_parts[-2]}"
        start_idx = start_path_parts[-1].split('.')[0]
        end_idx = end_path_parts[-1].split('.')[0]

        temporal_search_csv = f"video_file_name,start Idx,end Idx\n{video_file_name},{start_idx},{end_idx}\n"

        st.download_button(
           label="Create File Submission (Temporal Search)",
           data=temporal_search_csv,
           file_name="temporal_search_submission.csv",
           mime="text/csv",
        )
    except (IndexError, KeyError):
        st.warning("Could not generate temporal search CSV due to unexpected path format.")

    # Display frames
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Start Frame")
        if not safe_image_display(start_frame["path"]):
            display_image_placeholder("Start Frame")
        st.markdown(f"<p style='margin: 0.5rem 0; color: #666;'><strong>Path:</strong> {start_frame['path']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0.5rem 0; color: #666;'><strong>Score:</strong> {start_frame['score']:.3f}</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### End Frame")
        if not safe_image_display(end_frame["path"]):
            display_image_placeholder("End Frame")
        st.markdown(f"<p style='margin: 0.5rem 0; color: #666;'><strong>Path:</strong> {end_frame['path']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0.5rem 0; color: #666;'><strong>Score:</strong> {end_frame['score']:.3f}</p>", unsafe_allow_html=True)

    # Pivot selection slider
    try:
        start_frame_num = int(start_idx)
        end_frame_num = int(end_idx)
        if start_frame_num < end_frame_num:
            pivot_range = st.slider(
                "Select Pivot Frame in Range",
                min_value=start_frame_num,
                max_value=end_frame_num,
                value=start_frame_num
            )
            st.info(f"Selected Pivot Frame for review: **{pivot_range}**")
    except (ValueError, IndexError):
        st.warning("Could not determine frame range for pivot selection.")

elif st.session_state.temporal_results:
    st.markdown("### Temporal Search Results")
    st.warning("Temporal search did not return a valid start and end frame.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)