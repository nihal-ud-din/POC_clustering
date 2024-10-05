import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import read_pdf, split_document, topic_modeling, classify_and_cluster, create_wordcloud

import nltk
nltk.download('punkt')
import logging

# Set Streamlit page configuration to wide
st.set_page_config(layout="wide")

# Using HTML and CSS to style and center the title
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        width: 80%;
        margin: 5px auto; /* Center the div horizontally */
        padding: 20px 0;
    }
    </style>
    <div class="title">Document Clustering and Topic Modeling with Word Clouds</div>
    """,
    unsafe_allow_html=True
)

# File uploader for PDF
uploaded_files = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=True)







def process_uploaded_files(uploaded_files):
    try:
        all_doc_content_list = []
        for uploaded_file in uploaded_files:
            doc_content = read_pdf(uploaded_file)
            doc_content_list = split_document(doc_content)
            all_doc_content_list.extend(doc_content_list)
        return all_doc_content_list
    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        st.error(f"Error processing files: {str(e)}")
        return None



def perform_clustering(all_doc_content_list, n_clusters):
    try:
        tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(all_doc_content_list, n_clusters)
        return tfidf_matrix, clusters, topics, vectorizer
    except Exception as e:
        logging.error(f"Error clustering documents: {str(e)}")
        st.error(f"Error clustering documents: {str(e)}")
        return None, None, None, None
    
    
    
    

if uploaded_files is not None and len(uploaded_files) > 0:
    all_doc_content_list = process_uploaded_files(uploaded_files)
    
    if all_doc_content_list is not None:
        if len(all_doc_content_list) > 1:
            # Slider to select number of clusters
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=min(10, len(all_doc_content_list)), value=5)
            
            # Validate user input (number of clusters)
            if n_clusters < 2 or n_clusters > len(all_doc_content_list):
                st.error("Invalid number of clusters. Please select a value between 2 and " + str(len(all_doc_content_list)))
            else:
                # Apply KMeans clustering and topic modeling
                tfidf_matrix, clusters, topics, vectorizer = perform_clustering(all_doc_content_list, n_clusters)
                
                if tfidf_matrix is not None:
                    st.header("Clusters and Topics")
                    for i, cluster_topics in enumerate(topics):
                        st.write(f"Cluster {i + 1}:")
                        cluster_topics_str = [str(topic) for topic in cluster_topics]
                        topics_str = ", ".join(cluster_topics_str)
                        st.write(topics_str)
                    
                    st.header("Cluster Word Clouds")
                    create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)
        else:
            st.write("Document is too small to cluster.")
else:
    st.write("No files uploaded.")










# if uploaded_files is not None and len(uploaded_files) > 0:
#     all_doc_content_list = []

#     # Read content from each PDF and split into sentences/paragraphs
#     for uploaded_file in uploaded_files:
#         doc_content = read_pdf(uploaded_file)
#         doc_content_list = split_document(doc_content)
#         all_doc_content_list.extend(doc_content_list)  # Combine all documents into one list


#     if len(all_doc_content_list) > 1:
#         # Slider to select number of clusters
#         n_clusters = st.slider("Select number of clusters", min_value=2, max_value=min(10, len(all_doc_content_list)),
#                                value=5)

#         # Apply KMeans clustering and topic modeling
#         tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(all_doc_content_list, n_clusters)

#         st.header("Clusters and Topics")
#         # Display cluster and topic information
#         for i, cluster_topics in enumerate(topics):
#             st.write(f"Cluster {i + 1}:")
#             # Convert each topic to a string if it is not already
#             cluster_topics_str = [str(topic) for topic in cluster_topics]
#             # Join the topics into a single line
#             topics_str = ", ".join(cluster_topics_str)
#             st.write(topics_str)  # Print the topics as a single line


#         st.header("Cluster Word Clouds")
#         # Create and display word clouds for each cluster
#         create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)
#     else:
#         st.write("Document is too small to cluster.")
