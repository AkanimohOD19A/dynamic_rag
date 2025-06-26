# Dynamic RAG System - Proof of Concept
# A scalable document retrieval system with external knowledge base

import streamlit as st
from helper.ragsystems import *


# Streamlit UI
st.set_page_config(
    page_title="Dynamic RAG System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Dynamic RAG System POC")
st.markdown("A scalable document retrieval system with external knowledge base")

with st.sidebar:
    st.markdown("""Please provide an API KEY""")
    st.link_button("get one @ Cohere",
                   "https://dashboard.cohere.com/api-keys",
                   icon="🔗")
    API_KEY = st.text_input("password",
                            type="password",
                            label_visibility="collapsed")
    st.divider()

# Initialize system
if API_KEY and 'rag_system' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        st.session_state.rag_system = RAGSystem(cohere_api_key=API_KEY)

rag_system = st.session_state.rag_system

# Sidebar for system stats
with st.sidebar:
    st.header("📊 System Stats")
    stats = rag_system.get_system_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", stats['total_documents'])
    with col2:
        st.metric("Vectors", stats['total_vectors'])

    st.metric("Queue Size", stats['processing_queue_size'])

    if stats['recent_documents']:
        st.subheader("Recent Documents")
        for doc in stats['recent_documents']:
            st.text(f"• {doc}")

# Main interface
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📄 Add Document", "⚙️ System"])

with tab1:
    st.header("Search Documents")

    query = st.text_input("Enter your search query:",
                          placeholder="What would you like to know?")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_chunks = st.slider("Max chunks to retrieve", 1, 10, 5)
    with col2:
        max_tokens = st.slider("Token budget", 1000, 32000, 16000, step=1000)
    with col3:
        generate_response = st.checkbox("Generate response", value=True)

    search_btn = st.button("🔍 Search", type="primary")

    if search_btn and query:
        with st.spinner("Searching documents and generating response..."):
            results = rag_system.search_documents(
                query,
                max_chunks,
                max_tokens,
                generate_response=generate_response
            )

        if results.get('chunks'):
            st.success(f"Found {len(results['chunks'])} relevant chunks")

            if generate_response and 'response' in results:
                st.subheader("Generated Answer")
                st.markdown(results['response']['text'])

                if results['response'].get('citations'):
                    st.caption("Citations:")
                    for citation in results['response']['citations']:
                        st.json(citation)

            st.subheader("Retrieved Context")
            total_tokens = sum(len(chunk['content']) // 4 for chunk in results['chunks'])
            st.info(f"Total tokens used: {total_tokens:,} / {max_tokens:,}")

            for i, chunk in enumerate(results['chunks'], 1):
                with st.expander(f"Chunk {i} (From: {chunk['metadata']['document_title']})"):
                    st.markdown(chunk['content'])
                    st.caption(f"Document ID: {chunk['metadata']['document_id']} | "
                             f"Chunk ID: {chunk['metadata']['chunk_id']} | "
                             f"Score: {chunk['metadata']['score']:.3f}")

            if results.get('sources'):
                st.subheader("Source Documents")
                for source in results['sources']:
                    st.text(f"• {source['title']} (ID: {source['id']})")
        else:
            st.warning("No relevant documents found.")

with tab2:
    st.header("Add New Document")

    # Document input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Manual Text Input", "📄 File Upload"],
        horizontal=True
    )

    with st.form("add_document"):
        title = st.text_input("Document Title*", placeholder="Enter document title")

        content = ""
        file_metadata = {}

        if input_method == "📝 Manual Text Input":
            content = st.text_area("Document Content*",
                                   placeholder="Paste or type document content here...",
                                   height=300)
        else:
            st.subheader("📄 File Upload")

            # Show supported formats
            supported_formats = {
                "PDF": "✅" if PDF_AVAILABLE else "❌ (pip install PyPDF2)",
                "DOCX": "✅" if DOCX_AVAILABLE else "❌ (pip install python-docx)",
                "TXT/MD": "✅",
                "Excel": "✅" if EXCEL_AVAILABLE else "❌ (pip install openpyxl)",
                "Code files": "✅ (.py, .js, .html, .css, .json)"
            }

            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a file",
                    type=['pdf', 'docx', 'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'xlsx', 'xls'],
                    help="Upload a document to extract text automatically"
                )

            with col2:
                st.markdown("**Supported Formats:**")
                for format_name, status in supported_formats.items():
                    st.markdown(f"• {format_name}: {status}")

            # Process uploaded file
            if uploaded_file is not None:
                try:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        extracted_text, file_meta = FileProcessor.process_uploaded_file(uploaded_file)
                        content = extracted_text
                        file_metadata = file_meta

                    # Show file info
                    st.success(f"✅ File processed successfully!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("File Size", f"{file_metadata.get('file_size', 0):,} bytes")
                    with col2:
                        st.metric("File Type", file_metadata.get('file_type', 'unknown').upper())
                    with col3:
                        st.metric("Text Length", f"{len(content):,} chars")

                    # Show preview
                    with st.expander("📖 Preview Extracted Text"):
                        preview_text = content[:1000] + "..." if len(content) > 1000 else content
                        st.text_area("Extracted content:", preview_text, height=150, disabled=True)

                    # Allow title auto-fill from filename
                    if not title and uploaded_file.name:
                        suggested_title = uploaded_file.name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ')
                        if st.button(f"📝 Use filename as title: '{suggested_title}'"):
                            title = suggested_title

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    if "not available" in str(e):
                        st.info("💡 **Installation Required**: " + str(e))
                    content = ""

        # Metadata section
        st.subheader("📋 Metadata (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Author")
            category = st.text_input("Category")
        with col2:
            tags = st.text_input("Tags (comma-separated)")
            source_url = st.text_input("Source URL")

        # Submit button
        submitted = st.form_submit_button("📄 Add Document", type="primary")

        if submitted:
            if not title or not content:
                st.error("❌ Title and content are required!")
            else:
                # Combine metadata
                metadata = {
                    'author': author,
                    'category': category,
                    'tags': [tag.strip() for tag in tags.split(',') if tag.strip()],
                    'source_url': source_url,
                    **file_metadata  # Include file metadata if available
                }

                with st.spinner("Adding document to knowledge base..."):
                    doc_id = rag_system.add_document(title, content, metadata)

                st.success(f"✅ Document added successfully!")

                # Show document info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Document ID**: `{doc_id}`")
                with col2:
                    st.info(f"**Content Length**: {len(content):,} characters")

                st.info("🔄 Document is being processed in the background and will be available for search shortly.")

                # Show processing estimation
                estimated_chunks = len(content) // 500  # Rough estimate
                st.caption(
                    f"⏱️ Estimated processing time: ~{max(1, estimated_chunks * 2)} seconds ({estimated_chunks} chunks)")

    # Quick upload tips
    with st.expander("💡 Upload Tips & Best Practices"):
        st.markdown("""
        **📄 File Processing Tips:**
        - **PDF**: Works best with text-based PDFs (not scanned images)
        - **DOCX**: Extracts text from paragraphs, tables not fully supported
        - **Excel**: Converts sheets to tab-separated text format
        - **Large Files**: Files over 10MB may take longer to process

        **🎯 Content Guidelines:**
        - **Clear Structure**: Use headings and paragraphs for better chunking
        - **Relevant Content**: Remove unnecessary formatting or metadata
        - **Complete Documents**: Partial or truncated content may reduce search quality

        **🏷️ Metadata Best Practices:**
        - **Tags**: Use specific, searchable keywords
        - **Categories**: Group related documents for easier filtering
        - **Author**: Include for attribution and source tracking
        """)

with tab3:
    st.header("System Configuration")

    st.subheader("Architecture Overview")
    st.markdown("""
    This POC demonstrates the complete lifecycle of a dynamic RAG system:

    **🏗️ Architecture Components:**
    - **Document Database**: SQLite for document storage
    - **Vector Store**: FAISS for similarity search
    - **Background Indexer**: Asynchronous document processing
    - **Text Processor**: Chunking and embedding generation
    - **MMR Algorithm**: Diversity in retrieval results

    **🔄 Processing Pipeline:**
    1. Document ingestion → Database storage
    2. Background chunking → Semantic segmentation
    3. Embedding generation → Vector storage
    4. Query processing → Similarity search
    5. MMR application → Diverse results
    6. Token budgeting → Context optimization
    """)

    st.subheader("System Performance")

    # Performance metrics (mock data for demo)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Query Time", "145ms", "↓ 23ms")
    with col2:
        st.metric("Index Size", "2.3MB", "↑ 0.8MB")
    with col3:
        st.metric("Memory Usage", "127MB", "↑ 12MB")

    st.subheader("Configuration")
    st.markdown("""
    **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)  
    **Chunk Size**: 500 tokens with 50 token overlap  
    **Similarity Metric**: Cosine similarity (Inner Product)  
    **MMR Lambda**: 0.7 (70% relevance, 30% diversity)  
    """)

