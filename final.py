import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# ==========================
# CONFIGURATION
# ==========================
# WARNING: Keep your actual API key secure. This is a placeholder/example.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

# --- CRITICAL CHANGE: Static filename for continuous logging ---
LOG_FILENAME = "medqa_research_master_log.csv"
# --- NEW CONFIGURATION: Directory for persistent vector store ---
FAISS_INDEX_DIR = "faiss_medqa_index"
# ---------------------------------------------------------------

# Configure standard logging for internal messages (not for results CSV)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_medqa_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['question', 'answer'])
    return df

# ==========================
# FAISS PERSISTENCE FUNCTIONS (NEW)
# ==========================

def get_embeddings_model():
    """Initializes and returns the BGE embedding model."""
    return HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def build_and_save_vectorstore(df, index_dir: str, embeddings):
    """Builds the vector store, saves it to disk, and returns it."""
    texts = []
    for _, row in df.iterrows():
        q, a = row['question'], row['answer']
        texts.append(f"Question: {q}\nAnswer: {a}")
    
    # 1. Build the index
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    
    # 2. Save the index to disk
    try:
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        vectorstore.save_local(index_dir)
        logging.info(f"FAISS index successfully saved to {index_dir}")
    except Exception as e:
        logging.error(f"Failed to save FAISS index: {e}")
        
    return vectorstore

def load_vectorstore(index_dir: str, embeddings):
    """Loads the FAISS index from disk."""
    if os.path.exists(index_dir) and len(os.listdir(index_dir)) > 0:
        try:
            vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"FAISS index successfully loaded from {index_dir}")
            return vectorstore
        except Exception as e:
            logging.error(f"Failed to load FAISS index from disk: {e}")
            return None
    return None

# ==========================
# AGENT DEFINITIONS (Unchanged)
# ==========================

def create_symptom_extractor(llm):
    """Creates an LLMChain to extract symptoms."""
    template = """You are a medical symptom extraction specialist. Extract only key symptoms from the patient query.
    Input: {input}
    Output (comma separated symptoms):"""
    prompt = PromptTemplate(template=template, input_variables=["input"])
    return LLMChain(llm=llm, prompt=prompt)

def get_diagnoser(llm, vectorstore):
    """Creates a function that performs diagnosis using vector search context and returns context."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def diagnose(symptoms: str) -> Dict[str, str | List[Document]]:
        """Returns the diagnosis string and the retrieved documents."""
        docs = retriever.get_relevant_documents(symptoms)
        context = "\n---\n".join([d.page_content for d in docs])
        query = f"""Based on these symptoms: {symptoms}
Relevant medical cases from MedQA dataset:
---
{context}
---
What is the most likely diagnosis? Provide a brief explanation based on the context."""
        
        # Use invoke to get the structured content
        diagnosis_content = llm.invoke(query).content
        
        return {
            "diagnosis": diagnosis_content,
            "context_docs": docs
        }

    return diagnose

def create_treatment_planner(llm):
    """Creates an LLMChain to suggest a treatment plan."""
    template = """You are a medical assistant. Given a diagnosis, suggest an evidence-based treatment plan.
    Diagnosis: {diagnosis}
    Output:"""
    prompt = PromptTemplate(template=template, input_variables=["diagnosis"])
    return LLMChain(llm=llm, prompt=prompt)

# ==========================
# RESEARCH AND INSIGHTS LOGGING (Unchanged)
# ==========================

def run_research_batch(
    test_queries: List[str], 
    symptom_extractor: LLMChain, 
    diagnose_func: callable, 
    treatment_planner: LLMChain
) -> pd.DataFrame:
    """Runs a batch of queries through the pipeline and logs all steps for research."""
    
    results_list = []
    
    for i, user_input in enumerate(test_queries):
        logging.info(f"Processing research query {i+1}/{len(test_queries)}: '{user_input[:50]}...'")
        
        result: Dict[str, Any] = {
            "query_id": i + 1,
            "input_query": user_input,
            "timestamp": datetime.now().isoformat(),
            "symptoms_extracted": None,
            "diagnosis_output": None,
            "treatment_output": None,
            "context_document_1": None,
            "context_document_2": None,
            "context_document_3": None,
            "error": None
        }

        try:
            # STEP 1: Extract Symptoms
            symptoms = symptom_extractor.run(input=user_input).strip()
            result["symptoms_extracted"] = symptoms
            
            # STEP 2: Predict Diagnosis (using custom function)
            diagnosis_info = diagnose_func(symptoms)
            diagnosis = diagnosis_info["diagnosis"].strip()
            result["diagnosis_output"] = diagnosis
            
            # Log Context Documents
            for j, doc in enumerate(diagnosis_info["context_docs"]):
                if j < 3:
                    # Use || to separate lines for CSV readability and avoid issues with embedded commas
                    result[f"context_document_{j+1}"] = doc.page_content.replace('\n', ' || ') 

            # STEP 3: Plan Treatment
            treatment_output = treatment_planner.run(diagnosis=diagnosis)
            treatment = treatment_output.strip() if isinstance(treatment_output, str) else treatment_output.get('text', treatment_output.get('output', 'N/A')).strip()
            result["treatment_output"] = treatment

        except Exception as e:
            error_msg = f"Error during processing: {e}"
            logging.error(f"Query {i+1} failed: {error_msg}")
            result["error"] = error_msg
        
        results_list.append(result)
        
    return pd.DataFrame(results_list)

# ==========================
# STREAMLIT UI (Revised Logic)
# ==========================

def main():
    st.set_page_config(page_title="Groq + LangChain MedQA Multi-Agent Chatbot", layout="wide")
    st.title("🧠 Multi-Step Medical Chatbot (Groq + LangChain + MedQA)")
    st.subheader("For Research & Insights Recording")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY environment variable not set. Please set it to run the application.")
        return

    # Initialize state variables
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = None
    
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatGroq(temperature=0.3, model_name=MODEL_NAME, groq_api_key=GROQ_API_KEY)

    # Initialize Embeddings model once
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = get_embeddings_model()
        
    embeddings = st.session_state.embeddings
    
    # --- Vector Store Persistence Logic ---
    if 'vectorstore' not in st.session_state:
        # 1. Try to load from disk
        st.session_state.vectorstore = load_vectorstore(FAISS_INDEX_DIR, embeddings)
        if st.session_state.vectorstore:
            st.success(f"Loaded vector store from **{FAISS_INDEX_DIR}**. Ready to run!")
            st.session_state.is_indexed = True
            st.session_state.df_length = "Loaded from disk"
        else:
            st.warning(f"No existing vector store found in **{FAISS_INDEX_DIR}**. Please upload a dataset to build one.")
            st.session_state.is_indexed = False
            
    # File uploader and rebuild logic
    uploaded_file = st.file_uploader("Upload MedQA CSV dataset (only needed if building/rebuilding index)", type="csv", key="medqa_uploader")
    
    # If a file is uploaded OR if an index needs to be built/rebuilt
    if uploaded_file and (uploaded_file.name not in st.session_state or st.button("Rebuild Index")):
        with st.spinner(f"Building and saving vector database from {uploaded_file.name}..."):
            df = load_medqa_dataset(uploaded_file)
            # Use the new function to build AND save the index
            st.session_state.vectorstore = build_and_save_vectorstore(df, FAISS_INDEX_DIR, embeddings)
            st.session_state.vectorstore_file = uploaded_file.name
            st.session_state.df_length = len(df)
            st.session_state.is_indexed = True
        st.success(f"Indexed and saved {st.session_state.df_length} QA pairs to disk.")


    if st.session_state.get('is_indexed'):
        # Assign components from session state for easy access
        llm = st.session_state.llm
        vectorstore = st.session_state.vectorstore

        # Create components
        symptom_extractor = create_symptom_extractor(llm)
        diagnose_func = get_diagnoser(llm, vectorstore)
        treatment_planner = create_treatment_planner(llm)
        
        # ---
        ## 📊 Research Batch Testing
        # ---
        st.markdown("---")
        st.header("Research Logging & Batch Test")
        
        # ... (rest of the batch testing logic remains the same)
        test_queries_input = st.text_area(
            "Enter a batch of test queries (one per line):", 
            key="research_queries",
            height=200,
            value="I have a high fever, cough, and severe body aches.\nI'm experiencing blurred vision and frequent urination.\nSudden onset of crushing chest pain radiating to my left arm."
        )

        if st.button("Run Batch Test and Log Results"):
            if test_queries_input:
                test_queries = [q.strip() for q in test_queries_input.split('\n') if q.strip()]
                
                with st.spinner(f"Running {len(test_queries)} queries in batch mode. This may take a moment..."):
                    results_df = run_research_batch(
                        test_queries, 
                        symptom_extractor, 
                        diagnose_func, 
                        treatment_planner
                    )

                # --- CRITICAL CHANGE: Append Logic ---
                # Check if the file already exists
                file_exists = os.path.exists(LOG_FILENAME)
                
                # Determine mode: 'a' (append) if file exists, 'w' (write/create) if new
                mode = 'a' if file_exists else 'w'
                
                # Write header only if the file is new (mode 'w')
                header = not file_exists 

                results_df.to_csv(
                    LOG_FILENAME, 
                    index=False, 
                    mode=mode, 
                    header=header
                )
                # ------------------------------------

                st.success(f"Batch test complete! Results APPENDED to **{LOG_FILENAME}**.")
                
                # Display the just-added batch results
                st.dataframe(results_df)

                # Provide a download link for the entire master log
                try:
                    master_df = pd.read_csv(LOG_FILENAME)
                    csv_data = master_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Research Log CSV",
                        data=csv_data,
                        file_name=LOG_FILENAME,
                        mime='text/csv',
                    )
                except Exception as e:
                     st.error(f"Could not load master log for download: {e}")
            else:
                st.warning("Please enter at least one query for the batch test.")

        # ---
        ## 🩺 Single Query Interface
        # ---
        st.markdown("---")
        st.header("Single Query Diagnosis")

        user_input = st.text_area("Enter patient query (symptoms or question):", key="user_query_single")

        if st.button("Run Single Diagnosis") and user_input:
            with st.spinner("Processing through the multi-step pipeline..."):
                try:
                    # STEP 1: Extract Symptoms
                    symptoms = symptom_extractor.run(input=user_input).strip()
                    
                    # STEP 2: Predict Diagnosis (using custom function)
                    diagnosis_info = diagnose_func(symptoms)
                    diagnosis = diagnosis_info["diagnosis"].strip()
                    
                    # STEP 3: Plan Treatment
                    treatment_output = treatment_planner.run(diagnosis=diagnosis)
                    treatment = treatment_output.strip() if isinstance(treatment_output, str) else treatment_output.get('text', treatment_output.get('output', 'N/A')).strip()

                    # Store results for display
                    st.session_state.diagnosis_result = {
                        "symptoms": symptoms,
                        "diagnosis": diagnosis,
                        "treatment": treatment,
                        "context_docs": diagnosis_info["context_docs"] # Store docs for transparency
                    }
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    logging.error(f"Single query failed: {e}")
                    st.session_state.diagnosis_result = None
        
        # Display results if available
        if st.session_state.diagnosis_result and 'user_query_single' in st.session_state:
            results = st.session_state.diagnosis_result
            st.markdown("### 🔍 Step-by-Step Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 🩺 1. Extracted Symptoms")
                st.code(results['symptoms'], language="text")

            with col2:
                st.markdown("#### ⚕️ 2. Predicted Diagnosis")
                st.info(results['diagnosis'])

            with col3:
                st.markdown("#### 💊 3. Recommended Treatment")
                st.success(results['treatment'])

            st.markdown("---")
            st.markdown("### 📚 RAG Context (For Analysis)")
            for i, doc in enumerate(results['context_docs']):
                st.markdown(f"**Document {i+1}:**")
                st.code(doc.page_content, language="text")
    else:
        st.error("Please upload the MedQA CSV dataset to build the vector index and start the application.")


if __name__ == "__main__":
    main()