That's an impressive multi-step Streamlit application using Groq, LangChain, and a FAISS vector store for a RAG-powered medical QA system!

Here is the complete README.md file structured for a GitHub repository. You can copy and paste this directly.

🧠 Groq + LangChain MedQA Multi-Agent Chatbot (RAG for Research)
This project implements a multi-step medical diagnosis and treatment planning system using Groq's high-speed LLMs, LangChain, and a Retrieval-Augmented Generation (RAG) approach with a persisted FAISS vector store built on a medical QA dataset (e.g., MedQA).

It is specifically designed as a research and insights tool, featuring a batch testing mode that automatically logs every step—symptom extraction, diagnosis, treatment, and RAG context documents—to a persistent CSV file for later analysis.

✨ Features
⚡️ High-Speed Inference: Leverages Groq (e.g., llama-3.1-8b-instant) for rapid processing across multiple LLM calls in the pipeline.

🛠️ Multi-Step Pipeline: A sequential chain of agents for accurate, contextualized medical responses:

Symptom Extractor: Extracts key symptoms from the user query.

Diagnoser (RAG): Uses the extracted symptoms to retrieve relevant medical QA pairs from the FAISS vector store and generates a diagnosis with evidence.

Treatment Planner: Suggests an evidence-based treatment plan based on the final diagnosis.

💾 FAISS Persistence: The vector store index is built once from the uploaded CSV and saved to the configured directory (faiss_medqa_index), eliminating the need to re-index the data on every subsequent run.

📊 Research Logging: A dedicated Batch Test mode runs multiple queries and appends all inputs, intermediate steps, outputs, and RAG context to a master CSV log file (medqa_research_master_log.csv) for easy analysis.

⚛️ Streamlit Interface: A clean, interactive web application for managing the index, running single queries, and executing batch tests.

🚀 Setup and Installation
1. Prerequisites
You will need a Groq API Key.

2. Clone the Repository
Bash

git clone <your-repository-link>
cd <your-repository-name>
3. Environment Setup
Create a virtual environment and install the required dependencies.

First, create a requirements.txt file with the following contents:

streamlit
pandas
langchain
langchain-groq
langchain-community
huggingface-hub
sentence-transformers
python-dotenv
Then, install the dependencies:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
4. Configure API Key
Create a file named .env in the root directory and add your Groq API key:

# .env file
GROQ_API_KEY="gsk_..." 
5. Obtain Data
You will need a medical Question-Answer dataset in CSV format (e.g., a MedQA dataset) with columns named 'question' and 'answer'.

💻 Running the Application
1. Start the Streamlit App
Assuming your main application file is named med_rag_app.py:

Bash

streamlit run med_rag_app.py
2. Build the Vector Store (First Run)
Navigate to the Streamlit application in your browser.

Click the "Upload MedQA CSV dataset" file uploader.

Upload your medical QA CSV file.

The application will automatically Build and save the FAISS index to the configured directory (faiss_medqa_index/). A success message will confirm the process.

Note: In subsequent runs, the app will automatically load the persisted index from the folder, making startup much faster.

3. Use the Features
A. Single Query Diagnosis
Enter a patient query (e.g., "I'm a 50-year-old male with dull flank pain and blood in my urine.") into the Single Query Diagnosis text area.

Click Run Single Diagnosis.

The results will be displayed, showing the output of the three steps (Symptoms, Diagnosis, Treatment) and the raw RAG context documents used for the diagnosis.

B. Research Batch Testing (Log Insights)
Enter multiple research queries, one per line, in the Research Logging & Batch Test text area.

Click Run Batch Test and Log Results.

The results are processed and automatically appended to the medqa_research_master_log.csv file.

The application displays the new batch results and provides a button to Download Full Research Log CSV for comprehensive analysis.
