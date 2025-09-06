import streamlit as st
from PIL import Image

# Page Config
st.set_page_config(page_title="Qazi Adnan - Portfolio", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Home", "Skills", "Experience", "Projects", "Education", "Interests", "Contact"]
)

# Load Profile Image
image = Image.open("profile.jpg")

# Custom CSS for circular image
circle_style = """
<style>
img {
    border-radius: 50%;
    width: 180px;
    height: 180px;
    object-fit: cover;
    border: 4px solid #4CAF50;
}
</style>
"""
st.markdown(circle_style, unsafe_allow_html=True)

# Sections
if section == "Home":
    st.image(image, use_column_width=False)
    st.title("Qazi Adnan")
    st.subheader("AI Engineer | ML • DL • Generative AI • AWS")
    st.write("📍 Srinagar, Jammu and Kashmir")
    st.write("📧 qazi.adnan.2k1@gmail.com | 📱 6006758687")

    st.markdown("""
    **Professional Summary**  
    • AI Engineer with advanced expertise in Machine Learning, Deep Learning, and Generative AI, specializing in LLMs, RAG, and MLOps for enterprise applications on AWS.  
    • Expert in Amazon SageMaker (Training, Pipelines, Feature Store, Model Registry, Serverless/Multi-Model Endpoints) and Amazon Bedrock (Guardrails, Knowledge Bases, Agents).  
    • Skilled in deploying microservices with FastAPI and Flask, using EKS/ECS, Step Functions, EventBridge, and Lambda for scalable and observable AI platforms.  
    • Experienced in vector search, semantic retrieval, and evaluation using OpenSearch, FAISS, pgvector, and hybrid retrieval with reranking strategies.  
    • Strong in data engineering with S3 data lakes, Glue, Lake Formation, Athena, Redshift, and secure architectures using IAM, VPC, KMS, WAF, and CloudTrail.  
    """)

elif section == "Skills":
    st.header("Skills")

    st.write("### Programming")
    st.write("Python, FastAPI, Flask, SQL, Bash, TypeScript, Node.js")

    st.write("### ML/DL")
    st.write("PyTorch, TensorFlow, scikit-learn, XGBoost, ONNX Runtime, Triton, TensorRT")

    st.write("### LLM/GenAI")
    st.write("Amazon Bedrock, LangChain, LlamaIndex, RAG, Prompt Engineering, Eval Frameworks")

    st.write("### AWS AI/ML")
    st.write("SageMaker (Pipelines, HPO, Feature Store, Clarify, Wrangler, MME, Serverless), "
             "Bedrock (Guardrails, KBs, Agents), Kendra, OpenSearch, Comprehend, Textract")

    st.write("### Data/Streaming")
    st.write("S3, Glue, Lake Formation, Athena, Redshift, EMR, MSK Kafka, Kinesis")

    st.write("### Infra/Backend")
    st.write("Lambda, Step Functions, ECS, EKS, API Gateway, SQS, SNS, EventBridge")

    st.write("### Security/Ops")
    st.write("IAM, KMS, Secrets Manager, VPC, PrivateLink, WAF, CloudWatch, CloudTrail")

    st.write("### DevOps/IaC")
    st.write("Docker, Kubernetes, Terraform, AWS CDK, GitHub Actions, CodePipeline")

    st.write("### Databases/Search")
    st.write("PostgreSQL, MySQL, MongoDB, Redis, OpenSearch Vector, pgvector, FAISS")

    st.write("### Platforms")
    st.write("Azure AI Foundry, Vertex AI, Git, Jira, Postman, Swagger")

elif section == "Experience":
    st.header("Experience")

    with st.expander("AI Engineer – Elyspace (June 2024 – Present)"):
        st.write("""
        • Architected a production-grade GenAI platform on AWS using Amazon Bedrock Agents and Guardrails with OpenSearch vector indices and hybrid search.  
        • Built end-to-end MLOps pipelines in SageMaker: data prep, HPO, distributed training, registry approval workflows, and CI/CD with CodePipeline.  
        • Designed low-latency inference with SageMaker Multi-Model Endpoints and Serverless Inference, integrating blue/green and canary deployments.  
        • Implemented RAG pipelines with Bedrock Knowledge Bases, embeddings in OpenSearch/pgvector, reranking, and prompt safety filtering.  
        • Established observability with CloudWatch, Model Monitor, Clarify, and OpenTelemetry and defined SLOs for latency, throughput, and accuracy.  
        • Delivered secure data lakes with S3, Glue, and Lake Formation, fine-grained IAM, encrypted pipelines, and Athena/Redshift analytics.  
        """)

elif section == "Projects":
    st.header("Projects")

    with st.expander("Enterprise Retrieval-Augmented Generation (RAG) Platform"):
        st.write("""
        A production RAG platform integrating enterprise knowledge bases with vector search and LLM orchestration.  

        **Key Features:**  
        • Data ingestion pipelines (Confluence, SharePoint, PDFs, DB exports, HTML, S3).  
        • Embedding pipelines (OpenSearch + pgvector).  
        • Hybrid retrieval (BM25 + dense + reranking).  
        • Context injection with Bedrock KBs.  
        • Multi-tenant isolation with KMS/IAM.  

        **Tech Stack:** Python, FastAPI, SageMaker, Bedrock, OpenSearch, pgvector, FAISS, Glue, S3  
        """)

    with st.expander("Recommendation System for Job Portal and E-commerce App"):
        st.write("""
        End-to-end recommendation architecture for jobs & e-commerce personalization.  

        **Key Features:**  
        • Feature store & real-time pipelines with SageMaker Feature Store & Kinesis/MSK.  
        • Job matching via semantic embeddings, e-commerce candidates via hybrid signals.  
        • Ranking models (XGBoost, SASRec-style DL, two-tower retrieval).  
        • Contextual bandits for exploration/diversity.  
        • A/B testing, retraining, CI/CD pipelines.  

        **Tech Stack:** Python, SageMaker, FAISS, OpenSearch, XGBoost, PyTorch, Kafka/MSK, Glue, Terraform, Prometheus, Grafana  
        """)

    with st.expander("LLM Inference Mesh"):
        st.write("""
        Horizontally scalable inference mesh consolidating multiple LLMs for optimized GPU use.  

        **Key Features:**  
        • SageMaker Multi-Model Endpoints + Triton Inference Server.  
        • Dynamic model routing & tenant isolation.  
        • Autoscaling, batching, pre-warming strategies.  
        • Unified API with streaming + guardrails.  
        • Full observability with OpenTelemetry.  

        **Tech Stack:** SageMaker, Triton, Docker, EKS, FastAPI, PyTorch, Prometheus, CloudWatch  
        """)

    with st.expander("Document AI Pipeline"):
        st.write("""
        Serverless pipeline for extracting, summarizing, and querying large documents.  

        **Key Features:**  
        • Orchestrated with S3, Step Functions, Textract.  
        • Semantic embedding workflows → knowledge bases.  
        • Summarization & QA microservices with Bedrock + SageMaker.  
        • Compliance: PII detection, redaction, access governance.  

        **Tech Stack:** Textract, SageMaker, Bedrock, Comprehend, Glue, Lambda, OpenSearch  
        """)

    with st.expander("Customer Support Agentic AI"):
        st.write("""
        Agentic AI to automate customer support workflows.  

        **Key Features:**  
        • Multi-step agent workflows: retrieval, tool invocation, action execution.  
        • Connectors for Zendesk/Jira/ServiceNow, CRMs, order systems.  
        • Human-in-the-loop escalation triggers.  
        • Hallucination mitigation via provenance & verification.  

        **Tech Stack:** Bedrock Agents, SageMaker, OAuth2, OpenSearch, Terraform, Prometheus  
        """)

elif section == "Education":
    st.header("Education")
    st.write("🎓 **National Institute of Technology (NIT) Srinagar** — B.Tech in Computer Science & Engineering")

elif section == "Interests":
    st.header("Interests")
    st.write("""
    Applied ML and trustworthy AI, retrieval research, vector search,  
    LLM safety and evaluation, scalable inference systems, and developer platforms for AI.
    """)

elif section == "Contact":
    st.header("Contact Me")
    st.markdown("📧 [Email Me](mailto:qazi.adnan.2k1@gmail.com)")
    
