import os
import io
import re
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from IPython.display import display, HTML

from dotenv import load_dotenv

load_dotenv()

## PATH ##
ROOT = os.path.join(os.path.dirname(os.getcwd()), 'AgenticRAG')
FILE_PATH = os.path.join(ROOT, "database", "manual")

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found: {FILE_PATH}") # 파일이 없을 경우 에러 발생
    
# vector store path
VECTORSTORE_PATH = os.path.join(ROOT, "database/vectorstore")
if not os.path.exists(VECTORSTORE_PATH):
    os.makedirs(VECTORSTORE_PATH)
    
# embedding model path
EMBEDDING_MODEL_DIR_PATH = os.path.join(ROOT, "database/embedding_model/")
if not os.path.exists(EMBEDDING_MODEL_DIR_PATH):
    os.makedirs(EMBEDDING_MODEL_DIR_PATH)

# fdc database path
FDC_DB_PATH = os.path.join(ROOT, "data")
if not os.path.exists(FDC_DB_PATH):
    os.makedirs(FDC_DB_PATH)

anomaly_data = pd.read_csv(os.path.join(FDC_DB_PATH, "anomaly_history_data.csv"))
anomaly_data['start_time'] = pd.to_datetime(anomaly_data['start_time'])
anomaly_data['end_time'] = pd.to_datetime(anomaly_data['end_time'])
anomaly_data = anomaly_data[['start_time', 'end_time', 'cmk', 'threshold', 'pattern_name']]

raw_data = pd.read_csv(os.path.join(FDC_DB_PATH, "raw_data.csv"))

from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("./database/manual/조치이력보고서.docx")
docs = loader.load()


# 1. 워드 파일 로드
loader = Docx2txtLoader("./database/manual/조치이력보고서.docx")
docs = loader.load()

# 2. 전체 텍스트 합치기
full_text = "\n".join([doc.page_content for doc in docs])

# 3. '보고서' 기준으로 페이지 분할
pages = full_text.split("보고서")
pages = [page.strip() for page in pages if page.strip()]

# 최종 저장할 리스트
final_documents = []

# 4. 각 페이지 처리
for page in pages:
    # 문서 번호, 작성 일자, 작성자 추출
    doc_num_match = re.search(r"문서 번호:\s*(.+)", page)
    date_match = re.search(r"작성 일자:\s*(.+)", page)
    writer_match = re.search(r"작성자:\s*(.+)", page)
    
    doc_num = doc_num_match.group(1).strip() if doc_num_match else ""
    date = date_match.group(1).strip() if date_match else ""
    writer = writer_match.group(1).strip() if writer_match else ""
    
    # '조치 이력' 이후만 잘라서 오류 파싱
    after_history = page.split("[조치 이력]")[-1].strip()
    
    # "오류 1:", "오류 2:" 기준으로 오류 분리
    errors = re.split(r"오류\s*\d+\s*:", after_history)
    errors = [e.strip() for e in errors if e.strip()]
    
    # 오류 번호도 필요하니까 다시 찾자
    error_numbers = re.findall(r"오류\s*(\d+)\s*:", after_history)

    for error_content, error_no in zip(errors, error_numbers):
        document = Document(
            page_content=error_content,
            metadata={
                "문서 번호": doc_num,
                "작성 일자": date,
                "작성자": writer,
                "오류번호": int(error_no)
            }
        )
        final_documents.append(document)


os.environ["TRANSFORMERS_CACHE"] = EMBEDDING_MODEL_DIR_PATH
os.environ["HF_HOME"] = EMBEDDING_MODEL_DIR_PATH

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large-instruct",
)

vectorstore_report = FAISS.from_documents(
    final_documents,
    embedding_model,
)

retriever1 = vectorstore_report.as_retriever()

llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Latest Date - raw_data.csv 기준
time = pd.to_datetime(raw_data['TIME'])
latest_data_date = time.max().normalize().strftime("%Y-%m-%d")

agent_context = {
    "last_plot_date": None,
    "last_plot_columns": None,
    "last_anomaly_info": None,
}

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()


anomaly_code_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a data assistant that writes Python (pandas) code to filter a DataFrame named `anomaly_data`.

Here is the description of the columns in the 'anomaly_history' table:
- `start_time`: datetime when the anomaly event started (format: YYYY-MM-DD HH:MM:SS)
- `end_time`: datetime when the anomaly event ended (format: YYYY-MM-DD HH:MM:SS)
- `cmk`: numeric value indicating the anomaly confidence measure (higher values indicate stronger anomalies)
- `threshold`: numeric value representing the threshold for anomaly detection
- `pattern_name`: string describing the type of pattern detected (e.g., 'noise')

⚠️ Language clarification:
- The phrase "에러 패턴" or "error pattern" in user queries does NOT mean `pattern_name == 'error'`. 
- Instead, interpret it as: "retrieve the values in the `pattern` column for the corresponding date."

Write a single line of Python code that assigns the result to a variable named `result`. 
Only use the `anomaly_data` DataFrame and valid pandas syntax.

Only output a single line of code. Do NOT include markdown (like ```python) or explanation.

Query: {query}

Python code:
"""
)

anomaly_code_chain = LLMChain(llm=llm, prompt=anomaly_code_prompt)

def execute_anomaly_code(code: str, context: dict = None):
    context = context or {}
    local_vars = {"anomaly_data": anomaly_data, "pd": pd}
    local_vars.update(context)

    print(f"[DEBUG] Available Columns: {anomaly_data.columns}")

    try:
        # 🔧 문자열 전처리 추가
        cleaned_code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.MULTILINE)
        exec_globals = {}
        exec_locals = local_vars
        exec(cleaned_code, exec_globals, exec_locals)
        return exec_locals["result"]
    except Exception as e:
        return f"[ERROR in exec] {str(e)}"
    
def plot_sensor_data(columns: List[str], query:str, start_time=None, end_time=None) -> str:
    from IPython.display import Image 
    # Load the data
    df = raw_data.copy()
    df['TIME'] = pd.to_datetime(df['TIME'])
    
    if start_time:
        df = df[df['TIME'] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df['TIME'] <= pd.to_datetime(end_time)]
       
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in columns:
        ax.plot(df['TIME'], df[col], label=col)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f"Sensor Data: {', '.join(columns)}")
    ax.legend()
    ax.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return display(Image(data=buf.read()))

plot_query_prompt = PromptTemplate(
    input_variables=["query", "chat_history", "latest_data_date"],
    template="""
Extract the following information from the user's query and conversation:
1. "columns": One or more of ['Header_pressure_Act', 'Header_pressure_Set', 'cmk', 'error']
2. "date": Extract the date mentioned or implied. 
    - If user says "가장 최근" or "latest", use {latest_data_date}
    - If user says "그 전날" or "전날짜", use the date of the most recent assistant message containing "시각화한 날짜: YYYY-MM-DD" minus 1 day.
    - If user says "그 다음날" or "다음날짜", use the date of the most recent assistant message containing "시각화한 날짜: YYYY-MM-DD" plus 1 day.
    - If user says "그때" or "위 그래프", extract date from most recent assistant message containing "시각화한 날짜: YYYY-MM-DD"

Return JSON:
{{
  "columns": [...],
  "date": "YYYY-MM-DD",
  "message": "..."
}}

Return only valid JSON. Do NOT wrap it in triple backticks or markdown.

Query: {query}
Chat History:
{chat_history}
"""
)
plot_query_chain = LLMChain(llm=llm, prompt=plot_query_prompt)

anomaly_query_prompt = PromptTemplate(
    input_variables=["query", "chat_history", "latest_data_date", "last_plot_date", "last_plot_columns"],
    template="""
You are an assistant responsible for rewriting the user's query into a concrete anomaly search question.

Context interpretation rules:
- If the user says "latest" or "가장 최근", use {latest_data_date}.
- If the user says "the previous day" or "그 전날" or "전날짜", use {last_plot_date} minus 1 day.
- If the user says "the next day" or "그 다음날" or "다음날짜", use {last_plot_date} plus 1 day.
- If the user says "that time", "the graph above", or "그때" or "위 그래프", use the explicitly provided value: {last_plot_date}.

💡 The most recent plot date was: {last_plot_date}
💡 The most recently plotted columns were: {last_plot_columns}

From the user query, extract:
- A specific date
- A specific anomaly type (e.g., error, spike)

Then, rephrase the query into a format like:
"Show me the error pattern that occurred on 2025-04-24."

Only output the final rephrased query as a single string.

Query: {query}
Chat History:
{chat_history}
"""
)
anomaly_query_chain = LLMChain(llm=llm, prompt=anomaly_query_prompt)

recommend_query_prompt = PromptTemplate(
    input_variables=["query", "chat_history", "context"],
    template="""
Extract the relevant anomaly pattern name(s) from the query or from the most recent assistant message that provided anomaly information.

If the user says "그때 발생했던 각 이상들에 대한 조치는?", refer to the following structured anomaly result context:
{context}

From this, generate a query such as:
"spike 패턴에 대해 cmk=1.02, threshold=2.8일 때의 조치는?"

Query: 
{query}

Chat History:
{chat_history}
"""
)
recommend_query_chain = LLMChain(llm=llm, prompt=recommend_query_prompt)

def run_plot_tool(parsed_json_str: str):
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", parsed_json_str.strip(), flags=re.MULTILINE)
        parsed = json.loads(cleaned)

        date = parsed.get("date")
        columns = parsed.get("columns", [])

        if not columns or not date:
            return "[ERROR] columns 또는 date 누락됨"

        # Update context
        agent_context["last_plot_date"] = date
        agent_context["last_plot_columns"] = columns

        start_time = f"{date} 00:00:00"
        end_time = f"{date} 23:59:59"
        
        img = plot_sensor_data(columns, query="generated by agent", start_time=start_time, end_time=end_time)
        display(img)  # ✅ 여기서 셀 최상위 display 호출

        return f"{date}의 {', '.join(columns)} 그래프를 시각화했습니다."

    except Exception as e:
        return f"[ERROR] run_plot_tool 실패: {str(e)}"

def run_anomaly_tool(query: str, context: dict = None) -> str:
    try:
        last_plot_date = agent_context.get("last_plot_date")
        last_plot_columns = agent_context.get("last_plot_columns", [])
        
        if not last_plot_date:
            return "[ERROR] 최근에 시각화된 날짜 정보가 없습니다. 먼저 그래프를 그려야 합니다."

        transformed_query = anomaly_query_chain.run({
            "query": query,
            "chat_history": "\n".join([m.content for m in memory.chat_memory.messages]),
            "latest_data_date": latest_data_date,
            "last_plot_date": last_plot_date,
            "last_plot_columns": ", ".join(last_plot_columns)
        })

        code = anomaly_code_chain.run(transformed_query)
        print(f"[DEBUG] pandas code:\n{code}")
        result_df = execute_anomaly_code(code, context=context)

        # 조치 정보 저장
        agent_context["last_anomaly_info"] = {
            "pattern_name": result_df['pattern_name'].mode()[0],
            "cmk": round(result_df['cmk'].mean(), 2),
            "threshold": round(result_df['threshold'].mean(), 2)
        }

        return str(result_df)

    except Exception as e:
        return f"[ERROR] run_anomaly_tool failed: {str(e)}"

def run_recommend_tool(query: str) -> str:
    context = agent_context.get("last_anomaly_info", {})
    structured_context=(
        f"{context.get('pattern_name', '')} 패턴, "
        f"cmk={context.get('cmk', '')}, "
        f"threshold={context.get('threshold', '')}"
    )

    transformed_query = recommend_query_chain.run({
        "query": query,
        "chat_history": "\n".join([m.content for m in memory.chat_memory.messages]),
        "context": structured_context
    })

    retriever1_qa = RetrievalQA.from_chain(
        llm=llm,
        retriever=retriever1,
        chain_type="stuff",        
    )
    try:
        return retriever1_qa.invoke(query)
    except Exception as e:
        return f"[ERROR] run_recommend_tool 실패: {str(e)}"
    
tools = [
    Tool(
        name="PlotSensorData",
        description="시계열 센서 데이터를 시각화합니다. 예: '가장 최근 error 그래프', '그때의 설정값과 실제값 그래프'. 날짜가 명시되지 않으면 문맥을 통해 추론된 날짜를 사용합니다.",
        func=lambda q: run_plot_tool(plot_query_chain.run({
            "query": q,
            "chat_history": "\n".join([m.content for m in memory.chat_memory.messages]),
            "latest_data_date": latest_data_date
        })),
        return_direct=True
    ),
    Tool(
        name="QueryAnomalyHistory",
        description=(
            "Retrieve anomaly information from CSV based on past plot date, pattern_name, cmk, or threshold. "
            "For example: 'What was the pattern on the day of the last error graph?' or "
            "'Show anomalies with cmk over 1.5'. This tool transforms the query contextually before execution."
        ),
        func=lambda q: run_anomaly_tool(q, context={"anomaly_data": anomaly_data}),
    ),
    Tool(
        name="RecommendActions",
        description="이상 패턴에 대해 적절한 조치 방법을 추천합니다. 예: '그때 발생한 이상들에 대한 조치 방법은?'"
                    "이전에 검색된 패턴명, CMK, 임계값 정보를 활용해 조치 검색 쿼리를 생성합니다.",
        func=lambda q: run_recommend_tool(recommend_query_chain.run({
            "query": q,
            "chat_history": "\n".join([m.content for m in memory.chat_memory.messages]),
            "context": agent_context.get("last_anomaly_info", "")
        }))
    )
]

# === 5. REACT Agent 초기화 ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True,
    agent_type="tool-calling",
    handle_parsing_errors=True,
)