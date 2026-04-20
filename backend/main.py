from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List, Dict, Optional
import os

app = FastAPI()

# 配置 CORS 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议改为前端实际地址如 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 1. Neo4j 连接与查询封装
# ==========================================
class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_phenomenon(self, phenomenon: str):

        graph_query = """
        MATCH (n)
        WHERE (n:Entity OR n:Node)
        AND (
            $phenomenon = "" OR $phenomenon IS NULL
            OR toString(n.name) CONTAINS $phenomenon
            OR toString(n.semantic_description) CONTAINS $phenomenon
            OR toString(n.aliases) CONTAINS $phenomenon
        )
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT 100
        """

        nodes = []
        edges = []
        node_ids = set()

        with self.driver.session() as session:
            result = session.run(graph_query, phenomenon=phenomenon)

            for record in result:
                n = record["n"]
                r = record["r"]
                m = record["m"]

                def add_node(node):
                    if not node:
                        return None

                    # ✅ 核心修复：用业务字段做稳定ID（优先 name）
                    eid = node.get("id") or node.get("name")

                    if not eid:
                        eid = str(node.element_id)  # fallback

                    if eid in node_ids:
                        return eid

                    nodes.append({
                        "id": str(eid),
                        "name": node.get("name", "未知实体"),
                        "type": list(node.labels)[0] if node.labels else "Entity",
                        "semantic_description": node.get("semantic_description", ""),
                        "evidence_texts": node.get("evidence_texts", ""),
                        "aliases": node.get("aliases", "")
                    })

                    node_ids.add(str(eid))
                    return str(eid)

                source_id = add_node(n)
                target_id = add_node(m)

                if r and source_id and target_id:
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        # ✅ 正确获取关系类型
                        "label": type(r).__name__
                    })

        return nodes, edges


# 初始化 Neo4j 客户端
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")

try:
    neo4j_client = Neo4jGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print(f"成功连接到 Neo4j: {NEO4J_URI}")
except Exception as e:
    print(f"Neo4j 连接初始化失败: {e}")
    # 即使失败也创建一个实例，查询时会捕获异常并返回 Mock
    neo4j_client = Neo4jGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# ==========================================
# 2. 离线模型加载 (延迟加载以加快启动速度)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("LLM_MODEL_PATH", os.path.join(BASE_DIR, "models", "Qwen2-1.5B-Instruct"))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", os.path.join(BASE_DIR, "models", "bge-small-zh"))

_tokenizer = None
_model = None
_embedding_model = None


def get_llm():
    global _tokenizer, _model
    if _model is None:
        try:
            if os.path.exists(MODEL_PATH):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH, torch_dtype="auto", device_map="auto", local_files_only=True
                )
        except Exception as e:
            print(f"LLM 加载失败: {e}")
    return _tokenizer, _model


def get_embedding():
    global _embedding_model
    if _embedding_model is None:
        try:
            if os.path.exists(EMBEDDING_MODEL_PATH):
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        except Exception as e:
            print(f"Embedding 加载失败: {e}")
    return _embedding_model


# ==========================================
# 3. Prompt 构建方式 (Agentic Decision Pipeline)
# ==========================================
def build_prompt(user_message: str, context_texts: List[str], history: List[Dict[str, str]]):
    context_str = "\n".join(context_texts) if context_texts else "未检索到相关图谱信息。"

    sys_prompt = f"""你是一个专业的船舶故障诊断专家系统。请严格按照以下思考链路处理用户的问题，并以JSON格式输出。

【检索到的图谱证据】：
{context_str}

【处理链路要求】：
1. scoring（证据评分）：分析检索到的证据与用户症状的相关性和充分性。
2. confidence（置信度计算）：基于证据的充分性，给出一个0.0到1.0之间的置信度得分。
3. decision（决策核心）：
   - 高置信度 (>= 0.8) -> "answer" (给出明确的故障诊断和维修建议)
   - 中置信度 (0.4 ~ 0.79) -> "ask" (证据不足，向用户提出关键的排查追问)
   - 低置信度 (< 0.4) -> "refuse" (超出知识库范围，委婉拒绝或提示无法诊断)
4. reply（回复内容）：根据decision生成最终回复给用户的文本。

【输出格式要求】：
必须严格输出合法的JSON格式，包含以下4个字段，不要输出任何其他解释性文字：
{{
  "scoring": "对证据的分析过程...",
  "confidence": 0.85,
  "decision": "answer",
  "reply": "根据描述，初步诊断为..."
}}
"""
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})
    return messages


# ==========================================
# 4. FastAPI 完整接口代码
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


class GraphSearchRequest(BaseModel):
    keyword: str


@app.post("/api/graph/search")
async def graph_search_endpoint(request: GraphSearchRequest):
    keyword = request.keyword
    if not keyword:
        return {"nodes": [], "edges": []}

    # 默认 Mock 数据生成逻辑
    def get_mock_data(kw):
        if "主机" in kw:
            mock_nodes = [
                {"id": "n1", "name": "主机系统", "type": "设备"},
                {"id": "n2", "name": "燃油泵", "type": "部件"},
                {"id": "n3", "name": "启动困难", "type": "现象"},
                {"id": "n4", "name": "缸套", "type": "部件"},
            ]
            mock_edges = [
                {"source": "n1", "target": "n2", "label": "包含"},
                {"source": "n2", "target": "n3", "label": "可能导致"},
                {"source": "n1", "target": "n4", "label": "包含"},
            ]
        else:
            mock_nodes = [{"id": "m1", "name": f"{kw}相关设备", "type": "设备"}]
            mock_edges = []
        return mock_nodes, mock_edges

    try:
        # 1. 尝试从 Neo4j 检索
        nodes, edges = neo4j_client.query_phenomenon(keyword)

        # 2. 如果检索结果为空且连接正常，返回 Mock
        if not nodes:
            print(f"数据库中未找到 '{keyword}'，返回 Mock 数据")
            nodes, edges = get_mock_data(keyword)
            return {"nodes": nodes, "edges": edges, "source": "mock", "message": "未在数据库找到匹配项，已显示演示数据"}

        return {
            "nodes": nodes,
            "edges": edges,
            "source": "neo4j"
        }
    except Exception as e:
        print(f"Neo4j 连接或查询异常: {e}")
        # 即使报错也返回 Mock 数据，并附带错误信息
        nodes, edges = get_mock_data(keyword)
        return {
            "nodes": nodes,
            "edges": edges,
            "source": "mock_fallback",
            "error": str(e),
            "message": f"数据库连接异常，已自动切换为本地演示模式。错误详情: {str(e)}"
        }


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    # 模拟文件上传处理
    print(f"收到文件上传: {file.filename}")
    return {"status": "success", "filename": file.filename}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message

    # 1. Neo4j 检索
    try:
        nodes, edges = neo4j_client.query_phenomenon(user_input)
        context_texts = [f"【图谱关联】: {n['name']} ({n['type']})" for n in nodes[:5]]
    except:
        nodes, edges, context_texts = [], [], []

    # 2. 构建 Prompt
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    messages = build_prompt(user_input, context_texts, history_dicts)

    # 3. LLM 推理
    response_text = ""
    tokenizer, model = get_llm()
    if model is not None and tokenizer is not None:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"推理出错: {e}")
            response_text = '{"decision": "answer", "reply": "模型推理出错，请检查显存或模型配置。", "confidence": 0.0, "scoring": "推理异常"}'
    else:
        # Mock 数据逻辑
        if "启动" in user_input:
            response_text = '{"scoring": "发现启动困难相关证据", "confidence": 0.6, "decision": "ask", "reply": "请问主机启动时燃油压力是否正常？"}'
        elif "压力" in user_input:
            response_text = '{"scoring": "燃油压力异常指向燃油泵", "confidence": 0.9, "decision": "answer", "reply": "根据描述，初步诊断为燃油泵故障，建议检查燃油泵。\\n\\n维修步骤：\\n1. 关闭燃油阀门\\n2. 拆卸燃油泵\\n3. 检查泵体是否有磨损"}'
        else:
            response_text = '{"scoring": "未匹配到具体故障", "confidence": 0.2, "decision": "refuse", "reply": "抱歉，我没有在知识库中找到相关的故障信息，无法给出诊断建议。"}'

    # 4. 解析模型输出
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            result = {"decision": "answer", "reply": response_text, "confidence": 0.5, "scoring": "未按JSON格式输出"}
    except Exception as e:
        result = {"decision": "answer", "reply": response_text, "confidence": 0.5, "scoring": f"JSON解析异常: {e}"}

    res_type = result.get("decision", "answer")
    if res_type == "refuse":
        res_type = "answer"

    return {
        "type": res_type,
        "reply": result.get("reply", "抱歉，推理过程出错。"),
        "confidence": result.get("confidence", 0.85),
        "scoring": result.get("scoring", ""),
        "graph": {
            "nodes": nodes,
            "edges": edges
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
