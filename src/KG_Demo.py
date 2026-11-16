import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx

load_dotenv()

# --- STEP 1: Document Loader & Splitting ---
file_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.txt")
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found!")
    exit()

loader = TextLoader(file_path)
documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Step 1: Loaded {len(documents)} document(s) and split into {len(chunks)} chunks.")

# --- STEP 2: Entity & Relationship Extraction ---
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
graph_transformer = LLMGraphTransformer(llm=llm)

print("Step 2: Extracting entities and relationships using LLM...")
graph_documents = graph_transformer.convert_to_graph_documents(chunks)
print(f"Extracted {len(graph_documents[0].nodes)} nodes and {len(graph_documents[0].relationships)} relationships.")

# --- STEP 3: Graph Storage using NetworkX (Local In-Memory) ---
print("\nStep 3: Storing graph in local NetworkX graph...")
nx_graph = nx.DiGraph()

for graph_doc in graph_documents:
    for node in graph_doc.nodes:
        nx_graph.add_node(node.id, type=node.type)
    for rel in graph_doc.relationships:
        nx_graph.add_edge(rel.source.id, rel.target.id, relationship=rel.type)

print(f"Success! NetworkX graph created with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges.")

# --- STEP 4: Graph Query & Retrieval ---
print("\n--- STEP 4: Querying the Local Graph ---")

# Print all available nodes so we can see their exact names/casing
print("\nAvailable nodes in the graph:")
for node in nx_graph.nodes():
    print(f" - {node} (Type: {nx_graph.nodes[node].get('type')})")

# Let's search case-insensitively or find the node matching 'TechNova'
query_keyword = "tech"
print(f"\nSearching for nodes containing '{query_keyword}':")
matching_nodes = [n for n in nx_graph.nodes() if query_keyword.lower() in n.lower()]

for node in matching_nodes:
    print(f"\nOutgoing relationships from '{node}':")
    for neighbor in nx_graph.successors(node):
        rel_type = nx_graph.get_edge_data(node, neighbor)['relationship']
        print(f"  ({node}) --[{rel_type}]--> ({neighbor})")

# --- STEP 5: Graph RAG Answering (The Generation Phase) ---
print("\n--- STEP 5: Graph-Augmented Generation (KG-RAG) ---")

user_question = "What companies and projects are associated with Technova Corp?"

# 1. Gather context from our NetworkX graph for 'Technova Corp'
target_entity = "Technova Corp"
graph_context = []

if target_entity in nx_graph:
    for neighbor in nx_graph.successors(target_entity):
        rel_type = nx_graph.get_edge_data(target_entity, neighbor)['relationship']
        neighbor_type = nx_graph.nodes[neighbor].get('type', 'Entity')
        graph_context.append(f"- {target_entity} --[{rel_type}]--> {neighbor} ({neighbor_type})")

context_string = "\n".join(graph_context)

# 2. Construct the prompt with graph context injected
prompt = f"""You are an expert AI assistant. Answer the user's question using ONLY the provided Knowledge Graph context below.

Knowledge Graph Context:
{context_string}

User Question: {user_question}
Answer:"""

print("\nPrompt sent to LLM:\n", prompt)

# 3. Generate the final answer using the LLM
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
response = llm.invoke(prompt)

print(f"\n--- LLM RAG Response ---\n{response.content}")