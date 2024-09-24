import subprocess
import os
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Warning control
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

# Connect to the knowledge graph using LangChain
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_llama3_instruct(input_text):
    # Command to run the llama3:instruct model using ollama
    command = ['ollama', 'run', 'CustomLlama3:latest']
    # Start subprocess 
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stdout, stderr = process.communicate(input=input_text)
    # Check success
    if process.returncode == 0:
        return stdout
    else:
        return f"Error: {stderr}"
    
def calculate_embedding(text):
    embedding = model.encode(text)
    return embedding

def query_context(question_embedding):
    # Query to find context based on question embedding
    nodes = kg.query("""
    CALL db.index.vector.queryNodes(
        'fullindex', 
        10, 
        $question_embedding
        ) YIELD node AS n, score
        RETURN n.node_name AS name, n.category AS category, n.pharmacodynamics AS pharmacodynamics,
           n.group AS group, n.half_life AS half_life, n.indication AS indication,
           n.mechanism_of_action AS mechanism_of_action, n.protein_binding AS protein_binding,
           n.state AS state, n.node_index as node_index, n.node_type AS Type, n.node_source AS Source,
           n.mayo_causes AS mayo_causes, 
           n.mayo_complications AS mayo_complications, 
           n.mayo_risk_factors AS mayo_risk_factors, 
           n.mayo_symptoms AS mayo_symptoms, 
           n.mondo_definition AS mondo_definition, score
    """, params={"question_embedding": question_embedding})
    return nodes

def query_context_cypher(question):
    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the schema. Do not use any other relationship types or properties that are not provided.
    Database Schema:
        Node relationships:
            (:Node)-[:RELATION]->(:Node)

        Node properties:
            (:Node {node_id: INTEGER, node_index: INTEGER, node_name: STRING, node_source: STRING, node_type: STRING, atc_1: STRING, atc_2: STRING, atc_3: STRING, atc_4: STRING, category: STRING, description: STRING, group: STRING, indication: STRING, mechanism_of_action: STRING, pharmacodynamics: STRING, state: STRING,mondo_definition: STRING, orphanet_definition: STRING, umls_description: STRING, umls_description: STRING, mayo_complications: STRING, mayo_prevention: STRING, mayo_risk_factors: STRING, mayo_symptoms: STRING})    
    
        Relationship Properties:
            [:RELATION {display_relation: STRING, relation: STRING }]

    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    In the return clause, don't return the entire node, but only its attributes

     Examples: Here are a few examples of generated Cypher statements for particular questions:
    # What's associated with "malaria" AND  "duodenal ulcer"
    MATCH (n)-->(m)-->(r)
    WHERE  n.node_name contains "malaria" AND  r.node_name contains"duodenal ulcer"
    RETURN m.node_name, m.category, m.pharmacodynamics,
        m.group, m.half_life, m.indication,
        m.mechanism_of_action, m.protein_binding,
        m.state, m.node_index, m.mayo_causes, m.mayo_complications, m.mayo_risk_factors, m.mayo_symptoms, m.mondo_definition  limit 3
    
    # Out of the given list, which Gene is associated with Takayasu arteritis and cancer. Given list is: SHTN1, HLA-B, SLC14A2, BTBD9, DTNB
    MATCH (r)-->(n)-->(m)
    WHERE n.node_name IN ["SHTN1", "HLA-B", "SLC14A2", "BTBD9", "DTNB"] and m.node_name Contains "Takayasu arteritis" and r.node_name contains "cancer"
    RETURN n.node_name, n.node_type LIMIT 3

    #What are the causes of cancer?
    MATCH (n)
    WHERE n.node_name CONTAINS "cancer" 
    RETURN n.node_name, n.category, n.pharmacodynamics,
        n.group, n.half_life, n.indication,
        n.mechanism_of_action, n.protein_binding,
        n.state, n.node_index,n.mayo_causes, n.mayo_complications, n.mayo_risk_factors, n.mayo_symptoms, n.mondo_definition
    LIMIT 3

    #What are the symptoms of cancer?
    MATCH (n)
    WHERE n.node_name CONTAINS "cancer" 
    RETURN n.node_name, n.category, n.pharmacodynamics,
        n.group, n.half_life, n.indication,
        n.mechanism_of_action, n.protein_binding,
        n.state, n.node_index,n.mayo_causes, n.mayo_complications, n.mayo_risk_factors, n.mayo_symptoms, n.mondo_definition
    LIMIT 3

    # Tell me about Copper
    MATCH (n)
    WHERE n.node_name CONTAINS "Copper" 
    RETURN n.node_name, n.category, n.pharmacodynamics,
        n.group, n.half_life, n.indication,
        n.mechanism_of_action, n.protein_binding,
        n.state, n.node_index,n.mayo_causes, n.mayo_complications, n.mayo_risk_factors, n.mayo_symptoms, n.mondo_definition
    LIMIT 3

    """ +f"""
    The question is:{question}"""
    resp = query_llama3_instruct(CYPHER_GENERATION_TEMPLATE)
    print("\nQuery:")
    print(resp)
    node=kg.query(resp)
    print("contex:")
    if not node:
        print("Errore: Nessun nodo trovato.")
    else:
        for a in node:
            print(a)
    return node

def main():
    question = input("Inserisci la domanda: ")
   # Calculate the embedding
    question_embedding = calculate_embedding(question)
    context=query_context(question_embedding)
    context.sort(key=lambda obj: obj['score'], reverse=True)

    print("\nContesto:")
    for node in context[:3]:
       print(node['name'], node['score'])
    top_context = context[:2]
    # Concatenate context and question
    input_text = f"""TASK: You are an AI assistant. Respond in paragraph using only the context provided, giving more relevance to the one with the higher score.
    QUESTION: {question}
    CONTEXT: {top_context} """

    # Get the response from the model
    response = query_llama3_instruct(input_text)
    print("\nRisposta dal modello:")
    print(response)
   

    
    
    # Use llama for the query
    node=query_context_cypher(question)
    # Concatenate context and question
    input_text = f"""TASK: You are an AI assistant. Respond in paragraph using only the context provided, ignoring the 'embedding'.
    QUESTION: {question}
    CONTEXT: {node} """

    resp = query_llama3_instruct(input_text)
    print("\nRisposta dal modello cypher:")
    print(resp)

if __name__ == "__main__":
    main()
