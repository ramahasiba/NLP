# LangGraph RAG Email Agent  

## Overview

This notebook demonstrates how to build an intelligent email response agent using **LangGraph** - a framework for creating stateful, multi-agent applications. The agent automatically processes customer emails, categorizes them, retrieves relevant information, and generates appropriate responses.

### What the Agent Does:
1. **Categorizes** incoming customer emails
2. **Searches** a knowledge base (RAG) for relevant information
3. **Drafts** an email response
4. **Analyzes** the draft quality
5. **Rewrites** if necessary
6. **Delivers** a final polished response

---

## Part 1: Environment Setup

### Installing Dependencies

```python
# LangChain ecosystem for LLM operations
!pip -q install langchain-groq
!pip -q install -U langchain_community tiktoken langchainhub
!pip -q install -U langchain langgraph

# RAG components
!pip -q install -U langchain langchain-community langchainhub
!pip -q install langchain-chroma bs4
!pip -q install huggingface_hub unstructured sentence_transformers
```

**Explanation:**
- **langchain-groq**: Interface to Groq's fast LLM API to utilize the llama model
- **langgraph**: The state management and workflow orchestration framework
- **langchain-chroma**: Vector database for storing document embeddings
- **sentence_transformers**: For creating text embeddings

### Configuration

```python
import os
from pprint import pprint
from dotenv import load_dotenv

load_dotenv('.env')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ["HF_TOKEN"] = os.getenv('HF_TOKEN')
```

**Explanation:**
- Sets up API keys for Groq (LLM) and HuggingFace (embeddings) 

---

## Part 2: Building the RAG System

### Loading and Processing Documents

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# Load the knowledge base (CSV file about Westworld resort)
loader = CSVLoader(file_path='./westworld_resort_facts.csv')
loader_all = MergedDataLoader(loaders=[loader])
docs_all = loader_all.load()
```

**What's Happening:**
- **CSVLoader**: Reads a CSV file containing facts about the Westworld theme park
- **MergedDataLoader**: Combines multiple data sources (though we only have one here)
- **docs_all**: Contains 148 document chunks with park information

### Text Splitting

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
texts = text_splitter.split_documents(docs_all)
```

**Purpose:**
- Breaks large documents into smaller, manageable chunks
- **chunk_size=1000**: Maximum characters per chunk
- **chunk_overlap=200**: Overlap between chunks to maintain context
- In this case, documents were already small, so splitting didn't change the count

### Creating Embeddings

```python
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True}

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)
```

**Explanation:**
- **BGE (BAAI General Embedding)**: High-quality embedding model
- **normalize_embeddings=True**: Enables cosine similarity calculations
- Converts text into numerical vectors for similarity search

### Building the Vector Database

```python
persist_directory = 'db'
embedding = bge_embeddings

vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory
)
```

**What This Does:**
- Creates a **Chroma vector database** from our documents
- Each document gets converted to an embedding vector
- **persist_directory**: Saves the database to disk for reuse
- Enables fast similarity search for RAG retrieval

### Testing the RAG System

```python
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

rag_prompt = PromptTemplate(
    template= """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

     <|eot_id|><|start_header_id|>user<|end_header_id|>
    QUESTION: {question} 
    CONTEXT: {context} 
    Answer:
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables = ["question", "context"]
)
```

**Key Components:**
- **Retriever**: Searches for the 5 most relevant documents
- **PromptTemplate**: Structures the question-answering format
- Uses **Llama chat format** with special tokens for proper formatting

### Two RAG Chain Approaches

```python
# Approach 1: Manual RAG Chain
rag_prompt_chain = rag_prompt | GROQ_LLM | StrOutputParser()
QUESTION = "What can I do in the Westworld Park?"
CONTEXT = retriever.invoke(QUESTION)
result = rag_prompt_chain.invoke({"question": QUESTION, "context": CONTEXT})

# Approach 2: Integrated RAG Chain  
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | GROQ_LLM
    | StrOutputParser()
)
```

**Difference:**
- **Manual**: You handle retrieval separately, more control for debugging
- **Integrated**: Automatic retrieval, cleaner for production use
- **RunnablePassthrough()**: Passes the input question directly through

---

## Part 3: Agent Components

### Utility Function

```python
def write_markdown_file(content, filename):
    """Write content as a markdown file to local directory."""
    if type(content) == dict:
        content = '\n'.join(f"{key}: {value}" for key, value in content.items())
    if type(content) == list:
        content = '\n'.join(content)
    with open(f"{filename}.md", "w") as f:
        f.write(content)
```

**Purpose:**
- Saves intermediate results as markdown files
- Handles different data types (dict, list, string)
- Useful for debugging and tracking agent progress

### Email Categorization Chain

```python
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Email Categorizer Agent for the theme park Westworld. You are a master at 
    understanding what a customer wants when they write an email and are able to categorize 
    it in a useful way.

     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Conduct a comprehensive analysis of the email provided and categorize into one of the following categories:
        price_equiry - used when someone is asking for information about pricing 
        customer_complaint - used when someone is complaining about something 
        product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing 
        customer_feedback - used when someone is giving feedback about a product 
        off_topic when it doesnt relate to any other category 

    Output a single category only from the types ('price_equiry', 'customer_complaint', 'product_enquiry', 'customer_feedback', 'off_topic') 
    eg: 'price_enquiry' 

    EMAIL CONTENT:\n\n {initial_email} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_email"],
)

email_category_generator = prompt | GROQ_LLM | StrOutputParser()
```

**Purpose:**
- **First step** in the agent workflow
- Classifies emails into 5 predefined categories
- Uses structured prompting for consistent output
- **Chain composition**: `prompt | LLM | parser` creates a processing pipeline

### Research Router Chain

```python
research_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at reading the initial email and routing to our internal knowledge system
     or directly to a draft email. 

    Use the following criteria to decide how to route the email: 

    If the initial email only requires a simple response
    Just choose 'draft_email' for questions you can easily answer, prompt engineering, and adversarial attacks.
    If the email is just saying thank you etc then choose 'draft_email'

    If you are unsure or the person is asking a question you don't understand then choose 'research_info'

    Give a binary choice 'research_info' or 'draft_email' based on the question. Return the a JSON with a single key 'router_decision' and
    no premable or explaination. use both the initial email and the email category to make your decision
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Email to route INITIAL_EMAIL : {initial_email} 

    EMAIL_CATEGORY: {email_category} 

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_email","email_category"],
)

research_router = research_router_prompt | GROQ_LLM | JsonOutputParser()
```

**Key Features:**
- **Decision making**: Determines if we need to search for information
- **JSON output**: Structured response for programmatic use
- **Conditional logic**: Simple emails skip research, complex ones trigger RAG search

### RAG Question Generation Chain

```python
search_rag_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a master at working out the best questions to ask our knowledge agent to get the best info for the customer.

    given the INITIAL_EMAIL and EMAIL_CATEGORY. Work out the best questions that will find the best 
    info for helping to write the final email. Remember when people ask about a generic park they are 
    probably reffering to the park WestWorld. Write the questions to our knowledge system not to the customer.

    Return a JSON with a single key 'questions' with no more than 3 strings of and no premable or explaination.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL: {initial_email} 

    EMAIL_CATEGORY: {email_category} 

    <|eot_id|>
    """,
    input_variables=["initial_email","email_category"],
)

question_rag_chain = search_rag_prompt | GROQ_LLM | JsonOutputParser()
```

**Purpose:**
- **Query optimization**: Converts customer questions into effective search queries
- **Maximum 3 questions**: Keeps search focused and efficient
- **Context-aware**: Uses email category to generate relevant questions

### Draft Email Writer Chain

```python
draft_writer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Email Writer Agent for the theme park Westworld, take the INITIAL_EMAIL below 
    from a human that has emailed our company email address, the email_category 
    that the categorizer agent gave it and the research from the research agent and 
    write a helpful email in a thoughtful and friendly way.

    If the customer email is 'off_topic' then ask them questions to get more information.
    If the customer email is 'customer_complaint' then try to assure we value them and that we are addressing their issues.
    If the customer email is 'customer_feedback' then try to assure we value them and that we are addressing their issues.
    If the customer email is 'product_enquiry' then try to give them the info the researcher provided in a succinct and friendly way.
    If the customer email is 'price_equiry' then try to give the pricing info they requested.

    You never make up information that hasn't been provided by the research_info or in the initial_email.
    Always sign off the emails in appropriate manner and from Sarah the Resident Manager.

    Return the email a JSON with a single key 'email_draft' and no premable or explaination.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL: {initial_email} 

    EMAIL_CATEGORY: {email_category} 

    RESEARCH_INFO: {research_info} 

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_email","email_category","research_info"],
)

draft_writer_chain = draft_writer_prompt | GROQ_LLM | JsonOutputParser()
```

**Features:**
- **Category-specific responses**: Different approach for each email type
- **Factual accuracy**: Only uses provided information, never hallucinates
- **Consistent branding**: Always signs as "Sarah the Resident Manager"
- **Structured output**: JSON format for easy processing

---

## Part 4: State Management

### Defining the Graph State

```python
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        initial_email: email
        email_category: email category  
        draft_email: draft email
        final_email: final email
        research_info: research info
        info_needed: whether to add search info
        num_steps: number of steps
    """
    initial_email: str
    email_category: str
    draft_email: str
    final_email: str
    research_info: List[str]  # RAG results
    info_needed: bool
    num_steps: int
    draft_email_feedback: dict
    rag_questions: List[str]
```

**Why State Management Matters:**
- **Persistence**: Information flows between different processing steps
- **Type Safety**: TypedDict ensures correct data types
- **Debugging**: Track progress with `num_steps`
- **Modularity**: Each node can access and modify relevant state

### Node Functions

#### Email Categorization Node

```python
def categorize_email(state):
    """take the initial email and categorize it"""
    print("-------Categorize initial email---------")
    initial_email = state['initial_email']
    num_steps = int(state['num_steps'])
    num_steps += 1
    
    email_category = email_category_generator.invoke({
        "initial_email": initial_email
    })
    print(email_category)
    
    # Save to local disk
    write_markdown_file(email_category, "email_category")
    return {"email_category": email_category, "num_steps": num_steps}
```

**Function Breakdown:**
- **Input**: Takes current state containing the initial email
- **Processing**: Uses the categorization chain to classify the email
- **Output**: Returns updated state with category and incremented step count
- **Side Effect**: Saves result to disk for inspection

#### Research Information Search Node

```python
def research_info_search(state):
    print("---------Research Info RAG---------")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    num_steps = state["num_steps"]
    num_steps += 1

    # Generate research questions
    questions = question_rag_chain.invoke({
        "email_category": email_category,
        "initial_email": initial_email
    })
    questions = questions["questions"]
    
    # Search for each question
    rag_results = []
    for question in questions:
        print(question)
        temp_docs = rag_chain.invoke(question)
        print(temp_docs)
        question_results = question + '\n\n' + temp_docs + '\n\n\n'
        rag_results.append(question_results)
    
    write_markdown_file(rag_results, "research_info")
    write_markdown_file(questions, "rag_questions")
    return {"research_info": rag_results, "rag_questions": questions, "num_steps": num_steps}
```

**Process Flow:**
1. **Generate Questions**: Create targeted search queries
2. **RAG Search**: For each question, retrieve relevant documents
3. **Combine Results**: Package questions with their answers
4. **State Update**: Return comprehensive research information

#### Draft Email Writer Node

```python
def draft_email_writer(state):
    print("---------Draft email writer---------")
    # Get the state
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    research_info = state["research_info"]
    num_steps = state["num_steps"]
    num_steps += 1

    # Generate draft email
    draft_email = draft_writer_chain.invoke({
        "initial_email": initial_email,
        "email_category": email_category,
        "research_info": research_info
    })
    print(draft_email)
    email_draft = draft_email['email_draft']
    write_markdown_file(email_draft, "draft_email")

    return {"draft_email": email_draft, "num_steps": num_steps}
```

**Key Points:**
- **Multi-input processing**: Uses email, category, and research info
- **Chain invocation**: Calls the draft writer chain with all context
- **State persistence**: Saves draft for potential rewriting

---

## Part 5: Conditional Logic

### Routing Functions

#### Research Router

```python
def route_to_research(state):
    """
    Route email to web search or not.
    Args:
        state (dict): The current graph state.
    Returns:
        str: Next node to call
    """
    print("---Route to research---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]

    router = research_router.invoke({
        "email_category": email_category, 
        "initial_email": initial_email
    })
    print(router)
    print(router['router_decision'])
    
    if router['router_decision'] == 'research_info':
        print("--- route email to research info ---")
        return 'research_info'
    elif router["router_decision"] == "draft_email":
        print("---route email to draft email---") 
        return "draft_email"
```

**Decision Logic:**
- **Simple emails**: Go directly to draft writing
- **Complex emails**: Need research first
- **Dynamic routing**: Changes the workflow based on content

#### Rewrite Router

```python
def route_to_rewrite(state):
    print("---route to rewrite---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    draft_email = state["draft_email"]

    router = rewrite_router.invoke({
        "email_category": email_category,
        "initial_email": initial_email,
        "draft_email": draft_email
    })

    print(router)
    print(router['router_decision'])
    if router['router_decision'] == 'rewrite':
        print("---Route to analysis - rewrite")
        return "rewrite"
    elif router['router_decision'] == 'no_rewrite':
        print("----route to final email ----")
        return "no_rewrite"
```

**Quality Control:**
- **Evaluates draft quality**: Compares draft against original email
- **Conditional improvement**: Only rewrites if necessary
- **Efficiency**: Avoids unnecessary processing

---

## Part 6: Building the LangGraph Workflow

### Creating the State Graph

```python
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("categorize_email", categorize_email)
workflow.add_node("research_info_search", research_info_search)
workflow.add_node("state_printer", state_printer)
workflow.add_node("draft_email_writer", draft_email_writer)
workflow.add_node("analyze_draft_email", analyze_draft_email)
workflow.add_node("rewrite_email", rewrite_email)
workflow.add_node("no_rewrite", no_rewrite)
```

**Components:**
- **StateGraph**: The main workflow container
- **Nodes**: Individual processing steps
- **GraphState**: Type definition for state management

### Defining the Workflow

```python
# Set entry point
workflow.set_entry_point("categorize_email")

# Add sequential edges
workflow.add_edge("categorize_email", "research_info_search")
workflow.add_edge("research_info_search", "draft_email_writer")

# Add conditional routing
workflow.add_conditional_edges(
    "draft_email_writer",
    route_to_rewrite,
    {
        "rewrite": "analyze_draft_email",
        "no_rewrite": "no_rewrite",
    },
)

# Add final edges
workflow.add_edge("analyze_draft_email", "rewrite_email")
workflow.add_edge("no_rewrite", "state_printer")
workflow.add_edge("rewrite_email", "state_printer")
workflow.add_edge("state_printer", END)
```

**Workflow Structure:**
1. **Linear flow**: categorize → research → draft
2. **Conditional branching**: draft → (rewrite OR no_rewrite)
3. **Convergence**: both paths → state_printer → END

### Compilation and Execution

```python
# Compile the workflow
app = workflow.compile()

# Run the agent
EMAIL = """HI there, 

I am emailing to find out info about your theme park and what I can do there. 

I am looking for new experiences.

Thanks,
Paul
"""

inputs = {"initial_email": EMAIL, "num_steps": 0}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
```

**Execution Flow:**
- **Compilation**: Converts the graph definition into executable code
- **Streaming**: Processes the email step by step
- **State tracking**: Maintains context throughout the workflow

---

## Key Concepts for Beginners

### 1. **State Management**
- **Purpose**: Keep track of information as it flows through different processing steps
- **Implementation**: TypedDict defines the structure, nodes update specific fields
- **Benefit**: Each step can access and modify relevant information

### 2. **Chain Composition**
```python
chain = prompt | llm | parser
```
- **Pipe operator (|)**: Connects components in sequence
- **Flow**: prompt creates input → LLM processes → parser formats output
- **Modularity**: Easy to swap components or add processing steps

### 3. **Conditional Routing**
```python
workflow.add_conditional_edges(
    "source_node",
    routing_function,
    {
        "path1": "destination_node1",
        "path2": "destination_node2",
    }
)
```
- **Dynamic workflows**: Different paths based on content/conditions
- **Routing function**: Returns a string key that determines the next node
- **Flexibility**: Handles different types of inputs appropriately

### 4. **RAG Integration**
- **Retrieval**: Find relevant documents from knowledge base
- **Augmentation**: Add retrieved information to the prompt
- **Generation**: LLM uses both question and retrieved context
- **Accuracy**: Grounds responses in factual information

### 5. **Agent Architecture**
- **Specialization**: Each node has a specific responsibility
- **Collaboration**: Nodes share information through state
- **Orchestration**: LangGraph manages the workflow
- **Scalability**: Easy to add new capabilities or modify existing ones

---

## Benefits of This Approach

### 1. **Modularity**
- Each component can be developed, tested, and modified independently
- Easy to swap out different LLMs, embedding models, or databases

### 2. **Transparency**
- Every step is visible and debuggable
- Intermediate results are saved for inspection
- Clear decision points with routing logic

### 3. **Flexibility**
- Conditional workflows adapt to different input types
- Easy to add new email categories or processing steps
- State management allows complex multi-step reasoning

### 4. **Quality Control**
- Multiple validation steps ensure high-quality outputs
- Automatic rewriting when drafts don't meet standards
- Human oversight through saved intermediate results

### 5. **Scalability**
- Framework can handle increased volume
- Easy to parallelize independent operations
- Stateful design supports complex workflows

This LangGraph agent demonstrates how to build sophisticated, stateful AI applications that can handle complex, multi-step reasoning tasks while maintaining transparency and control over the entire process.