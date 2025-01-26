import os
import json
import requests
import time
import traceback
import decimal
import concurrent.futures
import queue

from flask import g
from openai import OpenAI, AssistantEventHandler
from pinecone import Pinecone

# SQLAlchemy models
from models.sql_models import (
    Users, 
    OpenAIUsageLog,
    Conditions,
    ConditionEmbedding,
    NexusTags,
    Tag,
    ServicePeriod
)

# Custom cost wrappers
from helpers.llm_wrappers import call_openai_chat_create, call_openai_embeddings

###############################################################################
# 1. ENV & GLOBAL SETUP
###############################################################################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"  # Adjust for your Pinecone region if needed

INDEX_NAME_CFR = "38-cfr-index"
INDEX_NAME_M21 = "m21-index"

EMBEDDING_MODEL_SMALL = "text-embedding-3-small"
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)
assistant_id = "asst_DPtDOmgeV83hOa0MWupf0qgw"  # Your Assistant's ID

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_cfr = pc.Index(INDEX_NAME_CFR)
index_m21 = pc.Index(INDEX_NAME_M21)

# URLs for Azure-stored JSON chunks
PART_3_URL = "https://vetdoxstorage.blob.core.windows.net/vector-text/part_3_flattened.json"
PART_4_URL = "https://vetdoxstorage.blob.core.windows.net/vector-text/part_4_flattened.json"
M21_1_URL = "https://vetdoxstorage.blob.core.windows.net/vector-text/m21_1_chunked3k.json"
M21_5_URL = "https://vetdoxstorage.blob.core.windows.net/vector-text/m21_5_chunked3k.json"

###############################################################################
# 2. QUERY CLEANUP
###############################################################################
def clean_up_query_with_llm(user_id: int, user_query: str) -> str:
    """
    Uses an OpenAI LLM to rewrite the user query in a more standardized,
    formal, or clarified way—removing slang, expanding contractions, etc.
    """
    print(f"[DEBUG] clean_up_query_with_llm called with user_id={user_id}, user_query={user_query}")
    system_message = (
        "You are a helpful assistant that rewrites user queries for better text embeddings. "
        "Expand or remove contractions, fix grammatical errors, and keep the original meaning. "
        "Be concise and ensure the question is still natural and complete. You will rewrite it "
        "professionally as if talking directly to a VA rater who could answer the question. "
        "Remove sentences not relevant to the question."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]

    response = call_openai_chat_create(
        user_id=user_id,
        model="gpt-4o",
        messages=messages,
        temperature=0
    )

    cleaned_query = response.choices[0].message.content
    print(f"[DEBUG] clean_up_query_with_llm returning: {cleaned_query}")
    return cleaned_query.strip()

###############################################################################
# 3. EMBEDDING FUNCTIONS
###############################################################################
def get_embedding_small(user_id: int, text: str) -> list:
    print(f"[DEBUG] get_embedding_small called with user_id={user_id}, text={text[:60]}...")
    cost_rate = decimal.Decimal("0.00000002")  # $0.02 / 1M tokens
    response = call_openai_embeddings(
        user_id=user_id,
        input_text=text,
        model=EMBEDDING_MODEL_SMALL,
        cost_per_token=cost_rate
    )
    emb = response.data[0].embedding
    print("[DEBUG] get_embedding_small returning embedding of length", len(emb))
    return emb

def get_embedding_large(user_id: int, text: str) -> list:
    print(f"[DEBUG] get_embedding_large called with user_id={user_id}, text={text[:60]}...")
    cost_rate = decimal.Decimal("0.00000013")  # $0.13 / 1M tokens
    response = call_openai_embeddings(
        user_id=user_id,
        input_text=text,
        model=EMBEDDING_MODEL_LARGE,
        cost_per_token=cost_rate
    )
    emb = response.data[0].embedding
    print("[DEBUG] get_embedding_large returning embedding of length", len(emb))
    return emb

###############################################################################
# 4. HELPER FUNCTIONS TO FETCH CFR / M21 TEXT
###############################################################################
def fetch_matches_content(search_results, max_workers=3) -> list:
    print("[DEBUG] fetch_matches_content called.")
    matches = search_results.get("matches", [])
    print(f"[DEBUG] Number of matches: {len(matches)}")

    def get_section_text(section_number: str, part_number: str) -> str:
        if part_number == "3":
            url = PART_3_URL
        elif part_number == "4":
            url = PART_4_URL
        else:
            return None

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                meta = item.get("metadata", {})
                if meta.get("section_number") == section_number:
                    return item.get("text")
        except Exception as e:
            print(f"[ERROR] Unable to fetch or parse JSON from {url}: {e}")
        return None

    matching_texts = []
    for match in matches:
        metadata = match.get("metadata", {})
        section_num = metadata.get("section_number")
        part_number = metadata.get("part_number")
        print(f"[DEBUG] fetch_matches_content -> section_num={section_num}, part_number={part_number}")
        if not section_num or not part_number:
            continue

        section_text = get_section_text(section_num, part_number)
        matching_texts.append({
            "section_number": section_num,
            "matching_text": section_text
        })

    print("[DEBUG] fetch_matches_content returning # of items:", len(matching_texts))
    return matching_texts

def fetch_matches_content_m21(search_results, max_workers=3) -> list:
    print("[DEBUG] fetch_matches_content_m21 called.")
    matches = search_results.get("matches", [])
    print(f"[DEBUG] Number of matches: {len(matches)}")

    def get_article_text(article_number: str, manual: str) -> str:
        if manual == "M21-1":
            url = M21_1_URL
        elif manual == "M21-5":
            url = M21_5_URL
        else:
            return None

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                meta = item.get("metadata", {})
                if meta.get("article_number") == article_number:
                    return item.get("text")
        except Exception as e:
            print(f"[ERROR] Unable to fetch or parse JSON from {url}: {e}")
        return None

    matching_texts = []
    for match in matches:
        metadata = match.get("metadata", {})
        article_num = metadata.get("article_number")
        manual_val = metadata.get("manual")
        print(f"[DEBUG] fetch_matches_content_m21 -> article_num={article_num}, manual={manual_val}")
        if not article_num or not manual_val:
            continue

        article_text = get_article_text(article_num, manual_val)
        matching_texts.append({
            "article_number": article_num,
            "matching_text": article_text
        })

    print("[DEBUG] fetch_matches_content_m21 returning # of items:", len(matching_texts))
    return matching_texts

###############################################################################
# 5. PINECONE SEARCH WRAPPERS
###############################################################################
def search_cfr_documents(user_id, query: str, top_k: int = 3) -> str:
    print(f"[DEBUG] search_cfr_documents called with user_id={user_id}, query={query}")
    cleaned_query = clean_up_query_with_llm(user_id, query)
    query_emb = get_embedding_small(user_id, cleaned_query)
    results = index_cfr.query(vector=query_emb, top_k=top_k, include_metadata=True)

    matching_sections = fetch_matches_content(results, max_workers=3)
    if not matching_sections:
        return "No sections found (CFR)."

    references_str = ""
    for item in matching_sections:
        sec_num = item["section_number"]
        text_snippet = item["matching_text"] or "N/A"
        references_str += f"\n---\nSection {sec_num}:\n{text_snippet}\n"
    print("[DEBUG] search_cfr_documents references_str:", references_str[:200], "...")
    return references_str.strip()

def search_m21_documents(user_id, query: str, top_k: int = 3) -> str:
    print(f"[DEBUG] search_m21_documents called with user_id={user_id}, query={query}")
    cleaned_query = clean_up_query_with_llm(user_id, query)
    query_emb = get_embedding_small(user_id, cleaned_query)
    results = index_m21.query(vector=query_emb, top_k=top_k, include_metadata=True)

    matching_articles = fetch_matches_content_m21(results, max_workers=3)
    if not matching_articles:
        return "No articles found (M21)."

    references_str = ""
    for item in matching_articles:
        article_num = item["article_number"]
        text_snippet = item["matching_text"] or "N/A"
        references_str += f"\n---\nArticle {article_num}:\n{text_snippet}\n"
    print("[DEBUG] search_m21_documents references_str:", references_str[:200], "...")
    return references_str.strip()

###############################################################################
# 6. ADDITIONAL CONDITION-SEARCH TOOLS
###############################################################################
def list_user_conditions(user_id: int) -> str:
    print(f"[DEBUG] list_user_conditions called with user_id={user_id}")
    session = g.session
    user_conditions = session.query(Conditions).filter(Conditions.user_id == user_id).all()

    if not user_conditions:
        return f"No conditions found for user_id={user_id}."

    results_list = []
    for cond in user_conditions:
        data = {
            "condition_id": cond.condition_id,
            "service_connected": cond.service_connected,
            "user_id": cond.user_id,
            "file_id": cond.file_id,
            "page_number": cond.page_number,
            "condition_name": cond.condition_name,
            "date_of_visit": cond.date_of_visit.isoformat() if cond.date_of_visit else None,
            "medical_professionals": cond.medical_professionals,
            "medications_list": cond.medications_list,
            "treatments": cond.treatments,
            "findings": cond.findings,
            "comments": cond.comments,
            "is_ratable": cond.is_ratable,
            "in_service": cond.in_service,
        }
        results_list.append(data)

    return json.dumps(results_list, indent=2, default=str)

def semantic_search_user_conditions(user_id: int, query_text: str, limit: int = 10) -> str:
    print(f"[DEBUG] semantic_search_user_conditions called with user_id={user_id}, query_text={query_text}, limit={limit}")
    session = g.session
    query_vec = get_embedding_large(user_id, query_text)

    results = (
        session.query(Conditions)
        .join(ConditionEmbedding, Conditions.condition_id == ConditionEmbedding.condition_id)
        .filter(Conditions.user_id == user_id)
        .order_by(ConditionEmbedding.embedding.op("<->")(query_vec))
        .limit(limit)
        .all()
    )

    if not results:
        return f"No semantically similar conditions found for user_id={user_id}."

    results_list = []
    for cond in results:
        data = {
            "condition_id": cond.condition_id,
            "condition_name": cond.condition_name,
            "service_connected": cond.service_connected,
            "in_service": cond.in_service,
            "date_of_visit": cond.date_of_visit.isoformat() if cond.date_of_visit else None,
            "medications_list": cond.medications_list,
            "treatments": cond.treatments,
            "findings": cond.findings,
            "comments": cond.comments,
        }
        results_list.append(data)

    print(f"[DEBUG] semantic_search_user_conditions returning {len(results_list)} results.")
    return json.dumps(results_list, indent=2, default=str)

def list_nexus_conditions(user_id: int) -> str:
    print(f"[DEBUG] list_nexus_conditions called with user_id={user_id}")
    session = g.session

    nexus_list = (
        session.query(NexusTags)
        .join(Tag, NexusTags.tag_id == Tag.tag_id)
        .filter(NexusTags.user_id == user_id)
        .filter(NexusTags.revoked_at.is_(None))
        .all()
    )

    if not nexus_list:
        return f"No nexus tags found for user_id={user_id}."

    results_list = []
    for nx in nexus_list:
        data = {
            "nexus_tags_id": nx.nexus_tags_id,
            "discovered_at": nx.discovered_at.isoformat() if nx.discovered_at else None,
            "revoked_at": nx.revoked_at.isoformat() if nx.revoked_at else None,
            "tag_id": nx.tag_id,
            "disability_name": nx.tag.disability_name,
            "description": nx.tag.description,
        }
        results_list.append(data)

    print(f"[DEBUG] list_nexus_conditions returning {len(results_list)} results.")
    return json.dumps(results_list, indent=2, default=str)


###############################################################################
# 7. EVENT HANDLER & MAIN STREAMING FUNCTION
###############################################################################
class MyEventHandler(AssistantEventHandler):
    def __init__(self, user_id, db_session):
        print("[DEBUG] MyEventHandler __init__ called.")
        super().__init__()
        self.user_id = user_id
        self.db_session = db_session
        
        self.final_text_chunks = []
        self.run_info = None
        
        # For real-time streaming
        self._token_queue = queue.Queue()
        self._finished = False
        
    def on_text_created(self, text):
        """
        Called once at the beginning with an initial text object.
        """
        string_val = str(text)
        print("[DEBUG] on_text_created:", string_val[:80], "..." if len(string_val) > 80 else "")
        if string_val:
            self.final_text_chunks.append(string_val)
            self._token_queue.put(string_val)
        
    def on_text_delta(self, delta, snapshot):
        """
        Called multiple times, each chunk is appended for streaming.
        """
        string_val = str(delta.value) if delta.value else ""
        print("[DEBUG] on_text_delta:", string_val[:80], "..." if len(string_val) > 80 else "")
        if string_val:
            self.final_text_chunks.append(string_val)
            self._token_queue.put(string_val)
    
    def on_tool_call_created(self, tool_call):
        """
        Called when the assistant requests a tool/function.
        """
        print("[DEBUG] on_tool_call_created:", tool_call)
        import json
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        
        print(f"[DEBUG] Tool call requested: {function_name} with args={function_args}")
        args = json.loads(function_args)
        result_str = None

        # Dispatch to local search functions, etc.
        if function_name == "search_cfr_documents":
            query_text = args["query"]
            top_k_arg = args.get("top_k", 3)
            result_str = search_cfr_documents(self.user_id, query_text, top_k=top_k_arg)

        elif function_name == "search_m21_documents":
            query_text = args["query"]
            top_k_arg = args.get("top_k", 3)
            result_str = search_m21_documents(self.user_id, query_text, top_k=top_k_arg)

        elif function_name == "list_user_conditions":
            user_id_arg = args["user_id"]
            result_str = list_user_conditions(user_id_arg)

        elif function_name == "list_nexus_conditions":
            user_id_arg = args["user_id"]
            result_str = list_nexus_conditions(user_id_arg)

        elif function_name == "semantic_search_user_conditions":
            user_id_arg = args["user_id"]
            query_txt = args["query_text"]
            limit_arg = args.get("limit", 10)
            result_str = semantic_search_user_conditions(user_id_arg, query_txt, limit_arg)

        else:
            result_str = f"No implementation for tool '{function_name}'."

        # Return the result to the LLM
        print("[DEBUG] Submitting tool output.")
        tool_call.submit_output(result_str)
    
    def on_run_finished(self, run, snapshot):
        """
        Called when the run completes successfully (status='completed') or otherwise.
        """
        print("[DEBUG] on_run_finished => status:", run.status)
        self.run_info = run
        self._finished = True

        if run.status == "completed":
            # Once complete, do usage logging & final text consolidation
            usage_obj = getattr(run, "usage", None)
            if usage_obj:
                prompt_tokens = usage_obj.prompt_tokens
                completion_tokens = usage_obj.completion_tokens
                total_tokens = usage_obj.total_tokens
                print("[DEBUG] usage tokens => prompt:", prompt_tokens,
                      "completion:", completion_tokens, "total:", total_tokens)

                user_row = self.db_session.query(Users).filter_by(user_id=self.user_id).first()
                if user_row:
                    # cost calculations, etc.
                    cost_per_prompt_token = decimal.Decimal("0.0000025")
                    cost_per_completion_token = decimal.Decimal("0.00001")
                    prompt_cost = prompt_tokens * cost_per_prompt_token
                    completion_cost = completion_tokens * cost_per_completion_token
                    total_cost = prompt_cost + completion_cost

                    # Log usage
                    usage_log = OpenAIUsageLog(
                        user_id=self.user_id,
                        model=run.model or "gpt-4o",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost=total_cost
                    )
                    self.db_session.add(usage_log)

                    # Deduct tokens from user's credits
                    user_row.credits_remaining -= total_tokens

                    self.db_session.commit()
                    print("[DEBUG] usage logged and credits updated.")
        
        # The final text is the sum of all chunks
        # (You might optionally store it in a property self.final_text)
        # self.final_text = "".join(self.final_text_chunks).strip()
    
    def on_run_failed(self, run, snapshot):
        """
        Called if the run fails (status='failed').
        """
        print("[DEBUG] on_run_failed => status:", run.status)
        self.run_info = run
        self._finished = True

    def gen(self):
        """
        The generator for chunked tokens.
        Yields chunks until the run is finished and queue is empty.
        """
        print("[DEBUG] MyEventHandler.gen => starting generator loop.")
        while not (self._finished and self._token_queue.empty()):
            try:
                chunk = self._token_queue.get(timeout=0.2)
                yield chunk
            except queue.Empty:
                continue
        print("[DEBUG] MyEventHandler.gen => done yielding.")


def continue_conversation(
    user_id: int,
    user_input: str,
    thread_id: str = None,
    system_msg: str = None
):
    """
    Create or continue a conversation. Returns the MyEventHandler instance 
    so we can stream tokens in real-time, regardless of final run status.
    
    If something fails *immediately* (e.g. cannot create thread), we return a dict with an error.
    Otherwise, we always return event_handler for streaming.
    """
    print(f"[DEBUG] continue_conversation called, user_id={user_id}, user_input={user_input[:80]}...")
    db_session = g.session

    # 1) Create or reuse a thread
    try:
        if not thread_id:
            new_thread = client.beta.threads.create()
            thread_id = new_thread.id
            print(f"[LOG] Created NEW thread: {thread_id}")

            # Optionally add a system message if provided
            if system_msg:
                client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=system_msg
                )
        else:
            print(f"[LOG] Reusing EXISTING thread: {thread_id}")

        # 2) Add the user's input
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )

    except Exception as e:
        print("[ERROR] Could not create/reuse thread or add message:", e)
        traceback.print_exc()
        return {
            "assistant_message": f"Failed to start conversation: {str(e)}",
            "thread_id": thread_id
        }

    # 3) Prepare our streaming event handler
    event_handler = MyEventHandler(user_id, db_session)

    # 4) Attempt to stream the run
    try:
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            # Strengthen instructions to encourage a single-turn final answer
            instructions=(
                "You are a helpful assistant. Provide one complete, conclusive answer. "
                "Do not ask for further input or say 'feel free to ask more questions.' "
                "End your answer definitively and do not wait for user follow-up."
            ),
            event_handler=event_handler,
            # If your library supports it, you could try max_tokens=512, stop=["<END>"], etc.
        ) as stream:
            # This blocks until the library sees "done" or times out
            stream.until_done()

    except Exception as e:
        print("[ERROR] Exception while streaming run:", e)
        traceback.print_exc()
        return {
            "assistant_message": f"Error streaming run: {str(e)}",
            "thread_id": thread_id
        }

    # 5) Always return the event handler, even if the run never signaled "completed"
    #    because we want to stream whatever tokens we got.
    return event_handler
