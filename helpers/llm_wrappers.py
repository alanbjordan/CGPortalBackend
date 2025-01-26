# llm_wrappers.py

import decimal
from datetime import datetime
from flask import g
from openai import OpenAI
from models.sql_models import Users, OpenAIUsageLog
import os
from database.session import ScopedSession

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def call_openai_chat_create(
    user_id: int,
    model: str,
    messages: list,
    temperature: float = 0.7,
    **kwargs
):
    """
    Wrapper for client.chat.completions.create(...) that logs usage in openai_usage_logs
    and updates the user's cost/credits.
    Blocks call if user does not have enough credits.
    """
    db_session = ScopedSession()
    try:
        # 1) Retrieve user
        user = db_session.query(Users).filter_by(user_id=user_id).first()
        if not user:
            raise ValueError(f"User not found (user_id={user_id}).")
        
        # 1a) Check user credits
        if user.credits_remaining <= 0:
            raise ValueError("User does not have enough credits to proceed.")

        # 2) Call the OpenAI ChatCompletion endpoint
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )

        # 3) Extract usage
        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            prompt_tokens = usage_obj.prompt_tokens
            completion_tokens = usage_obj.completion_tokens
            total_tokens = usage_obj.total_tokens
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        # 4) Calculate cost
        cost_per_prompt_token = decimal.Decimal("0.0000025")   # Example: $2.50 per 1M
        cost_per_completion_token = decimal.Decimal("0.00001") # Example: $10.00 per 1M

        prompt_cost = prompt_tokens * cost_per_prompt_token
        completion_cost = completion_tokens * cost_per_completion_token
        total_cost = prompt_cost + completion_cost

        # 5) Insert usage log
        usage_log = OpenAIUsageLog(
            user_id=user_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=total_cost,
            created_at=datetime.utcnow()
        )
        db_session.add(usage_log)

        # 6) Update user credits
        user.credits_remaining -= total_tokens

        db_session.commit()

        return response

    except Exception as e:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def call_openai_embeddings(
    user_id: int,
    input_text: str,
    model: str,
    cost_per_token: decimal.Decimal,
    **kwargs
):
    """
    A wrapper for client.embeddings.create(...) that:
      1) Looks up the user
      2) Blocks call if user has insufficient credits
      3) Calls the OpenAI embeddings endpoint
      4) Logs usage info in openai_usage_logs
      5) Updates user credits/cost
      6) Returns the raw response
    """
    db_session = ScopedSession()
    try:
        # 1) Retrieve user
        user = db_session.query(Users).filter_by(user_id=user_id).first()
        if not user:
            raise ValueError(f"User not found (user_id={user_id}).")

        # 1a) Check user credits
        if user.credits_remaining <= 0:
            raise ValueError("User does not have enough credits to proceed.")

        # 2) Make the embeddings call
        response = client.embeddings.create(
            input=input_text,
            model=model,
            **kwargs
        )

        # 3) Extract usage
        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            prompt_tokens = usage_obj.prompt_tokens
            total_tokens = usage_obj.total_tokens
        else:
            prompt_tokens = 0
            total_tokens = 0

        # 4) Calculate cost
        cost_for_this_call = total_tokens * cost_per_token

        # 5) Insert usage log
        usage_log = OpenAIUsageLog(
            user_id=user_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
            cost=cost_for_this_call,
            created_at=datetime.utcnow()
        )
        db_session.add(usage_log)

        # 6) Update user credits
        user.credits_remaining -= total_tokens

        db_session.commit()
        return response

    except Exception as e:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def call_openai_chat_parse(
    user_id: int,
    model: str,
    messages: list,
    response_format,
    cost_per_prompt_token: decimal.Decimal,
    cost_per_completion_token: decimal.Decimal,
    **kwargs
):
    """
    A wrapper for client.beta.chat.completions.parse(...)
    Logs usage in openai_usage_logs and updates user’s credits or balance.
    Blocks call if user does not have enough credits.
    """
    db_session = ScopedSession()
    try:
        # 1) Retrieve user
        user = db_session.query(Users).filter_by(user_id=user_id).first()
        if not user:
            raise ValueError(f"User not found (user_id={user_id}).")

        # 1a) Check user credits
        if user.credits_remaining <= 0:
            raise ValueError("User does not have enough credits to proceed.")

        # 2) Make the parse call
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )

        # 3) Extract usage
        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            prompt_tokens = usage_obj.prompt_tokens
            completion_tokens = usage_obj.completion_tokens
            total_tokens = usage_obj.total_tokens
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        # 4) Calculate cost
        prompt_cost = prompt_tokens * cost_per_prompt_token
        completion_cost = completion_tokens * cost_per_completion_token
        total_cost = prompt_cost + completion_cost

        # 5) Insert usage log
        usage_log = OpenAIUsageLog(
            user_id=user_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=total_cost,
            created_at=datetime.utcnow()
        )
        db_session.add(usage_log)

        # 6) Update user’s credits
        user.credits_remaining -= total_tokens

        db_session.commit()
        return response

    except Exception as e:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def stream_openai_chat_create(
    user_id: int,
    model: str,
    messages: list,
    temperature: float = 0.7,
    cost_per_prompt_token: decimal.Decimal = decimal.Decimal("0.0000025"),
    cost_per_completion_token: decimal.Decimal = decimal.Decimal("0.00001"),
    **kwargs
):
    """
    A *streaming* version of call_openai_chat_create. Instead of returning
    a single response, it yields partial text tokens as they arrive.
    
    At the end of the stream, it updates usage logs and user credits.
    """

    db_session = ScopedSession()
    try:
        # 1) Retrieve user
        user = db_session.query(Users).filter_by(user_id=user_id).first()
        if not user:
            raise ValueError(f"User not found (user_id={user_id}).")

        # 1a) Check user credits
        if user.credits_remaining <= 0:
            raise ValueError("User does not have enough credits to proceed.")

        # 2) Make a streaming call to the OpenAI ChatCompletion endpoint
        #    We set stream_options={"include_usage": True} so usage appears in the final chunk
        response_iter = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},  # usage in final chunk
            **kwargs
        )

        # We'll define a generator function that yields text tokens
        def token_generator():
            usage_obj = None  # We'll grab from the final chunk

            for chunk in response_iter:
                # chunk is an OpenAI ChatCompletionChunk object
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    # If there's new text content in this chunk
                    if getattr(delta, "content", None):
                        yield delta.content

                # The usage field only appears in the last chunk
                if chunk.usage is not None:
                    usage_obj = chunk.usage

            # Once streaming is done, usage_obj should be populated
            if usage_obj:
                _update_usage_and_credits(
                    db_session=db_session,
                    user=user,
                    usage_obj=usage_obj,
                    model=model,
                    cost_per_prompt_token=cost_per_prompt_token,
                    cost_per_completion_token=cost_per_completion_token
                )

        # Return that generator so the caller can iterate over partial tokens
        return token_generator()

    except Exception as e:
        db_session.rollback()
        raise
    finally:
        # We won't close the DB session immediately if we need it after the final chunk
        # but typically you'd do so after usage is recorded
        pass


def _update_usage_and_credits(
    db_session,
    user: Users,
    usage_obj,
    model: str,
    cost_per_prompt_token: decimal.Decimal,
    cost_per_completion_token: decimal.Decimal,
):
    """
    Helper function to insert usage log and update user's credits after streaming.
    usage_obj is typically chunk.usage from the final chunk.
    """
    prompt_tokens = usage_obj.prompt_tokens
    completion_tokens = usage_obj.completion_tokens
    total_tokens = usage_obj.total_tokens

    # Calculate cost
    prompt_cost = prompt_tokens * cost_per_prompt_token
    completion_cost = completion_tokens * cost_per_completion_token
    total_cost = prompt_cost + completion_cost

    # Insert usage log
    usage_log = OpenAIUsageLog(
        user_id=user.user_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=total_cost,
        created_at=datetime.utcnow()
    )
    db_session.add(usage_log)

    # Deduct tokens from user's credits
    user.credits_remaining -= total_tokens

    db_session.commit()
    db_session.close()