# routes/chatbot_route.py
from flask import Blueprint, request, jsonify, g, Response
from config import Config
from helpers.chatbot_helper import continue_conversation
from models.sql_models import Users, ServicePeriod
import traceback
from helpers.cors_helpers import cors_preflight
from helpers.stream_helper import stream_chat_response
from helpers.stream_chat_helper import continue_conversation

chatbot_bp = Blueprint("chatbot_bp", __name__)

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    """
    Handle a single user message and get a single assistant response, 
    with optional multi-turn memory via thread_id, and user identification via user_uuid.

    Expects JSON:
    {
        "message": "<User's question or statement>",
        "thread_id": "<optional existing thread ID>",
        "user_uuid": "<the user’s UUID (optional)>"
    }
    Returns JSON:
    {
        "assistant_message": "...",
        "thread_id": "..."
    }
    """
    try:
        # 2) Parse JSON request body
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        user_message = data.get("message", "")
        thread_id = data.get("thread_id")
        user_uuid = data.get("user_uuid")  # Retrieve the user’s UUID

        if not user_message:
            return jsonify({"error": "No 'message' provided"}), 400

        print(f"User UUID received: {user_uuid}")

        # 3) Look up the user in the DB
        user = g.session.query(Users).filter_by(user_uuid=user_uuid).first()
        if not user:
            print(f"Invalid user UUID: {user_uuid}")
            return jsonify({"error": "Invalid user UUID"}), 404
        
        # 3a) Look up the user credits in the DB and block if not enough
        if user.credits_remaining <= 0:
            return jsonify({"error": f"You do not have enough credits to continue this conversation. Your credit balance is {user.credits_remaining}. Please visit your account and purchase more credits."}), 403


        # Retrieve the user_id (and optionally first/last name, email, etc.)
        db_user_id = user.user_id
        first_name = user.first_name  # or user.last_name, user.email, etc.
        print(f"[DEBUG] Found user_id={db_user_id} for UUID={user_uuid} (First name: {first_name})")

        # 4) Retrieve the user's service periods
        service_periods = g.session.query(ServicePeriod).filter_by(user_id=db_user_id).all()
        if service_periods:
            formatted_service_periods = [
                f"{sp.branch_of_service} from {sp.service_start_date.strftime('%Y-%m-%d')} to {sp.service_end_date.strftime('%Y-%m-%d')}"
                for sp in service_periods
            ]
            service_periods_str = "; ".join(formatted_service_periods)
        else:
            service_periods_str = "No service periods found."

        print(f"[DEBUG] Service periods for user_id={db_user_id}: {service_periods_str}")

        # 5) Build the system_message
        system_message = (
            f"My first name is {first_name}, user_id is {db_user_id}, "
            f"and my service periods are: {service_periods_str}."
        )

        # 6) Call continue_conversation, passing system_msg
        result = continue_conversation(
            user_id=db_user_id,
            user_input=user_message,
            thread_id=thread_id,
            system_msg=system_message  # <--- pass here
        )

        # Now re-query the user so we get a fresh user object *still attached* to g.session
        updated_user = g.session.query(Users).filter_by(user_id=db_user_id).first()
        if not updated_user:
            # handle edge case: user got deleted somehow
            return jsonify({"error": "User record not found after update."}), 404
        
        # 7) Return the assistant response
        response_data = {
            "assistant_message": result["assistant_message"],
            "thread_id": result["thread_id"],
            "credits_remaining": updated_user.credits_remaining
        }
        return jsonify(response_data), 200

    except Exception as e:
        print("[ERROR] An exception occurred in /chat route:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    A route for streaming chat responses in real time, chunk-by-chunk (typing effect).
    Expects JSON: {
        "user_uuid": "...",
        "message": "...",
        "thread_id": "..." (optional)
    }
    Returns text chunks as they arrive (HTTP chunked response).
    """
    print("==> Entered /chat_stream route.")
    try:
        print("==> Attempting to parse incoming JSON...")
        data = request.get_json(force=True)
        if not data:
            return Response("Missing JSON body.", status=400)

        user_message = data.get("message", "")
        thread_id = data.get("thread_id")
        user_uuid = data.get("user_uuid")
        print(f"==> user_message: {user_message}")
        print(f"==> user_uuid: {user_uuid}")

        if not user_uuid or not user_message:
            print("==> Missing user_uuid or message. Returning 400.")
            return Response("Missing user_uuid or message.", status=400)

        print("==> Looking up user in database...")
        user = g.session.query(Users).filter_by(user_uuid=user_uuid).first()
        if not user:
            print(f"==> No user found for user_uuid={user_uuid}. Returning 404.")
            return Response("Invalid user UUID", status=404)

        print(f"==> Found user: user_id={user.user_id}, credits_remaining={user.credits_remaining}")
        if user.credits_remaining <= 0:
            print("==> User has insufficient credits. Returning 403.")
            return Response("User has insufficient credits.", status=403)

        # Optionally retrieve service periods
        service_periods = g.session.query(ServicePeriod).filter_by(user_id=user.user_id).all()
        if service_periods:
            formatted_service_periods = [
                f"{sp.branch_of_service} from {sp.service_start_date.strftime('%Y-%m-%d')} to {sp.service_end_date.strftime('%Y-%m-%d')}"
                for sp in service_periods
            ]
            service_periods_str = "; ".join(formatted_service_periods)
        else:
            service_periods_str = "No service periods found."
        print(f"[DEBUG] Service periods for user_id={user.user_id}: {service_periods_str}")

        # Build a system message
        system_message = (
            f"My first name is {user.first_name}, user_id is {user.user_id}, "
            f"and my service periods are: {service_periods_str}."
        )
        print("==> About to call continue_conversation for chunked streaming...")

        # 1) Call your streaming function
        result = continue_conversation(
            user_id=user.user_id,
            user_input=user_message,
            thread_id=thread_id,
            system_msg=system_message
        )

        # 2) If it's a dict, that means something failed before streaming
        if isinstance(result, dict):
            print("[DEBUG] continue_conversation returned dict => no streaming.")
            final_text = result.get("assistant_message", "No text generated.")
            return Response(final_text, mimetype="text/plain")

        # 3) Otherwise it's your MyEventHandler with a .gen() method => streaming
        print("[DEBUG] continue_conversation returned event_handler => streaming response.")

        def debug_wrapped_gen():
            """
            Wrap event_handler.gen() so we can log each chunk in the route
            right before sending it to the client.
            """
            for chunk in result.gen():
                # Print a snippet of the chunk to confirm server-side streaming
                print(f"[DEBUG] Route is about to yield chunk: {chunk[:50]!r}")
                yield chunk

        # 4) Return the wrapped generator as a streaming response
        return Response(debug_wrapped_gen(), mimetype="text/plain")

    except Exception as e:
        print("==> Exception occurred in /chat_stream route.")
        traceback.print_exc()
        return Response(str(e), status=500)
