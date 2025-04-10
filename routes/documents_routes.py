# routes/document_routes.py

from flask import Blueprint, request, jsonify, g
from models.sql_models import *  # Import File and Users models
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import time
from azure.storage.blob import BlobSasPermissions
from azure.storage.blob import generate_blob_sas
import logging
from azure.storage.blob import BlobServiceClient
from helpers.azure_helpers import *
from config import Config
from helpers.llm_helpers import *
from helpers.text_ext_helpers import *
from helpers.embedding_helpers import *
from sqlalchemy.orm.attributes import flag_modified
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers.visit_processor import process_visit
from helpers.diagnosis_worker import worker_process_diagnosis
from helpers.diagnosis_processor import process_diagnosis
from helpers.diagnosis_list import process_diagnosis_list
from helpers.sql_helpers import *
from helpers.cors_helpers import cors_preflight
from helpers.upload.validation_helper import validate_and_setup_request
from helpers.upload.usr_svcp_helpers import get_user_and_service_periods
# Celery task imports
from flask import Blueprint, request, jsonify, g
from models.sql_models import *  # Import File and Users models
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import logging

from azure.storage.blob import BlobServiceClient
from helpers.azure_helpers import upload_file_to_azure
from config import Config
from helpers.upload.validation_helper import validate_and_setup_request
from helpers.upload.usr_svcp_helpers import get_user_and_service_periods
from helpers.sql_helpers import discover_nexus_tags, revoke_nexus_tags_if_invalid
from celery import chain
from helpers.upload.upload_logic import can_user_afford_files

# Import your Celery tasks
from celery_app import extraction_task, process_pages_task, finalize_task

# Create a blueprint for document routes
document_bp = Blueprint('document_bp', __name__)

# ========== DOCUMENT CRUD ROUTES ==========
@document_bp.route('/upload', methods=['POST'])
def upload():
    logging.info("Upload route was hit")
    print("Upload route was hit")  # Keeping print statements as requested

    try:
        process_start_time = time.time()

        # 1. Validate the request and extract data
        validation_result, error_response = validate_and_setup_request(request)
        if error_response:
            return validation_result  # Early exit if validation fails

        user_uuid, uploaded_files = validation_result

        # 2. Retrieve user and service periods
        user_lookup_result, error_response = get_user_and_service_periods(user_uuid)
        if error_response:
            return user_lookup_result  # Early exit if user lookup fails
        user, service_periods = user_lookup_result

        # ------------------------------------------------
        # 2a) First, save all uploaded files to temp paths
        #     so we can do page-count logic below.
        # ------------------------------------------------
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            if uploaded_file.filename == '':
                logging.error("No selected file in the upload")
                print("No selected file in the upload")
                return jsonify({"error": "No selected file"}), 400

            temp_file_path = os.path.join(
                tempfile.gettempdir(),
                secure_filename(uploaded_file.filename)
            )
            uploaded_file.save(temp_file_path)

            logging.info(f"Saved file '{uploaded_file.filename}' to temp path '{temp_file_path}'")
            print(f"Saved file '{uploaded_file.filename}' to temp path '{temp_file_path}'")
            temp_file_paths.append(temp_file_path)

        # ------------------------------------------------
        # 2b) **Check credits** using the local temp files
        # create a list of temp file paths for each uploaded file
        # ------------------------------------------------
        try:
            affordable, total_pages, required_credits = can_user_afford_files(user, temp_file_paths)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": f"Could not count pages: {str(e)}"}), 500

        if not affordable:
            # If user cannot afford them, remove temp files and return error
            for path in temp_file_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

            return jsonify({
                "error": (
                    f"You need at least {required_credits} credits to process {total_pages} page(s). "
                    f"You have {user.credits_remaining} credits remaining."
                )
            }), 403

        # This will store info about each file we process
        uploaded_urls = []

        # ------------------------------------------------
        # 3) Now that we know user can afford them,
        #    process each temp file: upload to Azure, DB, etc.
        # ------------------------------------------------
        for i, uploaded_file in enumerate(uploaded_files):
            temp_file_path = temp_file_paths[i]

            # Determine file type based on extension
            file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
            file_type_mapping = {
                '.pdf': 'pdf',
                '.jpg': 'image',
                '.jpeg': 'image',
                '.png': 'image',
                '.mp4': 'video',
                '.mov': 'video',
                '.mp3': 'audio'
            }
            file_type = file_type_mapping.get(file_extension, 'unknown')
            logging.info(f"Determined file type '{file_type}' for extension '{file_extension}'")
            print(f"Determined file type '{file_type}' for extension '{file_extension}'")

            category = 'Unclassified'

            # 3a) Upload file to Azure
            blob_name = f"{user_uuid}/{category}/{uploaded_file.filename}"
            blob_url = upload_file_to_azure(temp_file_path, blob_name)
            logging.info(f"Uploaded file to Azure Blob Storage: {blob_url}")
            print(f"Uploaded file to Azure Blob Storage: {blob_url}")

            # 3b) Create DB record
            new_file = File(
                user_id=user.user_id,
                file_name=uploaded_file.filename,
                file_type=file_type,
                file_url=blob_url,
                file_date=datetime.now().date(),
                uploaded_at=datetime.utcnow(),
                file_size=os.path.getsize(temp_file_path),
                file_category=category,
                status="Uploading",
            )
            g.session.add(new_file)
            g.session.flush()  # get new_file.file_id
            file_id = new_file.file_id
            g.session.commit()

            logging.info(f"Inserted new file record with file_id={file_id}")
            print(f"Inserted new file record with file_id={file_id}")

            # 3c) Kick off Celery chain (extraction -> process_pages -> finalize)
            extraction = extraction_task.s(user.user_id, blob_url, file_type, file_id)
            processing = process_pages_task.s(
                user_id=user.user_id,
                user_uuid=user_uuid,
                file_info={
                    'service_periods': service_periods,
                    'file_id': file_id
                }
            )
            finalization = finalize_task.s(user.user_id, file_id)

            chain_result = (extraction | processing | finalization)()
            task_ids = {
                'extraction_task_id': extraction.freeze().id,
                'processing_task_id': processing.freeze().id,
                'finalization_task_id': finalization.freeze().id,
                'chain_task_id': chain_result.id
            }

            uploaded_urls.append({
                'category': category,
                "fileName": uploaded_file.filename,
                "blobUrl": blob_url,
                "task_ids": task_ids
            })

            # 3d) Clean up local temp file
            try:
                os.remove(temp_file_path)
                logging.info(f"Temporary file {temp_file_path} removed successfully.")
            except OSError as e:
                logging.warning(f"Failed to remove temporary file {temp_file_path}: {e}")
                print(f"Failed to remove temporary file {temp_file_path}: {e}")

        process_end_time = time.time()
        elapsed_time = process_end_time - process_start_time
        logging.info(f"Total time to queue Celery tasks: {elapsed_time:.2f} seconds.")
        print(f">>>>>>>>>>>>>>>>>>>>PROCESSING TIME (queuing tasks): {elapsed_time}<<<<<<<<<<<<<<<<<<<<<<")

        # Return response immediately, while tasks run in background
        return jsonify({
            "message": "File(s) uploaded and processing started",
            "files": uploaded_urls
        }), 202

    except Exception as e:
        g.session.rollback()
        logging.exception(f"Upload failed: {str(e)}")
        print(f"Upload failed: {str(e)}")
        return jsonify({"error": "Failed to upload file"}), 500


@document_bp.route('/documents', methods=['OPTIONS', 'GET', 'POST', 'PUT'])
def get_documents():
    # GET request logic
    user_uuid = request.args.get('userUUID')
    if not user_uuid:
        return jsonify({"error": "User UUID is required"}), 400

    user = g.session.query(Users).filter_by(user_uuid=user_uuid).first()
    if not user:
        return jsonify({"error": "Invalid user UUID"}), 404

    files = g.session.query(File).filter_by(user_id=user.user_id).order_by(File.uploaded_at.desc()).all()

    document_list = [{
        "id": file.file_id,
        "title": file.file_name,
        "file_category": file.file_category,
        "file_type": file.file_type,
        "size": f"{file.file_size / (1024 * 1024):.2f}MB" if file.file_size else "Unknown",
        "shared": "Only Me",
        "modified": file.uploaded_at.strftime("%d/%m/%Y"),
        "status": file.status  # <-- Add this line to include file status
    } for file in files]

    return jsonify(document_list), 200


@document_bp.route('/documents/delete/<int:file_id>', methods=['DELETE', 'OPTIONS'])
def delete_document(file_id):
    # 1) Grab the userUUID from the query param (e.g. ?userUUID=xxx)
    user_uuid = request.args.get('userUUID', None)
    if not user_uuid:
        return jsonify({'error': 'Missing user UUID'}), 400

    # 2) Look up the user by UUID
    user = g.session.query(Users).filter_by(user_uuid=user_uuid).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    try:
        # 3) Check if the file exists
        file_record = g.session.query(File).get(file_id)
        if not file_record:
            return jsonify({"error": "File not found"}), 404

        # 4) Confirm that this file belongs to the same user
        if file_record.user_id != user.user_id:
            return jsonify({"error": "You do not own this file"}), 403

        # 5) Delete the blob from Azure
        blob_name = '/'.join(file_record.file_url.split('/')[-3:])
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.delete_blob(blob_name)
        except Exception as e:
            logging.error(f"Failed to delete blob '{blob_name}' from Azure: {e}")
            return jsonify({"error": f"Failed to delete file from Azure: {e}"}), 500

        # 6) Remove the file record from the database
        g.session.delete(file_record)
        g.session.commit()

        # 7) Re-check user’s nexus tags
        revoke_nexus_tags_if_invalid(g.session, user.user_id)
        g.session.commit()

        return jsonify({"message": "File deleted successfully"}), 200

    except Exception as e:
        g.session.rollback()
        logging.error(f"Error deleting file with id {file_id}: {e}")
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

@document_bp.route('/documents/rename/<int:file_id>', methods=['PUT', 'OPTIONS'])
def rename_document(file_id):
    data = request.get_json()
    new_name = data.get('new_name')

    # Input validation
    if not new_name:
        return jsonify({"error": "New name is required"}), 400

    # Ensure the new name does not include the file extension
    new_name_without_extension = os.path.splitext(new_name)[0]

    # Find the file by ID
    file = g.session.query(File).filter_by(file_id=file_id).first()
    if not file:
        return jsonify({"error": "File not found"}), 404

    # Extract current blob details
    blob_url = file.file_url
    old_blob_name = extract_blob_name(blob_url)
    
    # Extract the original file extension from the current file name
    file_extension = os.path.splitext(file.file_name)[1]  # e.g., ".pdf"
    
    # Construct the new blob name without duplicating the extension
    new_blob_name = "/".join(old_blob_name.split("/")[:-1]) + f"/{new_name_without_extension}{file_extension}"

    try:
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)

        # Copy blob to a new blob with the new name
        source_blob = f"https://{account_name}.blob.core.windows.net/{container_name}/{old_blob_name}"
        new_blob_client = container_client.get_blob_client(new_blob_name)
        new_blob_client.start_copy_from_url(source_blob)

        # Ensure the new blob copy completes (it may take time)
        properties = new_blob_client.get_blob_properties()
        copy_status = properties.copy.status
        while copy_status == 'pending':
            time.sleep(1)
            properties = new_blob_client.get_blob_properties()
            copy_status = properties.copy.status

        if copy_status != "success":
            raise Exception("Blob copy operation failed.")

        # Delete the old blob
        container_client.delete_blob(old_blob_name)

        # Update the file name in the database
        file.file_name = f"{new_name_without_extension}{file_extension}"
        file.file_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{new_blob_name}"
        g.session.commit()
        logging.info(f"File '{old_blob_name}' renamed to '{new_blob_name}' successfully.")
        
        return jsonify({"message": f"File '{old_blob_name}' renamed to '{new_name_without_extension}' successfully."}), 200

    except Exception as e:
        g.session.rollback()
        logging.error(f"Failed to rename file ID {file_id}: {str(e)}")
        return jsonify({"error": f"Failed to rename file: {str(e)}"}), 500

@document_bp.route('/documents/change-category/<int:file_id>', methods=['PUT', 'OPTIONS'])
def change_document_category(file_id):
    # Extract the data from the request
    data = request.get_json()
    new_category = data.get('new_category')

    # Validate inputs
    if not new_category:
        return jsonify({"error": "New category is required"}), 400

    # Find the file by ID
    file = g.session.query(File).filter_by(file_id=file_id).first()
    if not file:
        return jsonify({"error": "File not found"}), 404

    # Extract current blob details
    blob_url = file.file_url
    old_blob_name = extract_blob_name(blob_url)
    old_category = file.file_category

    # Update the category in the blob path
    new_blob_name = old_blob_name.replace(f"/{old_category}/", f"/{new_category}/")

    try:
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)

        # Copy blob to new category path
        source_blob = f"https://{account_name}.blob.core.windows.net/{container_name}/{old_blob_name}"
        new_blob_client = container_client.get_blob_client(new_blob_name)
        copy_response = new_blob_client.start_copy_from_url(source_blob)

        # Ensure the new blob copy completes
        properties = new_blob_client.get_blob_properties()
        copy_status = properties.copy.status
        while copy_status == 'pending':
            time.sleep(1)
            properties = new_blob_client.get_blob_properties()
            copy_status = properties.copy.status

        if copy_status != "success":
            raise Exception("Blob copy operation failed.")

        # Delete the old blob
        container_client.delete_blob(old_blob_name)

        # Update the category and blob URL in the database
        file.file_category = new_category
        file.file_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{new_blob_name}"
        g.session.commit()

        logging.info(f"File category changed from '{old_category}' to '{new_category}' for file '{file.file_name}'.")
        
        return jsonify({"message": f"Category updated successfully to '{new_category}' for file '{file.file_name}'."}), 200

    except Exception as e:
        g.session.rollback()
        logging.error(f"Failed to change category for file ID {file_id}: {str(e)}")
        return jsonify({"error": f"Failed to change category: {str(e)}"}), 500

@document_bp.route('/documents/download/<int:file_id>', methods=['GET', 'OPTIONS'])
def download_document(file_id):
    try:
        # Fetch the file from the database
        file = g.session.query(File).filter_by(file_id=file_id).first()
        if not file:
            return jsonify({"error": "File not found"}), 404

        # Return the file URL directly for download
        return jsonify({"download_url": file.file_url}), 200

    except Exception as e:
        logging.error(f"Failed to get download URL for file ID {file_id}: {str(e)}")
        return jsonify({"error": f"Failed to get download URL: {str(e)}"}), 500

@document_bp.route('/documents/preview/<int:file_id>', methods=['GET', 'OPTIONS'])
def preview_document(file_id):
    
    try:
        # Fetch the file from the database
        file = g.session.query(File).filter_by(file_id=file_id).first()
        if not file:
            return jsonify({"error": "File not found"}), 404

        # Extract blob details from the URL
        blob_name = extract_blob_name(file.file_url)

        # Generate a SAS token for previewing the file (valid for 1 hour)
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )

        if not sas_token:
            logging.error(f"Failed to generate SAS token for file ID {file_id}")
            return jsonify({"error": "Failed to generate SAS token"}), 500

        # Construct the preview URL using the SAS token and set disposition to inline
        preview_url = f"{file.file_url}?{sas_token}&response-content-disposition=inline"

        return jsonify({"preview_url": preview_url}), 200

    except FileNotFoundError as e:
        logging.error(f"File not found error for file ID {file_id}: {str(e)}")
        return jsonify({"error": "File not found."}), 404
    except PermissionError as e:
        logging.error(f"Permission error for file ID {file_id}: {str(e)}")
        return jsonify({"error": "Permission denied."}), 403
    except Exception as e:
        logging.error(f"Failed to generate preview URL for file ID {file_id}: {str(e)}")
        return jsonify({"error": f"Failed to generate preview URL: {str(e)}"}), 500

@document_bp.route('/upload-file', methods=['POST', 'OPTIONS'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    document_type = request.form.get("document_type")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not document_type:
        return jsonify({"error": "Document type is required"}), 400

    # Define blob name based on document type and filename
    blob_name = f"{document_type}_{file.filename}"
    
    # Save the file temporarily to upload it
    temp_file_path = f"/tmp/{file.filename}"
    file.save(temp_file_path)
    
    # Upload file to Azure
    blob_url = upload_file_to_azure(temp_file_path, blob_name)
    
    # Remove the temporary file
    os.remove(temp_file_path)
    
    if blob_url:
        return jsonify({"message": "File uploaded successfully", "url": blob_url}), 200
    else:
        return jsonify({"error": "Failed to upload file"}), 500

@document_bp.route('/get-file-url/<blob_name>', methods=['GET', 'OPTIONS'])
def get_file_url(blob_name):
    try:
        # Generate the SAS URL using the helper function
        file_url = generate_sas_url(blob_name)
        
        # If SAS URL generation was successful, return it as a JSON response
        if file_url:
            return jsonify({"url": file_url}), 200
        else:
            return jsonify({"error": "Failed to generate SAS URL"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_blob_name(blob_url):
    # Example URL: "https://<account_name>.blob.core.windows.net/<container_name>/<blob_name>"
    return "/".join(blob_url.split("/")[4:])  # Gets the blob name from the full URL
