"""
SAT Forum Responder - Webhook Receiver
Flask server that receives webhook POST requests and processes them in background
"""

import json
import sqlite3
import logging
import hmac
import requests
from datetime import datetime
from threading import Thread
from queue import Queue, Full
from functools import wraps
from pathlib import Path

from flask import Flask, request, jsonify

from .app.config import config, WEBHOOK_API_KEY

# Setup logging
log_dir = config.LOG_DIR
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Processing queue
processing_queue = Queue(maxsize=config.MAX_QUEUE_SIZE)

# Statistics
stats = {
    'total_received': 0,
    'total_processed': 0,
    'total_success': 0,
    'total_failed': 0,
    'total_url_detected': 0,
    'total_auth_failed': 0,
    'total_images_transcribed': 0,
    'total_fetched_by_id': 0,
    'start_time': datetime.now().isoformat(),
    'last_webhook_time': None
}


# =============================================================================
# FETCH FORUM DATA BY CORRELATION ID
# =============================================================================

def fetch_forum_data_by_correlation_id(correlation_id: str) -> dict:
    """
    Fetch forum data from the LMS API using correlation ID.

    Args:
        correlation_id: The forum post correlation ID

    Returns:
        Forum data dictionary or None on failure
    """
    try:
        # Get API configuration
        neuron_config = config.get_neuron_get_api_config()
        base_url = neuron_config.get('url', '')
        api_key = neuron_config.get('api_key', '')

        if not base_url:
            logger.error("Neuron GET API URL not configured")
            return None

        # Build the full URL with correlation ID
        fetch_url = f"{base_url}/{correlation_id}"

        logger.info(f"[{correlation_id}] Fetching forum data from: {fetch_url}")

        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        response = requests.get(fetch_url, headers=headers, timeout=30)

        if response.status_code == 200:
            forum_data = response.json()
            logger.info(f"[{correlation_id}] Successfully fetched forum data - Keys: {list(forum_data.keys())}")
            stats['total_fetched_by_id'] += 1
            return forum_data
        else:
            logger.error(f"[{correlation_id}] Failed to fetch forum data - Status: {response.status_code}, Response: {response.text[:500]}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"[{correlation_id}] Request error fetching forum data: {e}")
        return None
    except Exception as e:
        logger.error(f"[{correlation_id}] Error fetching forum data: {e}")
        return None


# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_database():
    """Initialize SQLite database for webhook tracking"""
    db_path = config.WEBHOOK_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS webhooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            correlation_id TEXT UNIQUE,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            status TEXT DEFAULT 'pending',
            url_check BOOLEAN DEFAULT FALSE,
            urls_list TEXT,
            classification TEXT,
            processing_time_ms INTEGER,
            error_message TEXT,
            request_ip TEXT,
            request_headers TEXT,
            forum_post_status TEXT,
            forum_post_error TEXT,
            images_transcribed INTEGER DEFAULT 0
        )
    ''')

    # Add new columns if they don't exist
    try:
        cursor.execute('ALTER TABLE webhooks ADD COLUMN forum_post_status TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE webhooks ADD COLUMN forum_post_error TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE webhooks ADD COLUMN images_transcribed INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_correlation_id ON webhooks(correlation_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON webhooks(status)')

    conn.commit()
    conn.close()
    logger.info(f"Database initialized: {db_path}")


def save_webhook_received(correlation_id: str, request_ip: str, headers: dict) -> int:
    """Save received webhook to database"""
    try:
        conn = sqlite3.connect(str(config.WEBHOOK_DB_PATH))
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO webhooks (correlation_id, request_ip, request_headers, status)
            VALUES (?, ?, ?, 'pending')
        ''', (correlation_id, request_ip, json.dumps(dict(headers))))

        webhook_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return webhook_id
    except Exception as e:
        logger.error(f"Error saving webhook: {e}")
        return -1


def update_webhook_status(correlation_id: str, status: str, **kwargs):
    """Update webhook status in database"""
    try:
        conn = sqlite3.connect(str(config.WEBHOOK_DB_PATH))
        cursor = conn.cursor()

        updates = ['status = ?', 'processed_at = CURRENT_TIMESTAMP']
        values = [status]

        for key, value in kwargs.items():
            updates.append(f'{key} = ?')
            values.append(value)

        values.append(correlation_id)

        query = f"UPDATE webhooks SET {', '.join(updates)} WHERE correlation_id = ?"
        cursor.execute(query, values)

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating webhook status: {e}")


# =============================================================================
# AUTHENTICATION
# =============================================================================

def verify_api_key(provided_key: str) -> bool:
    """Verify API key using constant-time comparison"""
    if not provided_key:
        return False

    if provided_key.startswith('Bearer '):
        provided_key = provided_key[7:]

    return hmac.compare_digest(provided_key, WEBHOOK_API_KEY)


def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = None
        for header_name in config.API_KEY_HEADERS:
            api_key = request.headers.get(header_name)
            if api_key:
                break

        if not api_key:
            stats['total_auth_failed'] += 1
            logger.warning(f"Missing API key from {request.remote_addr}")
            return jsonify({
                'error': 'Missing API key',
                'message': f'Please provide API key in {config.API_KEY_HEADER} header'
            }), 401

        if not verify_api_key(api_key):
            stats['total_auth_failed'] += 1
            logger.warning(f"Invalid API key from {request.remote_addr}")
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is invalid'
            }), 403

        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# BACKGROUND PROCESSOR
# =============================================================================

def process_webhook_background(forum_data: dict, correlation_id: str):
    """Process webhook in background thread"""
    start_time = datetime.now()

    try:
        logger.info(f"Starting background processing for: {correlation_id}")

        from .forum_processor import ForumProcessor

        processor = ForumProcessor()
        results = processor.process_forum_post(forum_data)

        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Update stats
        images_transcribed = results.get('image_processing_stats', {}).get('total_images', 0)
        stats['total_images_transcribed'] += images_transcribed

        if results.get('processing_status') == 'completed':
            stats['total_success'] += 1
        elif results.get('processing_status') == 'url_detected':
            stats['total_url_detected'] += 1
            stats['total_success'] += 1
        else:
            stats['total_failed'] += 1

        stats['total_processed'] += 1

        # Save to Airtable and post to forum
        save_status = processor.save_results(results)

        # Update database
        update_webhook_status(
            correlation_id,
            status=results.get('processing_status', 'completed'),
            url_check=results.get('url_check', False),
            urls_list=json.dumps(results.get('urls_list', [])),
            classification=results.get('a2_result', {}).get('parsed', {}).get('classification') if results.get('a2_result') else None,
            processing_time_ms=processing_time_ms,
            forum_post_status=save_status.get('forum_post_status'),
            forum_post_error=save_status.get('forum_post_error'),
            images_transcribed=images_transcribed
        )

        logger.info(f"Completed processing for {correlation_id} in {processing_time_ms}ms - Status: {results.get('processing_status')}, Forum: {save_status.get('forum_post_status')}, Images: {images_transcribed}")

    except Exception as e:
        stats['total_failed'] += 1
        stats['total_processed'] += 1
        logger.error(f"Error processing webhook {correlation_id}: {e}")
        update_webhook_status(correlation_id, status='error', error_message=str(e))


def background_worker():
    """Background worker that processes webhooks from queue"""
    logger.info("Background worker started")

    while True:
        try:
            item = processing_queue.get()

            if item is None:
                break

            forum_data, correlation_id = item
            process_webhook_background(forum_data, correlation_id)

        except Exception as e:
            logger.error(f"Background worker error: {e}")
        finally:
            processing_queue.task_done()


# Start background workers
workers = []
for i in range(config.MAX_WORKERS):
    worker = Thread(target=background_worker, daemon=True)
    worker.start()
    workers.append(worker)
    logger.info(f"Started background worker {i + 1}/{config.MAX_WORKERS}")


# =============================================================================
# WEBHOOK ENDPOINTS
# =============================================================================

@app.route(config.WEBHOOK_ENDPOINT, methods=['POST'])
@require_api_key
def receive_webhook():
    """Receive webhook POST request"""
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Content-Type must be application/json'
            }), 400

        request_data = request.get_json()

        if not request_data:
            return jsonify({
                'error': 'Empty request body',
                'message': 'Request body must contain JSON data'
            }), 400

        # Handle nested structure where data is inside 'body' key (from n8n)
        if 'body' in request_data and isinstance(request_data['body'], dict):
            logger.info("Extracting forum data from nested 'body' key")
            request_data = request_data['body']

        # Check if this is just a correlation ID or full data
        correlation_id = request_data.get('correlationId') or request_data.get('Forum_Corr_ID')

        # If we only have correlation ID (minimal payload), fetch full data from API
        has_full_data = bool(request_data.get('questionDataVO') or request_data.get('forumPostText') or request_data.get('ForumPostText'))

        if correlation_id and not has_full_data:
            logger.info(f"[{correlation_id}] Received correlation ID only, fetching full data from API...")
            forum_data = fetch_forum_data_by_correlation_id(correlation_id)
            if not forum_data:
                return jsonify({
                    'error': 'Failed to fetch forum data',
                    'message': f'Could not retrieve data for correlation ID: {correlation_id}'
                }), 502
        else:
            forum_data = request_data

        if not correlation_id:
            return jsonify({
                'error': 'Missing correlation ID',
                'message': 'Request must contain correlationId or Forum_Corr_ID'
            }), 400

        # Normalize field names (handle case variations)
        # ForumPostText -> forumPostText
        if 'ForumPostText' in forum_data and 'forumPostText' not in forum_data:
            forum_data['forumPostText'] = forum_data['ForumPostText']
            logger.info(f"[{correlation_id}] Normalized ForumPostText -> forumPostText")

        # Log incoming webhook data
        logger.info(f"[{correlation_id}] ========== INCOMING SAT WEBHOOK DATA ==========")
        logger.info(f"[{correlation_id}] Keys received: {list(forum_data.keys())}")
        logger.info(f"[{correlation_id}] forumPostSubject: {forum_data.get('forumPostSubject', 'NOT PRESENT')}")
        logger.info(f"[{correlation_id}] forumPostText: {repr(forum_data.get('forumPostText', 'NOT PRESENT'))[:200]}")
        logger.info(f"[{correlation_id}] questionDataVO present: {bool(forum_data.get('questionDataVO'))}")
        logger.info(f"[{correlation_id}] environment: {forum_data.get('environment', 'NOT PRESENT')}")
        logger.info(f"[{correlation_id}] base64EncodedImages count: {len(forum_data.get('base64EncodedImages', []))}")
        logger.info(f"[{correlation_id}] ==============================================")

        stats['total_received'] += 1
        stats['last_webhook_time'] = datetime.now().isoformat()

        save_webhook_received(correlation_id, request.remote_addr, request.headers)

        try:
            processing_queue.put_nowait((forum_data, correlation_id))
            logger.info(f"Webhook received and queued: {correlation_id}")

            return jsonify({
                'status': 'accepted',
                'message': 'Webhook received and queued for processing',
                'correlation_id': correlation_id,
                'queue_position': processing_queue.qsize()
            }), 202

        except Full:
            logger.warning(f"Queue full, rejecting webhook: {correlation_id}")
            update_webhook_status(correlation_id, status='rejected', error_message='Queue full')

            return jsonify({
                'error': 'Queue full',
                'message': 'Server is busy, please retry later',
                'correlation_id': correlation_id
            }), 503

    except Exception as e:
        logger.error(f"Error receiving webhook: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route(config.HEALTH_ENDPOINT, methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'sat-forum-responder',
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'queue_size': processing_queue.qsize(),
        'workers': len(workers)
    }), 200


@app.route(config.STATS_ENDPOINT, methods=['GET'])
@require_api_key
def get_stats():
    """Get server statistics"""
    uptime_seconds = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()

    return jsonify({
        'status': 'ok',
        'uptime_seconds': int(uptime_seconds),
        'queue_size': processing_queue.qsize(),
        'max_queue_size': config.MAX_QUEUE_SIZE,
        'workers': len(workers),
        'statistics': stats
    }), 200


@app.route('/reprocess/<correlation_id>', methods=['POST'])
@require_api_key
def reprocess_by_correlation_id(correlation_id: str):
    """
    Reprocess a forum post by correlation ID.

    Flow:
    1. Fetch forum data from Neuron API
    2. Process through the normal pipeline
    3. Save to Airtable
    4. Post response to forum
    """
    try:
        logger.info(f"[REPROCESS] Starting reprocess for: {correlation_id}")

        neuron_config = config.get_neuron_get_api_config()
        base_url = neuron_config.get('url', '')
        api_key = neuron_config.get('api_key', '')

        if not base_url or not api_key:
            return jsonify({
                'error': 'Neuron API configuration missing',
                'correlation_id': correlation_id
            }), 500

        # Step 1: Fetch data from Neuron API
        url = f"{base_url}/{correlation_id}"
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        logger.info(f"[REPROCESS] Fetching forum data from Neuron API: {url}")
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            return jsonify({
                'error': f'Neuron API returned {response.status_code}',
                'message': response.text[:500],
                'correlation_id': correlation_id
            }), 500

        forum_data = response.json()
        logger.info(f"[REPROCESS] Fetched data from Neuron API: {list(forum_data.keys())}")

        # Normalize field names
        if 'ForumPostText' in forum_data and 'forumPostText' not in forum_data:
            forum_data['forumPostText'] = forum_data['ForumPostText']

        if 'correlationId' not in forum_data:
            forum_data['correlationId'] = correlation_id

        # Save to DB as received
        save_webhook_received(correlation_id, 'reprocess', {})

        # Step 2: Process through the pipeline
        from .forum_processor import ForumProcessor

        processor = ForumProcessor()
        start_time = datetime.now()

        results = processor.process_forum_post(forum_data)

        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Step 3 & 4: Save to Airtable and post to forum
        save_status = processor.save_results(results)

        # Update DB status
        images_transcribed = results.get("image_processing_stats", {}).get("total_images", 0)
        update_webhook_status(
            correlation_id,
            status=results.get('processing_status', 'unknown'),
            classification=results.get('a2_result', {}).get('parsed', {}).get('classification') if results.get('a2_result') else None,
            processing_time_ms=processing_time_ms,
            forum_post_status=save_status.get('forum_post_status'),
            forum_post_error=save_status.get('forum_post_error'),
            images_transcribed=images_transcribed
        )

        logger.info(f"[REPROCESS] Completed for {correlation_id} in {processing_time_ms}ms - Status: {results.get('processing_status')}, Forum: {save_status.get('forum_post_status')}")

        return jsonify({
            'status': 'completed',
            'correlation_id': correlation_id,
            'processing_status': results.get('processing_status'),
            'classification': results.get('a2_result', {}).get('parsed', {}).get('classification') if results.get('a2_result') else None,
            'hil_flag': results.get('hil_flag', False),
            'forum_post_status': save_status.get('forum_post_status'),
            'airtable_saved': save_status.get('airtable_saved'),
            'processing_time_ms': processing_time_ms,
            'images_transcribed': images_transcribed
        }), 200

    except Exception as e:
        logger.error(f"[REPROCESS] Error processing {correlation_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Processing failed',
            'message': str(e),
            'correlation_id': correlation_id
        }), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Simple dashboard"""
    try:
        conn = sqlite3.connect(str(config.WEBHOOK_DB_PATH))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT correlation_id, status, url_check, classification, processing_time_ms, received_at, forum_post_status, images_transcribed
            FROM webhooks
            ORDER BY received_at DESC
            LIMIT 20
        ''')
        recent = cursor.fetchall()

        cursor.execute('SELECT status, COUNT(*) FROM webhooks GROUP BY status')
        status_counts = dict(cursor.fetchall())

        conn.close()

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAT Forum Responder - Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
                .stat-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 120px; }}
                .stat-number {{ font-size: 32px; font-weight: bold; color: #4CAF50; }}
                .stat-label {{ color: #666; }}
                table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
                th {{ background: #4CAF50; color: white; }}
                tr:hover {{ background: #f5f5f5; }}
                .status-completed {{ color: green; }}
                .status-url_detected {{ color: orange; }}
                .status-error {{ color: red; }}
                .status-pending {{ color: blue; }}
                .status-hil_exception {{ color: #FF9800; }}
            </style>
            <meta http-equiv="refresh" content="30">
        </head>
        <body>
            <h1>SAT Forum Responder - Webhook Dashboard</h1>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{stats['total_received']}</div>
                    <div class="stat-label">Total Received</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{stats['total_success']}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{stats['total_failed']}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{stats['total_url_detected']}</div>
                    <div class="stat-label">URL Detected</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{stats['total_images_transcribed']}</div>
                    <div class="stat-label">Images Transcribed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{processing_queue.qsize()}</div>
                    <div class="stat-label">Queue Size</div>
                </div>
            </div>

            <h2>Recent Webhooks</h2>
            <table>
                <tr>
                    <th>Correlation ID</th>
                    <th>Status</th>
                    <th>Forum Post</th>
                    <th>Classification</th>
                    <th>Images</th>
                    <th>Time (ms)</th>
                    <th>Received</th>
                </tr>
                {''.join(f"""
                <tr>
                    <td>{row[0][:20]}...</td>
                    <td class="status-{row[1]}">{row[1]}</td>
                    <td class="status-{'completed' if row[6] == 'posted' else 'error' if row[6] == 'failed' else 'pending'}">{row[6] or '-'}</td>
                    <td>{row[3] or '-'}</td>
                    <td>{row[7] or 0}</td>
                    <td>{row[4] or '-'}</td>
                    <td>{row[5]}</td>
                </tr>
                """ for row in recent)}
            </table>

            <p style="color: #666; margin-top: 20px;">Auto-refreshes every 30 seconds | Model: Claude Opus 4.5</p>
        </body>
        </html>
        '''

        return html, 200

    except Exception as e:
        return f"Error loading dashboard: {e}", 500


@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'SAT Forum Responder Webhook Server',
        'version': '1.0',
        'model': 'Claude Opus 4.5',
        'features': ['Image URL transcription', 'Base64 image transcription', 'Inline image replacement'],
        'endpoints': {
            'webhook': config.WEBHOOK_ENDPOINT,
            'health': config.HEALTH_ENDPOINT,
            'stats': config.STATS_ENDPOINT,
            'dashboard': '/dashboard',
            'reprocess': '/reprocess/<correlation_id>'
        },
        'documentation': 'POST JSON to /webhook with X-Webhook-API-Key header'
    }), 200


# =============================================================================
# SERVER STARTUP
# =============================================================================

def start_server():
    """Start the Flask webhook server"""
    init_database()

    logger.info("=" * 60)
    logger.info("SAT Forum Responder - Webhook Server Starting")
    logger.info("=" * 60)
    logger.info(f"Host: {config.HOST}")
    logger.info(f"Port: {config.PORT}")
    logger.info(f"Workers: {config.MAX_WORKERS}")
    logger.info(f"Webhook Endpoint: {config.WEBHOOK_ENDPOINT}")
    logger.info(f"API Key: {WEBHOOK_API_KEY[:10]}...{WEBHOOK_API_KEY[-5:]}")
    logger.info(f"Model: Claude Opus 4.5")
    logger.info("=" * 60)

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )


if __name__ == '__main__':
    start_server()
