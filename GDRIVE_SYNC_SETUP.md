# Google Drive Sync Setup Guide

## Overview

This guide walks through setting up **incremental Google Drive folder sync** for the 2Meditate RAG chatbot. The setup uses a service account to enable automated weekly syncing of files from a Google Drive folder to your Firestore database.

**Key Architecture:**
- Google Cloud Project (Email A) - hosts the service account
- Google Drive folder (Email B) - contains files to sync
- Service account bridges both by having read-only access to the Drive folder

---

## Prerequisites

- ✅ Active Google Cloud Project: `gen-lang-client-0966906205`
- ✅ Two different Google email accounts (one for GCP, one for Drive)
- ✅ Access to both email accounts during setup
- ✅ Python 3.8+ with `pip` installed locally
- ✅ Google Drive folder already created (or will create during setup)

---

## Part 1: Create Service Account (Email A - GCP)

Service account is a special account that represents your application, not a person. It allows your code to access Google Drive without using a password.

### Step 1.1: Open Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. **Sign in with Email A** (the email tied to your GCP project)
3. At the top, ensure your **project is selected**: `gen-lang-client-0966906205`
   - Click the project dropdown if unsure

### Step 1.2: Navigate to Service Accounts

1. In the left sidebar, click **IAM & Admin**
2. In the dropdown menu, click **Service Accounts**
3. You'll see a list of existing service accounts (may be empty)

### Step 1.3: Create New Service Account

1. Click the **+ CREATE SERVICE ACCOUNT** button (top of page)
2. Fill in the form:
   - **Service account name**: `gdrive-sync` (or `google-drive-weekly-sync`)
   - **Service account ID**: Auto-filled (will look like `gdrive-sync@gen-lang-client-0966906205.iam.gserviceaccount.com`)
   - **Description**: `Service account for syncing Google Drive files weekly to Firestore`
3. Click **CREATE AND CONTINUE**

### Step 1.4: Skip Permissions (We'll Add Later)

1. On the **Grant this service account access to project** page:
   - Skip for now — leave empty
   - Click **CONTINUE**
2. On the **Grant users access to this service account** page:
   - Skip for now — leave empty
   - Click **DONE**

You're now back at the Service Accounts list. Your new account should be listed.

### Step 1.5: Create & Download JSON Key

1. In the Service Accounts list, click on your newly created service account (the row with name `gdrive-sync`)
2. You'll see the service account details page
3. Click the **KEYS** tab (top of page)
4. Click **ADD KEY** → **Create new key**
5. In the popup, select **JSON** and click **CREATE**
6. A JSON file will download automatically to your computer
   - Default filename: `gen-lang-client-0966906205-XXXXX.json` (or similar)
   - **Save this file** — you'll need it later
7. Close the dialog

### Step 1.6: Copy & Save the Service Account Email

1. Go back to the **Service Accounts** page
2. Find your `gdrive-sync` account in the list
3. Copy the **Service account ID** (email address)
   - Format: `gdrive-sync@gen-lang-client-0966906205.iam.gserviceaccount.com`
   - **Save this email** — you'll need it in Part 2

---

## Part 2: Enable Google Drive API (Email A - GCP)

Your service account needs permission to access Google Drive. This step enables the Drive API in your GCP project.

### Step 2.1: Open APIs & Services

1. In Google Cloud Console (still signed in as Email A)
2. In the left sidebar, click **APIs & Services**
3. In the dropdown, click **Library**
4. You'll see a search box at the top

### Step 2.2: Enable Google Drive API

1. In the search box, type: `Google Drive API`
2. Click on **Google Drive API** (the first result)
3. Click the blue **ENABLE** button
4. Wait for it to enable (takes 10-30 seconds)
5. You'll see a green checkmark and "API enabled"

### Step 2.3: Grant Service Account Access

1. In the left sidebar, click **IAM & Admin**
2. In the dropdown, click **IAM**
3. You'll see a list of members with roles
4. Click the **+ GRANT ACCESS** button (top of page)
5. A sidebar will appear on the right
6. In the **New principals** field, paste the service account email:
   ```
   gdrive-sync@gen-lang-client-0966906205.iam.gserviceaccount.com
   ```
7. In the **Select a role** dropdown, search for and select:
   - **Basic** → **Viewer** (if folder is read-only)
   - Or **Basic** → **Editor** (if you need to write/delete files)
   - For now, choose **Viewer** (safer)
8. Click **SAVE**

---

## Part 3: Share Google Drive Folder (Email B - Drive)

Now you'll share the Google Drive folder with the service account so it can read files.

### Step 3.1: Prepare the Folder

1. **Sign out of Google** (if needed)
2. **Sign in with Email B** (the email that owns the Drive folder)
3. Go to [Google Drive](https://drive.google.com/)
4. **Create a folder** (if you don't have one yet):
   - Right-click in empty space → **New** → **Folder**
   - Name it something like: `Swati_Audio_Files` or `2Meditate_Content`
   - Remember this folder name

### Step 3.2: Share Folder with Service Account

1. Find your folder in Google Drive
2. Right-click on the folder
3. Click **Share** (or click the folder and use the **Share** button at the top)
4. A sharing dialog will open
5. In the **Add people and groups** field, paste the service account email:
   ```
   gdrive-sync@gen-lang-client-0966906205.iam.gserviceaccount.com
   ```
6. In the permission dropdown (currently showing a role), select **Viewer**
7. **Uncheck** the box that says **Notify people** (service account won't receive emails)
8. Click **Share**
9. Close the dialog

### Step 3.3: Get the Folder ID

1. Open the shared folder (double-click it)
2. Look at the URL in your browser's address bar
   - Format: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
   - Copy the long ID after `/folders/`
   - Example: `1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p`
3. **Save this Folder ID** — you'll need it in your code

### Step 3.4: Upload Test Files (Optional)

1. Upload 1-2 test files to verify setup works:
   - Supported formats: `.pdf`, `.txt`, `.docx`, `.mp3`, `.m4a`
   - You can test the sync with these later

---

## Part 4: Install Python Dependencies

Now prepare your local environment to run the sync code.

### Step 4.1: Update requirements.txt

1. Open `requirements.txt` in your project
2. Add these lines at the end (if not already present):
   ```
   google-api-python-client>=2.100.0
   google-auth>=2.25.0
   google-auth-httplib2>=0.2.0
   ```
3. Save the file

### Step 4.2: Install Dependencies

1. Open Terminal/PowerShell in your project directory
2. Run:
   ```bash
   pip install -r requirements.txt
   ```
3. Wait for installation to complete

---

## Part 5: Set Up Local Environment Variables

Store your credentials securely using environment variables.

### Step 5.1: Create/Update .env File

1. In your project root, open or create `.env`
2. Add these lines:
   ```
   GDRIVE_FOLDER_ID=PASTE_FOLDER_ID_HERE
   GDRIVE_SERVICE_ACCOUNT_FILE=./service_account.json
   ```
3. Replace `PASTE_FOLDER_ID_HERE` with the Folder ID from Part 3.3
4. Save the file

### Step 5.2: Place JSON Key File

1. Take the JSON key file you downloaded in Part 1.5
2. Rename it to: `service_account.json` (for simplicity)
3. **Place it in your project root directory** (same folder as `server.py`)
4. ⚠️ **Important**: Add to `.gitignore` to prevent accidental commits:
   ```
   service_account.json
   ```

---

## Part 6: Test Service Account Access (Locally)

Verify the service account can access the Drive folder before implementing the full sync.

### Step 6.1: Create Test Script

1. Create a new file: `test_gdrive.py`
2. Copy this code:
   ```python
   from google.oauth2 import service_account
   from googleapiclient.discovery import build
   import os
   from dotenv import load_dotenv

   # Load environment variables
   load_dotenv()

   # Get credentials
   SERVICE_ACCOUNT_FILE = os.getenv('GDRIVE_SERVICE_ACCOUNT_FILE', './service_account.json')
   GDRIVE_FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID')

   print(f"Service account file: {SERVICE_ACCOUNT_FILE}")
   print(f"Folder ID: {GDRIVE_FOLDER_ID}")

   # Authenticate
   credentials = service_account.Credentials.from_service_account_file(
       SERVICE_ACCOUNT_FILE,
       scopes=['https://www.googleapis.com/auth/drive.readonly']
   )
   drive_service = build('drive', 'v3', credentials=credentials)
   print("✓ Authenticated with Google Drive API")

   # List files in folder
   try:
       results = drive_service.files().list(
           q=f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false",
           spaces='drive',
           fields='files(id, name, mimeType, size, modifiedTime)',
           pageSize=10
       ).execute()
       
       files = results.get('files', [])
       print(f"\n✓ Found {len(files)} files in folder:")
       for file in files:
           print(f"  - {file['name']} ({file['mimeType']}) - Modified: {file['modifiedTime']}")
       
       if len(files) == 0:
           print("  (No files yet - upload some to test)")
   
   except Exception as e:
       print(f"✗ Error listing files: {e}")
   ```
3. Save the file

### Step 6.2: Run Test Script

1. In Terminal/PowerShell, run:
   ```bash
   python test_gdrive.py
   ```
2. Expected output:
   ```
   Service account file: ./service_account.json
   Folder ID: 1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p
   ✓ Authenticated with Google Drive API
   ✓ Found 2 files in folder:
     - document1.pdf (application/pdf) - Modified: 2024-01-15T10:30:00Z
     - document2.txt (text/plain) - Modified: 2024-01-14T14:22:00Z
   ```

### Step 6.3: Troubleshooting

If you get an error, check:

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: service_account.json` | JSON file not in project root | Move it to the right location |
| `Permission denied (403)` | Folder not shared with service account | Verify sharing in Part 3.2 |
| `GDRIVE_FOLDER_ID` is `None` | Environment variable not set | Check `.env` file syntax |
| `Invalid Credentials` | JSON file is corrupted/wrong | Re-download from GCP |

---

## Part 7: Implement Sync Endpoint (server.py)

Now integrate Google Drive sync into your FastAPI application.

### Step 7.1: Add Imports & Initialization

Add this to the top of `server.py` (with other imports):

```python
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import io
```

### Step 7.2: Add Configuration Variables

Add this after the other constants (around line 100):

```python
# Google Drive Sync Configuration
GDRIVE_SERVICE_ACCOUNT_FILE = os.environ.get('GDRIVE_SERVICE_ACCOUNT_FILE', './service_account.json')
GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID', '')
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def _get_gdrive_service():
    """Initialize authenticated Google Drive service."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            GDRIVE_SERVICE_ACCOUNT_FILE,
            scopes=GDRIVE_SCOPES
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        logger.error(f"Failed to initialize Google Drive service: {e}")
        raise
```

### Step 7.3: Add Helper Functions

Add these functions after the helper functions section (around line 300):

```python
def _get_gdrive_sync_state():
    """Get last Google Drive sync state from Firestore."""
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    if not project:
        return None
    
    try:
        db_fs = _firestore_client()
        doc = db_fs.collection('system_config').document('gdrive_sync').get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.warning(f"Failed to read Google Drive sync state: {e}")
    
    return None

def _save_gdrive_sync_state(last_sync_time: str):
    """Save Google Drive sync state to Firestore."""
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    if not project:
        return
    
    try:
        db_fs = _firestore_client()
        db_fs.collection('system_config').document('gdrive_sync').set(
            {'last_sync_time': last_sync_time},
            merge=True
        )
        logger.info(f"Saved Google Drive sync state: {last_sync_time}")
    except Exception as e:
        logger.warning(f"Failed to save Google Drive sync state: {e}")

def _download_gdrive_file(drive_service, file_id: str) -> bytes:
    """Download a file from Google Drive as bytes."""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request, chunksize=1024*1024)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        return file_buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to download file {file_id}: {e}")
        raise
```

### Step 7.4: Add Sync Endpoint

Add this endpoint in the admin routes section (after `/onedrive/sync` if it exists):

```python
# ---------------------------------------------------------------------------
# Google Drive weekly sync (automatic new-file ingestion)
# ---------------------------------------------------------------------------
@app.post("/gdrive/sync")
def gdrive_sync():
    """
    List files in the shared Google Drive folder, ingest any added/modified files
    since the last sync, and update the last_sync_time in Firestore.
    Called weekly by Cloud Scheduler.
    """
    if not GDRIVE_FOLDER_ID:
        return {"synced": 0, "warning": "GDRIVE_FOLDER_ID not set"}
    
    try:
        drive_service = _get_gdrive_service()
    except Exception as e:
        logger.error(f"gdrive_sync: Failed to initialize Drive service: {e}")
        raise HTTPException(status_code=503, detail=f"Google Drive service unavailable: {e}")
    
    # Get last sync time
    last_sync_time = None
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    
    if project:
        sync_state = _get_gdrive_sync_state()
        if sync_state:
            last_sync_time = sync_state.get('last_sync_time')
    
    # Build query for files
    query = f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false"
    if last_sync_time:
        query += f" and modifiedTime > '{last_sync_time}'"
    
    # List files
    try:
        results = drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, mimeType, size, modifiedTime)',
            pageSize=100,
            orderBy='modifiedTime desc'
        ).execute()
        all_files = results.get('files', [])
    except Exception as e:
        logger.error(f"gdrive_sync: Graph API error: {e}")
        raise HTTPException(status_code=502, detail=f"Google Drive API error: {e}")
    
    # Filter for supported file types
    supported_types = [
        'application/pdf',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'audio/mpeg',
        'audio/mp4',
        'application/x-m4a'
    ]
    
    new_files = [f for f in all_files if f.get('mimeType') in supported_types]
    
    if not new_files:
        return {"synced": 0, "message": "No new supported files since last sync"}
    
    rag = _get_rag()
    total_chunks = 0
    synced_names = []
    max_ingested_time = last_sync_time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for item in new_files:
            file_id = item['id']
            file_name = item['name']
            
            try:
                # Download file
                file_bytes = _download_gdrive_file(drive_service, file_id)
                
                # Save to temp directory
                dest = os.path.join(tmpdir, file_name)
                with open(dest, 'wb') as f:
                    f.write(file_bytes)
                
                synced_names.append(file_name)
                max_ingested_time = item['modifiedTime']
                logger.info(f"gdrive_sync: downloaded {file_name}")
                
            except Exception as e:
                logger.warning(f"gdrive_sync: failed to download {file_name}: {e}")
                continue
        
        # Ingest files
        if synced_names:
            total_chunks = rag.ingest_folder(tmpdir, section='gdrive', recursive=False)
    
    # Save index
    if total_chunks > 0:
        _save_and_sync(rag)
    
    # Update sync state
    if project and synced_names:
        _save_gdrive_sync_state(max_ingested_time)
    
    return {
        "synced": len(synced_names),
        "chunks_stored": total_chunks,
        "files": synced_names,
        "next_sync_after": max_ingested_time
    }
```

---

## Part 8: Set Up Cloud Scheduler (Production)

Automate weekly syncing using Google Cloud Scheduler to call your sync endpoint every week.

**What is Cloud Scheduler?**
Cloud Scheduler is a managed cron job service. It will automatically call your `/gdrive/sync` endpoint on a schedule (e.g., every Monday at 9 AM) without any manual intervention.

---

### Step 8.0: Get Required Information (Prerequisites)

Before creating the scheduler job, gather these two pieces of information:

#### 8.0.1: Find Your Cloud Run Service URL

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in as **Email A** (GCP email)
3. Ensure project `gen-lang-client-0966906205` is selected
4. In the left sidebar, click **Cloud Run**
5. You should see a list of deployed services
6. Find the service named `swati-avatar` (or your service name)
7. Click on it to open the details page
8. **Copy the URL** from the top of the page
   - Format: `https://SERVICE-NAME-RANDOM.REGION.run.app`
   - Example: `https://swati-avatar-578167174494.asia-south1.run.app`
   - **Save this URL** — you'll need it in Step 8.4

#### 8.0.2: Find Your Cloud Run Service Account Email

1. Still in Cloud Run, find your `swati-avatar` service
2. Click the **SECURITY** tab (near the top)
3. Look for **Service account email**
   - Format: `service-account-name@PROJECT_ID.iam.gserviceaccount.com`
   - Example: `swati-avatar@gen-lang-client-0966906205.iam.gserviceaccount.com`
   - **Copy this email** — you'll need it in Step 8.3

---

### Step 8.1: Open Cloud Scheduler

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. **Sign in as Email A** (your GCP email)
3. At the top, ensure project **`gen-lang-client-0966906205`** is selected
   - Click the project name/dropdown if unsure
4. In the **left sidebar**, search for or scroll to find **Cloud Scheduler**
5. Click **Cloud Scheduler**
6. You'll see a list of existing jobs (may be empty)
7. Click the blue **+ CREATE JOB** button (top of page)

---

### Step 8.2: Enter Job Details (Name, Schedule, Timezone)

A form will appear with fields to configure your job.

#### 8.2.1: Job Name

1. In the **Name** field, enter:
   ```
   gdrive-weekly-sync
   ```
   - This is just a label to identify the job in Cloud Scheduler

#### 8.2.2: Frequency (Cron Expression)

1. In the **Frequency** field, enter the cron expression for when you want the sync to run

**Choose ONE of these based on your needs:**

- **Weekly (Monday 9 AM UTC)** — Recommended for production:
  ```
  0 9 * * 1
  ```

- **Daily (2 AM UTC)** — Good for testing:
  ```
  0 2 * * *
  ```

- **Every 6 hours** — For frequent syncing:
  ```
  0 */6 * * *
  ```

**Cron Format Explanation:**
```
0    9    *    *    1
│    │    │    │    │
│    │    │    │    └─ Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
│    │    │    └────── Month (1-12, * = any month)
│    │    └─────────── Day of month (1-31, * = any day)
│    └──────────────── Hour (0-23 in 24-hr format)
└───────────────────── Minute (0-59)
```

#### 8.2.3: Timezone

1. In the **Timezone** field, select your timezone
   - Click the dropdown
   - Find your timezone (e.g., `America/New_York`, `Europe/London`, `Asia/Kolkata`)
   - The times you see in the schedule will be adjusted to this timezone
   - Example: If you choose `Asia/Kolkata`, 9 AM UTC = 2:30 PM IST

2. After selecting timezone, you'll see the schedule in human-readable format (e.g., "Every Monday at 09:00")

#### 8.2.4: Click CONTINUE

Click the blue **CONTINUE** button to proceed to authentication settings.

---

### Step 8.3: Configure Authentication (OIDC Token)

Cloud Scheduler needs to authenticate when calling your Cloud Run endpoint. This uses OpenID Connect (OIDC) tokens.

#### 8.3.1: Select Authentication Type

1. You'll see a section labeled **Authentication**
2. Click the dropdown that says **Do not add authentication** (or similar)
3. Select **Add OIDC token**
4. New fields will appear below

#### 8.3.2: Choose Service Account

1. In the **Service account email** field, click the dropdown
2. Look for the Cloud Run service account email you copied in Step 8.0.2
   - Example: `swati-avatar@gen-lang-client-0966906205.iam.gserviceaccount.com`
3. Select it

If you don't see your service account:
- Make sure you're in the correct GCP project
- The service account must exist in your project
- If it doesn't exist, you'll need to create one (or use the default Cloud Run service account)

#### 8.3.3: Enter OIDC Token Audience

1. In the **OIDC token audience** field, paste your Cloud Run service URL from Step 8.0.1
   - Example: `https://swati-avatar-578167174494.asia-south1.run.app`
   - **Important:** Do NOT include a trailing `/` or any path

This tells the scheduler: "When you call this URL, prove that you're authorized to access this specific Cloud Run service."

---

### Step 8.4: Configure the HTTP Request

Now you'll specify what HTTP request Cloud Scheduler should make.

#### 8.4.1: HTTP Method

1. In the **HTTP method** dropdown, select:
   ```
   POST
   ```
   - Our `/gdrive/sync` endpoint expects POST requests

#### 8.4.2: Request URL

1. In the **URL** field, paste:
   ```
   https://swati-avatar-578167174494.asia-south1.run.app/gdrive/sync
   ```
   - Replace the base URL with yours from Step 8.0.1
   - Keep the `/gdrive/sync` path at the end

**Full URL format:**
```
{YOUR_CLOUD_RUN_URL}/gdrive/sync
```

Example with real values:
```
https://swati-avatar-578167174494.asia-south1.run.app/gdrive/sync
```

#### 8.4.3: Headers (Optional)

1. The **Headers** section can be left empty for this endpoint
   - Our `/gdrive/sync` endpoint doesn't require authentication headers
   - (It's protected by the OIDC token from Step 8.3)

#### 8.4.4: Request Body

1. Leave the **Body** field empty
   - The endpoint doesn't require a request body
   - It will read the `GDRIVE_FOLDER_ID` from environment variables

#### 8.4.5: Create the Job

Click the blue **CREATE** button to create the scheduler job.

You should see a success message and the job will appear in the Cloud Scheduler list.

---

### Step 8.5: Test the Scheduler Job Manually

Before waiting for the scheduled time, test it to ensure it works.

#### 8.5.1: Find Your Job

1. In Cloud Scheduler, find your job in the list
   - Look for `gdrive-weekly-sync`
2. Click on it to open the details

#### 8.5.2: Force Run (Manual Trigger)

1. Click the **⋮** (three dots) menu on the right side of the job
2. Select **Force run**
3. You'll see a message: "Job triggered successfully"

#### 8.5.3: Check Cloud Run Logs

1. Go to **Cloud Run** in the sidebar
2. Click on your `swati-avatar` service
3. Click the **LOGS** tab (top of page)
4. You should see recent logs including the sync execution
5. Look for log entries like:
   ```
   gdrive_sync: downloaded document1.pdf
   Saved Google Drive sync state: 2024-01-15T10:30:00Z
   ```

#### 8.5.4: Expected Success Response

If the sync worked, you should see in the logs:
```
{
  "synced": 2,
  "chunks_stored": 150,
  "files": ["document1.pdf", "document2.txt"],
  "next_sync_after": "2024-01-15T10:30:00Z"
}
```

---

### Step 8.6: Verify the Scheduler Is Working (Wait Test)

After testing manually, verify the scheduler will run automatically.

#### 8.6.1: Check Job Status

1. In Cloud Scheduler, click your `gdrive-weekly-sync` job
2. Look at the **Last execution** field
   - Should show a recent timestamp (from Step 8.5 manual run)
3. Look at the **Next execution** field
   - Should show when the job will run next based on your cron schedule

#### 8.6.2: Monitor Execution History

1. In the job details, scroll to see **Execution history** or click the **EXECUTION HISTORY** tab
2. You should see your manual test run logged with:
   - Status: `SUCCESS` (green checkmark)
   - Time: when you triggered it
   - Last execution time

#### 8.6.3: Wait for Scheduled Run (Optional)

1. If you set a frequent schedule (e.g., `0 2 * * *` for daily), wait for the next scheduled time
2. Check Cloud Run logs to verify it executed automatically
3. You should NOT need to manually trigger it again

---

### Step 8.7: Troubleshooting Scheduler Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Permission denied (403)** | Job status shows 403 error | Ensure Cloud Run service account has permission to call the endpoint. Check Cloud Run IAM settings. |
| **Job never runs** | Next execution keeps getting pushed back | Check cron format is valid. Verify timezone is correct. Try force-running manually. |
| **Connection timeout** | Log shows timeout error | Cloud Run service may be sleeping. Call the endpoint manually to wake it up. Check if service account has permissions. |
| **404 endpoint not found** | Job returns 404 error | Verify `/gdrive/sync` endpoint exists in `server.py`. Check Cloud Run service is deployed. |
| **Wrong authentication type** | Job shows auth failures | Ensure you used **OIDC token**, not "Add basic auth". Verify service account email is correct. |

---

### Step 8.8: Monitor Ongoing Scheduler Activity

Once the job is running, monitor it regularly:

#### Option A: Dashboard Monitoring (Easy)

1. Go to Cloud Scheduler
2. Click your job
3. Look for:
   - ✅ **Last execution result** — should show SUCCESS
   - 📅 **Next execution** — when it will run next
   - 📊 **Execution history** — list of past runs

#### Option B: Cloud Logging (Detailed)

1. Go to **Cloud Logging** in the sidebar
2. In the query, filter by:
   - Resource: `Cloud Run Service`
   - Service name: `swati-avatar`
   - Log name: `run.googleapis.com`
3. You'll see detailed logs for every scheduler execution

#### Option C: Alert if Job Fails (Recommended for Production)

1. Go to **Monitoring** → **Alerting**
2. Create an alert for Cloud Scheduler failures
3. Set notifications to email if job fails
4. This way you'll be notified if something breaks

---

### Step 8.9: Adjust Scheduler if Needed

Once running, you can modify the job:

#### Change Frequency

1. Click your job in Cloud Scheduler
2. Click **EDIT** (top right)
3. Change the **Frequency** cron expression
4. Click **UPDATE**

#### Change Timezone

1. Click **EDIT**
2. Change the **Timezone** dropdown
3. Click **UPDATE**

#### Pause/Resume Job

1. Click the **⋮** menu next to your job
2. Select **Pause** to stop it from running
3. Or select **Resume** to start it again

#### Delete Job

1. Click the **⋮** menu
2. Select **Delete**
3. Confirm deletion

---

## Part 9: Update cloudbuild.yaml (Production)

Make the Google Drive configuration persist through deployments.

### Step 9.1: Add Substitution Variables

Open `cloudbuild.yaml` and find the `substitutions:` section. Add:

```yaml
substitutions:
  _REGION:              asia-south1
  _SERVICE_NAME:        swati-avatar
  # ... (existing substitutions)
  _GDRIVE_FOLDER_ID:    ""
  _GDRIVE_SERVICE_ACCOUNT_FILE: ""
```

### Step 9.2: Store Service Account JSON as Secret

**Why:** Cloud Run containers need the `service_account.json` file but it can't be committed to git

**Steps:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Ensure project `gen-lang-client-0966906205` is selected
3. In the left sidebar, search for and click **Secret Manager**
4. Click **+ CREATE SECRET**
5. Fill in:
   - **Name:** `gdrive-service-account`
   - **Secret value:** Paste the entire contents of your `service_account.json` file
   - Leave "Replication" as automatic
6. Click **CREATE SECRET**
7. On the secret details page, note the **Secret resource ID:**
   ```
   projects/PROJECT_ID/secrets/gdrive-service-account/versions/latest
   ```
   You'll need this in Step 9.4

### Step 9.3: Grant Cloud Build Access to Secret

Cloud Build needs permission to read the secret during deployment

1. Still in Secret Manager, click your `gdrive-service-account` secret
2. Click the **PERMISSIONS** tab
3. Click **GRANT ACCESS**
4. In the "New principals" field, paste:
   ```
   SERVICE_ACCOUNT_ID@cloudbuild.gserviceaccount.com
   ```
   (Replace `SERVICE_ACCOUNT_ID` with your project number, found in project settings)
5. In the "Select a role" dropdown, search for and select: **Secret Accessor** (role/secretmanager.secretAccessor)
6. Click **SAVE**

### Step 9.4: Update cloudbuild.yaml to Use Secret

Now update `cloudbuild.yaml` to mount the secret when deploying:

**Find the deploy step** (around line 77-92) and update it:

```yaml
- name: gcr.io/google.com/cloudsdktool/cloud-sdk:slim
  entrypoint: gcloud
  args:
    - run
    - deploy
    - ${_SERVICE_NAME}
    - --platform=managed
    - --region=${_REGION}
    - --image=${_IMAGE}:$COMMIT_SHA
    - --allow-unauthenticated
    - --memory=2Gi
    - --cpu=2
    - --timeout=300
    - --update-env-vars=GOOGLE_CLOUD_PROJECT=${_PROJECT_ID},GCS_BUCKET=${_GCS_BUCKET},FIRESTORE_DB=${_FIRESTORE_DB},MAILJET_API_KEY=${_MAILJET_API_KEY},MAILJET_SECRET_KEY=${_MAILJET_SECRET_KEY},MAILJET_FROM_EMAIL=${_MAILJET_FROM_EMAIL},ADMIN_EMAIL=${_ADMIN_EMAIL},ADMIN_FRONTEND_URL=${_ADMIN_FRONTEND_URL},GDRIVE_FOLDER_ID=${_GDRIVE_FOLDER_ID},GDRIVE_SERVICE_ACCOUNT_FILE=/secrets/gdrive/service_account.json
    - --secret=GDRIVE_SERVICE_ACCOUNT=gdrive-service-account:latest
    - --quiet
```

**Key changes:**
- `--secret=GDRIVE_SERVICE_ACCOUNT=gdrive-service-account:latest` — Mounts the secret file
- `GDRIVE_SERVICE_ACCOUNT_FILE=/secrets/gdrive/service_account.json` — Points to mounted file

### Step 9.5: Set Cloud Build Substitution Variables

These are the values that get plugged into `cloudbuild.yaml` during deployment

1. Go to [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers)
2. Find your trigger (should be named something like `main` or `auto-deploy`)
3. Click **EDIT**
4. Scroll down to **Substitution variables**
5. Add or update these two:
   ```
   _GDRIVE_FOLDER_ID = 1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p  (your actual folder ID from Part 3.3)
   _GDRIVE_SERVICE_ACCOUNT_FILE = /secrets/gdrive/service_account.json
   ```
6. Click **SAVE**

---

## Part 10: Environment Variables Reference

### Local Development (.env)

```bash
# Google Drive Configuration
GDRIVE_FOLDER_ID=1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p
GDRIVE_SERVICE_ACCOUNT_FILE=./service_account.json

# Other existing vars
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_CLOUD_PROJECT=
```

### Cloud Run (Set in Console)

| Variable | Value |
|----------|-------|
| `GDRIVE_FOLDER_ID` | Your folder ID from Part 3.3 |
| `GDRIVE_SERVICE_ACCOUNT_FILE` | `/workspace/service_account.json` or stored as secret |

---

## Part 11: Testing & Verification

### Local Test

1. Run test script:
   ```bash
   python test_gdrive.py
   ```
   Expected: Lists files in your Drive folder

2. Start server locally:
   ```bash
   uvicorn server:app --reload --port 8000
   ```

3. Test sync endpoint:
   ```bash
   curl -X POST http://localhost:8000/gdrive/sync
   ```
   Expected response:
   ```json
   {
     "synced": 2,
     "chunks_stored": 150,
     "files": ["document1.pdf", "document2.txt"],
     "next_sync_after": "2024-01-15T10:30:00Z"
   }
   ```

### Production Test

1. Deploy to Cloud Run (push to main branch)
2. In Cloud Console → Cloud Scheduler, click **Force run** on your job
3. Check Cloud Run logs for success:
   ```
   gdrive_sync: downloaded document1.pdf
   Saved Google Drive sync state: 2024-01-15T10:30:00Z
   ```

---

## Troubleshooting

### Issue: "Permission denied (403)"

**Cause:** Service account doesn't have access to folder.

**Fix:**
1. Verify service account email in Part 3.2 sharing step
2. Check it's the exact email: `gdrive-sync@gen-lang-client-0966906205.iam.gserviceaccount.com`
3. In Google Drive, right-click folder → **Share** → verify it's in the list

### Issue: "No such file or directory: service_account.json"

**Cause:** JSON file not in project root.

**Fix:**
1. Move `service_account.json` to project root (same folder as `server.py`)
2. Or update `.env`: `GDRIVE_SERVICE_ACCOUNT_FILE=/path/to/service_account.json`

### Issue: Sync endpoint returns "GDRIVE_FOLDER_ID not set"

**Cause:** Environment variable not loaded.

**Fix:**
1. Check `.env` file exists in project root
2. Verify line: `GDRIVE_FOLDER_ID=1a2b3c4d5e...`
3. Restart the server after changing `.env`

### Issue: No files appear in sync despite uploading to Drive

**Cause:** Files not shared or wrong folder ID.

**Fix:**
1. Verify folder ID is correct (Part 3.3)
2. Verify you uploaded to that folder (not a subfolder)
3. Check file MIME type is in the supported list (Part 7.4)
4. Wait a few seconds after uploading before syncing

---

## Cloud Scheduler Troubleshooting

### Issue: Scheduler Job Shows 403 Unauthorized

**Cause:** The Cloud Run service account doesn't have permission to invoke the Cloud Run service.

**Fix:**
1. Go to **Cloud Run** in GCP Console
2. Click on your `swati-avatar` service
3. Click the **PERMISSIONS** tab (or **SECURITY**)
4. Ensure your Cloud Run service account has the role: `roles/run.invoker`
5. If missing, click **ADD MEMBER** and add the service account with that role

### Issue: Scheduler Job Status Shows 404 Not Found

**Cause:** The `/gdrive/sync` endpoint doesn't exist or the URL is wrong.

**Fix:**
1. Verify the endpoint exists in `server.py` (look for `@app.post("/gdrive/sync")`)
2. If not present, re-do Part 7 (Implement Sync Endpoint)
3. Verify Cloud Run has the latest code deployed
4. Verify the URL in scheduler is exactly: `{YOUR_CLOUD_RUN_URL}/gdrive/sync`
5. No trailing slashes or extra paths

### Issue: Scheduler Job Executes But Returns Empty Response or No Files Synced

**Cause:** Environment variables not set in Cloud Run or incorrect folder ID.

**Fix:**
1. Go to **Cloud Run** → your service → **EDIT AND DEPLOY**
2. In the **Environment variables** section, verify:
   - `GDRIVE_FOLDER_ID` is set to your folder ID
   - `GDRIVE_SERVICE_ACCOUNT_FILE` points to correct location
3. If missing, add them and redeploy
4. Check Cloud Run logs to see actual error messages

### Issue: Manual "Force Run" Works But Scheduled Run Never Triggers

**Cause:** Cron expression is wrong or scheduler is paused.

**Fix:**
1. In Cloud Scheduler, check the job status (should show as enabled/active)
2. Click **EDIT** and verify:
   - Frequency cron expression is valid (e.g., `0 9 * * 1`)
   - Timezone matches your local time
3. Check the **Next execution** timestamp is in the future
4. If stuck, click **PAUSE** then **RESUME** to reset

### Issue: Scheduler Keeps Showing "Next execution" in the Past

**Cause:** Cloud Run service is unreachable or permanently failing.

**Fix:**
1. Check Cloud Run service is deployed and running
2. Try manually calling the endpoint: `POST {URL}/gdrive/sync`
3. Check Cloud Run logs for errors
4. Verify GOOGLE_CLOUD_PROJECT environment variable is set (required for production mode)
5. Check service account JSON file is in correct location with correct permissions

---

## File Checklist

After completing setup, you should have:

**Local Development:**
- [ ] `service_account.json` in project root (added to `.gitignore`)
- [ ] `.env` file with `GDRIVE_FOLDER_ID` and `GDRIVE_SERVICE_ACCOUNT_FILE`
- [ ] Updated `requirements.txt` with Google Drive API packages
- [ ] `test_gdrive.py` for local testing (verified successful run)
- [ ] Updated `server.py` with sync endpoint code

**Production Deployment:**
- [ ] Updated `cloudbuild.yaml` with substitutions (for production)
- [ ] Cloud Run service deployed with latest code
- [ ] Environment variables set in Cloud Run console:
  - `GDRIVE_FOLDER_ID`
  - `GDRIVE_SERVICE_ACCOUNT_FILE` (or stored in Secret Manager)
  - `GOOGLE_CLOUD_PROJECT` (should already be set)

**Cloud Scheduler Setup:**
- [ ] Cloud Scheduler job created: `gdrive-weekly-sync`
- [ ] Job configured with correct cron frequency
- [ ] Authentication configured with OIDC token
- [ ] Service account email set correctly
- [ ] Cloud Run URL set correctly
- [ ] Endpoint URL set to `/gdrive/sync`
- [ ] Manual "Force run" test completed successfully
- [ ] Cloud Run logs checked for successful execution
- [ ] Next execution timestamp is in the future

---

## Quick Reference

| Item | Value | Where Found |
|------|-------|------------|
| GCP Project | `gen-lang-client-0966906205` | GCP Console |
| Service Account Email (Drive) | `gdrive-sync@gen-lang-client-0966906205.iam.gserviceaccount.com` | Part 1.6 |
| Service Account Email (Cloud Run) | `swati-avatar@gen-lang-client-0966906205.iam.gserviceaccount.com` | Part 8.0.2 |
| Folder ID | `1a2b3c4d5e...` | Part 3.3 |
| JSON File Location | `./service_account.json` | Project root |
| Sync Endpoint | `POST /gdrive/sync` | server.py |
| Cloud Run Service URL | `https://swati-avatar-578167174494.asia-south1.run.app` | Part 8.0.1 |
| Scheduler Job Name | `gdrive-weekly-sync` | Cloud Scheduler |
| Scheduler Frequency (Production) | `0 9 * * 1` (Monday 9 AM UTC) | Cloud Scheduler |
| Scheduler Frequency (Testing) | `0 2 * * *` (Daily 2 AM UTC) | Cloud Scheduler |

---

## Scheduler Cron Cheat Sheet

| Schedule | Cron Expression | Description |
|----------|-----------------|-------------|
| Every Monday 9 AM UTC | `0 9 * * 1` | Weekly production sync |
| Every day 2 AM UTC | `0 2 * * *` | Daily testing sync |
| Every 6 hours | `0 */6 * * *` | Frequent sync (0, 6, 12, 18 UTC) |
| Every Monday & Thursday 9 AM | `0 9 * * 1,4` | Twice-weekly sync |
| First day of month 9 AM | `0 9 1 * *` | Monthly sync |
| Every weekday (Mon-Fri) 9 AM | `0 9 * * 1-5` | Business days only |

---

## Support & Resources

- [Google Drive API Docs](https://developers.google.com/drive/api)
- [Service Account Guide](https://cloud.google.com/iam/docs/service-accounts)
- [Cloud Scheduler Guide](https://cloud.google.com/scheduler/docs)
- [Your Project Logs](https://console.cloud.google.com/logs)

