#!/usr/bin/env python3
"""
Test script for Google Drive API authentication and folder access.
Verifies that the service account can read files from the shared Drive folder.

Run: python test_gdrive.py
"""

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
SERVICE_ACCOUNT_FILE = os.getenv('GDRIVE_SERVICE_ACCOUNT_FILE', './service_account.json')
GDRIVE_FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID')
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

print("=" * 70)
print("Google Drive API Test Script")
print("=" * 70)
print()

# Step 1: Verify environment variables
print("📋 Step 1: Checking environment variables...")
print(f"  Service account file: {SERVICE_ACCOUNT_FILE}")
print(f"  Folder ID: {GDRIVE_FOLDER_ID}")
print()

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    print(f"❌ ERROR: Service account file not found: {SERVICE_ACCOUNT_FILE}")
    print("   Make sure service_account.json is in the project root")
    exit(1)

if not GDRIVE_FOLDER_ID:
    print("❌ ERROR: GDRIVE_FOLDER_ID not set in .env file")
    print("   Add this line to .env: GDRIVE_FOLDER_ID=your_folder_id")
    exit(1)

print("✅ Environment variables OK")
print()

# Step 2: Authenticate with service account
print("🔐 Step 2: Authenticating with Google Drive API...")
try:
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=SCOPES
    )
    print("✅ Authentication successful")
except Exception as e:
    print(f"❌ ERROR: Authentication failed: {e}")
    exit(1)

print()

# Step 3: Build the Drive service
print("🔧 Step 3: Building Google Drive service...")
try:
    drive_service = build('drive', 'v3', credentials=credentials)
    print("✅ Service built successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to build service: {e}")
    exit(1)

print()

# Step 4: List files in the folder
print("📁 Step 4: Listing files in folder...")
try:
    results = drive_service.files().list(
        q=f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false",
        spaces='drive',
        fields='files(id, name, mimeType, size, modifiedTime)',
        pageSize=20,
        orderBy='modifiedTime desc'
    ).execute()

    files = results.get('files', [])
    print(f"✅ Found {len(files)} files in folder")
    print()

    if not files:
        print("   (No files in folder yet)")
        print("   Upload some files to your Google Drive folder to test syncing")
    else:
        print("   Files found:")
        print()
        print(f"   {'Name':<40} {'Type':<35} {'Modified':<20}")
        print("   " + "-" * 95)
        for file in files:
            name = file['name'][:39]
            mime = file['mimeType'][:34]
            modified = file['modifiedTime'][:19]
            print(f"   {name:<40} {mime:<35} {modified:<20}")

except Exception as e:
    print(f"❌ ERROR: Failed to list files: {e}")
    print()
    print("Possible causes:")
    print("  1. Folder ID is incorrect")
    print("  2. Folder not shared with service account email")
    print("  3. Service account doesn't have permission to access folder")
    exit(1)

print()
print("=" * 70)
print("✅ All tests passed! Google Drive sync is ready to use.")
print("=" * 70)
print()
print("Next steps:")
print("  1. Make sure your server.py has the /gdrive/sync endpoint")
print("  2. Set up Cloud Scheduler to call the endpoint weekly")
print("  3. Or test locally: python -m pytest server.py -k gdrive")
print()
