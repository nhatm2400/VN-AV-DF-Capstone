import os
import json
import subprocess
import sys
import zipfile
import shutil
import time
import random
import builtins
import glob

# Force flush on all print statements for real-time progress
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(*args, **kwargs)

def extract_json(stdout_str):
    start = stdout_str.find('[')
    end = stdout_str.rfind(']')
    if start != -1 and end != -1:
        return json.loads(stdout_str[start:end+1])
    raise ValueError("Could not find JSON list in gdown output.")

def recover_filenames_from_bad_zip(zip_path):
    existing_files = set()
    if not os.path.exists(zip_path):
        return existing_files
        
    print(f"[Info] ZIP file '{os.path.basename(zip_path)}' is corrupted. Scanning local headers to recover downloaded list...")
    try:
        with open(zip_path, 'rb') as f:
            content = f.read()
            
        offset = 0
        while True:
            # Find the next local file header signature
            offset = content.find(b'PK\x03\x04', offset)
            if offset == -1:
                break
                
            # A local file header is at least 30 bytes long
            if offset + 30 > len(content):
                break
                
            # Read filename length (offset 26, 2 bytes)
            filename_len = int.from_bytes(content[offset+26:offset+28], byteorder='little')
            
            if offset + 30 + filename_len <= len(content):
                filename_bytes = content[offset+30:offset+30+filename_len]
                try:
                    filename = filename_bytes.decode('utf-8', errors='ignore')
                    # Make sure it matches our format (ends with .mp4 and doesn't contain path traversal)
                    if filename and filename.endswith('.mp4') and not filename.startswith('/') and not filename.startswith('\\'):
                        existing_files.add(filename)
                except Exception:
                    pass
            
            # Move offset past signature to find next one
            offset += 4
            
        print(f"[Info] Recovered {len(existing_files)} file(s) from '{os.path.basename(zip_path)}'.")
    except Exception as e:
        print(f"[Warning] Failed to scan '{os.path.basename(zip_path)}': {e}")
        
    return existing_files

def get_existing_files_from_zips(data_dir, zip_path):
    existing_files = set()
    zip_pattern = os.path.join(data_dir, "vn_av_df_capstone_dataset*.zip")
    zip_files = glob.glob(zip_pattern)
    
    for fpath in zip_files:
        print(f"Checking existing zip file: {fpath}")
        try:
            with zipfile.ZipFile(fpath, 'r') as zf:
                files_in_zip = set(zf.namelist())
                existing_files.update(files_in_zip)
                print(f"  --> Found {len(files_in_zip)} valid files in '{os.path.basename(fpath)}'.")
        except zipfile.BadZipFile:
            recovered = recover_filenames_from_bad_zip(fpath)
            existing_files.update(recovered)
            
            # If the bad zip is the main zip, rename it to preserve it, so we don't try to append to it
            if os.path.abspath(fpath) == os.path.abspath(zip_path):
                backup_name = os.path.join(
                    data_dir, 
                    f"vn_av_df_capstone_dataset_corrupted_{int(time.time())}.zip"
                )
                try:
                    os.rename(zip_path, backup_name)
                    print(f"[Warning] Renamed corrupted main ZIP to '{os.path.basename(backup_name)}' to preserve data.")
                except Exception as e:
                    print(f"[Warning] Failed to rename corrupted main ZIP: {e}")
                    
    return existing_files

def download_and_zip_streaming():
    folder_url = "https://drive.google.com/drive/folders/19JnStKpsur54BMOnU7U5KSFoYM8KBdPy?usp=drive_link"
    # Auto-copy local cookies.txt to gdown cache to authorize download requests
    local_cookies = "cookies.txt"
    cache_cookies_dir = os.path.expanduser("~/.cache/gdown")
    cache_cookies_path = os.path.join(cache_cookies_dir, "cookies.txt")
    
    if os.path.exists(local_cookies):
        try:
            os.makedirs(cache_cookies_dir, exist_ok=True)
            shutil.copy2(local_cookies, cache_cookies_path)
            print(f"[Info] Automatically copied cookies.txt to gdown cache: {cache_cookies_path}")
        except Exception as e:
            print(f"[Warning] Failed to copy cookies.txt to cache folder: {e}")
    else:
        print("[Warning] No local cookies.txt found in the project root. Downloads might be throttled/blocked.")
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "vn_av_df_capstone_dataset.zip")
    temp_file = os.path.join(data_dir, "temp_download.mp4")
    
    print("=" * 60)
    print("Google Drive Folder Stream-Zipper (Low Disk Space Mode)")
    print(f"Source Folder  : {folder_url}")
    print(f"Target Zip File: {os.path.abspath(zip_path)}")
    print("=" * 60)
    
    # Step 1: Get list of files in the folder using gdown --json
    print("Step 1: Fetching file list from Google Drive...")
    cmd_json = ["gdown", "--json", "--folder", folder_url]
    
    try:
        # Run gdown to get list of files
        result = subprocess.run(cmd_json, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f"Error fetching folder list (code {result.returncode}):")
            print(result.stderr)
            return
        
        files = extract_json(result.stdout)
        total_files = len(files)
        print(f"Found {total_files} files in the Google Drive folder.")
        
    except Exception as e:
        print(f"Failed to fetch file list: {e}")
        return
        
    # Step 2: Read already zipped files from all existing ZIP archives (including corrupted ones)
    print("Step 2: Checking existing files to resume...")
    existing_files = get_existing_files_from_zips(data_dir, zip_path)
    print(f"Total unique files already zipped: {len(existing_files)}")
            
    # Step 3: Stream download and zip
    print("\nStep 2: Starting streaming download and zip...")
    print("Files will be downloaded one by one, added to the zip, and deleted immediately.")
    print("-" * 60)
    
    success_count = len(existing_files)
    consecutive_failures = 0
    
    for idx, item in enumerate(files, 1):
        filename = item['path']
        download_url = item['url']
        
        # Skip if already in zip
        if filename in existing_files:
            continue
            
        print(f"[{idx}/{total_files}] Processing: {filename}")
        
        # Clean up any leftover temp file from a previous crash
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
            
        # Download the individual file with retries
        cmd_download = ["gdown", download_url, "-O", temp_file, "--quiet"]
        max_retries = 3
        download_success = False
        
        for attempt in range(1, max_retries + 1):
            try:
                # Normal sleep between requests to avoid rate limits
                if attempt == 1:
                    sleep_time = random.uniform(2.0, 4.0)  # Slightly longer default pause to be gentle
                    time.sleep(sleep_time)
                else:
                    # Exponential backoff for retries: 15s, 45s, 135s
                    sleep_time = 15 * (3 ** (attempt - 2))
                    print(f"      [Retry] Attempt {attempt}/{max_retries} failed. Waiting {sleep_time} seconds before retrying...")
                    time.sleep(sleep_time)
                
                dl_result = subprocess.run(cmd_download, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                
                if dl_result.returncode == 0 and os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    download_success = True
                    break
                else:
                    err_reason = dl_result.stderr.strip() if dl_result.stderr else "Empty file downloaded or non-zero exit code"
                    print(f"      [Attempt {attempt} failed] Reason: {err_reason}")
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass
            except Exception as e:
                print(f"      [Attempt {attempt} error] {e}")
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
        
        if not download_success:
            print(f"  --> Failed to download {filename} after {max_retries} attempts. Skipping this file for now...")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("\n[Warning] 3 consecutive files failed to download.")
                print("[Info] This likely means Google Drive is strictly rate limiting this IP address.")
                print("[Info] Pausing for 3 minutes to cool down before next attempts...")
                time.sleep(180)
                consecutive_failures = 0  # reset after pause
            continue
            
        # Reset consecutive failures on success
        consecutive_failures = 0
        
        try:
            # Append the file to the zip archive
            with zipfile.ZipFile(zip_path, 'a', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(temp_file, arcname=filename)
                
            # Delete the temp file immediately
            os.remove(temp_file)
            success_count += 1
            print(f"  --> Successfully added to ZIP and deleted temp file.")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Cleaning up...")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            print("You can run this script again to resume from where you stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"  --> Error processing {filename}: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
                
    print("-" * 60)
    print(f"Completed! {success_count}/{total_files} files are now in the ZIP archive.")
    print(f"Final Zip Path: {os.path.abspath(zip_path)}")

if __name__ == "__main__":
    download_and_zip_streaming()

