import praw
import os
from dotenv import load_dotenv
import csv
import datetime
import time

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
SEARCH_QUERY = 'gemini'  # <----------------------------------- 검색할 키워드
SUBREDDIT_TO_SEARCH = 'all' # <-------------------------------- 서브레딧
LIMIT_POSTS = 25  # <------------------------------------------ 최대로 가져올 게시글 수
CSV_FILENAME = '../raw_data/gemini/2025_6_gemini.csv'

CSV_HEADER = ['Post_ID', 'Type', 'Data', 'Timestamp']

# 날짜 범위 설정 (UTC 기준) <--------------------------------------- 날짜 수정

START_DATE = datetime.datetime(2025, 6, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
END_DATE = datetime.datetime(2025, 6, 18, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp()

# Helper function for timestamp formatting
def format_timestamp(utc_timestamp):
    """UTC 타임스탬프를 'YYYY-MM-DD HH:MM:S' 형식 문자열로 변환"""
    if utc_timestamp is None:
        return ""
    try:
        return datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(utc_timestamp)

try:
    # --- Initialize PRAW ---
    print("Initializing PRAW...")
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
    print(f"PRAW Initialized. Read-only: {reddit.read_only}")

    # --- Open CSV File and Write Header ---
    print(f"Opening {CSV_FILENAME} for writing...")
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        print("CSV header written.")

        print(f"Searching r/{SUBREDDIT_TO_SEARCH} for '{SEARCH_QUERY}' (limit: {LIMIT_POSTS or 'all'})...")
        post_count = 0
        processed_post_ids = set()

        # --- Outer Loop: Submissions ---
        search_limit = LIMIT_POSTS * 8 if LIMIT_POSTS else None
        for submission in reddit.subreddit(SUBREDDIT_TO_SEARCH).search(SEARCH_QUERY, limit=search_limit):
            time.sleep(1)
            # 게시글의 생성 시간 확인
            submission_time_utc = submission.created_utc
            if not (START_DATE <= submission_time_utc <= END_DATE):
                continue  # 날짜 범위 밖이면 건너뛰기

            if submission.id in processed_post_ids:
                continue

            if LIMIT_POSTS is not None and post_count >= LIMIT_POSTS:
                print(f"\nReached the specified limit of {LIMIT_POSTS} posts.")
                break

            post_count += 1
            processed_post_ids.add(submission.id)
            print(f"\nProcessing Post {post_count}/{LIMIT_POSTS or 'N/A'}: ID {submission.id} - '{submission.title[:60]}...'")

            post_id = submission.id
            submission_time = format_timestamp(submission_time_utc)

            # --- Row for Submission Title (Type 0) ---
            try:
                writer.writerow([post_id, 0, submission.title, submission_time])
            except Exception as write_error:
                print(f"  Error writing title for post {post_id}: {write_error}")

            # --- Row for Submission Body (Type 1) ---
            if submission.is_self and submission.selftext:
                try:
                    writer.writerow([post_id, 1, submission.selftext, submission_time])
                except Exception as write_error:
                    print(f"  Error writing body for post {post_id}: {write_error}")

            # --- Rows for Comments (Type 2) ---
            print(f"  Fetching comments for post {post_id}...")
            comment_count = 0
            try:
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if hasattr(comment, "body") and hasattr(comment, "created_utc"):
                        comment_body = comment.body
                        if not comment_body or comment_body in ['[deleted]', '[removed]']:
                            continue

                        comment_time = format_timestamp(comment.created_utc)
                        writer.writerow([post_id, 2, comment_body, comment_time])
                        comment_count += 1

                print(f"  Processed {comment_count} comments for post {post_id}.")

            except praw.exceptions.PRAWException as comment_praw_error:
                print(f"  PRAW Error fetching/processing comments for post {post_id}: {comment_praw_error}")
            except Exception as comment_error:
                print(f"  General Error fetching/processing comments for post {post_id}: {comment_error}")

        if post_count < (LIMIT_POSTS or 0):
            print(f"\nWarning: Found only {post_count} posts matching the query and date range.")
        elif not processed_post_ids:
            print("\nNo posts found matching the query and date range.")
        else:
            print(f"\nFinished processing {post_count} posts.")

    print(f"\nData successfully saved to {CSV_FILENAME}")

except praw.exceptions.PRAWException as e:
    print(f"\nA PRAW Error occurred: {e}")
    print("Check your Reddit API credentials, user agent, and network connection.")
except FileNotFoundError:
    print(f"Error: Could not open or write to the file {CSV_FILENAME}. Check permissions or path.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
