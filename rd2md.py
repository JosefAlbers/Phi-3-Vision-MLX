import praw
import markdown
import os
import requests
from datetime import datetime
from urllib.parse import urlparse
import re
import textwrap
import argparse

def get_reddit_instance(client_id=None, client_secret=None, user_agent="praw_bot"):
    client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "praw_bot")
    if not client_id or not client_secret:
        raise ValueError("Client ID and Client Secret must be provided either as arguments or environment variables.")
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

def is_interesting(post, score_threshold, comment_threshold):
    score_check = post.score > score_threshold
    comment_check = post.num_comments > comment_threshold
    return score_check and comment_check and not post.stickied

def is_image_url(url):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    parsed = urlparse(url)
    return parsed.path.lower().endswith(image_extensions)

def download_image(url, folder):
    try:
        if not url.startswith('http'):
            return None
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join(folder, os.path.basename(urlparse(url).path))
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except requests.exceptions.RequestException:
        print(f"Failed to download image from {url}")
    return None

def extract_image_urls(text):
    pattern = r'\[.*?\]\((https?://\S+\.(?:jpg|jpeg|png|gif))\)'
    return re.findall(pattern, text)

def format_comment(comment, depth=0, upvote_threshold=2):
    if comment.score < upvote_threshold:
        return ""
    indent = "  " * depth
    author_line = f"{indent}- u/{comment.author}:\n"
    dedented_body = textwrap.dedent(comment.body).strip()
    indented_body = textwrap.indent(dedented_body, indent + '  ')
    comment_block = f"{indent + '  '}```\n{indented_body}\n{indent + '  '}```\n\n"
    formatted = author_line + comment_block
    for reply in comment.replies:
        formatted += format_comment(reply, depth + 1, upvote_threshold)

    return formatted

def save_to_markdown(reddit, subreddit_name, limit, score_threshold, comment_threshold, comment_score_threshold, verbose, post_url):
    post_contents = []
    post_images = []
    if post_url:
        interesting_posts = [reddit.submission(url=post_url)]
    else:
        subreddit = reddit.subreddit(subreddit_name)
        interesting_posts = []
        for post in subreddit.hot(limit=None):
            if is_interesting(post, score_threshold, comment_threshold):
                interesting_posts.append(post)
                if len(interesting_posts) >= limit:
                    break
    if interesting_posts:
        date_str = datetime.now().strftime('%Y-%m-%d')
        base_folder = f"{subreddit_name}_posts_{date_str}"
        os.makedirs(base_folder, exist_ok=True)
        images_folder = os.path.join(base_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        for post in interesting_posts:
            post_content = []
            post_image = None
            post_content.append(f"## {post.title}\n\n")
            if verbose:
                post_content.append(f"* Author: u/{post.author}\n")
                post_content.append(f"* URL: {post.url}\n")
                post_content.append(f"* Score: {post.score}\n\n")
            post_content.append("### Post:\n\n")
            if post.is_self:
                content = post.selftext
                content = content.replace('\n#', '\n####') # md headings
                image_urls = extract_image_urls(content)
                for img_url in image_urls:
                    local_path = download_image(img_url, images_folder)
                    if local_path:
                        relative_path = os.path.relpath(local_path, base_folder)
                        content = content.replace(img_url, relative_path)
                post_content.append(f"{content}\n\n")
                post_image = image_urls if len(image_urls) > 0 else None
            elif is_image_url(post.url):
                local_path = download_image(post.url, images_folder)
                if local_path:
                    relative_path = os.path.relpath(local_path, base_folder)
                    post_content.append(f"![Post Image]({relative_path})\n\n")
                    post_image = local_path
            else:
                post_content.append(f"[Link to content]({post.url})\n\n")
            if post.thumbnail and post.thumbnail.startswith('http') and verbose:
                local_path = download_image(post.thumbnail, images_folder)
                if local_path:
                    relative_path = os.path.relpath(local_path, base_folder)
                    post_content.append(f"Thumbnail: ![Thumbnail]({relative_path})\n\n")
            post_content.append("### Comments:\n\n")
            post.comments.replace_more(limit=None)
            for comment in post.comments:
                post_content.append(format_comment(comment, upvote_threshold=comment_score_threshold))
            post_content.append("---\n\n")
            post_contents.append(''.join(post_content))
            post_images.append(post_image)
        all_content = ''.join(post_contents)
        if verbose:
            all_content = f"# Interesting posts from r/{subreddit_name}\n\n" + all_content
        filename = os.path.join(base_folder, f"{datetime.now().strftime('%H_%M_%S')}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(all_content)
        print(f"Saved interesting posts to {filename}")
        return filename, post_contents, post_images
    else:
        print("No interesting posts found.")
        return None, None, None

def rd2md(client_id=None, client_secret=None, user_agent="praw_bot", subreddit_name="LocalLLaMA", limit=3, score_threshold=30, comment_threshold=10, comment_score_threshold=2, verbose=False, post_url=None):
    reddit = get_reddit_instance(client_id, client_secret, user_agent)
    filename, list_contents, list_images = save_to_markdown(reddit, subreddit_name, limit, score_threshold, comment_threshold, comment_score_threshold, verbose, post_url)
    return filename, list_contents, list_images

def main():
    parser = argparse.ArgumentParser(description="Reddit Scraper Bot")
    parser.add_argument("--client_id", help="Reddit API client ID")
    parser.add_argument("--client_secret", help="Reddit API client secret")
    parser.add_argument("--user_agent", default="praw_bot", help="User agent for Reddit API")
    parser.add_argument("--subreddit", default="LocalLLaMA", help="Subreddit to scrape")
    parser.add_argument("--limit", type=int, default=3, help="Number of posts to scrape")
    args = parser.parse_args()
    rd2md(args.client_id, args.client_secret, args.user_agent, args.subreddit, args.limit)

if __name__ == "__main__":
    main()
