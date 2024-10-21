# usage: python scripts/create_post.py "Title of the post" "filename-of-the-post" [-d for draft]

import argparse
import os
from datetime import datetime, timezone, timedelta

def create_post(title, filename, draft):
    current_date = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()

    
    # Lorem ipsum is needed for parsing in main.py -- why not?
    header = f"""---
title: "{title}"
date: {current_date}
draft: {draft}
---

Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

"""
    with open("posts/" + filename + ".md", 'w') as f:
        f.write(header)
    
    print(f"Markdown file '{filename}' created successfully.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("title", help="Title of the post")
    argparser.add_argument("filename", help="File name of the post (exclude path and extension)")
    argparser.add_argument("-d", "--draft", help="Is the post a draft?", action="store_true")

    args = argparser.parse_args()

    draft = "true" if args.draft else "false"
    
    create_post(args.title, args.filename, draft)
