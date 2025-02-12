from fasthtml.common import *
from lucide_fasthtml import Lucide
import yaml

from datetime import datetime

import os

frankenui = (
    Link(rel='stylesheet', href='https://unpkg.com/franken-wc@0.1.0/dist/css/zinc.min.css'),
    Script(src='https://cdn.jsdelivr.net/npm/uikit@3.21.6/dist/js/uikit.min.js'),
    Script(src='https://cdn.jsdelivr.net/npm/uikit@3.21.6/dist/js/uikit-icons.min.js')
)
tailwind = Link(rel="stylesheet", href="/public/app.css", type="text/css")
favicon = Link(rel="icon", href="/public/favicon.png", type="image/png")
custom_styles = Link(rel="stylesheet", href="/public/custom.css", type="text/css")

app, rt = fast_app(
    pico=False,
    hdrs=(
        frankenui,
        tailwind,
        KatexMarkdownJS(),
        custom_styles,
        HighlightJS(langs=['python', 'bash', 'yaml', 'json'], light="atom-one-dark"),
        favicon
    ),
    static_dir='public'
)

def load_content(directory):
    items = []
    dir_path = directory

    for filename in os.listdir(dir_path):
        if filename.endswith('.md'):
            with open(os.path.join(dir_path, filename), 'r') as file:
                content = file.read()
                parts = content.split('---')
                if len(parts) > 2:
                    item = yaml.safe_load(parts[1])
                    item['slug'] = os.path.splitext(filename)[0]
                    item['content'] = parts[2].strip()
                    lines = item['content'].split('\n')
                    if 'excerpt' not in item:
                        for line in lines:
                            if line.strip() and not line.strip().startswith('!['):
                                item['excerpt'] = line.strip()
                                break
                    
                    # Convert date string to datetime object if it exists
                    if 'date' in item and isinstance(item['date'], str):
                        item['date'] = datetime.strptime(item['date'], "%Y-%m-%d")
                    
                    if not item.get("draft", False):
                        items.append(item)
    # Sort items by date, most recent first
    items.sort(key=lambda x: x.get('date', datetime.min), reverse=True)
    return items

def BlogCard(item, *args, **kwargs):
    return Div(
        Div(
            *args,
            A(
                H2(item["title"], cls="text-2xl font-bold font-heading tracking-tight"),
                P(item["date"].strftime("%B %d, %Y"), cls="uk-text-muted uk-text-small uk-text-italic"),
                P(item.get("excerpt", ""), cls="uk-text-muted uk-margin-small-top marked"),
                href=f"/{item['type']}/{item['slug']}",
            ),
            cls="uk-card-body",
        ),
        cls=f"uk-card {kwargs.pop('cls', '')}",
        **kwargs
    )

@rt('/')
def get():
    # Load blog posts
    posts = load_content('posts')
    for post in posts:
        post['type'] = 'posts'  # Add type to distinguish in URLs

    # Load LeetCode problems
    leetcode = load_content('leetcode')
    for problem in leetcode:
        problem['type'] = 'leetcode'  # Add type to distinguish in URLs

    return Title("Nic Hoffs' Blog"), Div(
        H1("my (nic hoffs') blog", cls="text-4xl font-bold font-heading tracking-tight "),
        P(
            "I love training and researching deep models, architecting autonomous race cars, developing web applications, practicing Brazilian-Jiu-Jitsu, playing water polo, and lots of other sh*t.",
            cls="text-lg uk-text-muted uk-margin-small-top"
        ),
        P(
            "Shoutout to",
            A(Strong(" Marius"), href="https://blog.mariusvach.com", style="color: dimgray;"),
            " for this fantastic blog template.",
            cls="text-lg uk-text-muted uk-margin-small-top"
        ),
        Div(
            A(Lucide('mail', cls="w-4 h-4 mr-2"), "Email me", href="mailto:nicthoffs@gmail.com", cls="uk-button uk-button-primary uk-margin-small-top uk-margin-small-right"),
            A(Lucide('github', cls="w-4 h-4 mr-2 text-white"), "GitHub", href="https://github.com/nichoffs", cls="uk-button uk-button-primary  uk-margin-small-right uk-margin-small-top"),
            A(Lucide('scroll-text', cls="w-4 h-4 mr-2 text-white"), "Resume", href="/resume", cls="uk-button uk-button-primary  uk-margin-small-right uk-margin-small-top"),
        ),
        H2("Here are some things I wrote:", cls="text-3xl font-bold font-heading tracking-tight uk-margin-medium-top"),
        Div(
            *[BlogCard(post) for post in posts], 
            cls="md:grid md:grid-cols-3 md:gap-4 uk-margin-top space-y-4 md:space-y-0",
        ),
        # Separator
        Hr(cls="my-8 border-gray-300"),
        H2("Daily Leet:", cls="text-3xl font-bold font-heading tracking-tight uk-margin-medium-top"),
        Div(
            *[BlogCard(problem) for problem in leetcode], 
            cls="md:grid md:grid-cols-3 md:gap-4 uk-margin-top space-y-4 md:space-y-0",
        ),
        cls="uk-container uk-container-xl py-16",
    )

@rt('/posts/{slug}')
def get_post(slug: str):
    with open(f'posts/{slug}.md', 'r') as file:
        content = file.read()
        
    post_content = content.split('---')[2]
    
    frontmatter = yaml.safe_load(content.split('---')[1])
    
    return Title(f"{frontmatter['title']} - Nic Hoffs' Blog"), Div(
        A(Lucide('arrow-left', cls="w-4 h-4 text-black mr-2"), 'Go Back', href='/', cls="absolute md:top-0 left-0 top-2 md:-ml-48 md:mt-16 uk-button uk-button-ghost"),
        H1(frontmatter["title"], cls="text-4xl font-bold font-heading tracking-tight uk-margin-small-bottom"),
        P(frontmatter['date'].strftime("%B %d, %Y"), " by Nicholas Hoffs", cls="uk-text-muted uk-text-small uk-text-italic"),
        Div(post_content, cls="marked prose mx-auto uk-margin-top"),
        cls="uk-container max-w-[65ch] mx-auto relative py-16",
    )

@rt('/leetcode/{slug}')
def get_leetcode(slug: str):
    with open(f'leetcode/{slug}.md', 'r') as file:
        content = file.read()
        
    problem_content = content.split('---')[2]
    
    frontmatter = yaml.safe_load(content.split('---')[1])
    
    return Title(f"{frontmatter['title']} - LeetCode Problem - Nic Hoffs' Blog"), Div(
        A(Lucide('arrow-left', cls="w-4 h-4 text-black mr-2"), 'Go Back', href='/', cls="absolute md:top-0 left-0 top-2 md:-ml-48 md:mt-16 uk-button uk-button-ghost"),
        H1(frontmatter["title"], cls="text-4xl font-bold font-heading tracking-tight uk-margin-small-bottom"),
        P(frontmatter['date'].strftime("%B %d, %Y"), " by Nicholas Hoffs", cls="uk-text-muted uk-text-small uk-text-italic"),
        Div(problem_content, cls="marked prose mx-auto uk-margin-top"),
        cls="uk-container max-w-[65ch] mx-auto relative py-16",
    )

@rt("/resume")
async def get_resume(): 
    return FileResponse(f'public/resume.pdf')

serve()
