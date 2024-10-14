import os
import inspect
import markdown
from fasthtml.components import html2ft
from fasthtml.common import (
    FastHTML,
    serve,
    H1,
    H2,
    H3,
    Code,
    Pre,
    A,
    Ul,
    Li,
    Div,
    Main,
    Style,
    P,
    FileResponse,
    KatexMarkdownJS,
    HighlightJS,
    Hr,
)


custom_css = Style("""
@font-face {
    font-family: "commit";
    src: url("./fonts/CommitMono-400-Regular.otf") format("opentype");
    font-weight: normal;
    font-style: normal;
}
@font-face {
    font-family: "commit";
    src: url("fonts/CommitMono-700-Regular.otf") format("opentype");
    font-weight: bold;
    font-style: normal;
}
@font-face {
    font-family: "commit";
    src: url("fonts/CommitMono-400-Italic.otf") format("opentype");
    font-weight: normal;
    font-style: italic;
}
@font-face {
    font-family: "commit";
    src: url("fonts/CommitMono-700-Italic.otf") format("opentype");
    font-weight: bold;
    font-style: italic;
}
* {
    background-color: #eeeeee;
}

main {
    margin: 15px auto;
    width: 900px;
    font-family: "commit";
    font-weight: normal;
}

a {
    color: black;
}
hr {
    border: 1px dashed black;
}
""")


app = FastHTML(hdrs=(custom_css))


@app.get("/")
def home():
    intro = (
        H1("nic hoffs' website"),
        P(
            """I'm posting mostly deep-learning-related stuff here.
            I've recently decided to adopt Jupyter Notebooks as my primary medium for writing code, blog posts, and notes,
            so all the posts are simply Notebooks exported to HTML.
            """
        ),
        P("""Projects, on the other hand, are specific to this website and are written in Python using the fasthtml library
        (which also powers the rest of the non-post parts of the site).
        """),
    )
    info = (
        Div(
            H2("contact"),
            Ul(
                Li(
                    "Email:",
                    A("nicthoffs@gmail.com", href="mailto: nicthoffs@gmail.com"),
                ),
                Li("Phone: 949-280-2672"),
            ),
            H2("work"),
            Ul(
                Li(A("resume", href="/resume")),
                Li(A("github", href="https://github.com/nichoffs")),
            ),
        ),
    )
    posts = (
        H2("posts"),
        Ul(
            *[Li(A(post, href=f"/posts/{post}")) for post in os.listdir("public/posts")]
        ),
    )
    projects = (H2("projects"), P("Nothing here yet... coming soon!"))
    return Main(
        Div(intro, info, posts, projects),
    )


@app.get("/resume")
def resume():
    return FileResponse(f"public/resume.pdf")


@app.get("/{fname:path}.{ext:static}")
async def get(fname: str, ext: str):
    return FileResponse(f"public/{fname}.{ext}")


if __name__ == "__main__":
    serve()
