# data_downloader.py

# Define a list of URLs for the documentation of HTML, CSS, JS, React, Git, and TS.
# We will use these URLs to download the content and extract code snippets.

URLS = {
    "html": "https://developer.mozilla.org/en-US/docs/Web/HTML",
    "css": "https://developer.mozilla.org/en-US/docs/Web/CSS",
    "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "react": "https://react.dev/",
    "git": "https://git-scm.com/doc",
    "typescript": "https://www.typescriptlang.org/docs/",
}

# Add imports for requests and Beautiful Soup
import requests
from bs4 import BeautifulSoup

def download_content(url: str) -> str | None:
    """Downloads the content from a given URL.

    Args:
        url: The URL to download the content from.

    Returns:
        The content of the URL as a string, or None if an error occurs.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_code_snippets(html_content: str) -> list[str]:
    """Extracts code snippets from HTML content.

    Args:
        html_content: The HTML content to extract code snippets from.

    Returns:
        A list of code snippets found in the HTML content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    code_snippets = []
    # This is a basic implementation, it might need to be adjusted based on the
    # specific structure of the documentation websites.
    for code_tag in soup.find_all(["code", "pre"]):
        code_snippets.append(code_tag.get_text())
    return code_snippets

def save_snippets_to_file(tech: str, snippets: list[str]):
    """Saves the extracted code snippets to a file.

    Args:
        tech: The technology name (e.g., html, css).
        snippets: A list of code snippets to save.
    """
    # Create a directory for the snippets if it doesn't exist
    import os
    output_dir = "code_snippets"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{tech}_snippets.txt")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for i, snippet in enumerate(snippets):
                f.write(f"--- Snippet {i+1} ---\n")
                f.write(snippet)
                f.write("\n\n")
        print(f"Successfully saved snippets for {tech} to {filepath}")
    except IOError as e:
        print(f"Error saving snippets for {tech} to {filepath}: {e}")

if __name__ == "__main__":
    for tech, url in URLS.items():
        print(f"Processing documentation for {tech} from: {url}")
        content = download_content(url)
        if content:
            print(f"Successfully downloaded content for {tech}.")
            snippets = extract_code_snippets(content)
            if snippets:
                print(f"Found {len(snippets)} code snippets for {tech}.")
                save_snippets_to_file(tech, snippets)
            else:
                print(f"No code snippets found for {tech}.")
        else:
            print(f"Failed to download content for {tech}.")
