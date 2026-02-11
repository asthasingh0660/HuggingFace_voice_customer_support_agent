import requests
from bs4 import BeautifulSoup

URLS = [
    "https://huggingface.co/docs/transformers/index",
    "https://huggingface.co/docs/transformers/installation",
    "https://huggingface.co/docs/transformers/quicktour"
]

def fetch_docs():
    all_text = []

    for url in URLS:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

        # Hugging Face docs put real content inside <main> Navigation/menu/footer are removed
        main = soup.find("main")
        if not main:
            continue

        text = main.get_text(separator="\n")
        all_text.append(text)

    return "\n\n".join(all_text)


if __name__ == "__main__":
    data = fetch_docs()
    with open("hf_docs.txt", "w", encoding="utf-8") as f:
        f.write(data)

    print("Docs saved to hf_docs.txt")
