"""
preprocessor.py
---------------
Handles loading and cleaning of the 20 Newsgroups corpus.

DESIGN DECISIONS:
- We use sklearn's load_files to read from the locally extracted dataset folder
  (./20_newsgroups) instead of fetch_20newsgroups auto-download. This is because
  the user has the dataset already extracted locally.
- encoding='latin-1' is critical: these are 1990s Usenet posts and are NOT utf-8
  encoded. Using utf-8 will crash with a UnicodeDecodeError on many files.
  decode_error='replace' handles any remaining bad bytes gracefully.
- We strip headers, footers, and quotes manually via clean_text() for the same
  reasons as before:
    * Headers (From:, Subject:, NNTP-Posting-Host:) are metadata, not semantics.
      They introduce spurious similarity (e.g. two posts from the same server
      look similar even if their topics differ).
    * Quoted reply chains (lines starting with '>') are duplicated content from
      other posts and add noise to embeddings.
    * Footers (.sig blocks, disclaimers) are boilerplate.
- We apply a minimum word-count filter of 50 words. Very short posts
  (1-2 sentences) don't embed meaningfully — the model has too little signal
  to place them accurately in the semantic space.
- We do NOT apply stemming or lemmatization. Sentence-transformers are
  trained on raw text and handle morphology internally. Pre-processing
  that strips word endings would degrade embedding quality.
"""

import re
import os
from sklearn.datasets import load_files


MIN_WORD_COUNT = 50       # Posts shorter than this are too sparse to embed reliably
DATASET_PATH = "./20_newsgroups"  # Path to the locally extracted dataset folder


def load_raw_corpus(subset: str = "all") -> tuple[list[str], list[int], list[str]]:
    """
    Loads the 20 Newsgroups dataset from the local extracted folder.

    Args:
        subset: ignored here (local folder has all data together),
                kept for API compatibility with the rest of the codebase.

    Returns:
        texts       - list of raw document strings
        labels      - list of integer category labels (0–19)
        label_names - the 20 category names in index order
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder '{DATASET_PATH}' not found.\n"
            f"Please extract twenty_newsgroups.zip and then run:\n"
            f"  tar -xzf 20_newsgroups.tar.gz\n"
            f"so that a '20_newsgroups/' folder exists in your project directory."
        )

    print(f"[Preprocessor] Loading dataset from '{DATASET_PATH}'...")

    dataset = load_files(
        DATASET_PATH,
        encoding="latin-1",       # critical — these files are NOT utf-8
        decode_error="replace",   # replace any remaining undecodable bytes
    )

    return dataset.data, dataset.target.tolist(), dataset.target_names


def clean_text(text: str) -> str:
    """
    Light-touch cleaning of a single document.

    We deliberately avoid heavy NLP preprocessing (stop-word removal, stemming)
    because sentence-transformers expect natural language input. Over-cleaning
    hurts embedding quality more than it helps.

    We manually strip:
    - Email header lines (From:, Subject:, Lines:, etc.)
    - Quoted reply lines (starting with '>')
    - Signature blocks (lines after '-- ')
    - Lines that are purely punctuation/symbols (common in newsgroup ASCII art)
    """
    # Remove email header block — everything before the first blank line
    # Headers always appear at the top and are separated from the body by \n\n
    if "\n\n" in text:
        text = text.split("\n\n", 1)[1]

    # Remove quoted reply lines (lines starting with '>' possibly preceded by spaces)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()

        # Skip quoted lines
        if stripped.startswith(">"):
            continue

        # Stop at signature block
        if stripped == "--":
            break

        # Skip lines that are purely punctuation/symbols (ASCII art, separators)
        if len(re.sub(r"[^a-zA-Z0-9]", "", stripped)) <= 2:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse excessive whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def build_corpus(subset: str = "all") -> tuple[list[str], list[int], list[str]]:
    """
    Full pipeline: load → clean → filter short docs.

    Returns parallel lists: (texts, labels, label_names)
    Indices align — texts[i] has ground-truth label labels[i].
    """
    raw_texts, labels, label_names = load_raw_corpus(subset)

    cleaned_texts, cleaned_labels = [], []
    skipped = 0

    for text, label in zip(raw_texts, labels):
        cleaned = clean_text(text)
        word_count = len(cleaned.split())

        if word_count < MIN_WORD_COUNT:
            # Too short to embed meaningfully — discard
            skipped += 1
            continue

        cleaned_texts.append(cleaned)
        cleaned_labels.append(label)

    print(f"[Preprocessor] Loaded {len(cleaned_texts)} documents "
          f"(skipped {skipped} below {MIN_WORD_COUNT} words)")

    return cleaned_texts, cleaned_labels, label_names