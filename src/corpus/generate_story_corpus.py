import re
from pathlib import Path

import config
import logging
from ai_client.tokenizer import load_tokenizer


logger = logging.getLogger(__name__)

ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.",
    "Jr.", "Sr.", "vs.", "etc.", "e.g.", "i.e."
]


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences while avoiding common
    abbreviation breaks (e.g. Mr., Dr., etc.).
    """

    placeholder_map = {}

    # Protect abbreviations
    for i, abbr in enumerate(ABBREVIATIONS):
        placeholder = f"__ABBR{i}__"
        text = text.replace(abbr, placeholder)
        placeholder_map[placeholder] = abbr

    # Split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Restore abbreviations
    restored = []
    for s in sentences:
        for placeholder, abbr in placeholder_map.items():
            s = s.replace(placeholder, abbr)
        s = s.strip()
        if s:
            restored.append(s)

    return restored


def load_chapter_text(path: Path) -> str:
    """
    Load chapter text while skipping the title line.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ""
    return " ".join(lines[1:]).strip()


def build_story_prompts(chapter_path: Path, output_dir: Path):
    """
    Generate synthetic user prompts from a chapter using a token window.

    Each prompt:
      • begins with "[Trey] "
      • contains roughly DAY_CONTEXT_TOKENS * 3/4 tokens
      • ends on a sentence boundary
      • overlaps the next prompt by one sentence

    Output files are written to:
        ../data/corpus/story_telling

    File naming:
        <chapter_name>_story_<index>.txt
    """

    tokenizer = load_tokenizer()

    max_tokens = int(config.DAY_CONTEXT_TOKENS * 3 / 4)

    text = load_chapter_text(chapter_path)
    sentences = split_sentences(text)

    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    i = 0

    while i < len(sentences):
        current = []
        token_count = 0
        j = i

        # Normal forward window
        while j < len(sentences):
            candidate = " ".join(current + [sentences[j]])
            tokens = tokenizer.encode(candidate, add_special_tokens=False)

            if len(tokens) > max_tokens:
                break

            current.append(sentences[j])
            token_count = len(tokens)
            j += 1

        # If we reached the end and the chunk is too small,
        # expand backward to fill the token budget
        if j == len(sentences) and token_count < max_tokens:
            start = i
            while start > 0:
                candidate = " ".join([sentences[start - 1]] + current)
                tokens = tokenizer.encode(candidate, add_special_tokens=False)

                if len(tokens) > max_tokens:
                    break

                start -= 1
                current.insert(0, sentences[start])
                token_count = len(tokens)

            i = start

        prompt_text = "[Trey] Continuing the story: " + " ".join(current)
        prompts.append(prompt_text)

        if j >= len(sentences):
            break

        # Maintain overlap using the last sentence
        i = max(j - 1, i + 1)

    base_name = chapter_path.stem.replace(" ", "_").replace(",", "")

    for idx, prompt in enumerate(prompts, start=1):
        filename = f"{base_name}_story_{idx:03d}.txt"
        out_path = output_dir / filename
        out_path.write_text(prompt, encoding="utf-8")

        token_count = len(
            tokenizer.encode(prompt, add_special_tokens=False)
        )

        logger.info(
            "Story prompt saved | file=%s | tokens=%d",
            out_path.name,
            token_count,
        )

    return len(prompts)


def process_story_corpus(
    chapters_dir: str = "../data/corpus/chapters",
    output_dir: str = "../data/corpus/story_telling",
    name_filter: str | None = None,
):
    """
    Process chapter files in the corpus directory.

    Parameters
    ----------
    chapters_dir
        Directory containing chapter text files.
    output_dir
        Directory where generated story prompts will be saved.
    name_filter
        Optional substring filter. If provided, only chapter
        filenames containing this text will be processed.
    """

    chapters_path = Path(chapters_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if name_filter:
        chapters = sorted(
            p for p in chapters_path.glob("*.txt")
            if name_filter.lower() in p.name.lower()
        )
    else:
        chapters = sorted(chapters_path.glob("*.txt"))

    total = 0

    for chapter in chapters:
        count = build_story_prompts(chapter, output_path)
        total += count

    logger.info("Generated %d story prompts.", total)