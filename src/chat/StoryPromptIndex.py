import logging
import pickle
from pathlib import Path
from typing import Self

from chat import StoryPromptIndexEntry


STORY_PROMPT_PATH = Path("../data/corpus/story_telling")
STORY_PROMPT_INDEX = STORY_PROMPT_PATH / "index.pkl"
logger = logging.getLogger(__name__)


class StoryPromptIndex:
    """
    Tracks how far into the story_telling prompts we have gone
    and provides the next story prompt on-demand.
    """


    entries: list[StoryPromptIndexEntry]


    def __init__(self) -> None:
        self.entries = []

    
    def __iter__(self):
        return self.entries.__iter__()

    
    def add_entry(self, path: Path) -> StoryPromptIndexEntry:
        new_entry = StoryPromptIndexEntry(path)
        self.entries.append(new_entry)
        return new_entry


    def reset(self) -> None:
        for e in self.entries:
            e.reset()


    def get_next_entry(self) -> StoryPromptIndexEntry | None:
        entry = next((e for e in self.entries if not e.is_complete), None)
        if entry is None:
            return None
        return entry
        

    def build_index() -> Self:
        story_index = StoryPromptIndex()
        files = sorted(
            f for f in STORY_PROMPT_PATH.glob("*.txt")
        )
        for path in files:
            logger.info("Path: %s", path)
            story_index.add_entry(path)
        return story_index


    def save(self) -> None:
        index_file = STORY_PROMPT_INDEX.open("wb")
        pickle.dump(self, index_file)
        index_file.close()
        logger.info(f"Saved story index to %s.", STORY_PROMPT_INDEX)

    
    def load() -> Self:
        try:
            index_file = STORY_PROMPT_INDEX.open("rb")
            story_index = pickle.load(index_file)
            index_file.close()
            logger.info(f"Story index loaded.")
            return story_index
        except FileNotFoundError:
            story_index = StoryPromptIndex.build_index()
            story_index.save()
            return story_index
        except PermissionError:
            logger.critical("Permission denied to access or create the file.")
            raise
