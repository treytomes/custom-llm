import logging
import pickle
from pathlib import Path
from typing import Self

import chat.repl
from chat.StoryPromptIndex import StoryPromptIndex


STORY_PROMPT_PATH = Path("../data/corpus/story_telling")
STORY_PROMPT_INDEX = STORY_PROMPT_PATH / "index.pkl"
logger = logging.getLogger(__name__)


def run_story_repl() -> None:
    story_index = StoryPromptIndex.load()
    
    # for entry in story_index:
    #     logger.info("Story entry: %s", entry)

    while True:
        entry = story_index.get_next_entry()
        prompt = entry.get_prompt()
        logger.info("Next session: '%s", str(entry))
        logger.info("Prompt: '%s'", prompt)

        # Execute REPL against `prompt``
        logger.info("Beginning story-telling session.")
        chat.repl.run_chat_repl(prompt, False, True)
        logger.info("Story-telling complete for the day.")

        # Mark the story entry as complete and save the update.
        entry.mark_complete()
        story_index.save()

    return
