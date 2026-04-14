from pathlib import Path


class StoryPromptIndexEntry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.is_complete = False
    
    def __str__(self) -> str:
        return f"{self.path}, completed: {self.is_complete}"
    

    def get_prompt(self) -> str:
        return self.path.read_text()
    

    def mark_complete(self) -> None:
        self.is_complete = True

    def reset(self) -> None:
        self.is_complete = False
