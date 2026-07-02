from dataclasses import dataclass


@dataclass
class Config:
    name: str
    timeout: int = 30


class BaseHandler:
    def __init__(self, config: Config) -> None:
        self._config = config

    async def handle(self, event: str) -> None:
        result = await self._process(event)
        print(result)

    def _process(self, event: str) -> str:
        return f"processed: {event}"


def compute(a: int, b: int) -> int:
    return a + b
