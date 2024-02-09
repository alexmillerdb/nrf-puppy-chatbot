import abc
import asyncio
from typing import Protocol, TypeVar, AsyncIterator

import chainlit as cl
from pydantic import BaseModel


class Response(BaseModel):
    response: str

class StreamingResponse(Response):
    pass

class BatchResponse(Response):
    pass

class ChainlitChat(abc.ABC):

    @abc.abstractmethod
    def intro_message(self) -> cl.Message | None:
        ...

    @abc.abstractmethod
    async def complete(self, content: str, input_message: cl.Message, response: cl.Message) -> (
            StreamingResponse | BatchResponse):
        ...

    @abc.abstractmethod
    def complete_sync(self, content: str, input_message: cl.Message, response: cl.Message) -> (
            StreamingResponse | BatchResponse):
        ...


class HasMessage(Protocol):
    message: str


T = TypeVar('T', bound=HasMessage)


class AsyncGeneratorWrapper(AsyncIterator[T]):
    def __init__(self, gen):
        self.gen = gen

    def __aiter__(self) -> 'AsyncGeneratorWrapper[T]':
        return self

    async def __anext__(self) -> T:
        try:
            # Use asyncio to yield control and create asynchronous behavior
            await asyncio.sleep(0)
            return next(self.gen)
        except StopIteration:
            raise StopAsyncIteration
