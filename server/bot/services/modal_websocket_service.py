from loguru import logger

from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame, 
    StartFrame, 
    EndFrame, 
    CancelFrame, 
)
from pipecat.services.websocket_service import WebsocketService

import modal

class ModalWebsocketService(WebsocketService):
    def __init__(
        self, 
        websocket_url: str = None,
        dict_name: str = None,
        dict_url_key: str = None,
        reconnect_on_error: bool = True,
        **kwargs
    ):
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)
        # self._register_event_handler("on_connection_error")
        self._websocket_url = websocket_url
        if dict_name:
            key = dict_url_key or "websocket_url"
            url_dict = modal.Dict.from_name(dict_name)
            self._websocket_url = url_dict.get(key)
        if not self._websocket_url:
            raise ValueError("Websocket URL is required")
        else:
            print(f"Websocket URL: {self._websocket_url}")
        self._receive_task = None

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error(error)

    async def _connect(self):
        """Connect to WebSocket and start background tasks."""
        print(f"Connect: {self._websocket_url}")
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
    
    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks."""
        try:
            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            # Reset state only after everything is cleaned up
            self._websocket = None

    async def _connect_websocket(self):
        """Establish WebSocket connection to API."""
        print(f"Connecting to WebSocket: {self._websocket_url}")
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._websocket = await websocket_connect(
                self._websocket_url,
            )
            logger.debug("Connected to Modal Websocket")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Modal Websocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

