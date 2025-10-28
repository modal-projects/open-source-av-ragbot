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
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)
        self._register_event_handler("on_connection_error")
        super().__init__(**kwargs)
        self._websocket_url = websocket_url
        if dict_name:
            key = dict_url_key or "websocket_url"
            url_dict = modal.Dict.from_name(dict_name)
            self._websocket_url = url_dict.get(key)
        if not self._websocket_url:
            raise ValueError("Websocket URL is required")
        self._receive_task = None

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error(error)

    async def _connect(self):
        """Connect to WebSocket and start background tasks."""
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

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns the active WebSocket connection instance, raising an exception
        if no connection is currently established.

        Returns:
            The active WebSocket connection instance.

        Raises:
            Exception: If no WebSocket connection is currently active.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the Websocket service.

        Initializes the service by constructing the WebSocket URL with all configured
        parameters and establishing the connection to begin transcription processing.

        Args:
            frame: The start frame containing initialization parameters and metadata.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Websocket service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()
