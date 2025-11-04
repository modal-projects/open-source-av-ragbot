from loguru import logger
import uuid
import asyncio 

from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import WebsocketService

import modal

class ModalWebsocketService(WebsocketService):
    def __init__(
        self, 
        app_name: str = None,
        cls_name: str = None,
        websocket_url: str = None,
        reconnect_on_error: bool = True,
        **kwargs
    ):
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)
        self._websocket_url = websocket_url
        self._app_name = app_name
        self._cls_name = cls_name

        self._receive_task = None

        self.call_id = None
        self.modal_client_id = str(uuid.uuid4())
        self.registry_dict = modal.Dict.from_name(f"{self.modal_client_id}-websocket-client-registry", create_if_missing=True)
        self.registry_dict.put(self.modal_client_id, True)
        if self._app_name and self._cls_name:
            print(f"Spawning service for {self._app_name}.{self._cls_name} with client id {self.modal_client_id}")
            ws_service = modal.Cls.from_name(self._app_name, self._cls_name)()
            self.call_id = ws_service.register_client.spawn(self.registry_dict, self.modal_client_id)
        elif not self._websocket_url:
            raise Exception("Either app_name and cls_name or websocket_url must be provided")
        else:
            print(f"Using websocket URL: {self._websocket_url}")
        

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error(error)

    async def _connect(self):
        """Connect to WebSocket and start background tasks."""

        retries = 240 # 2 minutes
        while self._websocket_url is None and retries > 0:
            retries -= 1
            self._websocket_url = await self.registry_dict.get.aio("url")
            await asyncio.sleep(0.500)
        if self._websocket_url is None:
            raise Exception("Failed to get websocket URL")
        
        print(f"Connectingo to: {self._websocket_url}")
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        print(f"Connected to: {self._websocket_url}")
    
    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks."""
        try:
            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

            if self.call_id:
                self.registry_dict.put(self.modal_client_id, False)
                self.call_id.gather()
                self.call_id = None
    
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            # Reset state only after everything is cleaned up
            self._websocket = None
            if self.call_id:
                self.call_id.cancel()
                self.call_id = None
            modal.Dict.objects.delete(f"{self.modal_client_id}-websocket-client-registry")

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

