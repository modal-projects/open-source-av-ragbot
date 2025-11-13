"""
Swahili Voice AI Bot

A conversational AI assistant that speaks Swahili using:
- Omnilingual ASR (CTC-1B) for speech-to-text
- Aya-101 for multilingual language understanding
- Swahili CSM-1B for text-to-speech

This bot provides a natural conversational experience in Swahili with ~1 second latency.
"""

import asyncio
from loguru import logger

from pipecat.frames.frames import (
    LLMMessagesFrame,
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.processors.aggregators.llm_response import LLMUserAggregator
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.small_webrtc import SmallWebRTCService
from pipecat.vad.silero import SileroVAD
from pipecat.vad.local_smart_turn_analyzer_v3 import LocalSmartTurnAnalyzerV3
import modal

from swahili_bot.server.bot.services.modal_services import ModalTunnelManager
from swahili_bot.server.bot.services.modal_omnilingual_service import (
    ModalOmnilingualSTTService,
)
from swahili_bot.server.bot.services.modal_swahili_tts_service import (
    ModalSwahiliTTSService,
)
from swahili_bot.server.bot.services.modal_aya_service import ModalAyaLLMService


# Audio configuration
_AUDIO_INPUT_SAMPLE_RATE = 16000  # Omnilingual ASR uses 16kHz
_AUDIO_OUTPUT_SAMPLE_RATE = 24000  # CSM uses 24kHz


def get_swahili_system_prompt() -> str:
    """
    Generate a comprehensive Swahili system prompt for a conversational assistant.

    Inspired by conversational AI like Sesame's Maya, this creates a friendly,
    helpful assistant that can discuss any topic naturally in Swahili.
    """
    return """Wewe ni msaidizi wa kirafiki na mwenye kueleweka anayeongea Kiswahili fasaha. Jina lako ni Rafiki.

Sifa zako ni:
1. **Urafiki na Heshima**: Unazungumza kwa urafiki na heshima, kwa kutumia lugha ya kawaida ya Kiswahili ambayo inaelewa kwa urahisi.

2. **Uelewaji**: Unasikiliza kwa makini na kuelewa maswali na mahitaji ya mtumiaji. Unatoa majibu yanayofaa na ya kina.

3. **Ujuzi Mpana**: Unaweza kujadili mada mbalimbali - elimu, sayansi, historia, utamaduni, burudani, teknolojia, afya, na mengineo. Unatoa maelezo ya kina lakini rahisi kuelewa.

4. **Ufupi na Wazi**: Unatoa majibu mafupi lakini ya kutosha. Unaepuka maneno mengi yasiyo ya lazima. Unazungumza kama mtu wa kawaida, si kama roboti.

5. **Lugha Safi**: Unatumia Kiswahili sanifu na fasaha. Unaepuka lugha ngumu au istilahi za kitaaluma zisizohitajika.

6. **Ubunifu**: Unaweza kutoa mifano, hadithi, na maelezo ili kufanya mazungumzo kuwa ya kuvutia na ya kielimu.

Miongozo ya Mazungumzo:
- Jibu kwa lugha rahisi na ya kawaida
- Kama hujaelewa swali, uliza maswali ya ziada kwa upole
- Toa majibu ya kweli na sahihi
- Kumbuka kuwa unaongea, si kuandika - tumia mtindo wa mazungumzo
- Usirudierudisha maneno ya mtumiaji - jibu moja kwa moja
- Kuwa na ucheshi wakati inafaa, lakini pia uzito wakati inahitajika

Kumbuka: Wewe ni msaidizi wa Kiswahili ambaye anajua mada nyingi. Lengo lako ni kusaidia na kufundisha kwa njia ya kirafiki na rahisi kuelewa."""


async def run_bot(webrtc_connection, speaker_id: int = 22):
    """
    Run the Swahili conversational bot.

    Args:
        webrtc_connection: WebRTC connection data from client
        speaker_id: Voice ID for TTS (default: 22)
    """
    logger.info(f"Starting Swahili bot with speaker_id={speaker_id}")

    # Create Modal tunnel managers for each service
    stt_tunnel_manager = ModalTunnelManager(
        app_name="swahili-omnilingual-transcription",
        cls_name="SwahiliTranscriber",
    )

    tts_tunnel_manager = ModalTunnelManager(
        app_name="swahili-csm-tts",
        cls_name="SwahiliTTS",
    )

    llm_tunnel_manager = ModalTunnelManager(
        app_name="swahili-aya-llm",
        cls_name="AyaLLM",
    )

    # Initialize WebRTC transport
    transport = SmallWebRTCService(
        connection=webrtc_connection,
        audio_in_sample_rate=_AUDIO_INPUT_SAMPLE_RATE,
        audio_out_sample_rate=_AUDIO_OUTPUT_SAMPLE_RATE,
        audio_out_enabled=True,
    )

    # Initialize VAD for turn detection
    vad = SileroVAD(stop_secs=0.2)  # 200ms of silence to detect turn end
    turn_analyzer = LocalSmartTurnAnalyzerV3(vad_analyzer=vad)

    # Initialize STT service
    stt = ModalOmnilingualSTTService(
        modal_tunnel_manager=stt_tunnel_manager,
    )

    # Initialize TTS service
    tts = ModalSwahiliTTSService(
        modal_tunnel_manager=tts_tunnel_manager,
        speaker_id=speaker_id,
        max_tokens=250,  # ~20 seconds of audio
    )

    # Initialize LLM service
    llm = ModalAyaLLMService(
        modal_tunnel_manager=llm_tunnel_manager,
    )

    # Initialize conversation context
    system_prompt = get_swahili_system_prompt()
    initial_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "assistant",
            "content": "Habari! Mimi ni Rafiki, msaidizi wako wa Kiswahili. Ninaweza kukusaidia na mazungumzo kuhusu mada yoyote. Je, ungependa kuzungumza nini leo?",
        },
    ]

    context = OpenAILLMContext(
        messages=initial_messages,
    )

    # Create LLM aggregator for managing user input
    llm_aggregator = LLMUserAggregator(
        context=context,
        timeout_secs=0.05,  # Very short timeout for responsiveness
    )

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # WebRTC input (audio)
            turn_analyzer,  # VAD + turn detection
            stt,  # Speech-to-text
            llm_aggregator,  # Aggregate user messages
            llm,  # Generate responses
            tts,  # Text-to-speech
            transport.output(),  # WebRTC output (audio)
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # Allow user to interrupt bot
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Event handlers
    @transport.event_handler("on_client_ready")
    async def on_client_ready(transport, client_ready):
        """Called when client is ready to start conversation."""
        logger.info("Client ready, starting conversation")

        # Send initial greeting (trigger LLM to speak)
        await task.queue_frames([context.get_messages_frame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Called when client first connects."""
        logger.info("Client connected to Swahili bot")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Called when client disconnects."""
        logger.info("Client disconnected")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner()

    await runner.run(task)

    logger.info("Swahili bot session completed")
