import React, { useState, useCallback, useRef, useEffect } from 'react';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import './app.css';

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected';

export default function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle');
  const [speakerId, setSpeakerId] = useState<number>(22);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [isMuted, setIsMuted] = useState(false);

  const transportRef = useRef<SmallWebRTCTransport | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const connect = useCallback(async () => {
    if (connectionState === 'connected') {
      // Disconnect
      transportRef.current?.disconnect();
      transportRef.current = null;
      setConnectionState('idle');
      setTranscript([]);
      return;
    }

    setConnectionState('connecting');
    setTranscript(['Inaunganisha... Tafadhali subiri.']);

    try {
      // Create transport
      const transport = new SmallWebRTCTransport({
        audioElement: audioRef.current!,
        callbacks: {
          onConnected: () => {
            console.log('Connected to bot');
            setConnectionState('connected');
            setTranscript(['Imeunganishwa! Anza kuongea...']);
          },
          onDisconnected: () => {
            console.log('Disconnected from bot');
            setConnectionState('disconnected');
            setTranscript(prev => [...prev, 'Umeondoka.']);
          },
          onBotTranscript: (text: string) => {
            console.log('Bot transcript:', text);
            setTranscript(prev => [...prev, `Rafiki: ${text}`]);
          },
          onUserTranscript: (text: string) => {
            console.log('User transcript:', text);
            setTranscript(prev => [...prev, `Wewe: ${text}`]);
          },
          onError: (error: Error) => {
            console.error('Transport error:', error);
            setConnectionState('disconnected');
            setTranscript(prev => [...prev, `Hitilafu: ${error.message}`]);
          },
        },
      });

      transportRef.current = transport;

      // Connect to backend
      const response = await fetch('/offer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: transport.offer,
          speaker_id: speakerId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to connect to bot');
      }

      const data = await response.json();
      await transport.setRemoteSDP(data.sdp);

    } catch (error) {
      console.error('Connection error:', error);
      setConnectionState('idle');
      setTranscript(['Hitilafu wakati wa kuunganisha. Tafadhali jaribu tena.']);
    }
  }, [connectionState, speakerId]);

  const toggleMute = useCallback(() => {
    if (transportRef.current) {
      const newMutedState = !isMuted;
      transportRef.current.enableMic(!newMutedState);
      setIsMuted(newMutedState);
    }
  }, [isMuted]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (transportRef.current) {
        transportRef.current.disconnect();
      }
    };
  }, []);

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">🎤 Rafiki</h1>
          <p className="subtitle">Msaidizi Wako wa Kiswahili</p>
        </header>

        <div className="controls">
          <div className="speaker-control">
            <label htmlFor="speaker-id">
              Sauti (Speaker ID):
              <input
                id="speaker-id"
                type="number"
                min="0"
                max="100"
                value={speakerId}
                onChange={(e) => setSpeakerId(parseInt(e.target.value) || 22)}
                disabled={connectionState === 'connected'}
                className="speaker-input"
              />
            </label>
            <span className="speaker-hint">
              Badilisha nambari ili kubadilisha sauti (kwa mfano: 22, 25, 30)
            </span>
          </div>

          <button
            onClick={connect}
            disabled={connectionState === 'connecting'}
            className={`connect-button ${connectionState === 'connected' ? 'connected' : ''}`}
          >
            {connectionState === 'idle' && '🎙️ Anza Mazungumzo'}
            {connectionState === 'connecting' && '⏳ Inaunganisha...'}
            {connectionState === 'connected' && '🔴 Ondoka'}
            {connectionState === 'disconnected' && '🔄 Unganisha Tena'}
          </button>

          {connectionState === 'connected' && (
            <button
              onClick={toggleMute}
              className={`mute-button ${isMuted ? 'muted' : ''}`}
            >
              {isMuted ? '🔇 Washa Sauti' : '🔊 Zima Sauti'}
            </button>
          )}
        </div>

        <div className="transcript-container">
          <h2 className="transcript-title">Mazungumzo</h2>
          <div className="transcript">
            {transcript.length === 0 ? (
              <p className="transcript-empty">
                Bonyeza "Anza Mazungumzo" ili kuanza kuzungumza na Rafiki.
              </p>
            ) : (
              transcript.map((line, index) => (
                <p key={index} className="transcript-line">
                  {line}
                </p>
              ))
            )}
          </div>
        </div>

        <div className="info">
          <h3>📚 Maelekezo</h3>
          <ul>
            <li>Bonyeza "Anza Mazungumzo" ili kuunganisha</li>
            <li>Zungumza Kiswahili na Rafiki atakujibu</li>
            <li>Unaweza kubadilisha Speaker ID ili kubadilisha sauti</li>
            <li>Rafiki anaweza kujadili mada yoyote - elimu, historia, sayansi, n.k.</li>
          </ul>
        </div>

        <footer className="footer">
          <p>
            Powered by{' '}
            <a href="https://modal.com" target="_blank" rel="noopener noreferrer">
              Modal
            </a>
            {' '}&amp;{' '}
            <a href="https://pipecat.ai" target="_blank" rel="noopener noreferrer">
              Pipecat
            </a>
          </p>
        </footer>
      </div>

      <audio ref={audioRef} autoPlay playsInline />
    </div>
  );
}
