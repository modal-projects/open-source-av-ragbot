/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * RTVI Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebRTC (via Daily).
 * It handles audio/video streaming and manages the connection lifecycle.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 * - The server must implement the /connect endpoint that returns Daily.co room credentials
 * - Browser with WebRTC support
 */

import { RTVIClient, RTVIEvent } from '@pipecat-ai/client-js';
import {
  SmallWebRTCTransport
} from "@pipecat-ai/small-webrtc-transport";

/**
 * ChatbotClient handles the connection and media management for a real-time
 * voice and video interaction with an AI bot.
 */
class ChatbotClient {
  constructor() {
    // Initialize client state
    this.rtviClient = null;
    this.setupDOMElements();
    this.initializeClientAndTransport();
    this.setupEventListeners();
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  setupDOMElements() {
    // Get references to UI control elements
    this.connectBtn = document.getElementById('connect-btn');
    this.disconnectBtn = document.getElementById('disconnect-btn');
    this.statusSpan = document.getElementById('connection-status');
    this.debugLog = document.getElementById('debug-log');
    this.botVideoContainer = document.getElementById('bot-video-container');
    this.deviceSelector = document.getElementById('device-selector');
    this.mediaContainer = document.querySelector('.media-container');
    this.userVideoContainer = document.getElementById('user-video-container');
    this.conversationLog = document.getElementById('conversation-log');
    
    // Track current bot message for adding extras (code blocks, links)
    this.currentBotMessage = null;
    // Track the last message source for grouping
    this.lastMessageSource = null;

    // Create an audio element for bot's voice output
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    this.botAudio.playsInline = true;
    document.body.appendChild(this.botAudio);
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  setupEventListeners() {
    this.connectBtn.addEventListener('click', () => this.connect());
    this.disconnectBtn.addEventListener('click', () => this.disconnect());

    // Populate device selector
    this.rtviClient.getAllMics().then((mics) => {
      console.log('Available mics:', mics);
      mics.forEach((device) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Microphone ${device.deviceId}`;
        this.deviceSelector.appendChild(option);
      });
    });
    this.deviceSelector.addEventListener('change', (event) => {
      const selectedDeviceId = event.target.value;
      console.log('Selected device ID:', selectedDeviceId);
      this.rtviClient.updateMic(selectedDeviceId);
    });

    // Handle mic mute/unmute toggle
    const micToggleBtn = document.getElementById('mic-toggle-btn');

    micToggleBtn.addEventListener('click', () => {
      let micEnabled = this.rtviClient.isMicEnabled;
      micToggleBtn.textContent = micEnabled ? 'Unmute Mic' : 'Mute Mic';
      this.rtviClient.enableMic(!micEnabled);
      // Add logic to mute/unmute the mic
      if (micEnabled) {
        console.log('Mic muted');
        // Add code to mute the mic
      } else {
        console.log('Mic unmuted');
        // Add code to unmute the mic
      }
    });

    // Display user's webcam in user-video-container
    this.setupUserWebcam();
  }

  /**
   * Set up the RTVI client and Daily transport
   */
  async initializeClientAndTransport() {
    // Initialize the RTVI client with a DailyTransport and our configuration
    const transport = new SmallWebRTCTransport(
      {
        waitForICEGathering: true,
        iceServers: [
          {
            urls: 'stun:stun.l.google.com:19302',
          },
        ],
      }
    );
    const RTVIConfig = {
      params: {
        // REPLACE WITH YOUR MODAL URL ENDPOINT
        baseUrl:
          'https://modal-labs-shababo-dev--moe-and-dal-ragbot-bot-server.modal.run',
        endpoints: {
          connect: '/offer',
        },
        requestData: {
          bot_name: 'gemini',
        },
      },
      transport: transport,
      enableMic: true, // Enable microphone for user input
      enableCam: false,
      callbacks: {
        // Handle connection state changes
        onConnected: () => {
          this.updateStatus('Connected');
          // this.connectBtn.disabled = true;
          this.disconnectBtn.disabled = false;
          this.log('Client connected');
        },
        onDisconnected: () => {
          this.updateStatus('Disconnected');
          this.connectBtn.disabled = false;
          this.disconnectBtn.disabled = true;
          this.log('Client disconnected');
        },
        // Handle transport state changes
        onTransportStateChanged: (state) => {
          this.updateStatus(`Transport: ${state}`);
          this.log(`Transport state changed: ${state}`);
          if (state === 'connecting') {
            window.startTime = Date.now();
          }
          if (state === 'ready') {
            this.setupMediaTracks();
            console.warn('TIME TO BOT READY:', Date.now() - window.startTime);
          }
        },
        // Handle bot connection events
        onBotConnected: (participant) => {
          this.log(`Bot connected: ${JSON.stringify(participant)}`);
        },
        onBotDisconnected: (participant) => {
          this.log(`Bot disconnected: ${JSON.stringify(participant)}`);
        },
        onBotReady: (data) => {
          this.log(`Bot ready: ${JSON.stringify(data)}`);
          this.setupMediaTracks();
        },
        // Transcript events
        onUserTranscript: (data) => {
          // Only log final transcripts
          if (data.final) {
            this.addUserMessage(data.text);
            this.log(`User: ${data.text}`);
          }
        },
        onBotTtsText: (data) => {
          this.addBotMessage(data.text);
          this.log(`Bot: ${data.text}`);
        },
        // Error handling
        onMessageError: (error) => {
          console.log('Message error:', error);
        },
        onMicUpdated: (data) => {
          console.log('Mic updated:', data);
          this.deviceSelector.value = data.deviceId;
        },
        onError: (error) => {
          console.log('Error:', JSON.stringify(error));
        },
        onServerMessage: (data) => {
          console.log('Server message:', data);
          this.handleServerMessage(data);
        },
      },
    };

    
    RTVIConfig.customConnectHandler = () => Promise.resolve();
    this.rtviClient = new RTVIClient(RTVIConfig);
    
    this.smallWebRTCTransport = transport
    window.client = this.rtviClient;
  }

  /**
   * Add a timestamped message to the debug log
   */
  log(message) {
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;

    // Add styling based on message type
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3'; // blue for user
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50'; // green for bot
    }

    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    console.log(message);
  }

  /**
   * Update the connection status display
   */
  updateStatus(status) {
    this.statusSpan.textContent = status;
    this.log(`Status: ${status}`);
  }

  /**
   * Check for available media tracks and set them up if present
   * This is called when the bot is ready or when the transport state changes to ready
   */
  setupMediaTracks() {
    if (!this.rtviClient) return;

    // Get current tracks from the client
    const tracks = this.rtviClient.tracks();

    // Set up any available bot tracks
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
    if (tracks.bot?.video) {
      this.setupVideoTrack(tracks.bot.video);
    }
  }

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.rtviClient) return;

    // Listen for new tracks starting
    this.rtviClient.on(RTVIEvent.TrackStarted, (track, participant) => {
      // Only handle non-local (bot) tracks
      if (!participant?.local) {
        if (track.kind === 'audio') {
          this.setupAudioTrack(track);
        } else if (track.kind === 'video') {
          this.setupVideoTrack(track);
        }
        this.log(
          `Track started event: ${track.kind} from ${
            participant?.name || 'unknown'
          }`
        );
      } else {
        this.log('Local mic unmuted');
      }
    });

    // Listen for tracks stopping
    this.rtviClient.on(RTVIEvent.TrackStopped, (track, participant) => {
      if (participant.local) {
        this.log('Local mic muted');
        return;
      }
      this.log(
        `Track stopped event: ${track.kind} from ${
          participant?.name || 'unknown'
        }`
      );
    });
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  setupAudioTrack(track) {
    this.log('Setting up audio track');
    // Check if we're already playing this track
    if (this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    // Create a new MediaStream with the track and set it as the audio source
    this.botAudio.srcObject = new MediaStream([track]);
  }

  /**
   * Set up a video track for display
   * Handles both initial setup and track updates
   */
  setupVideoTrack(track) {
    this.log('Setting up video track');
    const videoEl = document.createElement('video');
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    videoEl.muted = true;
    videoEl.style.width = '100%';
    videoEl.style.height = '100%';
    videoEl.style.objectFit = 'cover';

    // Check if we're already displaying this track
    if (this.botVideoContainer.querySelector('video')?.srcObject) {
      const oldTrack = this.botVideoContainer
        .querySelector('video')
        .srcObject.getVideoTracks()[0];
      if (oldTrack?.id === track.id) return;
    }

    // Create a new MediaStream with the track and set it as the video source
    videoEl.srcObject = new MediaStream([track]);
    this.botVideoContainer.innerHTML = '';
    this.botVideoContainer.appendChild(videoEl);
  }

  /**
   * Initialize and connect to the bot
   * This sets up the RTVI client, initializes devices, and establishes the connection
   */
  async connect() {
    try {
      this.connectBtn.disabled = true;
      
      // Clear any previous content
      this.clearDisplayedContent();
      
      // await this.initializeClientAndTransport();
      console.log(this.rtviClient.params.requestData);

      // // Initialize audio/video devices
      // this.log('Initializing devices...');
      await this.rtviClient.initDevices();

      // Set up listeners for media track events
      this.setupTrackListeners();

      // Connect to the bot
      this.log(`Connecting to bot`);
      await this.rtviClient.connect();

      this.log('Connection complete');
    } catch (error) {
      this.connectBtn.disabled = false;
      // Handle any errors during connection
      console.error('Connection error:', error);
      this.log(`Error connecting: ${JSON.stringify(error.message)}`);
      this.log(`Error stack: ${error.stack}`);
      this.updateStatus('Error');

      // Clean up if there's an error
      if (this.rtviClient) {
        try {
          await this.rtviClient.disconnect();
        } catch (disconnectError) {
          this.log(`Error during disconnect: ${disconnectError.message}`);
        }
      }
    }
  }

  /**
   * Disconnect from the bot and clean up media resources
   */
  async disconnect() {
    if (this.rtviClient) {
      try {
        // Disconnect the RTVI client
        await this.rtviClient.disconnect();

        // Clean up audio
        if (this.botAudio.srcObject) {
          this.botAudio.srcObject.getTracks().forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }

        // Clean up video
        if (this.botVideoContainer.querySelector('video')?.srcObject) {
          const video = this.botVideoContainer.querySelector('video');
          video.srcObject.getTracks().forEach((track) => track.stop());
          video.srcObject = null;
        }
        this.botVideoContainer.innerHTML = '';

        // Clear displayed conversation
        this.clearDisplayedContent();
      } catch (error) {
        this.log(`Error disconnecting: ${error.message}`);
      }
    }
  }

  /**
   * Set up the user's webcam and display it in the user-video-container
   */
  async setupUserWebcam() {
    if (!this.userVideoContainer) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      let userVideo = document.createElement('video');
      userVideo.autoplay = true;
      userVideo.playsInline = true;
      userVideo.muted = true;
      userVideo.style.width = '100%';
      userVideo.style.height = '100%';
      userVideo.style.objectFit = 'cover';
      userVideo.srcObject = stream;
      this.userVideoContainer.innerHTML = '';
      this.userVideoContainer.appendChild(userVideo);
    } catch (err) {
      this.log('Could not access webcam: ' + err.message);
      this.userVideoContainer.innerHTML = '<div style="color: #f44336; text-align: center;">Webcam unavailable</div>';
    }
  }

  /**
   * Handle server messages for code blocks and links
   */
  handleServerMessage(data) {
    if (data.type === 'code_blocks' && data.payload) {
      this.addCodeBlocksToCurrentBotMessage(data.payload);
    } else if (data.type === 'links' && data.payload) {
      this.addLinksToCurrentBotMessage(data.payload);
    }
  }

  /**
   * Add a user message to the conversation
   */
  addUserMessage(text) {
    if (!this.conversationLog || !text.trim()) return;

    // Remove empty state if it exists
    const emptyState = this.conversationLog.querySelector('.empty-state');
    if (emptyState) {
      emptyState.remove();
    }

    // Check if we should group with the last message
    if (this.lastMessageSource === 'user') {
      // Append to existing user message container
      this.appendToLastMessage(text);
    } else {
      // Create new user message container
      this.createNewUserMessage(text);
      this.lastMessageSource = 'user';
    }

    this.scrollToBottom(this.conversationLog);
  }

  /**
   * Add a bot message to the conversation
   */
  addBotMessage(text) {
    if (!this.conversationLog || !text.trim()) return;

    // Remove empty state if it exists
    const emptyState = this.conversationLog.querySelector('.empty-state');
    if (emptyState) {
      emptyState.remove();
    }

    // Check if we should group with the last message
    if (this.lastMessageSource === 'bot') {
      // Append to existing bot message container
      this.appendToLastMessage(text);
    } else {
      // Create new bot message container
      this.createNewBotMessage(text);
      this.lastMessageSource = 'bot';
    }

    this.scrollToBottom(this.conversationLog);
  }

  /**
   * Create a new user message container
   */
  createNewUserMessage(text) {
    const messageEl = document.createElement('div');
    messageEl.className = 'message-container user-message';
    messageEl.innerHTML = `
      <div class="message-header">
        <span class="message-avatar">👤</span>
        <span>You</span>
        <span style="margin-left: auto; font-size: 10px;">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="message-content">
        <div class="message-text">${this.escapeHtml(text)}</div>
      </div>
    `;

    this.conversationLog.appendChild(messageEl);
  }

  /**
   * Create a new bot message container
   */
  createNewBotMessage(text) {
    const messageEl = document.createElement('div');
    messageEl.className = 'message-container bot-message';
    messageEl.innerHTML = `
      <div class="message-header">
        <span class="message-avatar">🤖</span>
        <span>Modal Assistant</span>
        <span style="margin-left: auto; font-size: 10px;">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="message-content">
        <div class="message-text">${this.escapeHtml(text)}</div>
      </div>
      <div class="message-extras"></div>
    `;

    this.conversationLog.appendChild(messageEl);
    this.currentBotMessage = messageEl;
  }

  /**
   * Append text to the last message container
   */
  appendToLastMessage(text) {
    const lastMessage = this.conversationLog.lastElementChild;
    if (lastMessage && lastMessage.classList.contains('message-container')) {
      const messageContent = lastMessage.querySelector('.message-content');
      if (messageContent) {
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.textContent = text;
        messageContent.appendChild(messageText);

        // Update current bot message reference if this is a bot message
        if (lastMessage.classList.contains('bot-message')) {
          this.currentBotMessage = lastMessage;
        }
      }
    }
  }

  /**
   * Add code blocks to the current bot message
   */
  addCodeBlocksToCurrentBotMessage(codeBlocks) {
    if (!this.currentBotMessage || !Array.isArray(codeBlocks) || codeBlocks.length === 0) {
      return;
    }

    const extrasContainer = this.currentBotMessage.querySelector('.message-extras');
    if (!extrasContainer) return;

    // Create code blocks section
    const codeSection = document.createElement('div');
    codeSection.className = 'extras-section';
    codeSection.innerHTML = `
      <div class="extras-header">📋 Code Examples</div>
      <div class="code-blocks-container"></div>
    `;

    const codeContainer = codeSection.querySelector('.code-blocks-container');

    codeBlocks.forEach((code, index) => {
      const codeBlockEl = document.createElement('div');
      codeBlockEl.className = 'code-block';
      codeBlockEl.innerHTML = `
        <div class="code-block-header">
          <span>Example ${index + 1}</span>
          <button class="copy-btn" onclick="this.parentElement.parentElement.copyCode()">📋 Copy</button>
        </div>
        <pre><code>${this.escapeHtml(code)}</code></pre>
      `;

      // Add copy functionality
      codeBlockEl.copyCode = () => {
        navigator.clipboard.writeText(code).then(() => {
          const copyBtn = codeBlockEl.querySelector('.copy-btn');
          const originalText = copyBtn.textContent;
          copyBtn.textContent = '✅ Copied!';
          copyBtn.style.color = '#28a745';
          setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.color = '';
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy code:', err);
        });
      };

      codeContainer.appendChild(codeBlockEl);
    });

    extrasContainer.appendChild(codeSection);
    this.scrollToBottom(this.conversationLog);
  }

  /**
   * Add links to the current bot message
   */
  addLinksToCurrentBotMessage(links) {
    if (!this.currentBotMessage || !Array.isArray(links) || links.length === 0) {
      return;
    }

    const extrasContainer = this.currentBotMessage.querySelector('.message-extras');
    if (!extrasContainer) return;

    // Create links section
    const linksSection = document.createElement('div');
    linksSection.className = 'extras-section';
    linksSection.innerHTML = `
      <div class="extras-header">🔗 Helpful Links</div>
      <div class="links-container"></div>
    `;

    const linksContainer = linksSection.querySelector('.links-container');

    links.forEach((link) => {
      const linkEl = document.createElement('div');
      linkEl.className = 'link-item';
      
      // Extract domain for display
      let displayText = link;
      let icon = '🔗';
      
      try {
        const url = new URL(link);
        displayText = url.hostname + url.pathname;
        
        // Use different icons based on domain
        if (url.hostname.includes('modal.com')) {
          icon = '📖';
        } else if (url.hostname.includes('github.com')) {
          icon = '⚡';
        } else if (url.hostname.includes('docs.')) {
          icon = '📋';
        }
      } catch {
        // If URL parsing fails, use the original link as display text
      }

      linkEl.innerHTML = `
        <a href="${link}" target="_blank" rel="noopener noreferrer">
          <span class="link-icon">${icon}</span>
          <span class="link-text">${this.escapeHtml(displayText)}</span>
        </a>
      `;

      linksContainer.appendChild(linkEl);
    });

    extrasContainer.appendChild(linksSection);
    this.scrollToBottom(this.conversationLog);
  }

  /**
   * Scroll to bottom of container
   */
  scrollToBottom(container) {
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }

  /**
   * Escape HTML to prevent XSS
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Clear conversation log
   */
  clearDisplayedContent() {
    if (this.conversationLog) {
      this.conversationLog.innerHTML = '<div class="empty-state">Your conversation will appear here</div>';
    }
    this.currentBotMessage = null;
    this.lastMessageSource = null;
  }
}

// Initialize the client when the page loads
window.addEventListener('DOMContentLoaded', () => {
  new ChatbotClient();
});