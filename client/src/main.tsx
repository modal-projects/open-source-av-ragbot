import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@pipecat-ai/voice-ui-kit/styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <FullScreenContainer>
        <ConsoleTemplate
          title="Open Source AV RAGbot"
          transportType="smallwebrtc"
          connectParams={{
            connectionUrl: "/offer",
          }}
          transportOptions={{
            waitForICEGathering: true,
            iceServers: [
              {
                urls: "stun:stun.l.google.com:19302",
              },
            ],
          }}
          noUserVideo={true}
        />
      </FullScreenContainer>
    </ThemeProvider>
  </StrictMode>
);
