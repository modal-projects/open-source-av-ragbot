import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Logo from "./logo";

import "./global.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <FullScreenContainer>
        <ConsoleTemplate
          title="Open Source AV RAGbot"
          logoComponent={
            <Logo width="auto" height={26} className="vkui:w-auto" />
          }
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
          conversationElementProps={{
            assistantLabel: "moe-and-dal",
          }}
        />
      </FullScreenContainer>
    </ThemeProvider>
  </StrictMode>
);
