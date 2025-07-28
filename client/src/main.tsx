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
          transportType="smallwebrtc"
          connectParams={{
            connectionUrl:
              "https://modal-labs-shababo-dev--moe-and-dal-ragbot-bot-server.modal.run/offer",
          }}
          noUserVideo={true}
        />
      </FullScreenContainer>
    </ThemeProvider>
  </StrictMode>
);
