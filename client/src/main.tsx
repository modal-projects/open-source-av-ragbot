import { ThemeProvider } from "@pipecat-ai/voice-ui-kit";
import { createRoot } from "react-dom/client";
import App from "./components/app";

import "./global.css";

createRoot(document.getElementById("root")!).render(
  <ThemeProvider>
    <App />
  </ThemeProvider>
);
