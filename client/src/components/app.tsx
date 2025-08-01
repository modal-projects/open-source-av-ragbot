import {
  ConsoleTemplate,
  FullScreenContainer,
  type ConversationMessage,
} from "@pipecat-ai/voice-ui-kit";
import { useCallback, useRef } from "react";
import Logo from "./logo";

import { Code } from "./code";
import { Links } from "./links";

export default function App() {
  const injectMessageRef = useRef<
    ((message: Pick<ConversationMessage, "role" | "content">) => void) | null
  >(null);

  const handleOnInjectMessage = useCallback(
    (
      injectFn: (message: Pick<ConversationMessage, "role" | "content">) => void
    ) => {
      injectMessageRef.current = injectFn;
    },
    []
  );

  const handleOnServerMessage = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (data: any) => {
      if (injectMessageRef.current) {
        let content;
        if (!data.payload || data.payload.length === 0) {
          return;
        }
        switch (data.type) {
          case "links":
            content = <Links links={data.payload} />;
            break;
          case "code_blocks":
            content = <Code codeBlocks={data.payload} />;
            break;
          default:
            break;
        }

        if (content) {
          injectMessageRef.current({
            role: "system",
            content,
          });
        }
      }
    },
    []
  );

  return (
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
        onInjectMessage={handleOnInjectMessage}
        onServerMessage={handleOnServerMessage}
      />
    </FullScreenContainer>
  );
}
