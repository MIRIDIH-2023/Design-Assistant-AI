import { Collapse, ScrollArea, Stack, Sx } from "@mantine/core";
import { ChatInput } from "./ChatInput";
import { FormEvent, useEffect, useRef, useState } from "react";
import { DialogBubble } from "./DialogBubble";
import { useClickOutside } from "@mantine/hooks";
import { useChat } from "ai/react";
import { ImageSelector } from "./ImageSelector";
import { Text, useMantineTheme, Button, Textarea } from "@mantine/core";
import { ConfirmArea } from "./ConfirmArea";
import { getBackgroundImages } from "../utils/api";

export interface Props {
  sx?: Sx | (Sx | undefined)[];
  onGenerate?: (text: string, imgUrl?: string) => void;
}

export function DialogBox({ sx, onGenerate }: Props) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editBoxOpen, setEditBoxOpen] = useState(false);
  const [imageSelecting, setImageSelecting] = useState(false);
  const [generatedText, setGeneratedText] = useState("");
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const ref = useClickOutside(() => setDialogOpen(false));
  const {
    messages,
    handleInputChange,
    handleSubmit,
    input,
    isLoading: chatEndpointIsLoading,
    setMessages,
  } = useChat({
    api: "api/chat/agents",
  });
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const openAndCloseBox = () => {
    setEditBoxOpen(!editBoxOpen);
  };

  const dialogClose = () => {
    setDialogOpen(false);
  };

  useEffect(() => {
    const scrollArea = scrollAreaRef.current;
    if (!scrollArea) return;
    scrollArea.scrollTo({
      top: scrollAreaRef.current.scrollHeight,
      behavior: "instant",
    });
  }, [messages, editBoxOpen, imageSelecting]);

  async function sendMessage(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!messages.length)
      await new Promise((resolve) => setTimeout(resolve, 300));
    if (chatEndpointIsLoading) return;
    handleSubmit(e);
  }

  return (
    <Stack
      spacing={10}
      sx={{
        backgroundColor: "white",
        borderRadius: "15px",
        padding: "10px",
        boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.1)",
        ...sx,
      }}
      ref={ref}
    >
      <Collapse in={dialogOpen}>
        <ScrollArea h={500} viewportRef={scrollAreaRef}>
          <Stack spacing={10}>
            {messages.map((message, index) => (
              <DialogBubble
                key={index}
                type={message.role == "user" ? "user" : "system"}
                content={message.content}
              />
            ))}
            <ConfirmArea
              content={
                messages.length > 0 ? messages[messages.length - 1].content : ""
              }
              openAndCloseBox={openAndCloseBox}
              dialogClose={dialogClose}
              isLoading={chatEndpointIsLoading}
              onGenerate={(text, isChanged) => {
                if (isChanged) {
                  const textvalue = `What I changed:\n${text}`;
                  // console.log(textvalue);
                  const messagesUserReply = messages.concat({
                    id: messages.length.toString(),
                    content: textvalue,
                    role: "user",
                  });
                  setMessages(messagesUserReply);
                }
                setGeneratedText(text);
                getBackgroundImages(text.split("\n"), 10).then((res) => {
                  setImageUrls(res);
                  setImageSelecting(true);
                });
              }}
            />
            {imageSelecting && (
              <ImageSelector
                imgURLs={imageUrls}
                onSelect={(imgURL) => {
                  console.log("이미지 선택됨");
                  console.log(imgURL);
                  setImageSelecting(false);
                  onGenerate?.(generatedText, imgURL);
                  dialogClose();
                }}
              />
            )}
          </Stack>
        </ScrollArea>
      </Collapse>
      <ChatInput
        onSubmit={sendMessage}
        onChange={handleInputChange}
        onFocus={() => setDialogOpen(true)}
        placeholder="메시지를 입력하세요."
        isLoading={chatEndpointIsLoading}
        value={input}
      />
    </Stack>
  );
}
