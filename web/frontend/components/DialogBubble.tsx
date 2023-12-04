import { Box, Sx, Text, useMantineTheme, Button, Textarea, Image } from "@mantine/core";
import { ReactNode } from "react";
import React, { useState, ChangeEvent } from 'react';
import image from "../public/mascotLeftFace.png";

interface Props {
  content: string;
  type: "system" | "user";
}

export function DialogBubble({ content, type }: Props) {
  const theme = useMantineTheme();
  const textColor = type === "system" ? theme.colors.dark[9] : theme.white;
  const backgroundColor =
    type === "system" ? theme.colors.gray[2] : theme.colors.blue[5];

  return (
    <Box
      sx={{
        display: content.slice(0, 15) === `What I changed:` ? "none" : "",
        padding: "10px 15px",
        borderRadius:
          type === "system" ? "15px 15px 15px 0px" : "15px 15px 0px 15px",
        backgroundColor,
      }}
    >
      <div style={{display: "flex"}}>
        <div style={{marginRight: "4px"}}>
          {type === "system" && <Image src={image.src} width="1.5rem" />}
        </div>
        <div style={{color: `${textColor}`, fontSize: "1rem", whiteSpace: "pre-wrap"}}>
          {content}
        </div>
      </div>
    </Box>
  );
}
