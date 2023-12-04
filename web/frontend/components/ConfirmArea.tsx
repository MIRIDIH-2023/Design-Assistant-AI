import {
  Box,
  Sx,
  Flex,
  Text,
  useMantineTheme,
  Button,
  Textarea,
} from "@mantine/core";
import { useSetState } from "@mantine/hooks";
import { ReactNode } from "react";
import React, {
  useState,
  ChangeEvent,
  useEffect,
  useRef,
  forwardRef,
} from "react";

interface Props {
  content: string;
  openAndCloseBox: () => void;
  dialogClose: () => void;
  isLoading?: boolean;
  onGenerate?: (text: string, isChanged: boolean) => void;
}

function extractContentsFromText(text: string): string[] {
  // 정규 표현식을 사용하여 "-"로 시작하는 항목 추출
  const pattern: RegExp = /-\s+(.*?)(?=\n|$)/g;
  const matches: RegExpMatchArray | null = text.match(pattern);

  // const contents: string[] = [];

  // if (matches) {
  //   matches.forEach((match: string) => {
  //     const cleanedMatch = match.replace(/-\s*/, "").trim();
  //     const parts = cleanedMatch.split(': ');
  //     contents.push(...parts);
  //   });
  // }

  // 추출된 항목을 배열로 저장
  const contents: string[] =
    matches?.map((match: string) => match.replace(/-\s*/, "").trim()) || [];

  return contents;
}

export function ConfirmArea({
  content,
  openAndCloseBox,
  dialogClose,
  isLoading,
  onGenerate,
}: Props) {
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [textValue, setTextValue] = useState<string>("");
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [originalValue, setoriginalValue] = useState<string>("");

  const theme = useMantineTheme();
  const textColor = theme.black;
  const backgroundColor = theme.white;

  // let text = content;
  useEffect(() => {
    setIsCompleted(false);
    let text = content;
    const extracted = extractContentsFromText(content);
    if (extracted.length !== 0) {
      setIsCompleted(true);
      text = extracted.join("\n");
      setTextValue(text);
      setIsEditing(false);
      setoriginalValue(text);
    }
  }, [content]);

  const handleEditClick = () => {
    setIsEditing(true);
    openAndCloseBox();
  };

  const handleGenerateClick = () => {
    let isChanged = false;
    if (textValue !== originalValue) {
      isChanged = true;
      // console.log("Changed!!!!!!!");
    }
    onGenerate?.(textValue, isChanged);
    setIsCompleted(false);
    setIsEditing(false);
    setoriginalValue(textValue);
    console.log(textValue.split("\n"));
  };

  const handleCancelClick = () => {
    setIsEditing(false);
    setTextValue(originalValue);
  };

  const handleTextChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setTextValue(event.target.value);
  };

  return (
    <>
      {isCompleted && (
        <div
          style={{
            padding: "10px 15px",
            borderRadius: "15px 15px 15px 15px",
            border: "2px solid",
            borderTopColor: `${
              isEditing ? theme.colors.blue[5] : backgroundColor
            }`,
            borderBottomColor: `${
              isEditing ? theme.colors.blue[5] : backgroundColor
            }`,
            borderRightColor: `${
              isEditing ? theme.colors.blue[5] : backgroundColor
            }`,
            borderLeftColor: `${
              isEditing ? theme.colors.blue[5] : backgroundColor
            }`,
            backgroundColor: `${backgroundColor}`,
            whiteSpace: "pre-wrap",
          }}
        >
          {isEditing ? (
            <>
              <Textarea
                value={textValue}
                onChange={handleTextChange}
                autosize
                minRows={2}
              />
              <Flex
                mih={50}
                gap="md"
                justify="center"
                align="flex-end"
                direction="row"
                wrap="nowrap"
              >
                <Button onClick={handleGenerateClick}>Generate</Button>
                <Button variant="outline" onClick={handleCancelClick}>
                  Cancel
                </Button>
              </Flex>
            </>
          ) : (
            <>
              <Flex
                mih={50}
                gap="md"
                justify="center"
                align="center"
                direction="row"
                wrap="nowrap"
              >
                <Button disabled={isLoading} onClick={handleGenerateClick}>
                  Generate
                </Button>
                <Button
                  disabled={isLoading}
                  variant="outline"
                  onClick={handleEditClick}
                >
                  Edit
                </Button>
              </Flex>
            </>
          )}
        </div>
      )}
    </>
  );
}
