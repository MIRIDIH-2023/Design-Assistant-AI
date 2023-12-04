import {
  TextInput,
  TextInputProps,
  ActionIcon,
  useMantineTheme,
  UnstyledButton,
  Textarea,
  TextareaProps,
  Loader,
} from "@mantine/core";
import {
  IconSearch,
  IconArrowRight,
  IconArrowLeft,
  IconMessage,
} from "@tabler/icons-react";
import { ChangeEvent, FormEvent, useState } from "react";

export interface Props extends Omit<TextareaProps, "onSubmit"> {
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  isLoading?: boolean;
}

export function ChatInput({
  onSubmit,
  onChange,
  isLoading = false,
  ...others
}: Props) {
  const theme = useMantineTheme();

  const handleTextareaKeyPress = (e: React.KeyboardEvent<HTMLFormElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      onSubmit(e);
    }
  };

  return (
    <form
      onSubmit={(e) => {
        onSubmit(e);
      }}
      onKeyDown={handleTextareaKeyPress}
    >
      <Textarea
        icon={<IconMessage size="1.3rem" stroke={1.5} />}
        radius="xl"
        size="md"
        onChange={onChange}
        autosize
        maxRows={4}
        rightSection={
          <ActionIcon
            type="submit"
            disabled={isLoading}
            size={32}
            radius="xl"
            color={theme.primaryColor}
            variant={isLoading ? "default" : "filled"}
          >
            {isLoading ? (
              <Loader size="xs" />
            ) : (
              <IconArrowRight size="1.1rem" stroke={1.5} />
            )}
          </ActionIcon>
        }
        rightSectionWidth={42}
        {...others}
      />
    </form>
  );
}
