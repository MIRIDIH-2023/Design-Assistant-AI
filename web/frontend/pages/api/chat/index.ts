import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { ChatOpenAI } from "langchain/chat_models/openai";
import { BytesOutputParser } from "langchain/schema/output_parser";
import {
  PromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";

export const runtime = "edge";

const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

// const TEMPLATE = `You are a pirate named Patchy. All responses must be extremely verbose and in pirate dialect.

// Current conversation:
// {chat_history}

// User: {input}
// AI:`;

const en_SystemPromt = SystemMessagePromptTemplate.fromTemplate(
  "You are text contents generation robot. " +
  "You need to gather information about the users goals, objectives, and other relevant context. " +
  "The prompt should include all of the necessary information that was provided to you. " +
  "Ask follow up questions to the user until you have confident you can produce a perfect text contents. " +
  "Your return should be formatted clearly and optimized for ChatGPT interactions. " +
  "You should ask the user the goals, and any additional information you may need. " +
  "The user's prompt must be related to making text contents for one slide in PowerPoint Presentations. So you should generate the contents. " +
  "The slide can be a cover, table of contents, introduction, body, and conclusion. " +
  "When you can produce a perfect text contents, your return that are text contents should be noun phrases. " +
  "The text contents must be separated into new lines. Put the prefix: \'-\'."
);

// 'The text contents have prefix: "content:"'  seperated '-', and new line, not numbering "You answer on the premise that the user wants to make text contents in PowerPoint Presentations. "

/**
 * This handler initializes and calls a simple chain with a prompt,
 * chat model, and output parser. See the docs for more information:
 *
 * https://js.langchain.com/docs/guides/expression_language/cookbook#prompttemplate--llm--outputparser
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;
    // console.log(messages, formattedPreviousMessages, currentMessageContent);
    // const prompt = PromptTemplate.fromTemplate<{
    //   chat_history: string;
    //   input: string;
    // }>(TEMPLATE);
    const prompt = ChatPromptTemplate.fromPromptMessages<{
      chat_history: string;
      input: string;
    }>([
      en_SystemPromt,
      HumanMessagePromptTemplate.fromTemplate(
        "Current conversation:\n{chat_history}\n\nUser: {input}\nAI:"
      ),
    ]);
    /**
     * You can also try e.g.:
     *
     * import { ChatAnthropic } from "langchain/chat_models/anthropic";
     * const model = new ChatAnthropic({});
     *
     * See a full list of supported models at:
     * https://js.langchain.com/docs/modules/model_io/models/
     */
    const model = new ChatOpenAI({
      modelName: "gpt-4",
      temperature: 0.8,
    });
    /**
     * Chat models stream message chunks rather than bytes, so this
     * output parser handles serialization and byte-encoding.
     */
    const outputParser = new BytesOutputParser();

    /**
     * Can also initialize as:
     *
     * import { RunnableSequence } from "langchain/schema/runnable";
     * const chain = RunnableSequence.from([prompt, model, outputParser]);
     */
    const chain = prompt.pipe(model).pipe(outputParser);

    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join("\n"),
      input: currentMessageContent,
    });

    return new StreamingTextResponse(stream);
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export default POST;
