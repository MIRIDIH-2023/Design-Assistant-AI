import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { DynamicTool } from "langchain/tools";

import { AIMessage, ChatMessage, HumanMessage } from "langchain/schema";
import { BufferMemory, ChatMessageHistory } from "langchain/memory";

export const runtime = "edge";

const convertVercelMessageToLangChainMessage = (message: VercelChatMessage) => {
  if (message.role === "user") {
    return new HumanMessage(message.content);
  } else if (message.role === "assistant") {
    return new AIMessage(message.content);
  } else {
    return new ChatMessage(message.content, message.role);
  }
};

const TEMPLATE =
  "You are robot generating contents in Presentations. " +
  "You should generate only the contents that will fit on one slide. The slide can be a Cover, Table of contents, Introduction, Body, and Conclusion. " +
  "Ask follow up questions by way of example that relate to the characteristics of a particular slide, not design elements, and questions that are needed to create more complete and detailed contents to the user after you make a text contents as complete as possible. " +
  "Your return that are text contents should be noun phrases. " +
  "The text contents must be separated into new lines, and five or less, and should be key. Put the prefix marks: '-'.";

/**
 * This handler initializes and calls an OpenAI Functions agent.
 * See the docs for more information:
 *
 * https://js.langchain.com/docs/modules/agents/agent_types/openai_functions_agent
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    /**
     * We represent intermediate steps as system messages for display purposes,
     * but don't want them in the chat history.
     */
    const messages = (body.messages ?? []).filter(
      (message: VercelChatMessage) =>
        message.role === "user" || message.role === "assistant"
    );
    // const returnIntermediateSteps = body.show_intermediate_steps;
    const previousMessages = messages
      .slice(0, -1)
      .map(convertVercelMessageToLangChainMessage);
    const currentMessageContent = messages[messages.length - 1].content;

    const PREFIX_TEMPLATE =
      "If user ask something in Korean, Your response must be Korean. " +
      TEMPLATE;

    // Requires process.env.SERPAPI_API_KEY to be set: https://serpapi.com/
    // const tools = [new Calculator()];
    const tools = [
      new DynamicTool({
        name: "Error",
        description:
          "call this if user try to talk without instructions, and the talk is not related to making contents in PPT. input should be user's prompt.",
        func: async (input) =>
          `I am a robot generating presentation. I can not respone with the ${input}. Give examples, and ask what kind of slide the user want to make.`,
      }),
      //call this if the input is list of contents, or the output is AI's response that contains contents in PPT slide. Do not call if the input is an interrogative sentence, or an instruction. input should be string.
      new DynamicTool({
        name: "Organize-user-input",
        description:
          "call this if input is list of contents in PPT slide. input should be string.",
        func: async (list) => {
          const divided = list.split(":");
          const res = `The inputs are ${divided}. Put \'-\' before each content to arrange ${divided} lines by line, and ask if there's anything to add or modify.`;

          return res;
        },
      }),
      new DynamicTool({
        name: "Organize-Generation",
        description:
          "call this if generation is list of contents in PPT slide. generation should be string.",
        func: async (list) => {
          const divided = list.split(":");
          const res = `The inputs are ${divided}. Put \'-\' before each content to arrange ${divided} lines by line, and ask if there's anything to add or modify.`;

          return res;
        },
      }),
    ];
    const chat = new ChatOpenAI({ modelName: "gpt-4", temperature: 0 });

    /**
     * The default prompt for the OpenAI functions agent has a placeholder
     * where chat messages get injected - that's why we set "memoryKey" to
     * "chat_history". This will be made clearer and more customizable in the future.
     */
    const executor = await initializeAgentExecutorWithOptions(tools, chat, {
      agentType: "openai-functions",
      verbose: true,
      maxIterations: 5,
      memory: new BufferMemory({
        memoryKey: "chat_history",
        chatHistory: new ChatMessageHistory(previousMessages),
        returnMessages: true,
        outputKey: "output",
      }),
      agentArgs: {
        prefix: PREFIX_TEMPLATE,
      },
    });

    const result = await executor.call({
      input: currentMessageContent,
    });

    const textEncoder = new TextEncoder();
    const fakeStream = new ReadableStream({
      async start(controller) {
        for (const character of result.output) {
          controller.enqueue(textEncoder.encode(character));
          await new Promise((resolve) => setTimeout(resolve, 30));
        }
        controller.close();
      },
    });

    return new StreamingTextResponse(fakeStream);
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export default POST;
