import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

import { ChatOpenAI } from "langchain/chat_models/openai";
import { 
  PromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { JsonOutputFunctionsParser } from "langchain/output_parsers";

export const runtime = "edge";

// const TEMPLATE = `Extract the requested fields from the input.

// The field "entity" refers to the first mentioned entity in the input.

// Input:

// {input}`;

const en_SystemPromt = SystemMessagePromptTemplate.fromTemplate(
  "You are text contents generation robot. " +
  "You need to gather information about the users goals, objectives, and other relevant context. " +
  "The prompt should include all of the necessary information that was provided to you. " +
  "Ask follow up questions to the user until you have confident you can produce a perfect text contents. " +
  "Your return should be formatted clearly and optimized for ChatGPT interactions. " +
  "You should ask the user the goals, and any additional information you may need. " +
  "The user's prompt must be related to making text contents for one slide in PowerPoint Presentations. So you should generate the contents. " +
  "The slide can be a cover, table of contents, introduction, body, and conclusion. " +
  "When you can produce a perfect text contents, your return that are text contents should be noun phrases."
);

const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

/**
 * This handler initializes and calls an OpenAI Functions powered
 * structured output chain. See the docs for more information:
 *
 * https://js.langchain.com/docs/modules/chains/popular/structured_output
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;
    console.log(messages, formattedPreviousMessages, currentMessageContent);

    // const prompt = PromptTemplate.fromTemplate<{ input: string }>(TEMPLATE);
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
     * Function calling is currently only supported with ChatOpenAI models
     */
    const model = new ChatOpenAI({
      temperature: 0.8,
      modelName: "gpt-4",
    });

    /**
     * We use Zod (https://zod.dev) to define our schema for convenience,
     * but you can pass JSON Schema directly if desired.
     */
    const schema = z.object({
      chat_response: z.string().describe("A response to the human's input"),
      perfect_text_contents: z
        .optional(z.array(z.string()))
        .describe("A response that is perfect text contents AI generated."),
    });

    /**The final punctuation mark in the input, if any.
     * Bind the function and schema to the OpenAI model.
     * Future invocations of the returned model will always use these arguments.
     *
     * Specifying "function_call" ensures that the provided function will always
     * be called by the model.
     */
    const functionCallingModel = model.bind({
      functions: [
        {
          name: "output_formatter",
          description: "Should always be used to properly format output",
          parameters: zodToJsonSchema(schema),
        },
      ],
      function_call: { name: "output_formatter" },
    });

    /**
     * Returns a chain with the function calling model.
     */
    const chain = prompt
      .pipe(functionCallingModel)
      .pipe(new JsonOutputFunctionsParser());

    const result = await chain.invoke({
      chat_history: formattedPreviousMessages.join("\n"),
      input: currentMessageContent,
    });
    // NextResponse.json(result, { status: 200 })
    return NextResponse.json(result, { status: 200 });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export default POST;
