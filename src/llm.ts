import { infer, ZodType } from "zod";
import { OpenAI as BaseOpenAI } from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import ollama from "ollama";
import zodToJsonSchema from "zod-to-json-schema";

export const OLLAMA_CONTENT_LIMIT = 1000;

export interface Request {
  model?: string;
  messages: RequestMessage[];
}

export interface RequestMessage {
  role: string;
  content: string;
}

export interface RequestWithFormat<T> extends Request {
  format: ZodType<T>;
}

export interface LLMInterface {
  chat(_: Request): Promise<Response<string>>;
  chatFormat<T>(r: RequestWithFormat<T>): Promise<Response<T>>;
}

export interface Response<T> {
  content: T;
  usage: any;
}

export interface OllamaOptions {
  model: string;
}

export class Ollama implements LLMInterface {
  constructor(private options: OllamaOptions) {}

  async chat(r: Request): Promise<Response<string>> {
    const response = await ollama.chat({
      model: r.model ?? this.options.model,
      messages: r.messages,
      options: { num_ctx: OLLAMA_CONTENT_LIMIT + 1000 },
    });
    return { content: response.message.content, usage: null };
  }

  async chatFormat<T>(r: RequestWithFormat<T>): Promise<Response<T>> {
    const response = await ollama.chat({
      model: r.model ?? this.options.model,
      messages: r.messages,
      options: { num_ctx: OLLAMA_CONTENT_LIMIT + 1000 },
      format: zodToJsonSchema(r.format),
    });
    return {
      content: r.format.parse(JSON.parse(response.message.content)),
      usage: null,
    };
  }
}

export class OpenAI implements LLMInterface {
  constructor(private openai: BaseOpenAI) {}
  async chat(r: Request): Promise<Response<string>> {
    const response = await this.openai.beta.chat.completions.parse({
      model: r.model,
      messages: r.messages,
      reasoning_effort: "high",
    });

    return {
      content: response.choices[0].message.content ?? "",
      usage: response.usage,
    };
  }

  async chatFormat<T>(r: RequestWithFormat<T>): Promise<Response<T>> {
    const response = await this.openai.beta.chat.completions.parse({
      model: r.model,
      messages: r.messages,
      reasoning_effort: "high",
      response_format: zodResponseFormat(r.format, "response"),
    });

    return {
      content: response.choices[0].message.parsed,
      usage: r.usage,
    };
  }
}
