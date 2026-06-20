import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

const ORIGINAL_FETCH = globalThis.fetch;

// Cerebras-native default ids (gpt-oss-120b, zai-glm-4.7) serve directly via
// the Cerebras key; OpenRouter is the on-error backup for the SAME model. The
// free-tier Cerebras key rate-limits at 5 req/min, so a 429 must soft-degrade
// to OpenRouter instead of surfacing to the user.
delete process.env.BITROUTER_API_KEY;
delete process.env.AI_GATEWAY_API_KEY;
delete process.env.AIGATEWAY_API_KEY;
delete process.env.ANTHROPIC_API_KEY;
delete process.env.OPENAI_API_KEY;
delete process.env.GROQ_API_KEY;
process.env.CEREBRAS_API_KEY = "test-cerebras-key";
process.env.OPENROUTER_API_KEY = "test-openrouter-key";
delete process.env.OPENROUTER_BASE_URL;

mock.module("@/lib/utils/logger", () => ({
  logger: {
    debug: () => {},
    error: () => {},
    info: () => {},
    warn: () => {},
  },
}));

const { generateText } = await import("ai");
const { getLanguageModel, resolveAiProviderSource } = await import("./language-model");

function hostOf(url: RequestInfo | URL): "openrouter" | "cerebras" | "other" {
  const u = String(url);
  if (u.includes("openrouter.ai")) return "openrouter";
  if (u.includes("cerebras.ai")) return "cerebras";
  return "other";
}

function completion(model: string, content: string): Response {
  return new Response(
    JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 0,
      model,
      choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

function tooManyRequests(): Response {
  return new Response(JSON.stringify({ error: { message: "Rate limit exceeded" } }), {
    status: 429,
  });
}

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
});

describe("getLanguageModel cerebras-direct → OpenRouter fallback", () => {
  let hosts: Array<"openrouter" | "cerebras" | "other">;

  beforeEach(() => {
    hosts = [];
  });

  test("a bare Cerebras id serves directly via cerebras.ai on the happy path", async () => {
    globalThis.fetch = (async (url: RequestInfo | URL) => {
      hosts.push(hostOf(url));
      return completion("gpt-oss-120b", "from-cerebras");
    }) as typeof fetch;

    const result = await generateText({
      model: getLanguageModel("gpt-oss-120b"),
      prompt: "hi",
      maxRetries: 0,
    });

    expect(result.text).toBe("from-cerebras");
    expect(hosts).toEqual(["cerebras"]);
  });

  test("falls over to OpenRouter when Cerebras returns a retryable 429", async () => {
    globalThis.fetch = (async (url: RequestInfo | URL) => {
      const host = hostOf(url);
      hosts.push(host);
      return host === "openrouter"
        ? completion("openai/gpt-oss-120b:nitro", "from-openrouter")
        : tooManyRequests();
    }) as typeof fetch;

    // The decorated id dedicated agents emit; still a Cerebras-native model.
    const result = await generateText({
      model: getLanguageModel("openai/gpt-oss-120b:nitro"),
      prompt: "hi",
      maxRetries: 0,
    });

    expect(result.text).toBe("from-openrouter");
    expect(hosts).toEqual(["cerebras", "openrouter"]);
  });

  test("does not fall over on a non-retryable error (400)", async () => {
    globalThis.fetch = (async (url: RequestInfo | URL) => {
      hosts.push(hostOf(url));
      return new Response(JSON.stringify({ error: { message: "bad request" } }), { status: 400 });
    }) as typeof fetch;

    await expect(
      generateText({
        model: getLanguageModel("gpt-oss-120b"),
        prompt: "hi",
        maxRetries: 0,
      }),
    ).rejects.toBeDefined();
    // A 400 is a real request error, not an outage: OpenRouter is never reached.
    expect(hosts).toEqual(["cerebras"]);
  });

  test("happy-path billing still attributes to cerebras", () => {
    // resolveAiProviderSource stays "cerebras" — the fallback only changes the
    // on-error path, not the canonical happy-path attribution.
    expect(resolveAiProviderSource("gpt-oss-120b")).toBe("cerebras");
    expect(resolveAiProviderSource("openai/gpt-oss-120b:nitro")).toBe("cerebras");
  });
});
