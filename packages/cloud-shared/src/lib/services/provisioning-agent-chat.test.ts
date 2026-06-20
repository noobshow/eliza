import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

const ORIGINAL_FETCH = globalThis.fetch;

// The provisioning chat persona runs on the bare Cerebras small model routed
// through the shared language-model layer, so it inherits cerebras-direct →
// OpenRouter fallback. Drive Cerebras-direct + the on-error backup here.
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
  logger: { debug: () => {}, error: () => {}, info: () => {}, warn: () => {} },
}));

// In-memory cache so we can assert history persistence without Redis.
const store = new Map<string, unknown>();
mock.module("@/lib/cache/client", () => ({
  cache: {
    get: async (key: string) => store.get(key) ?? null,
    set: async (key: string, value: unknown) => {
      store.set(key, value);
    },
  },
}));

// No sandbox rows: the chat still works and reports status "none".
mock.module("@/db/repositories/agent-sandboxes", () => ({
  agentSandboxesRepository: {
    findByIdAndOrg: async () => undefined,
    listByOrganization: async () => [],
  },
}));

const { provisioningAgentChat } = await import("./provisioning-agent-chat");

function hostOf(url: RequestInfo | URL): "openrouter" | "cerebras" | "other" {
  const u = String(url);
  if (u.includes("openrouter.ai")) return "openrouter";
  if (u.includes("cerebras.ai")) return "cerebras";
  return "other";
}

function completion(content: string): Response {
  return new Response(
    JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 0,
      model: "gpt-oss-120b",
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

beforeEach(() => {
  store.clear();
});

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
});

describe("provisioningAgentChat routes through the shared language-model layer", () => {
  test("serves directly via Cerebras on the happy path and persists history", async () => {
    const hosts: string[] = [];
    globalThis.fetch = (async (url: RequestInfo | URL) => {
      hosts.push(hostOf(url));
      return completion("Hello from the provisioning persona!");
    }) as typeof fetch;

    const result = await provisioningAgentChat("user-1", "org-1", "hi there");

    expect(result.reply).toBe("Hello from the provisioning persona!");
    expect(result.containerStatus).toBe("none");
    expect(hosts).toEqual(["cerebras"]);
    // History contains the user message + assistant reply.
    expect(result.history).toEqual([
      { role: "user", content: "hi there" },
      { role: "assistant", content: "Hello from the provisioning persona!" },
    ]);
    expect(store.get("prov-chat:user-1")).toEqual(result.history);
  });

  test("falls over to OpenRouter when Cerebras returns a retryable 429", async () => {
    const hosts: string[] = [];
    globalThis.fetch = (async (url: RequestInfo | URL) => {
      const host = hostOf(url);
      hosts.push(host);
      return host === "openrouter" ? completion("recovered via backup") : tooManyRequests();
    }) as typeof fetch;

    const result = await provisioningAgentChat("user-2", "org-1", "are you there?");

    // The 429 soft-degrades to OpenRouter instead of the static apology.
    expect(result.reply).toBe("recovered via backup");
    expect(hosts).toEqual(["cerebras", "openrouter"]);
  });

  test("falls back to the static apology on a non-retryable upstream failure", async () => {
    // A 400 is non-retryable, so the shared layer never reaches OpenRouter and
    // generateText throws — exercising the service's static-apology catch.
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ error: { message: "bad request" } }), {
        status: 400,
      })) as typeof fetch;

    const result = await provisioningAgentChat("user-3", "org-1", "hello?");

    expect(result.reply).toContain("brief moment of difficulty");
    // Even the degraded turn persists the apology so history stays coherent.
    expect(result.history.at(-1)).toEqual({
      role: "assistant",
      content: result.reply,
    });
  });
});
