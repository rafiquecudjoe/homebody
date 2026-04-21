import "dotenv/config";
import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import Anthropic from "@anthropic-ai/sdk";
import { randomUUID } from "node:crypto";
import { ROOM_PROMPTS, ROOM_IDS } from "./prompts.js";

const {
  ANTHROPIC_API_KEY,
  PORT = 3000,
  MODEL = "claude-sonnet-4-6",
  MAX_TOKENS = "1000",
  ALLOWED_ORIGINS = "https://cherrymenlove.com,https://www.cherrymenlove.com",
  RATE_LIMIT_WINDOW_MS = "60000",
  RATE_LIMIT_MAX = "20",
  DAILY_TOKEN_CAP = "2000000",
  UPSTREAM_TIMEOUT_MS = "30000",
  LOG_LEVEL = "info",
} = process.env;

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };
const minLevel = LEVELS[LOG_LEVEL] ?? LEVELS.info;
function log(level, msg, fields = {}) {
  if (LEVELS[level] < minLevel) return;
  const line = { t: new Date().toISOString(), level, msg, ...fields };
  const out = JSON.stringify(line);
  if (level === "error") console.error(out);
  else console.log(out);
}

if (!ANTHROPIC_API_KEY) {
  log("error", "FATAL: ANTHROPIC_API_KEY env var not set");
  process.exit(1);
}

const rawOrigins = ALLOWED_ORIGINS.split(",").map((s) => s.trim()).filter(Boolean);
const openCors = rawOrigins.includes("*");
const allowedOrigins = new Set(openCors ? [] : rawOrigins);

log("info", "boot", {
  port: Number(PORT),
  model: MODEL,
  maxTokens: Number(MAX_TOKENS),
  rateLimit: `${RATE_LIMIT_MAX}/${Number(RATE_LIMIT_WINDOW_MS) / 1000}s`,
  dailyTokenCap: Number(DAILY_TOKEN_CAP),
  upstreamTimeoutMs: Number(UPSTREAM_TIMEOUT_MS),
  allowedOrigins: openCors ? "* (open)" : [...allowedOrigins],
  logLevel: LOG_LEVEL,
});

const anthropic = new Anthropic({
  apiKey: ANTHROPIC_API_KEY,
  timeout: Number(UPSTREAM_TIMEOUT_MS),
});
const app = express();

const dailyCap = Number(DAILY_TOKEN_CAP);
let dailyUsage = { day: new Date().toISOString().slice(0, 10), tokens: 0 };
function recordTokens(n) {
  const today = new Date().toISOString().slice(0, 10);
  if (dailyUsage.day !== today) {
    log("info", "daily_usage_reset", { previous: dailyUsage });
    dailyUsage = { day: today, tokens: 0 };
  }
  dailyUsage.tokens += n;
}
function dailyCapExceeded() {
  const today = new Date().toISOString().slice(0, 10);
  if (dailyUsage.day !== today) return false;
  return dailyUsage.tokens >= dailyCap;
}

app.set("trust proxy", 1);

app.use((req, res, next) => {
  req.id = randomUUID().slice(0, 8);
  req.startedAt = Date.now();
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
  res.setHeader("X-Request-Id", req.id);

  res.on("finish", () => {
    const ms = Date.now() - req.startedAt;
    log("info", "request", {
      reqId: req.id,
      method: req.method,
      path: req.path,
      status: res.statusCode,
      ms,
      origin: req.get("origin") ?? null,
      ip: req.ip,
      ua: req.get("user-agent") ?? null,
    });
  });
  next();
});

const corsMw = cors({
  origin: openCors
    ? true
    : (origin, cb) => {
        if (!origin) return cb(null, false);
        cb(null, allowedOrigins.has(origin));
      },
  methods: ["POST", "OPTIONS"],
  maxAge: 600,
});

const chatLimiter = rateLimit({
  windowMs: Number(RATE_LIMIT_WINDOW_MS),
  max: Number(RATE_LIMIT_MAX),
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests, please slow down." },
  handler: (req, res, _next, options) => {
    log("warn", "rate_limited", { reqId: req.id, ip: req.ip, limit: options.max });
    res.status(options.statusCode).json(options.message);
  },
});

const MAX_MESSAGES = 30;
const MAX_TEXT_LEN = 4000;
const MAX_IMAGE_BYTES = 1_500_000;
const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);

function validateContent(content) {
  if (typeof content === "string") {
    if (content.length > MAX_TEXT_LEN) return "text too long";
    return null;
  }
  if (!Array.isArray(content)) return "content must be string or array";
  if (content.length === 0 || content.length > 4) return "content array length out of range";
  for (const block of content) {
    if (!block || typeof block !== "object") return "content block must be object";
    if (block.type === "text") {
      if (typeof block.text !== "string") return "text block missing text";
      if (block.text.length > MAX_TEXT_LEN) return "text block too long";
    } else if (block.type === "image") {
      const src = block.source;
      if (!src || src.type !== "base64") return "only base64 image source allowed";
      if (typeof src.data !== "string") return "image data missing";
      if (!ALLOWED_IMAGE_TYPES.has(src.media_type)) return "unsupported image media type";
      if (!/^[A-Za-z0-9+/]+={0,2}$/.test(src.data)) return "image data not valid base64";
      const decodedBytes = Buffer.byteLength(src.data, "base64");
      if (decodedBytes > MAX_IMAGE_BYTES) return "image too large";
    } else {
      return "unknown content block type";
    }
  }
  return null;
}

function validateMessages(messages) {
  if (!Array.isArray(messages)) return "messages must be array";
  if (messages.length === 0 || messages.length > MAX_MESSAGES) return "messages length out of range";
  for (const m of messages) {
    if (!m || typeof m !== "object") return "message must be object";
    if (m.role !== "user" && m.role !== "assistant") return "invalid role";
    const err = validateContent(m.content);
    if (err) return err;
  }
  return null;
}

function getSeason() {
  const m = new Date().getMonth();
  if (m >= 2 && m <= 4) return { name: "Spring", note: "Time to sow and tend" };
  if (m >= 5 && m <= 7) return { name: "Summer", note: "Time to feast and flourish" };
  if (m >= 8 && m <= 10) return { name: "Autumn", note: "Time to harvest and preserve" };
  return { name: "Winter", note: "Time to restore and dream" };
}

app.get("/healthz", (_req, res) => res.json({ ok: true, dailyUsage }));

app.options("/api/chat", corsMw);
app.post(
  "/api/chat",
  corsMw,
  chatLimiter,
  express.json({ limit: "2mb" }),
  async (req, res) => {
    const origin = req.get("origin");
    if (!openCors && (!origin || !allowedOrigins.has(origin))) {
      log("warn", "origin_rejected", { reqId: req.id, origin, ip: req.ip });
      return res.status(403).json({ error: "Origin not allowed" });
    }

    const { roomId, messages } = req.body ?? {};

    if (typeof roomId !== "string" || !ROOM_IDS.has(roomId)) {
      log("warn", "invalid_roomId", { reqId: req.id, roomId });
      return res.status(400).json({ error: "Invalid roomId" });
    }
    const msgErr = validateMessages(messages);
    if (msgErr) {
      log("warn", "invalid_messages", {
        reqId: req.id,
        reason: msgErr,
        msgCount: Array.isArray(messages) ? messages.length : null,
      });
      return res.status(400).json({ error: msgErr });
    }

    if (dailyCapExceeded()) {
      log("warn", "daily_cap_exceeded", { reqId: req.id, dailyUsage });
      return res.status(503).json({ error: "Daily quota reached, please try again tomorrow." });
    }

    const season = getSeason();
    const system = `${ROOM_PROMPTS[roomId]} The current season is ${season.name}. ${season.note}.`;

    log("debug", "anthropic_request", {
      reqId: req.id,
      roomId,
      msgCount: messages.length,
      lastRole: messages[messages.length - 1]?.role,
    });

    const t0 = Date.now();
    try {
      const response = await anthropic.messages.create({
        model: MODEL,
        max_tokens: Number(MAX_TOKENS),
        system,
        messages,
      });
      const usage = response.usage ?? {};
      const totalTokens = (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0);
      recordTokens(totalTokens);
      const text = (response.content ?? [])
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");

      log("info", "anthropic_ok", {
        reqId: req.id,
        roomId,
        ms: Date.now() - t0,
        inputTokens: usage.input_tokens ?? 0,
        outputTokens: usage.output_tokens ?? 0,
        stopReason: response.stop_reason,
        replyLen: text.length,
        dailyTokens: dailyUsage.tokens,
      });

      res.json({ text });
    } catch (err) {
      log("error", "anthropic_error", {
        reqId: req.id,
        roomId,
        ms: Date.now() - t0,
        status: err?.status ?? null,
        errName: err?.name ?? null,
        errMessage: err?.message ?? String(err),
      });
      res.status(502).json({ error: "Upstream request failed" });
    }
  },
);

// JSON-only error handler — replaces Express default HTML stack trace
app.use((err, req, res, _next) => {
  if (err?.type === "entity.parse.failed" || err instanceof SyntaxError) {
    log("warn", "bad_json", { reqId: req.id, errMessage: err.message });
    return res.status(400).json({ error: "Invalid JSON body" });
  }
  if (err?.type === "entity.too.large") {
    log("warn", "payload_too_large", { reqId: req.id });
    return res.status(413).json({ error: "Payload too large" });
  }
  log("error", "unhandled_error", { reqId: req.id, errMessage: err?.message ?? String(err) });
  res.status(500).json({ error: "Internal server error" });
});

app.listen(PORT, () => {
  log("info", "listening", { port: Number(PORT) });
});
