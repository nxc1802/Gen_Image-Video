import { NextRequest, NextResponse } from "next/server";

const FALLBACK_TUNNEL_URL =
  "https://palace-dealing-optional-layer.trycloudflare.com";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function handleProxy(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
): Promise<Response> {
  const { path } = await context.params;
  const subPath = "/" + (path || []).join("/");
  const search = req.nextUrl.search || "";

  const headerTarget = req.headers
    .get("x-target-base-url")
    ?.trim()
    .replace(/\/+$/, "");
  const primaryBaseUrl = headerTarget || FALLBACK_TUNNEL_URL;

  const forwardHeaders = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) {
    forwardHeaders.set("content-type", contentType);
  }
  const auth = req.headers.get("authorization");
  if (auth) {
    forwardHeaders.set("authorization", auth);
  }
  const accept = req.headers.get("accept");
  if (accept) {
    forwardHeaders.set("accept", accept);
  }

  let bodyBuffer: ArrayBuffer | undefined;
  if (req.method !== "GET" && req.method !== "HEAD") {
    bodyBuffer = await req.arrayBuffer();
  }

  const tryFetch = async (baseUrl: string) => {
    const targetUrl = `${baseUrl}${subPath}${search}`;
    return fetch(targetUrl, {
      method: req.method,
      headers: forwardHeaders,
      body: bodyBuffer ? Buffer.from(bodyBuffer) : undefined,
      cache: "no-store",
    });
  };

  let upstreamRes: Response;
  try {
    upstreamRes = await tryFetch(primaryBaseUrl);
    // If primaryBaseUrl is a dead Cloudflare tunnel (530 / 502) and differs from FALLBACK_TUNNEL_URL, retry with fallback
    if (
      (upstreamRes.status === 530 || upstreamRes.status === 502) &&
      primaryBaseUrl !== FALLBACK_TUNNEL_URL
    ) {
      upstreamRes = await tryFetch(FALLBACK_TUNNEL_URL);
    }
  } catch (err) {
    if (primaryBaseUrl !== FALLBACK_TUNNEL_URL) {
      try {
        upstreamRes = await tryFetch(FALLBACK_TUNNEL_URL);
      } catch (err2) {
        return NextResponse.json(
          {
            error: "Upstream tunnel unreachable",
            detail: err2 instanceof Error ? err2.message : String(err2),
          },
          { status: 502 }
        );
      }
    } else {
      return NextResponse.json(
        {
          error: "Upstream tunnel unreachable",
          detail: err instanceof Error ? err.message : String(err),
        },
        { status: 502 }
      );
    }
  }

  const respHeaders = new Headers();
  const upstreamContentType = upstreamRes.headers.get("content-type");
  if (upstreamContentType) {
    respHeaders.set("content-type", upstreamContentType);
  }
  respHeaders.set("cache-control", "no-store");

  if (upstreamRes.status >= 400) {
    const errText = await upstreamRes.text();
    if (subPath === "/v1/audio/diagnostics" && upstreamRes.status === 500) {
      return NextResponse.json(
        {
          status: "disabled",
          models: { asr: "standby", tts: "standby", xtts: "standby" },
        },
        { status: 200 }
      );
    }
    console.warn(
      `[Proxy Upstream ${upstreamRes.status}] ${req.method} ${subPath}: ${errText.slice(0, 500)}`
    );
    return new Response(errText, {
      status: upstreamRes.status,
      statusText: upstreamRes.statusText,
      headers: respHeaders,
    });
  }

  if (upstreamContentType?.includes("text/event-stream") && upstreamRes.body) {
    respHeaders.set("content-type", "text/event-stream; charset=utf-8");
    const reader = upstreamRes.body.getReader();
    const stream = new ReadableStream({
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
          } else {
            controller.enqueue(value);
          }
        } catch {
          controller.close();
        }
      },
      cancel() {
        reader.cancel().catch(() => {});
      },
    });
    return new Response(stream, {
      status: upstreamRes.status,
      headers: respHeaders,
    });
  }

  return new Response(upstreamRes.body, {
    status: upstreamRes.status,
    statusText: upstreamRes.statusText,
    headers: respHeaders,
  });
}

export async function GET(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleProxy(req, context);
}

export async function POST(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleProxy(req, context);
}

export async function PATCH(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleProxy(req, context);
}

export async function PUT(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleProxy(req, context);
}

export async function DELETE(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleProxy(req, context);
}
