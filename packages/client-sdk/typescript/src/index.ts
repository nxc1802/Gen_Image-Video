/**
 * Official TypeScript Client SDK for Factory v4 Control API.
 */

export interface FactoryClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
}

export interface JobSubmissionPayload {
  service: string;
  model: string;
  task: string;
  inputs: Record<string, any>;
  priority?: number;
  idempotency_key?: string;
}

export class FactoryClient {
  public readonly baseUrl: string;
  public readonly apiKey: string;
  public readonly timeout: number;

  constructor(options: FactoryClientOptions = {}) {
    this.baseUrl = (options.baseUrl || "http://localhost:8000").replace(/\/+$/, "");
    this.apiKey = (
      options.apiKey ??
      (typeof process !== "undefined" && process.env?.FACTORY_API_KEY ? process.env.FACTORY_API_KEY : "")
    ).trim();
    this.timeout = options.timeout ?? 120000;
  }

  private getHeaders(isJson = true, extraHeaders?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = {};
    if (isJson) {
      headers["Content-Type"] = "application/json";
    }
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (extraHeaders) {
      Object.assign(headers, extraHeaders);
    }
    return headers;
  }

  public async getHealth(): Promise<Record<string, any>> {
    const res = await fetch(`${this.baseUrl}/health`, {
      method: "GET",
      headers: this.getHeaders(false),
    });
    if (!res.ok) {
      throw new Error(`Health check failed with status ${res.status}: ${await res.text()}`);
    }
    return res.json();
  }

  public async getServices(): Promise<any[]> {
    const res = await fetch(`${this.baseUrl}/v1/services`, {
      method: "GET",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch services (${res.status}): ${await res.text()}`);
    }
    const data = await res.json();
    return Array.isArray(data) ? data : data.data || [];
  }

  public async getModels(): Promise<any[]> {
    const res = await fetch(`${this.baseUrl}/v1/models`, {
      method: "GET",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch models (${res.status}): ${await res.text()}`);
    }
    const data = await res.json();
    return Array.isArray(data) ? data : data.data || [];
  }

  public async getCapabilities(): Promise<any[]> {
    const res = await fetch(`${this.baseUrl}/v1/capabilities`, {
      method: "GET",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch capabilities (${res.status}): ${await res.text()}`);
    }
    const data = await res.json();
    return Array.isArray(data) ? data : data.data || [];
  }

  public async submitJob(payload: JobSubmissionPayload): Promise<Record<string, any>> {
    const res = await fetch(`${this.baseUrl}/v1/jobs`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`Job submission failed (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }

  public async getJob(jobId: string): Promise<Record<string, any>> {
    const res = await fetch(`${this.baseUrl}/v1/jobs/${jobId}`, {
      method: "GET",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to get job ${jobId} (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }

  public async listJobs(options: {
    limit?: number;
    offset?: number;
    service?: string;
    state?: string;
  } = {}): Promise<any[]> {
    const params = new URLSearchParams();
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.offset !== undefined) params.set("offset", String(options.offset));
    if (options.service) params.set("service", options.service);
    if (options.state) params.set("state", options.state);

    const query = params.toString() ? `?${params.toString()}` : "";
    const res = await fetch(`${this.baseUrl}/v1/jobs${query}`, {
      method: "GET",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to list jobs (${res.status}): ${await res.text()}`);
    }
    const data = await res.json();
    return Array.isArray(data) ? data : data.jobs || [];
  }

  public async cancelJob(jobId: string): Promise<Record<string, any>> {
    const res = await fetch(`${this.baseUrl}/v1/jobs/${jobId}/cancel`, {
      method: "POST",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to cancel job ${jobId} (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }

  public async retryJob(jobId: string): Promise<Record<string, any>> {
    const res = await fetch(`${this.baseUrl}/v1/jobs/${jobId}/retry`, {
      method: "POST",
      headers: this.getHeaders(),
    });
    if (!res.ok) {
      throw new Error(`Failed to retry job ${jobId} (${res.status}): ${await res.text()}`);
    }
    return res.json();
  }
}
