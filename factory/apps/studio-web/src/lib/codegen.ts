// Factory Studio v4 Client-Side Code Generator
import type { CodegenResult, JobSubmissionPayload } from "@/types/studio";

export function generateClientSnippets(
  payload: JobSubmissionPayload,
  baseUrl = "https://jobs.ai.example.com",
  apiKey = "$FACTORY_API_KEY"
): CodegenResult {
  const service = payload.service || "image";
  const model = payload.model || "qwen-image-2.1";
  const task = payload.task || "text_to_image";
  const inputs = payload.inputs || {};

  const cleanBaseUrl = baseUrl.replace(/\/+$/, "");

  // 1. cURL
  const curlJson = JSON.stringify(
    {
      service,
      model,
      task,
      inputs,
    },
    null,
    2
  );
  const curl = `curl -X POST ${cleanBaseUrl}/v1/jobs \\
  -H "Authorization: Bearer ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '${curlJson}'`;

  // 2. Python httpx
  const pyInputs = JSON.stringify(inputs, null, 8).replace(
    /{\n        }/g,
    "{}"
  );
  const python = `import os
import httpx

API_KEY = os.getenv("FACTORY_API_KEY", "${apiKey.startsWith("$") ? "sk-factory-studio" : apiKey}")
BASE_URL = os.getenv("FACTORY_BASE_URL", "${cleanBaseUrl}")

with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
    response = client.post(
        "/v1/jobs",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "service": "${service}",
            "model": "${model}",
            "task": "${task}",
            "inputs": ${pyInputs.trim()},
        },
    )
    response.raise_for_status()
    job = response.json()
    print(f"Submitted Job: {job.get('job_id')} | State: {job.get('state')}")
`;

  // 3. TypeScript fetch
  const tsInputs = JSON.stringify(inputs, null, 4).replace(/{\n    }/g, "{}");
  const typescript = `const apiKey = process.env.FACTORY_API_KEY || "${apiKey.startsWith("$") ? "sk-factory-studio" : apiKey}";
const baseUrl = process.env.FACTORY_BASE_URL || "${cleanBaseUrl}";

async function submitJob() {
  const response = await fetch(\`\${baseUrl}/v1/jobs\`, {
    method: "POST",
    headers: {
      "Authorization": \`Bearer \${apiKey}\`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      service: "${service}",
      model: "${model}",
      task: "${task}",
      inputs: ${tsInputs.trim()},
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(\`Job submission failed (\${response.status}): \${errorText}\`);
  }

  const job = await response.json();
  console.log("Job Submitted Successfully:", job.job_id, "State:", job.state);
  return job;
}

submitJob().catch(console.error);
`;

  // 4. OpenAI SDK (for text/vlm subsystems)
  const promptValue =
    typeof inputs.prompt === "string"
      ? inputs.prompt
      : "Explain the architecture of Factory v4 in one paragraph.";

  const openai = `import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("FACTORY_CONTROL_URL", "${cleanBaseUrl}/v1"),
    api_key=os.getenv("FACTORY_API_KEY", "${apiKey.startsWith("$") ? "sk-factory-studio" : apiKey}"),
)

response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant running on Factory v4."},
        {"role": "user", "content": ${JSON.stringify(promptValue)}},
    ],
    stream=True,
)

for chunk in response:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
`;

  return {
    curl,
    python,
    typescript,
    openai,
  };
}
