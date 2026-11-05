"""
deployments.kaggle.runner
Kaggle Dual-T4 Certification Runner Infrastructure using Kaggle REST API v1.

Features:
- Bearer token authentication support (token format: KGAT_...)
- Pushes certification kernel payload configuring dual-T4 GPU (enableGpu=True, enableInternet=True)
- Monitors execution status (queued, running, complete, error)
- Streams stdout/stderr output logs
- Clean error handling (auth, timeouts, HTTP errors, kernel failure)
"""
import argparse
import base64
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("kaggle_runner")

KAGGLE_API_BASE_URL = "https://www.kaggle.com/api/v1"


class KaggleAPIError(Exception):
    """Base exception for Kaggle REST API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KaggleAuthError(KaggleAPIError):
    """Raised when authentication with Kaggle fails (HTTP 401/403 or missing credentials)."""
    pass


class KaggleTimeoutError(KaggleAPIError):
    """Raised when kernel execution or polling times out."""
    pass


class KaggleKernelFailedError(KaggleAPIError):
    """Raised when a Kaggle kernel execution finishes in error/failed status."""
    def __init__(self, message: str, failure_message: Optional[str] = None):
        super().__init__(message)
        self.failure_message = failure_message


def resolve_kaggle_username(explicit_username: Optional[str] = None) -> str:
    """
    Dynamically resolves the Kaggle username with the following precedence:
    1. Explicitly passed parameter
    2. KAGGLE_USERNAME environment variable
    3. ~/.kaggle/kaggle.json (or $KAGGLE_CONFIG_DIR/kaggle.json) 'username' field
    Raises KaggleAuthError if username cannot be resolved.
    """
    if explicit_username and explicit_username.strip():
        return explicit_username.strip()

    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user and env_user.strip():
        return env_user.strip()

    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    kaggle_json_path = (Path(config_dir) if config_dir else Path.home() / ".kaggle") / "kaggle.json"
    if kaggle_json_path.is_file():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_user = data.get("username")
                if file_user and isinstance(file_user, str) and file_user.strip():
                    return file_user.strip()
        except Exception as e:
            logger.debug(f"Failed to read Kaggle username from {kaggle_json_path}: {e}")

    raise KaggleAuthError(
        "Kaggle username not configured. Provide via parameter, "
        "KAGGLE_USERNAME environment variable, or ~/.kaggle/kaggle.json"
    )


def resolve_kaggle_api_token(explicit_api_key: Optional[str] = None) -> str:
    """
    Dynamically resolves the Kaggle API token/key with the following precedence:
    1. Explicitly passed parameter
    2. KAGGLE_API_TOKEN environment variable (standardized)
    3. KAGGLE_KEY environment variable (legacy fallback)
    4. ~/.kaggle/kaggle.json (or $KAGGLE_CONFIG_DIR/kaggle.json) 'key' field
    Raises KaggleAuthError if token cannot be resolved.
    """
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    env_token = os.environ.get("KAGGLE_API_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    legacy_key = os.environ.get("KAGGLE_KEY")
    if legacy_key and legacy_key.strip():
        return legacy_key.strip()

    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    kaggle_json_path = (Path(config_dir) if config_dir else Path.home() / ".kaggle") / "kaggle.json"
    if kaggle_json_path.is_file():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_key = data.get("key")
                if file_key and isinstance(file_key, str) and file_key.strip():
                    return file_key.strip()
        except Exception as e:
            logger.debug(f"Failed to read Kaggle key from {kaggle_json_path}: {e}")

    raise KaggleAuthError(
        "Kaggle API key/token not configured. Provide via parameter, "
        "KAGGLE_API_TOKEN or KAGGLE_KEY environment variable, or ~/.kaggle/kaggle.json"
    )


class KaggleAPIClient:
    """Low-level REST API v1 client for Kaggle Kernels."""

    def __init__(
        self,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = KAGGLE_API_BASE_URL,
    ):
        self.username = resolve_kaggle_username(username)
        self.api_key = resolve_kaggle_api_token(api_key)
        self.base_url = base_url.rstrip("/")

    def __repr__(self) -> str:
        masked = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "***"
        return f"KaggleAPIClient(username='{self.username}', api_key='{masked}', base_url='{self.base_url}')"

    def get_auth_header(self) -> str:
        """Returns Authorization header using Bearer token for KGAT_ tokens, or Basic auth."""
        if self.api_key.startswith("KGAT_"):
            return f"Bearer {self.api_key}"
        cred = f"{self.username}:{self.api_key}".encode("utf-8")
        return f"Basic {base64.b64encode(cred).decode('ascii')}"

    def _request(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query_params:
            url += f"?{urllib.parse.urlencode(query_params)}"

        headers = {
            "Authorization": self.get_auth_header(),
            "User-Agent": "Factory-V4-Kaggle-Runner/4.0",
        }

        body_bytes = None
        if json_data is not None:
            headers["Content-Type"] = "application/json"
            body_bytes = json.dumps(json_data).encode("utf-8")

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = resp.read().decode("utf-8")
                if not resp_data.strip():
                    return {}
                return json.loads(resp_data)
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")
            if he.code in (401, 403):
                raise KaggleAuthError(
                    f"Kaggle authentication failed ({he.code}): {err_body}",
                    status_code=he.code,
                    response_body=err_body,
                )
            raise KaggleAPIError(
                f"Kaggle API error ({he.code}) on {method} {path}: {err_body}",
                status_code=he.code,
                response_body=err_body,
            )
        except urllib.error.URLError as ue:
            raise KaggleAPIError(f"Network error connecting to Kaggle API: {ue.reason}")
        except Exception as e:
            raise KaggleAPIError(f"Unexpected error calling Kaggle API: {e}")

    def push_kernel(
        self,
        slug: str,
        title: str,
        code_content: str,
        is_private: bool = True,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        kernel_type: str = "script",
        language: str = "python",
        dataset_sources: Optional[List[str]] = None,
        competition_sources: Optional[List[str]] = None,
        kernel_sources: Optional[List[str]] = None,
        model_sources: Optional[List[str]] = None,
        session_timeout_seconds: int = 7200,
    ) -> Dict[str, Any]:
        """
        Pushes kernel payload to Kaggle with Dual-T4 GPU configuration.
        POST https://www.kaggle.com/api/v1/kernels/push
        """
        payload = {
            "slug": f"{self.username}/{slug}",
            "newTitle": title,
            "text": code_content,
            "language": language,
            "kernelType": kernel_type,
            "isPrivate": is_private,
            "enableGpu": enable_gpu,
            "enableInternet": enable_internet,
            "datasetDataSources": dataset_sources or [],
            "competitionDataSources": competition_sources or [],
            "kernelDataSources": kernel_sources or [],
            "modelDataSources": model_sources or [],
            "sessionTimeoutSeconds": max(7200, session_timeout_seconds),
        }

        resp = self._request("POST", "/kernels/push", json_data=payload)
        if resp.get("hasError"):
            err_msg = resp.get("error") or resp.get("errorNullable") or "Unknown Kaggle error"
            raise KaggleAPIError(f"Kaggle API reported error pushing kernel: {err_msg}")
        return resp

    def get_kernel_status(self, slug: str) -> Dict[str, Any]:
        """
        Retrieves current execution status of kernel.
        GET https://www.kaggle.com/api/v1/kernels/status?userName={username}&kernelSlug={slug}
        """
        params = {"userName": self.username, "kernelSlug": slug}
        return self._request("GET", "/kernels/status", query_params=params)

    def get_kernel_output(self, slug: str) -> Dict[str, Any]:
        """
        Retrieves stdout/stderr and output files of kernel.
        GET https://www.kaggle.com/api/v1/kernels/output?userName={username}&kernelSlug={slug}
        """
        params = {"userName": self.username, "kernelSlug": slug}
        return self._request("GET", "/kernels/output", query_params=params)

    def poll_until_complete(
        self,
        slug: str,
        poll_interval_sec: float = 5.0,
        timeout_sec: float = 1800.0,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Monitors kernel status and streams output until completion or timeout.
        """
        start_time = time.time()
        last_logged_lines = 0
        out_resp: Dict[str, Any] = {}

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                raise KaggleTimeoutError(
                    f"Kaggle kernel '{slug}' timed out after {timeout_sec:.0f} seconds."
                )

            status_resp = self.get_kernel_status(slug)
            status = (status_resp.get("status") or "unknown").lower()
            failure_msg = status_resp.get("failureMessage")

            # Stream logs if available
            try:
                out_resp = self.get_kernel_output(slug)
                log_text = out_resp.get("log", "")
                if log_text:
                    lines = log_text.splitlines()
                    if len(lines) > last_logged_lines:
                        new_lines = lines[last_logged_lines:]
                        last_logged_lines = len(lines)
                        if output_callback:
                            for l in new_lines:
                                output_callback(l)
            except Exception:
                pass

            if status == "complete":
                return out_resp or self.get_kernel_output(slug)
            elif status in ("error", "failed", "cancelacknowledgement"):
                raise KaggleKernelFailedError(
                    f"Kaggle kernel '{slug}' failed with status '{status}'.",
                    failure_message=failure_msg,
                )

            time.sleep(poll_interval_sec)


class KaggleDualT4Runner:
    """
    High-level orchestrator for Kaggle Dual-T4 certification execution.
    Can deploy kernel script remotely to Kaggle or run direct certification locally.
    """

    def __init__(
        self,
        client: Optional[KaggleAPIClient] = None,
        workspace_dir: Optional[str | Path] = None,
        mode: str = "remote",
        **kwargs: Any,
    ):
        if isinstance(client, str) and client.lower() in ("preflight", "remote", "auto", "local"):
            mode = client
            client = None

        self.mode = mode.lower().strip() if isinstance(mode, str) else "remote"
        self._client = client
        self.workspace_dir = Path(workspace_dir or Path(__file__).resolve().parent.parent.parent)
        self.kwargs = kwargs
        self.extra_options = kwargs

    @property
    def client(self) -> KaggleAPIClient:
        if self._client is None:
            self._client = KaggleAPIClient()
        return self._client

    def prepare_kernel_script(self) -> str:
        """Reads certify_kernel.py and wraps it for standalone execution in Kaggle."""
        cert_kernel_path = self.workspace_dir / "deployments" / "kaggle" / "certify_kernel.py"
        if not cert_kernel_path.exists():
            raise FileNotFoundError(f"Certification kernel script not found at {cert_kernel_path}")
        with open(cert_kernel_path, "r", encoding="utf-8") as f:
            code = f.read()
        return code

    def deploy_and_certify(
        self,
        slug: str = "factory-dual-t4-certification",
        title: str = "Factory v4 Dual-T4 Live Certification",
        poll_interval_sec: float = 5.0,
        timeout_sec: float = 1800.0,
    ) -> Dict[str, Any]:
        """Pushes certification kernel to Kaggle and streams results."""
        code_content = self.prepare_kernel_script()

        push_resp = self.client.push_kernel(
            slug=slug,
            title=title,
            code_content=code_content,
            is_private=True,
            enable_gpu=True,
            enable_internet=True,
            kernel_type="script",
            language="python",
        )

        version_num = push_resp.get("versionNumber") or push_resp.get("versionNumberNullable")
        actual_slug = slug
        if push_resp.get("ref"):
            actual_slug = push_resp["ref"].strip("/").split("/")[-1]
        elif push_resp.get("url"):
            actual_slug = push_resp["url"].strip("/").split("/")[-1]
        logger.info(f"Pushed kernel {actual_slug} version {version_num} to Kaggle.")

        def _log_streamer(line: str):
            logger.info(f"[Kaggle Remote] {line}")

        output_resp = self.client.poll_until_complete(
            slug=actual_slug,
            poll_interval_sec=poll_interval_sec,
            timeout_sec=timeout_sec,
            output_callback=_log_streamer,
        )

        return output_resp

    def run_local_certification(self, output_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Direct local certification run using Tier1ModelCertifier in preflight contract simulation mode."""
        from deployments.kaggle.certify_kernel import Tier1ModelCertifier

        out = output_path or (self.workspace_dir / "deployments" / "kaggle" / "certification_results.json")
        certifier_mode = self.mode if self.mode in ("preflight", "live", "auto") else "preflight"
        certifier = Tier1ModelCertifier(
            manifests_dir=self.workspace_dir / "manifests",
            mode=certifier_mode,
        )
        results = certifier.execute_all(output_path=out)
        return results

    def run(
        self,
        output_path: Optional[str | Path] = None,
        slug: str = "factory-dual-t4-certification",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Unified entrypoint dispatching to local or remote certification based on self.mode."""
        if self.mode in ("preflight", "local"):
            return self.run_local_certification(output_path=output_path)
        return self.deploy_and_certify(slug=slug, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Dual-T4 Certification Runner")
    parser.add_argument("--slug", default="factory-dual-t4-certification", help="Kaggle kernel slug")
    parser.add_argument("--token", default=None, help="Kaggle API Bearer Token")
    parser.add_argument("--user", "--username", default=None, help="Kaggle username")
    parser.add_argument("--local", action="store_true", help="Run certification locally without pushing to Kaggle")
    parser.add_argument("--out", default="deployments/kaggle/certification_results.json", help="Output JSON path")
    args = parser.parse_args()

    mode = "preflight" if args.local else "remote"
    client = None
    if not args.local and (args.user or args.token):
        client = KaggleAPIClient(username=args.user, api_key=args.token)
    runner = KaggleDualT4Runner(client=client, mode=mode)

    if args.local:
        print("Running Tier 1 model certification locally...")
        res = runner.run_local_certification(output_path=args.out)
        print(f"Certification finished. Success: {res.get('all_gates_passed')}")
        sys.exit(0 if res.get("all_gates_passed") else 1)
    else:
        print(f"Deploying certification kernel to Kaggle ({args.slug})...")
        try:
            res = runner.deploy_and_certify(slug=args.slug)
            print("Kaggle certification execution completed successfully.")
        except Exception as e:
            print(f"Kaggle deployment failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
