"""CPU-only lifecycle contract fixture. Not a model/backend inference server."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys

parser = argparse.ArgumentParser()
for name in ["port", "model", "nonce", "graph", "token_file"]:
    parser.add_argument("--" + name, required=True)
parser.add_argument("--stubborn-child", action="store_true")
args = parser.parse_args()
if args.stubborn_child:
    child = subprocess.Popen([sys.executable, "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(300)"])
    print(child.pid, flush=True)
token = Path(args.token_file).read_text().strip()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer " + token:
            self.send_error(401)
            return
        body = json.dumps({"model":args.model, "boot_nonce":args.nonce, "checkpoint_graph_sha256":args.graph,
            "model_loaded":True, "contract_fixture":True, "inference_verified":False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
HTTPServer(("127.0.0.1", int(args.port)), Handler).serve_forever()
