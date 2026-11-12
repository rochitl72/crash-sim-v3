"""CrashLab Studio: local web app. Paste/upload a PC-Crash values export (+ officer notes) ->
AI scene description -> CrashLab simulation -> 3D case viewer with an AI report.

Run:  python3 studio/server.py      then open http://127.0.0.1:8765
The Groq key is read from .env (GROQ_API_KEY=...) on this machine and never sent to the browser."""
import json, os, sys, re, traceback, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
import pipeline  # noqa: E402

CASES = os.path.join(ROOT, "cases")
PORT = int(os.environ.get("PORT", "8765"))
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def log_message(self, fmt, *a):
        sys.stderr.write("[studio] " + (fmt % a) + "\n")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, open(os.path.join(HERE, "index.html"), encoding="utf-8").read())
        if p == "/api/sample":
            txt = open(os.path.join(ROOT, "samples", "pc_crash_sample_1.txt"), encoding="latin-1").read()
            return self._send(200, txt, "text/plain; charset=utf-8")
        if p == "/api/cases":
            items = []
            if os.path.isdir(CASES):
                for f in sorted(os.listdir(CASES), reverse=True)[:30]:
                    if f.endswith(".json"):
                        try:
                            items.append(json.load(open(os.path.join(CASES, f), encoding="utf-8")))
                        except Exception:
                            pass
            return self._send(200, json.dumps(items), "application/json")
        m = re.match(r"^/case/([0-9A-Za-z-]+)$", p)
        if m and os.path.exists(os.path.join(CASES, m.group(1) + ".html")):
            return self._send(200, open(os.path.join(CASES, m.group(1) + ".html"), encoding="utf-8").read())
        if p == "/demo":
            return self._send(200, open(os.path.join(ROOT, "dist", "crashlab.html"), encoding="utf-8").read())
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/reconstruct":
            return self._send(404, "not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 2_000_000:
                return self._send(413, json.dumps(dict(error="Input too large")), "application/json")
            req = json.loads(self.rfile.read(n) or b"{}")
            text, notes = str(req.get("text", "")), str(req.get("notes", ""))[:4000]
            if len(text.strip()) < 40:
                return self._send(400, json.dumps(dict(error="Paste the PC-Crash values export first.")), "application/json")
            with LOCK:
                base, pay = pipeline.run(text, notes, use_ai=bool(req.get("ai", True)))
                cid, path = pipeline.save_case(base, CASES)
            ai = pay["ai"]
            meta = dict(id=cid, title=pay["title"], subtitle=pay["subtitle"], model=ai["model"], error=ai["error"],
                        impacts=len(pay["events"]), seconds=ai["seconds"],
                        summary=(ai["report"] or {}).get("summary") if ai["report"] else None)
            json.dump(meta, open(os.path.join(CASES, cid + ".json"), "w", encoding="utf-8"))
            return self._send(200, json.dumps(meta), "application/json")
        except ValueError as e:
            return self._send(400, json.dumps(dict(error=str(e))), "application/json")
        except Exception as e:  # noqa
            traceback.print_exc()
            return self._send(500, json.dumps(dict(error=f"{type(e).__name__}: {e}")), "application/json")


def _is_studio(port):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            return b"CrashLab Studio" in r.read(4000)
    except Exception:
        return False


if __name__ == "__main__":
    srv, port = None, PORT
    for port in range(PORT, PORT + 10):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            if _is_studio(port):
                url = f"http://127.0.0.1:{port}"
                print(f"CrashLab Studio is already running at {url}. Opening it.")
                webbrowser.open(url)
                sys.exit(0)
            print(f"Port {port} is busy, trying {port + 1}...")
    if srv is None:
        sys.exit("No free port between %d and %d" % (PORT, PORT + 9))
    url = f"http://127.0.0.1:{port}"
    print(f"CrashLab Studio running at {url}  (Ctrl+C to stop)")
    if "--no-open" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
